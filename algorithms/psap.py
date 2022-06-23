import torch
import numpy as np

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task, get_info
from .modules.similarity import gram_linear, cka


def regularizer_null(**kwargs):
    return torch.Tensor([0])

def regularizer_l2(alfas, params, **kwargs):
    summ = torch.sigmoid(alfas[0]) * torch.sum(params[0]**2)
    for i in range(1, len(params)):
        summ = summ + torch.sigmoid(alfas[i//2]) * torch.sum(params[i]**2)
    return summ


# Regularizer for number of parameters
def regularizer_np(alfas, num_params, **kwargs):
    # num_params is a tensor [n1, n2, n3, n4] where ni is 
    # the num of params in layer i
    summ = torch.sigmoid(alfas[0]) * num_params[0]
    for i in range(1, len(alfas)):
        summ = summ + torch.sigmoid(alfas[i]) * num_params[i]
    return summ


class PSAP(Algorithm):
    """Structured Adaptation Prior
    
    ...

    Attributes
    ----------
    baselearner_fn : constructor function
        Constructor function for the base-learner
    baselearner_args : dict
        Dictionary of keyword arguments for the base-learner
    opt_fn : constructor function
        Constructor function for the optimizer to use
    T : int
        Number of update steps to parameters per task
    train_batch_size : int
        Indicating the size of minibatches that are sampled from meta-train tasks
    test_batch_size : int
        Size of batches to sample from meta-[val/test] tasks
    lr : float
        Learning rate for the optimizer
    validation : boolean
        Whether this model should use meta-validation
    dev : str
        Device identifier
    trainable : boolean
        Whether it can train on meta-trrain tasks
    episodic : boolean
        Whether to sample tasks or mini batches for training
        
    Methods
    -------
    train(train_x, train_y, test_x, test_y)
        Perform a single training step on a given task
    
    evaluate(train_x, train_y, test_x, test_y)
        Evaluate the performance on the given task
        
    dump_state()
        Dump the meta-learner state s.t. it can be loaded again later
        
    load_state(state)
        Set meta-learner state to provided @state 
    """
    
    def __init__(self, train_base_lr, gamma, base_lr, second_order, grad_clip=None, reg="num_params", 
                 meta_batch_size=1, sine_constr=None, var_updates=False, pretrain=False, freeze_init=False, freeze_transform=False, 
                 force_nopretrain=False, learn_alfas=False, exp1=False, solid=float("inf"), free_arch=False, trans_net=False, 
                 image=False, **kwargs):
        """Initialization of model-agnostic meta-learner
        
        Parameters
        ----------
        T_test : int
            Number of updates to make at test time
        base_lr : float
            Learning rate for the base-learner 
        second_order : boolean
            Whether to use second-order gradient information
        grad_clip : float
            Threshold for gradient value clipping
        meta_batch_size : int
            Number of tasks to compute outer-update
        **kwargs : dict
            Keyword arguments that are ignored
        """
        
        super().__init__(**kwargs)
        self.sine = False
        self.train_base_lr = train_base_lr
        self.base_lr = base_lr
        self.grad_clip = grad_clip        
        self.second_order = second_order
        self.gamma = gamma
        self.meta_batch_size = meta_batch_size 
        self.log_test_norm = False
        self.disabled = False
        self.sine_constr = sine_constr
        self.var_updates = var_updates
        self.freeze_init = freeze_init
        self.freeze_transform = freeze_transform
        self.do_pretraining = pretrain
        self.learn_alfas = learn_alfas
        self.solid = solid
        self.free_arch = free_arch
        self.trans_net = trans_net
        self.image = image


        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        self.global_counter = 0
        
        # Maintain train loss history
        self.train_losses = []

        # Get random initialization point for baselearner
        self.baselearner_args["trans_net"] = self.trans_net
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.base_params = [p.clone().detach().to(self.dev) for p in self.baselearner.model.parameters()]
        if self.trans_net:
            self.base_params += [p.clone().detach().to(self.dev) for p in self.baselearner.transform.parameters()]

        self.alfas = [p.clone().detach().to(self.dev) for p in self.baselearner.alfas.parameters()]
        self.alfa_mask = set() # will contain indices that will be frozen (activated/de-activated)
        self.counts = np.zeros((len(self.alfas)))

        if exp1:
            assert not self.learn_alfas, "alfas were not learnable in experiment 1"
            assert not self.free_arch, "exp1 is incompatible with free_arch"
            self.alfas[0] = torch.Tensor([999]).squeeze()
            self.alfas[1] = torch.Tensor([-999]).squeeze()
            self.alfas[2] = torch.Tensor([-999]).squeeze()
            self.alfas[3] = torch.Tensor([999]).squeeze()
            print("Preset activations to", [torch.sigmoid(x).item() for x in self.alfas])
            self.alfas = [p.clone().detach().to(self.dev) for p in self.alfas]
        

        for b in [self.base_params, self.alfas]:
            # Enable gradient tracking for the initialization parameters
            for p in b:
                p.requires_grad = True


        self.regmap = {
            "num_params": regularizer_np,
            "l2": regularizer_l2,
            "null": regularizer_null
        }
        self.regularizer = self.regmap[reg]

        if not self.do_pretraining:
            # else it is taken care of in pretrain() function
            adjustable_params = []
            # Initialize the meta-optimizer
            if not self.freeze_init:
                if self.learn_alfas:
                    adjustable_params += (self.base_params + self.alfas)
                    print("Alfas are learnable")
                else:
                    adjustable_params += self.base_params
            if len(adjustable_params) > 0:
                self.optimizer = self.opt_fn(adjustable_params, lr=self.lr)
        

        pls = np.array([p.numel() for p in self.baselearner.alfas.parameters()])
        self.num_params = torch.Tensor(pls)


        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]  
        
        if pretrain and not force_nopretrain and not self.image:
            self.pretrain()


    def pretrain(self):
        x = np.random.uniform(-5.0-1, 5.0+1, 1028).reshape(-1, 1).astype('float32')
        ampl1, phase1 = 1, 0


        best_loss = 999
        best_init_weights = [p.clone().detach() for p in self.base_params]

        def fn(x, ampl, phase):
            return ampl*np.sin(x + phase)
        import matplotlib.pyplot as plt

        y = fn(x, ampl=ampl1, phase=phase1).reshape(-1, 1).astype('float32')
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        net = self.sine_constr(**self.baselearner_args).to(self.dev)
        optim = self.opt_fn(self.base_params, lr=self.lr)

        for t in range(10000):
            indices = np.random.permutation(len(x))[:128]

            preds = net.forward_weights(x[indices], self.base_params)
            loss = net.criterion(preds, y[indices])
            loss.backward()
            optim.step()
            optim.zero_grad()

            if t % 500 == 0:
                print(f"Loss: {loss.item():.3f}")
                x_plot, y_plot, pred_plot = x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), net.forward_weights(x, self.base_params).detach().numpy().reshape(-1)
                plt.figure()
                plt.scatter(x_plot, y_plot, color='blue', label="ground-truth")
                plt.scatter(x_plot, pred_plot, color='red', label='pred')
                plt.savefig(f"plt{t}.png")
                plt.close()

                with torch.no_grad():
                    full_preds = net.forward_weights(x, self.base_params)
                    full_loss = net.criterion(full_preds, y).item()
                    if full_loss < best_loss:
                        best_loss = full_loss
                        best_init_weights = [p.clone().detach() for p in self.base_params]

        self.base_params = [p.clone().detach() for p in best_init_weights]
        for p in self.base_params:
            p.requires_grad = True

        
        adjustable_params = []
        # Initialize the meta-optimizer
        if not self.freeze_init:
            if self.learn_alfas:
                adjustable_params += (self.base_params + self.alfas)
                print("Alfas are learnable")
            else:
                adjustable_params += self.base_params
        if not self.freeze_transform:
            adjustable_params += self.transform_params
        if len(adjustable_params) > 0:
            self.optimizer = self.opt_fn(adjustable_params, lr=self.lr)

    def _forward(self, x):
        return self.baselearner.transform_forward(x, bweights=self.base_params, 
                                                  weights=self.alfas)

    def _fast_weights(self, params, gradients, train_mode=False, freeze=False):
        """Compute task-specific weights using the gradients
        
        Apply a single step of gradient descent using the provided gradients
        to compute task-specific, or equivalently, fast, weights.
        
        Parameters
        ----------
        params : list
            List of parameter tensors
        gradients : list
            List of torch.Tensor variables containing the gradients per layer
        """
        lr = self.base_lr if not train_mode else self.train_base_lr

        # Clip gradient values between (-10, +10)
        if not self.grad_clip is None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients]
        
        fast_weights = [(params[i] - lr * gradients[i]) if (not freeze or i >= len(gradients) - 2) else params[i]\
                        for i in range(len(gradients))] # not start at 1 in regular case
        
        return fast_weights
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, **kwargs):
        """Run DOSO on a single task to get the loss on the query set
        
        1. Evaluate the base-learner loss and gradients on the support set (train_x, train_y)
        using our initialization point.
        2. Make a single weight update based on this information.
        3. Evaluate and return the loss of the fast weights (initialization + proposed updates)
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        train_mode : boolean
            Whether we are in training mode or test mode

        Returns
        ----------
        test_loss
            Loss of the base-learner on the query set after the proposed
            one-step update
        """
        
        # if not train mode and self.special, we need to use val_params instead of self.init
        fast_transform_params = [p.clone() for p in self.alfas]
        fast_init_params = [p.clone() for p in self.base_params]

        learner = self.baselearner

        loss_history = None
        if not train_mode: loss_history = []

        if train_mode or not self.var_updates:
            for step in range(T):     
                preds = learner.transform_forward(train_x, bweights=fast_init_params, 
                                                weights=fast_transform_params)
                pred_loss = learner.criterion(preds, train_y)
                # grads = torch.autograd.grad(pred_loss, fast_transform_params, create_graph=self.second_order, 
                #                     retain_graph=T > 1 or self.second_order)
                grads = torch.autograd.grad(pred_loss, fast_init_params + fast_transform_params, create_graph=self.second_order and train_mode, 
                                    retain_graph=(T > 1 or self.second_order) and train_mode)
                grads = list(grads)
                init_grads = grads[:len(self.base_params)]
                trans_grads = grads[len(self.base_params):]
                fast_init_params = self._fast_weights(params=fast_init_params, gradients=init_grads, train_mode=train_mode)
                # fast_init_params = self._fast_weights(params=fast_init_params, gradients=init_grads, train_mode=train_mode)
        else:
            best_init_weights = [p.clone().detach() for p in fast_init_params]
            best_transform_weights = [p.clone().detach() for p in fast_transform_params]
            best_acc = 9999
            count_not_improved = 0
            t=0
            t_max = 500
            while True:
                # preds = learner.transform_forward(train_x, bweights=self.base_params, 
                #                                   weights=fast_transform_params, alfas=self.alfas, raw_activs=[torch.Tensor([x], device=self.dev).squeeze() for x in [1,0,0,1]])
                preds = learner.transform_forward(train_x, bweights=fast_init_params, 
                                                weights=fast_transform_params)
                pred_loss = learner.criterion(preds, train_y)
                # grads = torch.autograd.grad(pred_loss, fast_transform_params, create_graph=self.second_order, 
                #                     retain_graph=T > 1 or self.second_order)
                grads = torch.autograd.grad(pred_loss, fast_init_params + fast_transform_params, create_graph=False, 
                                    retain_graph=False)
                grads = list(grads)
                init_grads = grads[:len(self.base_params)]
                trans_grads = grads[len(self.base_params):]
                fast_init_params = self._fast_weights(params=fast_init_params, gradients=init_grads, train_mode=train_mode)
                # fast_init_params = self._fast_weights(params=fast_init_params, gradients=init_grads, train_mode=train_mode)
                # print(trans_grads[1], trans_grads[-1])

                t+=1
                with torch.no_grad():
                    test_preds = learner.transform_forward(test_x, bweights=fast_init_params, 
                                                weights=fast_transform_params)
                    acc = learner.criterion(test_preds, test_y).item()
                    #print(pred_loss.item(), acc, best_acc)
                    if acc >= best_acc - 0.1:
                        count_not_improved += 1
                    else:
                        count_not_improved = 1
                        best_init_weights = [p.clone().detach() for p in fast_init_params]
                        best_transform_weights = [p.clone().detach() for p in fast_transform_params]
                        best_acc = acc
                
                if count_not_improved >= 30 or t == t_max: # or
                    #print(count_not_improved >= 30, t == t_max)
                    #print(f"made {t} updates")
                    break
                
            fast_init_params = [p.clone().detach() for p in best_init_weights]
            fast_transform_params = [p.clone().detach() for p in best_transform_weights]

        

        if train_mode and T > 0:
            self.train_losses.append(pred_loss.item())


        # Get and return performance on query set
        # test_preds = learner.transform_forward(test_x, bweights=self.base_params, 
        #                                       weights=fast_transform_params, alfas=self.alfas)
        test_preds = learner.transform_forward(test_x, bweights=fast_init_params, 
                                              weights=fast_transform_params)
        test_loss = learner.criterion(test_preds, test_y)
        test_loss = test_loss #+ self.gamma * self.regularizer(alfas=self.alfas, num_params=self.num_params, params=fast_transform_params)

        if not train_mode: loss_history.append(test_loss.item())
        return test_loss, test_preds, loss_history
    
    def train(self, train_x, train_y, test_x, test_y):
        """Train on a given task
        
        Start with the common initialization point and perform a few
        steps of gradient descent from there using the support set
        (rain_x, train_y). Observe the error on the query set and 
        propagate the loss backwards to update the initialization.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """ 
        
        # Put baselearner in training mode
        self.baselearner.train()
        self.task_counter += 1
        self.global_counter += 1

        # Compute the test loss after a single gradient update on the support set
        if not self.freeze_init or not self.freeze_transform:
            # Put all tensors on right device
            train_x, train_y, test_x, test_y = put_on_device(
                                                self.dev,
                                                [train_x, train_y,
                                                test_x, test_y])
        
        
            test_loss, _,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)

            # Propagate the test loss backwards to update the initialization point
            test_loss.backward()
                
            # Clip gradients
            if not self.grad_clip is None:
                for b in [self.base_params, self.alfas]:
                    # Enable gradient tracking for the initialization parameters
                    for p in b:
                        p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)
                        if p.grad is None:
                            continue

            
            if self.task_counter % self.meta_batch_size == 0: 
                # for p in self.base_params:
                #     print(p.grad)
                # for p in self.alfas:
                #     print(p.grad)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.task_counter = 0


    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False, return_preds=False):
        """Evaluate on a given task
        
        Use the support set (train_x, train_y) and/or 
        the query set (test_x, test_y) to evaluate the performance of 
        the model.
        
        Parameters
        ----------
        train_x : torch.Tensor
            Inputs of the support set
        train_y : torch.Tensor
            Outputs of the support set
        test_x : torch.Tensor
            Inputs of the query set
        test_y : torch.Tensor
            Outputs of the query set
        """
        loss_history = []
        # Put baselearner in evaluation mode
        self.baselearner.eval()
        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test
        

        # Compute the test loss after a single gradient update on the support set
        test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)


        if self.operator == min:
            if return_preds:
                return test_loss.item(), loss_history, preds.detach()
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            test_acc = accuracy(preds, test_y)
            if return_preds:
                return test_acc, loss_history, preds.detach()
            return test_acc, loss_history
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.base_params],\
               [p.clone().detach() for p in self.alfas],\
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        base, alfas = state
        self.base_params = state[0]
        self.alfas = state[1]

        for s in ["base_params", "alfas"]:
            for p in eval(f"self.{s}"):
                p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.base_params = [p.to(device) for p in self.base_params]
        self.alfas = [p.to(device) for p in alfas]