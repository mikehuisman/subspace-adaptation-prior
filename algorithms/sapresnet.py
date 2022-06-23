from re import L
import torch
import torch.nn.functional as F
import numpy as np


from copy import deepcopy
from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task, get_info,\
                            ParamType
from .modules.similarity import gram_linear, cka


def regularizer_null(device, **kwargs):
    return torch.zeros(1, device=device)

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

def regularizer_l1(alfas, params, **kwargs):
    i = 0
    loss = None
    for als in alfas:
        if len(als) > 1:
            probs = torch.softmax(als, dim=0)[1:]
            inc = len(als)-1
        else:
            probs = torch.sigmoid(als)
            inc = 1
        
        #print("PROBS:", probs)
        
        pls = params[i: i+inc]
        #print(probs.size(), len(pls))
        for a, p in zip(probs, pls):
            # print("probability:", a)
            # print("param:", p)
            # print("penalt:y", a*torch.norm(p, p=1))
            if loss is None:
                loss = a * torch.norm(p, p=1)
            else:
                loss = loss + a*torch.norm(p, p=1)
        i += inc
    return loss



def alfa_regularizer_entropy(alfas, params, free_net=True, **kwargs):
    # Punish uncertainty -> sparsify the paths
    if free_net:
        loss = None
        for als in alfas:
            if len(als) > 1:
                probs = torch.softmax(als, dim=0)
                penalty = torch.sum(torch.log(probs) * probs)
            else:
                probs = torch.sigmoid(als)
                penalty = torch.log(probs)*probs + (1-probs)*torch.log(1-probs)
            
            if loss is None:
                loss = penalty
            else:
                loss = loss + penalty
    return loss


def weight_entropy_regularizer(alfas, params, gammas, free_net=False, **kwargs):
    if free_net:
        loss = gammas[0]*regularizer_l1(alfas, params, free_net=free_net) +\
               gammas[1]*alfa_regularizer_entropy(alfas, params, free_net=free_net) 
    return loss

def max_binary_mask(distributions, device, **kwargs):
    max_masks = []
    for distr in distributions:
        max_masks.append( torch.zeros(2,device=device).float() )
        max_index = distr.argmax(dim=0)
        max_masks[-1][max_index] = 1
    return max_masks

def gumbel_binary_mask(distributions, **kwargs):
    binary_masks = []
    for i in range(len(distributions)):
        binary_masks.append( ( F.gumbel_softmax(torch.log(1e-6 + torch.softmax(distributions[i], dim=0)), hard=True) ) )
    return binary_masks

def soft_mask(distributions, temperature, **kwargs):
    # soft mask through a softmax
    masks = []
    for i in range(len(distributions)):
        masks.append( torch.softmax(distributions[i]/temperature + 1e-12, dim=0) )
    return masks

def create_grad_mask(grad_masks, device):
    hard_masks = []
    for p in grad_masks:
        params = torch.stack([p, torch.zeros(p.size(0), device=device)],dim=1)
        logits = torch.log(torch.softmax(params,dim=1)+1e-10)
        mask = F.gumbel_softmax(logits, hard=True)
        real_mask = mask[:,0]
        hard_masks.append(real_mask)
    return hard_masks


def create_deterministic_mask(grad_masks, **kwargs):
    hard_masks = []
    for p in grad_masks:
        real_mask = (p > 0).float()
        hard_masks.append(real_mask)
    return hard_masks

class SAPResNet(Algorithm):
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
    
    def __init__(self, train_base_lr, gamma, base_lr, second_order, grad_clip=None, transform="interp", reg="num_params", 
                 meta_batch_size=1, sine_constr=None, var_updates=False, pretrain=False, freeze_init=False, freeze_transform=False, 
                 force_nopretrain=False, learn_alfas=False, exp1=False, solid=float("inf"), free_arch=False, relu=False, image=False, 
                 channel_scale=False, free_net=False, unfreeze_init=False, boil=False, svd=False, linear_transform=False, 
                 max_pool_before_transform=False, old=False, discrete_ops=False, trans_before_relu=False, train_iters=None, 
                 anneal_temp=False, soft=False, warm_start=0, tnet=False, avg_grad=False, swap_base_trans=False, 
                 use_grad_mask=False, train_all_dense=False, warpgrad=False, train_curve=False, warpgrad_optimizer=False, top=0, 
                 max_freq_train=3, xrange=8, comp_pretrain=False, freeze_pretrained=False, nn=False, use_logits=False, adapt=False, 
                 simple_linear=False, enable_sap=False, **kwargs):
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
        self.image = image # whether we are doing image classification
        self.channel_scale = channel_scale
        self.free_net = free_net
        self.unfreeze_init = unfreeze_init
        self.simple_linear = simple_linear
        self.boil = boil
        self.svd = svd
        self.soft = soft
        self.adapt = adapt
        self.enable_sap = enable_sap
        self.avg_grad = avg_grad
        if self.boil: assert self.image, "boil only supported for image"
        self.linear_transform = linear_transform
        self.max_pool_before_transform = max_pool_before_transform
        self.old = old
        self.discrete_ops = discrete_ops
        self.warm_start = warm_start
        self.use_grad_mask = use_grad_mask
        self.swap_base_trans = swap_base_trans
        self.train_all_dense = train_all_dense
        self.warpgrad = warpgrad
        assert not (self.swap_base_trans and tnet), "tnet is incompatible with swap_base_trans"
        self.training = True
        self.train_curve = train_curve
        self.warpgrad_optimizer = warpgrad_optimizer
        self.top = top
        self.max_freq_train = max_freq_train
        self.comp_pretrain = comp_pretrain
        self.xrange = xrange
        self.freeze_pretrained = freeze_pretrained
        self.nn = nn
        if self.warm_start != 0:
            self.done_warmup = False
        self.use_logits = use_logits

            
        self.trans_before_relu = trans_before_relu
        assert not (discrete_ops and not self.svd), "Discrete ops only supported when --svd is true"
        assert not (discrete_ops and not self.old), "Discrete ops only supported when --old is true"
        assert not (trans_before_relu and not self.svd), "trans before relu only implemented when --svd is true"
        assert not (trans_before_relu and not self.old), "trans before relu only implemented when --old is true"
        self.tnet = tnet

        self.train_iters = train_iters
        if not self.train_iters is None:
            self.meta_iters = self.train_iters / self.meta_batch_size
        print("Number of train_iters:", self.train_iters)
        self.temperature = 1
        self.anneal_temp = anneal_temp
        assert not (self.anneal_temp and not self.discrete_ops), "Temperature annealing only makes sense when discrete_ops is true"
        if self.anneal_temp:
            self.make_train_masks = soft_mask
        else:
            self.make_train_masks = gumbel_binary_mask

        assert not (self.soft and not self.discrete_ops), "--soft requires --discrete_ops"
        assert not (self.soft and not self.anneal_temp), "--soft requires --anneal_temp"
        if self.soft:
            self.make_test_masks = soft_mask
        else:
            self.make_test_masks = max_binary_mask

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        self.global_counter = 0
        
        # Maintain train loss history
        self.train_losses = []
        self.train_scores = []

        # Get random initialization point for baselearner
        self.baselearner_args["free_arch"] = self.free_arch
        self.baselearner_args["relu"] = relu
        self.baselearner_args["channel_scale"] = self.channel_scale
        self.baselearner_args["use_grad_mask"] = self.use_grad_mask
        self.baselearner_args["nearest_neighbor"] = self.nn
        self.baselearner_args["use_logits"] = self.use_logits
        self.baselearner_args["adapt"] = self.adapt
        self.baselearner_args["simple_linear"] = self.simple_linear
        self.baselearner_args["enable_sap"] = self.enable_sap
        if self.warpgrad:
            assert self.image, "warpgrad only implemented for image data"
            self.baselearner_args["warpgrad"] = self.warpgrad
        if self.image and self.svd:
            self.baselearner_args["max_pool_before_transform"] = max_pool_before_transform
            self.baselearner_args["linear_transform"] = linear_transform
            self.baselearner_args["old"] = self.old
            self.baselearner_args["discrete_ops"] = self.discrete_ops
            self.baselearner_args["trans_before_relu"] = self.trans_before_relu
        
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        if self.tnet:
            assert not (self.learn_alfas and self.tnet), "can't learn alfas when tnet"
            assert not (self.svd and self.tnet), "svd doesnt work with tnet"
        if self.warpgrad:
            assert not (self.learn_alfas and self.warpgrad), "can't learn alfas when warpgrad"
            assert not (self.svd and self.warpgrad), "svd doesnt work with warpgrad"


        base_params = list(self.baselearner.base_params())
        transform_params = list(self.baselearner.transform_params())
        alfas = list(self.baselearner.get_alfas())
        print("Got:", len(base_params) + len(transform_params) + len(alfas), "params")
        print("Expected:", len(list(self.baselearner.parameters())))
        self.num_real_alfas = len(alfas)

        self.alfa_mask = set() # will contain indices that will be frozen (activated/de-activated)
        self.counts = np.zeros((len(alfas)))
        if not self.free_net and not self.svd:
            self.alfa_history = [ np.array([x.item() for x in alfas]) ]
        else:
            asl = []
            for x in alfas:
                asl.append([j.item() for j in x])

            self.alfa_history = [ np.array(asl) ]
        
        if self.discrete_ops:
            distribution_history = []
            for x in self.baselearner.distributions:
                distribution_history.append([j.item() for j in x])
            self.distribution_history = [ np.array(distribution_history) ]


        if exp1:
            assert not self.learn_alfas, "alfas were not learnable in experiment 1"
            assert not self.free_arch, "exp1 is incompatible with free_arch"
            self.alfas[0] = torch.Tensor([999]).squeeze()
            self.alfas[1] = torch.Tensor([-999]).squeeze()
            self.alfas[2] = torch.Tensor([-999]).squeeze()
            self.alfas[3] = torch.Tensor([999]).squeeze()
            print("Preset activations to", [torch.sigmoid(x).item() for x in self.alfas])
            self.alfas = [p.clone().detach().to(self.dev) for p in self.alfas]

        self.regmap = {
            "num_params": regularizer_np,
            "l2": regularizer_l2,
            "null": regularizer_null,
            "l1": regularizer_l1,
            "entropy": alfa_regularizer_entropy,
            "we": weight_entropy_regularizer,
        }
        self.regularizer = self.regmap[reg]

        self.base_idx, self.transform_idx, self.alfa_idx = 0, 1, 2
        self.optimizer = self.opt_fn([
            {'params': base_params},
            {'params': transform_params},
            {'params': alfas}
        ], lr=self.lr)
        
        #print(self.optimizer.param_groups)
        for gid, group in enumerate(self.optimizer.param_groups):
            groupstr = {
                0: "base params",
                1: "transform params",
                2: "alfas"
            }
            print(f"Group {groupstr[gid]}")
            print('-'*40)
            for param in group['params']:
                print(param.size())
            print('-'*40)

        if not self.free_arch and not self.free_net and not self.svd and not self.tnet and not self.warpgrad:
            if not self.image:
                pls = np.array([p.numel() for p in self.baselearner.transform.parameters()] + [0])
                self.num_params = torch.Tensor(pls.reshape(len(pls)//2,2).sum(axis=1))
            else:
                pls = np.array([p.numel() for p in transform_params])
                self.num_params = torch.Tensor(pls.reshape(len(pls)//2,2).sum(axis=1))
        else:
            pls = np.array([p.numel() for p in transform_params])
            self.num_params = torch.Tensor(pls)


        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]  
        
        if pretrain and not force_nopretrain:
            self.pretrain()

        self.current_meta_iter = 0


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


    def pretrain_compsine(self):
        x = np.linspace(-self.xrange, self.xrange, 1028).reshape(-1, 1).astype('float32')
        ampl1, phase1 = 1, 0


        best_loss = 999
        best_init_weights = [p.clone().detach() for p in self.base_params]

        def fn(x, ampl, phase, freq):
            return ampl*np.sin(freq*x + phase)
        #import matplotlib.pyplot as plt

        y = fn(x, ampl=ampl1, phase=phase1, freq=self.max_freq_train).reshape(-1, 1).astype('float32')
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        net = self.sine_constr(**self.baselearner_args).to(self.dev)
        optim = self.opt_fn(self.base_params, lr=self.lr)

        for t in range(50000):
            indices = np.random.permutation(len(x))[:128]

            preds = net.forward_weights(x[indices], self.base_params)
            loss = net.criterion(preds, y[indices])
            loss.backward()
            optim.step()
            optim.zero_grad()

            if t % 500 == 0:
                print(f"Loss: {loss.item():.3f}")
                x_plot, y_plot, pred_plot = x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), net.forward_weights(x, self.base_params).detach().numpy().reshape(-1)
                # plt.figure()
                # plt.scatter(x_plot, y_plot, color='blue', label="ground-truth")
                # plt.scatter(x_plot, pred_plot, color='red', label='pred')
                # plt.savefig(f"plt{t}.png")
                # plt.close()

                with torch.no_grad():
                    full_preds = net.forward_weights(x, self.base_params)
                    full_loss = net.criterion(full_preds, y).item()
                    if full_loss < best_loss:
                        best_loss = full_loss
                        best_init_weights = [p.clone().detach() for p in self.base_params]

        self.base_params = [p.clone().detach() for p in best_init_weights]
        for p in self.base_params:
            p.requires_grad = True

        
        # else it is taken care of in pretrain() function
        # init params no longer adjusted, only the transform params and alfas
        adjustable_params = []
        # Initialize the meta-optimizer
        if not self.freeze_init:
            if self.learn_alfas and self.warm_start == 0:
                adjustable_params += self.alfas
                print("Added alfas to meta-learnable paramset")
        if not self.freeze_transform and self.warm_start == 0:
            adjustable_params += self.transform_params
            print("Added transform parameters to meta-learnable paramset")
            
        if self.use_grad_mask:
            adjustable_params += self.grad_masks
        if len(adjustable_params) > 0:
            self.optimizer = self.opt_fn(adjustable_params, lr=self.lr)
        
        for p in adjustable_params:
            print(p.size())


    def _forward(self, x):
        return self.baselearner.transform_forward(x, bweights=self.base_params, 
                                                  weights=self.transform_params, alfas=self.alfas)
    
    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, return_fw=False, **kwargs):
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
        learner = self.baselearner
        loss_history = None
        if not train_mode: loss_history = []

        if self.nn:
            copy_weights = [p.clone().detach() for p in self.baselearner.parameters()]     #deepcopy(self.baselearner.state_dict())
            if self.adapt:
                # let's try the version whhere we remove and recreate transform_params (but may break comp graph)
                # alternative: create new clone

                conv_supp_embeddings = self.baselearner._forward(train_x)
                supp_embeddings = self.baselearner._fc_forward(conv_supp_embeddings)
                centroids = []
                for c in range(self.baselearner.num_classes):
                    indices = train_y == c
                    centroid = supp_embeddings[indices].mean(dim=0).unsqueeze(0)
                    centroids.append(centroid)
                centroids = torch.cat(centroids, dim=0) #[num_classes, num features]

                # use centroids to initialize weights of adaptable_linear: (num_classes, in_features)
                biases = - (centroids * centroids).sum(dim=1)
                self.baselearner.adaptable_linear.weight.data = 2*centroids.data
                self.baselearner.adaptable_linear.bias.data = biases.data

                adjust_params = list(self.baselearner.adaptable_linear.parameters())
                if self.enable_sap:
                    for p in self.baselearner.transform_params():
                        adjust_params.append(p)

                for t in range(self.T):
                    if not self.enable_sap:
                        preds = self.baselearner.adaptable_linear(supp_embeddings) # supp embeddings dont change
                    else:
                        conv_supp_embeddings = self.baselearner._forward(train_x)
                        supp_embeddings = self.baselearner._fc_forward(conv_supp_embeddings)
                        preds = self.baselearner.adaptable_linear(supp_embeddings)
                        a = torch.sigmoid(self.baselearner.fin_alfa)
                        preds = (1-a)*preds + a*self.baselearner.fintransform(preds)
                        
                    loss = self.baselearner.criterion(preds, train_y)
                    grad = torch.autograd.grad(loss, adjust_params)
                    # for p, g in zip(adjust_params, grad):
                    #     if g is None:
                    #         print(p.size(), g)
                    # import sys; sys.exit()
                    for p, g in zip(adjust_params, grad):
                        p.data = p.data - self.base_lr * g

                test_preds = self.baselearner(x=train_x, y=train_y, xquery=test_x, yquery=test_y)
            else:
                test_preds = self.baselearner(x=train_x, y=train_y, xquery=test_x, yquery=test_y)

            test_loss = learner.criterion(test_preds, test_y)
            if train_mode:
                test_loss.backward()
            raw_loss = test_loss.item()

            # reset weights
            # init_grads = [p.grad for p in transform_params]
            #self.baselearner.load_state_dict(copy_weights)
            for old_weight, param in zip(copy_weights, self.baselearner.parameters()):
                param.data = old_weight.data
            if not train_mode: loss_history.append(test_loss.item())
            if return_fw:
                return test_loss, test_preds, loss_history, raw_loss, [None, None]
            return test_loss, test_preds, loss_history, raw_loss


        # let's try the version whhere we remove and recreate transform_params (but may break comp graph)
        # alternative: create new clone
        copy_weights = [p.clone().detach() for p in self.baselearner.parameters()]     #deepcopy(self.baselearner.state_dict())


        transform_params = self.optimizer.param_groups[self.transform_idx]['params']
        #print('-'*40)
        for step in range(T):     
            preds = self.baselearner(train_x)
            pred_loss = self.baselearner.criterion(preds, train_y)
            #print(pred_loss.item())
            grads = torch.autograd.grad(pred_loss, transform_params, create_graph=False, 
                                retain_graph=False, allow_unused = self.top!=0) # allow_unused = self.top != 0
            #print(grads)
            grads = list(grads)
            #before_params = [p.clone().detach() for p in transform_params]
            for p, grad in zip(transform_params,grads):
                p.data.sub_(self.base_lr*grad)
            
            # after_params = [p.clone().detach() for p in transform_params]
            # for p,v in zip(before_params, after_params):
            #     print(torch.all(p.data==v.data))
            # import sys; sys.exit()
                
        # Get and return performance on query set
        test_preds = self.baselearner(test_x)
        test_loss = learner.criterion(test_preds, test_y)
        if train_mode:
            test_loss.backward()
        
        # reset weights
        # init_grads = [p.grad for p in transform_params]
        #self.baselearner.load_state_dict(copy_weights)
        for old_weight, param in zip(copy_weights, self.baselearner.parameters()):
            param.data = old_weight.data

        # after_grads = [p.grad for p in transform_params]
        # for p,v in zip(init_grads, after_grads):
        #     print(torch.all(p==v), torch.all(p==0))
        # print('-'*40)

        #print(transform_params[0].grad)
        #import sys; sys.exit()
        
        if train_mode and T > 0:
            if self.train_curve:
                acc = accuracy(torch.argmax(test_preds, dim=1), test_y)
                self.train_losses.append(pred_loss.item())
                self.train_scores.append(acc)
            else:
                self.train_losses.append(pred_loss.item())
        
        
        
        
        raw_loss = test_loss.item()
        if not train_mode: loss_history.append(test_loss.item())
        if return_fw:
            return test_loss, test_preds, loss_history, raw_loss, [None, None]
        return test_loss, test_preds, loss_history, raw_loss
    
    def train(self, train_x, train_y, test_x, test_y, return_fw=False):
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
        self.training = True

        # Compute the test loss after a single gradient update on the support set
        if not self.freeze_init or not self.freeze_transform:
            # Put all tensors on right device
            train_x, train_y, test_x, test_y = put_on_device(
                                                self.dev,
                                                [train_x, train_y,
                                                test_x, test_y])
        

            if not return_fw:
                test_loss, preds,_,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T, return_fw=return_fw)
            else:
                test_loss, preds,_,_,fw = self._deploy(train_x, train_y, test_x, test_y, True, self.T, return_fw=return_fw)


            if return_fw:
                return fw


            if self.task_counter % self.meta_batch_size == 0: 
                # Clip gradients
                if not self.grad_clip is None:
                    for group in self.optimizer.param_groups:
                        for p in group["params"]:
                            # Enable gradient tracking for the initialization parameters
                            if p.grad is None:
                                continue
                            p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)
                            

                if not self.free_arch and not self.image and not self.free_net and not self.svd and not self.tnet and not self.warpgrad:
                    #print('here-1')
                    self.transform_params[0].grad = None # zero out grad


                # if self.current_meta_iter < self.warm_start:
                #     # Mask gradients of alfas
                #     for p in self.alfas:
                #         p.grad = None

                if self.discrete_ops:
                    #print('here1')
                    for b in [self.base_params, self.transform_params, self.alfas]:
                        for p in b:
                            if not torch.all(~torch.isnan(p.grad)):
                                p.grad = None

                # if self.tnet and not self.image:
                #     for b in [self.base_params, self.transform_params]:
                #         for p in b:
                #             print(p.grad)
                            # if not torch.all(~torch.isnan(p.grad)):
                            #     p.grad = None
                    #import sys; sys.exit()



                if self.use_grad_mask:
                    #print('here2')
                    for pid, p in enumerate(self.grad_masks):
                        # print(pid, p.grad)
                        if torch.isnan(p.grad).any():
                            print("Found NAN gradients") 
                            import sys; sys.exit()
                    # import sys;sys.exit(0)
                # for p in self.alfas[self.num_real_alfas:]:
                #     print(p, p.grad)

                #transform_params_before = [p.clone().detach() for p in self.optimizer.param_groups[self.alfa_idx]['params']]
                if self.freeze_pretrained:
                    for p in self.optimizer.param_groups[self.base_idx]['params']:
                        p.grad = None

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.task_counter = 0
                self.current_meta_iter += 1

                #transform_params_after = [p.clone().detach() for p in self.optimizer.param_groups[self.alfa_idx]['params']]

                # for p,v in zip(transform_params_after, transform_params_before):
                #     print(torch.all(p.data==v.data))
                # print('-'*40)

                if self.warm_start != 0:
                    if self.current_meta_iter >= self.warm_start and not self.freeze_transform and not self.done_warmup:
                        print("Warmstarting phase ended")
                        if not self.freeze_init:
                            adaptable_params = (self.base_params + self.alfas + self.transform_params)
                        else:
                            adaptable_params = (self.alfas + self.transform_params)
                        self.optimizer = self.opt_fn(adaptable_params, lr=self.lr)
                        self.done_warmup = True

                if self.anneal_temp:
                    self.temperature -= 1/self.meta_iters

                # if len(self.alfa_mask) != len(self.alfas) and self.solid != float("inf"):
                #     for i in range(len(self.alfas)):
                #         if self.counts[i] >= 0:
                #             # if sigmoid > 0.5 
                #             if self.alfas[i].item() >= 0:
                #                 self.counts[i] += 1
                #             elif self.alfas[i].item() < 0:
                #                 self.counts[i] = 0 # reset counter   
                #         elif self.counts[i] < 0:
                #             if self.alfas[i].item() < 0:
                #                 self.counts[i] -= 1
                #             elif self.alfas[i].item >= 0:
                #                 self.counts[i] = 0

                #         # Solidify the operation into the architecture
                #         if self.counts[i] >= self.solid or self.counts[i] <= -self.solid:
                #             self.alfa_mask.add(i)
                #             value = 999 if self.counts[i] > 0 else -999
                #             with torch.no_grad():
                #                 self.alfas[i] = self.alfas[i] - self.alfas[i] +  torch.Tensor([value]).squeeze()

                if not self.free_net and not self.svd:
                    self.alfa_history.append( np.array([x.item() for x in self.alfas]) )
                else:
                    asl = []
                    for x in self.optimizer.param_groups[self.alfa_idx]["params"]:
                        asl.append([j.item() for j in x])

                    self.alfa_history.append(np.array(asl))

                    if self.discrete_ops:
                        distribution_history = []
                        for x in self.alfas[self.num_real_alfas:]:
                            distribution_history.append([j.item() for j in x])
                        self.distribution_history.append( np.array(distribution_history) )


    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False, return_preds=False, return_fw=False,):
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
        

        if self.training:
            if self.use_grad_mask:
                for p in self.grad_masks:
                    print(p)
            self.training = False
        
        # Compute the test loss after a single gradient update on the support set
        if not return_fw:
            test_loss, preds, loss_history, raw_loss = self._deploy(train_x, train_y, test_x, test_y, False, T)
        else:
            test_loss, preds, loss_history, raw_loss, fw = self._deploy(train_x, train_y, test_x, test_y, False, T, return_fw=return_fw)
            return fw

        if self.operator == min:
            if return_preds:
                return raw_loss, loss_history, preds.detach()
            return raw_loss, loss_history
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

        return deepcopy(self.baselearner.state_dict())

    
    def load_state(self, state, reset=True, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.baselearner.load_state_dict(state)
        
        if reset:
            print("within reset")
            ignore = False
            # extract the top transformations and use them
            if self.top != 0:
                print("ALFAS:", self.alfas)
                detached = [als.detach() for als in self.alfas]
                for als in detached:
                    highest_weights = torch.argsort(als, descending=True)
                    for lid in highest_weights[self.top:]:
                        als[lid] = 0
                    for hid in highest_weights[:self.top]:
                        als[hid] = 1
                    
                    # only true for linear transform
                    if len(als) == 1 and als[0] <= 0:
                        self.baselearner.ignore_linear_transform = True
                        ignore = True
                        print("ignoring linear transform")

                self.alfas = detached

                # reset parameters of network
                self.base_params = [p.clone().detach().to(self.dev) for p in self.baselearner.model.parameters()]
                self.transform_params = [p.clone().detach().to(self.dev) for p in self.baselearner.transform.parameters()]
                for p in self.base_params: p.requires_grad = True
                for p in self.transform_params: p.requires_grad = True

                if ignore:
                    self.transform_params[-1].grad = torch.zeros_like(self.transform_params[-1], device=self.transform_params[-1].device)
                    self.transform_params[-2].grad = torch.zeros_like(self.transform_params[-2], device=self.transform_params[-2].device)

                # renew optimizer
                adjustable_params = self.base_params + self.transform_params
                self.optimizer = self.opt_fn(adjustable_params, lr=self.lr)

                print(f"TOP {self.top} transformations locked. New alfs:", self.alfas)

        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.base_params = [p.to(device) for p in self.base_params]
        self.transform_params = [p.to(device) for p in self.transform_params]
        self.alfas = [p.to(device) for p in self.alfas]