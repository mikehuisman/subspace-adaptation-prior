import torch
import numpy as np
import os
import psutil
import GPUtil as GPU

from .algorithm import Algorithm
from .modules.utils import put_on_device, set_weights, get_loss_and_grads,\
                           empty_context, accuracy, deploy_on_task, get_info
from .modules.similarity import gram_linear, cka


class MAML(Algorithm):
    """Model-Agnostic Meta-Learning
    
    Meta-learning algorithm that attempts to obtain a good common 
    initialization point (base-learner parameters) across tasks.
    From this initialization point, we want to be able to make quick
    task-specific updates to achieve good performance from just few
    data points.
    Our implementation performs a single step of gradient descent
    
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
    
    def __init__(self, train_base_lr, base_lr, second_order, grad_clip=None, meta_batch_size=1, special=False, 
                log_norm=False, random=False, var_updates=False, arch=None, trans_net=False, sign_sgd=False, 
                train_only_sign_sgd=False, avg_grad=False, train_curve=False, finite_diff=False, 
                special_opt=None, **kwargs):
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
        self.meta_batch_size = meta_batch_size
        self.special = special     
        self.log_norm = log_norm   
        self.log_test_norm = False
        self.random = random
        self.var_updates = var_updates
        self.trans_net = trans_net
        self.sign_sgd = sign_sgd
        self.train_only_sign_sgd = train_only_sign_sgd
        self.avg_grad = avg_grad
        self.measure_distances = False
        self.train_curve = train_curve
        self.finite_diff = finite_diff

        SPECIAL_OPT_FN={
            'fdtrunc': self._iterative_diff_truncated,
            'osass': self._iterative_diff_assumption,
        }

        if not special_opt is None:
            self.special_opt = SPECIAL_OPT_FN[special_opt]
        else:
            self.special_opt = None

        self.supp_dists = []
        self.query_dists = []
        

        # Increment after every train step on a single task, and update
        # init when task_counter % meta_batch_size == 0
        self.task_counter = 0 
        self.global_counter = 0
        self.log_interval = 80
        assert self.log_interval % self.meta_batch_size == 0, "log_interval must be divisble by meta_batch_size"

        self.gpu_usage = []
        self.cpu_usage = []
        self.using_gpu = ":" in self.dev
        if self.using_gpu:
            gpu_id = int(self.dev.split(":")[-1]) 
            self.gpu = GPU.getGPUs()[gpu_id]
        
        # Maintain train loss history
        self.train_losses = []
        self.train_scores = []
        if arch is not None:
            self.baselearner_args["arch"] = arch
        # Get random initialization point for baselearner
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in self.baselearner.parameters()]

        if self.special:
            self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)

        # Store gradients across tasks
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]

        # Enable gradient tracking for the initialization parameters
        for p in self.initialization:
            p.requires_grad = True
                
        # Initialize the meta-optimizer
        self.optimizer = self.opt_fn(self.initialization, lr=self.lr)
        count_params = sum([p.numel() for p in self.baselearner.parameters()])
        print("Number of parameters:", count_params)
        
        # All keys in the base learner model that have nothing to do with batch normalization
        self.keys = [k for k in self.baselearner.state_dict().keys()\
                     if not "running" in k and not "num" in k]
        
        self.bn_keys = [k for k in self.baselearner.state_dict().keys()\
                        if "running" in k or "num" in k]

        if self.log_norm:
            # wandb.init(project="MAML-norms")
            self.init_norms = []
            self.final_norms = []
            self.t_iter = 0
        
        self.test_losses = []
        self.test_norms = []  
        self.test_perfs = []    
        self.angles = []
        self.distances = []
        self.gangles = []
        self.gdistances = []      



    # Optimization functions
    ####################################################################################################
    def _iterative_diff_truncated(self, T, weights, qvector, x, y,):
        summ = qvector
        # for every term in the equation
        for t in range(T):
            # approximately differentiate w.r.t. initialization theta_0, and subtract from qvector
            diff = self._diff_truncated(t, weights, qvector, x, y, )
            if summ is None:
                summ = diff
            else:
                summ = [s - self.base_lr * d for (s,d) in zip(summ, diff)]# summ - self.base_lr*diff
        return summ


    def _diff_truncated(self, idx, weights, qvector, x, y,):
        R, plus_grads, min_grads, est_grad = self._hessian_vector_product(weights[idx], qvector, x, y,)
        if idx == 0:
            return [p/(2*R) for p in est_grad]

        # if not idx ==0, we have to subtract and add two summations from est_grad 
        plus_sum = [p-p for p in est_grad] # add plus_grads
        neg_sum = [p-p for p in est_grad] # subtract neg_grads

        for k in range(idx):
            # plus grads prime, min grads prime, est grad prime
            egp = self._diff_truncated(k, weights, plus_grads, x, y,)
            egn = self._diff_truncated(k, weights, min_grads, x, y,)

            plus_sum = [p+v for (p,v) in zip(plus_sum, egp)] 
            neg_sum = [n+v for (n,v) in zip(neg_sum,egn)]

        return [(p - self.base_lr*v + self.base_lr*n)/(2*R) for (p,v,n) in zip(est_grad, plus_sum, neg_sum)]

    def _hessian_vector_product(self, weight, vector, x, y, r=1e-2):
        # x: inputs
        # y: outputs 
        # r: the small constant in which we move the params
        # vector: gradients w.r.t. final fast weights on query set
        R = r / self._concat(vector).norm()
        
        #print("norm of qgrad:", vector.norm(), "disivor:", R)


        init_params = [p.clone().detach() for p in weight] #weight.clone().detach()
        for p in init_params: p.requires_grad = True


        temp_weights = [p + R*v for (p,v) in zip(init_params, vector)]# init_params.clone() + R*vector
        test_preds = self.baselearner.forward_weights(x, temp_weights)
        test_loss = self.baselearner.criterion(test_preds, y)
        plus_grads = torch.autograd.grad(test_loss, temp_weights)

        temp_weights = [p - R*v for (p,v) in zip(init_params, vector)]
        test_preds = self.baselearner.forward_weights(x, temp_weights)
        test_loss = self.baselearner.criterion(test_preds, y)
        min_grads = torch.autograd.grad(test_loss, temp_weights)

        return R, plus_grads, min_grads, [p-v for (p,v) in zip(plus_grads, min_grads)]#(plus_grads-min_grads)


    def _iterative_diff_assumption(self, T, weights, qvector, x, y, ):
        summ = qvector
        # for every term in the equation
        for t in range(T):
            # approximately differentiate w.r.t. initialization theta_0, and subtract from qvector
            diff = self._diff_assumption(t, weights, qvector, x, y, )
            if summ is None:
                summ = diff
            else:
                summ = [s - self.base_lr * d for (s,d) in zip(summ, diff)]
        return summ


    def _diff_assumption(self, idx, weights, qvector, x, y, ):
        R, _, _, est_grad = self._hessian_vector_product(weights[idx], qvector, x, y, )
        return [p/(2*R) for p in est_grad] #est_grad/(2*R)
    #########################################################################################








    def _forward(self, x):
        return self.baselearner.forward_weights(x, self.initialization)

    def _get_params(self):
        return [p.clone().detach() for p in self.initialization]


    def _concat(self, params):
        return torch.cat([x.view(-1) for x in params])


    def _iterative_diff(self, T, weights, qvector, input, target, inner_lr):
        summ = qvector
        # for every term in the equation
        for t in range(T):
            # approximately differentiate w.r.t. initialization theta_0, and subtract from qvector
            diff = self._diff(t, weights, qvector, input, target, inner_lr)
            if summ is None:
                summ = diff
            else:
                summ = [p-inner_lr*g for p,g in zip(summ, diff)]
        return summ

    # Approximate the terms in the expansion of theta^idx using finite diff
    # def _diff(self, idx, weights, qvector, input, target, inner_lr):
    #     R, plus_grads, min_grads, est_grad = self._hessian_vector_product(weights[idx], qvector, input, target)
    #     if idx == 0:
    #         return [p/(2*R) for p in est_grad]
        
    #     # if not idx ==0, we have to subtract and add two summations from est_grad 
    #     plus_sum = [p-p for p in est_grad] # add plus_grads
    #     neg_sum = [p-p for p in est_grad] # subtract neg_grads

    #     for k in range(idx):
    #         # plus grads prime, min grads prime, est grad prime
    #         egp = self._diff(k, weights, plus_grads, input, target, inner_lr)
    #         egn = self._diff(k, weights, min_grads, input, target, inner_lr)

    #         plus_sum = [p + v for p,v in zip(plus_sum, egp)]
    #         neg_sum = [p + v for p,v in zip(neg_sum, egn)]
        

    #     return [(v-inner_lr*w+inner_lr*z)/(2*R) for v,w,z in zip(est_grad, plus_sum, neg_sum)]

    # def _hessian_vector_product(self, weight, vector, input, target, r=1e-2):
    #     # r: the small constant in which we move the params
    #     # vector: gradients w.r.t. final fast weights on query set
    #     R = r / self._concat(vector).norm()

    #     init_params = [p.clone().detach() for p in weight]
    #     for p in init_params: p.requires_grad = True

    #     # plus weights
    #     temp_weights = [init_params[i].clone() + R*vector[i] for i in range(len(init_params))]
    #     _, plus_grads = get_loss_and_grads(self.baselearner, input, target, 
    #                                         weights=temp_weights, 
    #                                         create_graph=False,
    #                                         retain_graph=False,
    #                                         flat=False)

    #     # min weights
    #     temp_weights = [init_params[i].clone() - R*vector[i] for i in range(len(init_params))]
    #     _, min_grads = get_loss_and_grads(self.baselearner, input, target, 
    #                                         weights=temp_weights, 
    #                                         create_graph=False,
    #                                         retain_graph=False,
    #                                         flat=False)
        
    #     # return estimated gradient for alphas with respect to the gradients in vector?
    #     return R, plus_grads, min_grads, [(x-y) for x,y in zip(plus_grads, min_grads)]


    def _fast_weights(self, params, gradients, train_mode=False, freeze=False, sign_sgd=False, train_only_sign_sgd=False):
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
        
        if sign_sgd:
            if (train_mode and train_only_sign_sgd) or not train_only_sign_sgd:
                gradients = [torch.sign(x) for x in gradients]      

        fast_weights = [(params[i] - lr * gradients[i]) if ((not freeze or i >= len(gradients) - 2) and not gradients[i] is None) else params[i]\
                        for i in range(len(gradients))]
        
        return fast_weights


    def compute_distances(self, x, y, weights):
        CosineDist = 0
        EuclidDist = 1

        # get penultimate representations of inputs
        representations = self.baselearner.forward_weights(x, weights, flat=True) # flat=True to get penultimate activations
        indices = (y.float().reshape(-1,1)+1) * 1/(y.float().reshape(1,-1)+1) # entries that are 1 have the same class

        # mask for where classes are the same
        intra_indices = indices==1
        # mask for where classes are different
        inter_indices = indices!=1

        # amount of inter and intra indices
        num_inter = inter_indices.sum().item()
        num_intra = intra_indices.sum().item()

        dists = []

        for dist_type in [CosineDist, EuclidDist]:
            if dist_type == CosineDist:
                normalized_reprs = representations / representations.norm(p=2, dim=1).unsqueeze(1)
                distances = torch.matmul(normalized_reprs, normalized_reprs.T)
            else:
                distances = torch.cdist(representations, representations)

            # compute average distances
            inter_distances = distances[inter_indices].sum().item()
            intra_distances = distances[intra_indices].sum().item()

            if num_inter != 0:
                inter_distances /= num_inter
            if num_intra != 0:
                intra_distances /= num_intra
            
            dists.append(inter_distances)
            dists.append(intra_distances)
        
        inter_cosine, intra_cosine, inter_euclid, intra_euclid = dists
        return inter_cosine, intra_cosine, inter_euclid, intra_euclid

        # with torch.no_grad():
            

        #     # get pairwise distances
        #     distances = torch.cdist(representations, representations)
            

        #     # mask for where classes are the same
        #     intra_indices = indices==1
        #     # mask for where classes are different
        #     inter_indices = indices!=1
        #     # amount of inter and intra indices
        #     num_inter = inter_indices.sum().item()
        #     num_intra = intra_indices.sum().item()

        #     # compute average distances
        #     inter_distances = distances[inter_indices].sum().item()
        #     intra_distances = distances[intra_indices].sum().item()

        #     if num_inter != 0:
        #         inter_distances /= num_inter
        #     if num_intra != 0:
        #         intra_distances /= num_intra

            

        #     print(y)


        # # loop over all combinations of variables
        # for i in range(len(x)-1):
        #     for j in range(i, len(x)):
        #         euclid_dist = torch.sum((representations[i] - representations[j])**2).item()**0.5
        #         norm_i, norm_j = torch.norm(representations[i],p=2), torch.norm(representations[j],p=2)
        #         cosine_dist = (torch.sum(representations[i]*representations[j])/(norm_i * norm_j)).item()
        #         if y[i] == y[j]:
        #             intra_count += 1
        #             intra_euclid += euclid_dist
        #             intra_cosine += cosine_dist
        #         else:
        #             inter_count += 1
        #             inter_euclid += euclid_dist
        #             inter_cosine += cosine_dist
        
        # # take the average
        # if not inter_count == 0:
        #     inter_euclid /= inter_count
        #     inter_cosine /= inter_count
        # if not intra_count == 0:
        #     intra_euclid /= intra_count
        #     intra_cosine /= intra_count
        # print(inter_euclid, inter_cosine, intra_euclid, intra_cosine)
        # print(y.size())
        # return inter_euclid, inter_cosine, intra_euclid, intra_cosine


    def _deploy(self, train_x, train_y, test_x, test_y, train_mode, T, compute_cka=False, var_updates=False):
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
        if self.special and not train_mode:
            learner = self.val_learner
            fast_weights = [p.clone() for p in self.val_params]
        else:

            fast_weights = [p.clone() for p in self.initialization]   
            learner = self.baselearner


            if not train_mode and self.random:
                self.baselearner.freeze_layers(False)
                fast_weights = fast_weights[:-2] + [p.clone().detach() for p in list(self.baselearner.parameters())[-2:]]
                for p in fast_weights[-2:]:
                    p.requires_grad = True
        
        loss_history = None
        if not train_mode: loss_history = []
        if train_mode:
            # If batching episodic data, create random batches and train on them
            if self.batching_eps:
                #Create random permutation of rows in test set
                perm = torch.randperm(test_x.size()[0])
                data_x = test_x[perm]
                data_y = test_y[perm]

                # Compute batches
                batch_size = test_x.size()[0]//T
                batches_x = torch.split(data_x, batch_size)
                batches_y = torch.split(data_y, batch_size)

                batches_x = [torch.cat((train_x, x)) for x in batches_x]
                batches_y = [torch.cat((train_y, y)) for y in batches_y]

            
        if self.log_norm and not train_mode:
            loss, grads = get_loss_and_grads(learner, test_x, test_y, 
                                        weights=fast_weights, 
                                        create_graph=self.second_order,
                                        retain_graph=True,
                                        flat=False)
            init_norm = None
            with torch.no_grad():
                for p in grads:
                    if init_norm is None:
                        init_norm = torch.sum(p**2)
                    else:
                        init_norm = init_norm  + torch.sum(p**2)
                init_norm = torch.sqrt(init_norm)
        
        if not train_mode and self.log_test_norm:
            init_params = [p.clone().detach().to("cpu") for p in self.initialization]


        if not var_updates or train_mode:

            if self.measure_distances:
                supp_dists = []
                query_dists = []
                supp_dists.append(list(self.compute_distances(train_x, train_y, fast_weights)))
                query_dists.append(list(self.compute_distances(test_x, test_y, fast_weights)))
                print(supp_dists[-1])


            if self.finite_diff and train_mode:
                W_hist = [[p.clone().detach() for p in fast_weights]]
                for p in W_hist[-1]: p.requires_grad = True

            # maintain list of fast weights
            if train_mode and not self.special_opt is None:
                FLS = [[p.detach() for p in fast_weights]]

            for step in range(T):     
                if self.batching_eps and train_mode:
                    xinp, yinp = batches_x[step], batches_y[step]
                else:
                    xinp, yinp = train_x, train_y

                # if self.special and not train mode, use val_learner instead
                loss, grads = get_loss_and_grads(learner, xinp, yinp, 
                                            weights=fast_weights, 
                                            create_graph=self.second_order,
                                            retain_graph=T > 1 or self.second_order,
                                            flat=False)
                
                
                if not train_mode: 
                    loss_history.append(loss)
                    if self.log_test_norm:
                        init_norm = None
                        with torch.no_grad():
                            for p in grads:
                                if init_norm is None:
                                    init_norm = torch.sum(p**2)
                                else:
                                    init_norm = init_norm  + torch.sum(p**2)
                            init_norm = torch.sqrt(init_norm)
                        self.test_norms.append(init_norm.item())
                        self.test_losses.append(loss)


                fast_weights = self._fast_weights(params=fast_weights, gradients=grads, train_mode=train_mode, sign_sgd=self.sign_sgd, train_only_sign_sgd=self.train_only_sign_sgd)

                if train_mode and not self.special_opt is None:
                    FLS.append([p.detach() for p in fast_weights])

                if self.finite_diff and train_mode:
                    W_hist.append([p.clone().detach() for p in fast_weights])
                    for p in W_hist[-1]: p.requires_grad = True

                if self.measure_distances:
                    supp_dists.append(list(self.compute_distances(train_x, train_y, fast_weights)))
                    query_dists.append(list(self.compute_distances(test_x, test_y, fast_weights)))
            
            if self.measure_distances:
                self.supp_dists.append(supp_dists)
                self.query_dists.append(query_dists)
                print(supp_dists[-1])

        else:
            best_weights = [p.clone().detach() for p in fast_weights]
            best_acc = -1 if self.operator == max else 999
            count_not_improved = 0
            t=0
            t_max = 500
            while True:     
                xinp, yinp = train_x, train_y

                # if self.special and not train mode, use val_learner instead
                loss, grads = get_loss_and_grads(learner, xinp, yinp, 
                                            weights=fast_weights, 
                                            create_graph=False,
                                            retain_graph=False,
                                            flat=False)
                
                fast_weights = self._fast_weights(params=fast_weights, gradients=grads, train_mode=train_mode, freeze=False, sign_sgd=self.sign_sgd, train_only_sign_sgd=self.train_only_sign_sgd)
                t += 1

                if not train_mode: 
                    loss_history.append(loss)
                    if self.log_test_norm:
                        init_norm = None
                        with torch.no_grad():
                            for p in grads:
                                if init_norm is None:
                                    init_norm = torch.sum(p**2)
                                else:
                                    init_norm = init_norm  + torch.sum(p**2)
                            init_norm = torch.sqrt(init_norm)
                        self.test_norms.append(init_norm.item())
                        self.test_losses.append(loss)

                
                with torch.no_grad():
                    if self.operator == max:
                        test_preds = torch.argmax(learner.forward_weights(test_x, fast_weights), dim=1)
                        acc = accuracy(test_preds, test_y)

                        if acc <= best_acc:
                            count_not_improved += 1
                        else:
                            count_not_improved = 1
                            best_weights = [p.clone().detach() for p in fast_weights]
                            best_acc = acc
                    else:
                        test_preds = learner.forward_weights(test_x, fast_weights)
                        acc = learner.criterion(test_preds, test_y).item()
                        if acc >= best_acc - 0.1:
                            count_not_improved += 1
                        else:
                            count_not_improved = 1
                            best_weights = [p.clone().detach() for p in fast_weights]
                            best_acc = acc
                
                if count_not_improved >= 30 or t == t_max:
                    break

            fast_weights = [p.clone().detach() for p in best_weights]


        if not train_mode and self.log_test_norm:
            final_params = [p.clone().detach().to("cpu") for p in fast_weights]
            angles, distances, global_angle, global_distance = get_info(init_params, final_params)
            self.angles.append(angles)
            self.distances.append(distances)
            self.gangles.append(global_angle)
            self.gdistances.append(global_distance)


        if compute_cka:
            return fast_weights

        xinp, yinp = test_x, test_y


        if (not self.finite_diff and self.special_opt is None) or not train_mode:
            # Get and return performance on query set
            test_preds = learner.forward_weights(xinp, fast_weights)
            test_loss = learner.criterion(test_preds, yinp)
        else:
            test_preds = None
            test_loss, test_grads = get_loss_and_grads(learner, xinp, yinp, 
                                            weights=fast_weights, 
                                            create_graph=False,
                                            retain_graph=False,
                                            flat=False)

            if self.finite_diff:
                estimated_grads = self._iterative_diff(T, W_hist, test_grads, train_x, train_y, self.base_lr)
            elif not self.special_opt is None:
                estimated_grads = self.special_opt(T, FLS, test_grads, train_x, train_y, )
            for p, g in zip(self.initialization, estimated_grads):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad = p.grad + g



        if train_mode and T > 0:
            if self.train_curve:
                acc = accuracy(torch.argmax(test_preds, dim=1), yinp)
                self.train_losses.append(test_loss.item())
                self.train_scores.append(acc)
            else:
                self.train_losses.append(loss)


        if self.log_norm:
            final_norm = None
            grads = torch.autograd.grad(test_loss, fast_weights, retain_graph=True)
            with torch.no_grad():
                for p in grads:
                    if final_norm is None:
                        final_norm = torch.sum(p**2)
                    else:
                        final_norm = final_norm  + torch.sum(p**2)
                final_norm = torch.sqrt(final_norm)

                self.init_norms.append(init_norm.item())
                self.final_norms.append(final_norm.item())

                if train_mode:
                    self.t_iter += 1
                    if (self.t_iter+1) % 2500 == 0:
                        inits = np.array(self.init_norms)
                        finals = np.array(self.final_norms)
                        np.save("inits.npy", inits)
                        np.save("finals.npy", finals)
                        #table = wandb.Table(data=[[x,y] for (x,y) in zip(self.init_norms, self.final_norms)], columns = ["init_norm", "final_norm"])
                        #wandb.log({"custom": wandb.plot.scatter(table, "init_norm", "final_norm")})
                #wandb.log({"final_norm": final_norm.item(), "init_norm": init_norm.item()})

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

        # Put all tensors on right device
        train_x, train_y, test_x, test_y = put_on_device(
                                            self.dev,
                                            [train_x, train_y,
                                            test_x, test_y])
        
        # Compute the test loss after a single gradient update on the support set
        test_loss, _,_ = self._deploy(train_x, train_y, test_x, test_y, True, self.T)

        # if finite diff, grads are already set in the buffers
        if not self.finite_diff and self.special_opt is None:
            # Propagate the test loss backwards to update the initialization point
            test_loss.backward()
            
        # Clip gradients
        if not self.grad_clip is None:
            for p in self.initialization:
                p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)

        self.grad_buffer = [self.grad_buffer[i] + self.initialization[i].grad if not self.initialization[i].grad is None else self.grad_buffer[i] for i in range(len(self.initialization))]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0: 
            if self.global_counter % self.log_interval == 0 and self.using_gpu:
                self.gpu_usage.append(self.gpu.memoryUsed)
            # Copy gradients from self.grad_buffer to gradient buffers in the initialization parameters
            for i, p in enumerate(self.initialization):
                if not self.avg_grad:
                    p.grad = self.grad_buffer[i]
                else:
                    p.grad = self.grad_buffer[i]/self.meta_batch_size
            self.optimizer.step()
            
            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in self.initialization]
            self.task_counter = 0
            self.optimizer.zero_grad()

    def evaluate(self, train_x, train_y, test_x, test_y, val=True, compute_cka=False):
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
        
        if self.special:
            # copy initial weights except for bias and weight of final dense layer
            val_init = [p.clone().detach() for p in self.initialization[:-2]]
            self.val_learner.eval()
            # forces the network to get a new final layer consisting of eval_N classes
            self.val_learner.freeze_layers(freeze=False)
            newparams = [p.clone().detach() for p in self.val_learner.parameters()][-2:]

            self.val_params = val_init + newparams
            for p in self.val_params:
                p.requires_grad = True
            
            # Compute the test loss after a single gradient update on the support set
            test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)
            
        else:

            if self.test_adam:
                opt = self.opt_fn(self.baselearner.parameters(), self.lr)
                for p,q in zip(self.initialization, self.baselearner.parameters()):
                    q.data = p.data
                self.baselearner.train()
                test_acc, loss_history = deploy_on_task(
                                        model=self.baselearner, 
                                        optimizer=opt,
                                        train_x=train_x, 
                                        train_y=train_y, 
                                        test_x=test_x, 
                                        test_y=test_y, 
                                        T=T, 
                                        test_batch_size=self.test_batch_size,
                                        cpe=0.5,
                                        init_score=self.init_score,
                                        operator=self.operator        
                                    )
                return test_acc, loss_history 
            

            if compute_cka:
                _, initial_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=self.initialization)
                fast_weights = self._deploy(train_x, train_y, test_x, test_y, False, T, compute_cka=True)
                _, final_features = self.baselearner.forward_weights_get_features(torch.cat((train_x, test_x)), weights=fast_weights)
                ckas = []
                dists = []
                for features_x, features_y in zip(initial_features, final_features):
                    ckas.append( cka(gram_linear(features_x), gram_linear(features_y), debiased=True) )
                    dists.append( np.mean(np.sqrt(np.sum((features_y - features_x)**2, axis=1))) )
                # Compute the test loss after a single gradient update on the support set
                test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T)
                if self.sine:
                    return test_loss.item(), ckas, dists
                
                preds = torch.argmax(preds, dim=1)
                return accuracy(preds, test_y), ckas, dists
            



            # Compute the test loss after a single gradient update on the support set
            test_loss, preds, loss_history = self._deploy(train_x, train_y, test_x, test_y, False, T, 
                                                var_updates=(self.var_updates and not val) or (self.var_updates and self.operator==min) )


        

        if self.operator == min:
            return test_loss.item(), loss_history
        else:
            # Turn one-hot predictions into class preds
            preds = torch.argmax(preds, dim=1)
            test_acc = accuracy(preds, test_y)
            if self.log_test_norm:
                self.test_perfs.append(test_acc)
            return test_acc, loss_history
    
    def dump_state(self):
        """Return the state of the meta-learner
        
        Returns
        ----------
        initialization
            Initialization parameters
        """
        return [p.clone().detach() for p in self.initialization]
    
    def load_state(self, state, **kwargs):
        """Load the given state into the meta-learner
        
        Parameters
        ----------
        state : initialization
            Initialization parameters
        """
        
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
        
    def to(self, device):
        self.baselearner = self.baselearner.to(device)
        self.initialization = [p.to(device) for p in self.initialization]