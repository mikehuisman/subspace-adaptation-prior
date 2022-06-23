import pdb
import copy
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

# Used to decouple direction and norm of tensors which allow us
# to update only the direction (which is what we want for baseline++)
from torch.nn.utils.weight_norm import WeightNorm
from functools import partial
from algorithms.modules.utils import ParamType

from collections import OrderedDict


class LinearNet(nn.Module):

    def __init__(self, criterion, **kwargs):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.criterion = criterion
        self.pow = 2

    
    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """
        pred = self.coeff*torch.pow(x,self.pow) + self.bias
        return pred


    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        pred = weights[0]*torch.pow(x,self.pow) + weights[1]
        return pred
    
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI): return
    
    def freeze_layers(self): return


def identity(x, **kwargs):
    return x

def relu_fn(x, f):
    return f(x)

class TransformSineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, relu=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.free_arch = free_arch
        self.transform_type = transform
        self.relu = nn.ReLU()
        self.z = identity if not relu else partial(relu_fn, f=self.relu)
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })

        if not self.free_arch:
            use_final_bias = False
            num_alfa = 4
        else:
            use_final_bias = True
            num_alfa = 8
        
        self.transform = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('tinput', nn.Linear(in_dim, in_dim)),
            ('tlin1', nn.Linear(40, 40)),
            ('tlin2', nn.Linear(40, 40)),
            ('tlin3', nn.Linear(1,1,bias=use_final_bias))]))
        })

        # Initialize to leave original inputs unaffected
        nn.init.ones_(self.transform.features.tinput.weight)
        nn.init.eye_(self.transform.features.tlin1.weight)
        nn.init.eye_(self.transform.features.tlin2.weight)
        nn.init.ones_(self.transform.features.tlin3.weight)

        self.alfas = [nn.Parameter(torch.zeros(1).squeeze()) for _ in range(num_alfa)]
        self.alfas = nn.ParameterList(self.alfas)
        
        
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

            for m in self.transform.modules():
                if isinstance(m, nn.Linear):
                    if not m.bias is None:
                        m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, raw_activs=None, **kwargs):
        # Transform the input
        if raw_activs is not None:
            activs = raw_activs
        else:
            if alfas is None:
                activs = [torch.sigmoid(a) for a in self.alfas]
            else:
                activs = [torch.sigmoid(a) for a in alfas] 

        if not self.free_arch:
            if weights is None:
                if self.transform_type == "interp":
                    x = (1 - activs[0])*x + activs[0]*self.transform.features.tinput(x)
                    # Regular first layer
                    x = self.model.features.relu1(F.linear(x, bweights[0], bweights[1]))
                    # Transform representation of first layer
                    x = (1-activs[1])*x + activs[1]*self.transform.features.tlin1(x)
                    # Second layer
                    x = self.model.features.relu2(F.linear(x, bweights[2], bweights[3]))
                    # Transform pen-ultimate layer representation
                    x = (1-activs[2])*x + activs[2]*self.transform.features.tlin2(x)
                else:
                    x = activs[0]*self.transform.features.tinput(x)
                    # Regular first layer
                    x = self.model.features.relu1(F.linear(x, bweights[0], bweights[1]))
                    # Transform representation of first layer
                    x = activs[1]*self.transform.features.tlin1(x)
                    # Second layer
                    x = self.model.features.relu2(F.linear(x, bweights[2], bweights[3]))
                    # Transform pen-ultimate layer representation
                    x = activs[2]*self.transform.features.tlin2(x)

            else:
                if self.transform_type == "interp":
                    x = (1 - activs[0])*x + activs[0]*(F.linear(x, weights[0], weights[1]))
                    # Regular first layer
                    x = self.model.features.relu1(F.linear(x, bweights[0], bweights[1]))
                    # Transform representation of first layer
                    x = (1-activs[1])*x + activs[1]*F.linear(x, weights[2], weights[3])
                    # Second layer
                    x = self.model.features.relu2(F.linear(x, bweights[2], bweights[3]))
                    # Transform pen-ultimate layer representation
                    x = (1-activs[2])*x + activs[2]*F.linear(x, weights[4], weights[5])
                else:
                    x = activs[0]*(F.linear(x, weights[0], weights[1]))
                    # Regular first layer
                    x = self.model.features.relu1(F.linear(x, bweights[0], bweights[1]))
                    # Transform representation of first layer
                    x = activs[1]*F.linear(x, weights[2], weights[3])
                    # Second layer
                    x = self.model.features.relu2(F.linear(x, bweights[2], bweights[3]))
                    # Transform pen-ultimate layer representation
                    x = activs[2]*F.linear(x, weights[4], weights[5])

            out = F.linear(x, bweights[4], bweights[5])
            if not weights is None:
                if self.transform_type == "interp":
                    out = (1-activs[3])*out + activs[3]*F.linear(out, weights[6], None)
                else:
                    out = activs[3]*F.linear(out, weights[6], None)
        else:
            if weights is None:
                print("code not implemented, line 279 in networks.py")
                import sys; sys.exit()
            else:
                if self.transform_type == "interp":
                    x = (1 - activs[0])*x + activs[0]*(F.linear(x, weights[0], bias=None))
                    x = (1 - activs[1])*x + activs[1]*(x + weights[1])

                    # Regular first layer
                    x = self.model.features.relu1(F.linear(x, bweights[0], bweights[1]))

                    # Transform representation of first layer
                    x = (1-activs[2])*x + activs[2]*F.linear(x, weights[2], bias=None)
                    x = (1-activs[3])*x + activs[3]*self.z(x+weights[3])


                    # Second layer
                    x = self.model.features.relu2(F.linear(x, bweights[2], bweights[3]))


                    # Transform pen-ultimate layer representation
                    x = (1-activs[4])*x + activs[4]*F.linear(x, weights[4], bias=None)
                    x = (1-activs[5])*x + activs[5]*self.z(x+weights[5])

                    out = F.linear(x, bweights[4], bweights[5])
                    if not weights is None:
                        if self.transform_type == "interp":
                            out = (1-activs[6])*out + activs[6]*F.linear(out, weights[6], bias=None)
                            out = (1-activs[7])*out + activs[7]*(out + weights[7])
        
        return out
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


# Experimental free transform network

class FreeTransformSineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, relu=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.free_arch = free_arch
        self.transform_type = transform
        self.relu = nn.ReLU()
        self.z = identity if not relu else partial(relu_fn, f=self.relu)
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        layers = 3

        # there are 3 layers (counting the output layer too) so we need 


        tparams = []
        alfas = []
        for layer in range(layers+1):
            indim = 1 if layer == 0 or layer==layers else 40
            if indim == 1:
                tparams.append( nn.Parameter(torch.ones(1)) ) # single scalar multiplication -- with *
                tparams.append( nn.Parameter(torch.zeros(1)) ) # single constant shift -- +
                alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this
                alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this 
            else:
                # For multiplication
                tparams.append( nn.Parameter(torch.ones(1)) ) # single scalar multiplication -- with *
                tparams.append( nn.Parameter(torch.ones(indim)) ) # vector multiplication -- with *
                tparams.append( nn.Parameter(torch.ones(indim, indim)) )
                nn.init.eye_(tparams[-1])  

                # For shifting
                tparams.append( nn.Parameter(torch.zeros(1)) ) # single constant shift -- +
                tparams.append( nn.Parameter(torch.zeros(indim)) ) # vector shift   -- +   

                # add alfas
                alfas.append( nn.Parameter(torch.zeros(4)) ) # unnormalized activation strengths for the original + transformations 
                alfas.append( nn.Parameter(torch.zeros(3)) ) # unnormalized activation strengths for the original + transformations 




        self.transform = nn.ParameterList(tparams)
        self.alfas = nn.ParameterList(alfas)
        
        
        
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))


    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    # def transform_forward_measure_effect(self, x, bweights=None, weights=None, alfas=None, **kwargs):
    #     # Multiplication:
    #     # 0: original
    #     # 1: single scalar multiplication
    #     # 2: vector multiplication  
    #     # 3: matrix multiplication 

    #     # Shifting
    #     # 0: original
    #     # 1: constant shift
    #     # 2: vector shift 

    #     s = nn.Softmax()
    #     activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]
    #     effects = []; neffects = []

    #     # Input transform
    #     # 0: original
    #     # 1: single scalar multiplication
    #     # 2: constant shift

    #     x1 = (1-activs[0])*x + activs[0]*x*weights[0]
    #     eo = ((1-activs[0])*x/(x1+1e-6)).mean().item(); es = 1 - eo
    #     effects.append([eo, es])

    #     norm = x1.norm(p=2, dim=1)**2 + 1e-6 # to prevent zero-division
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = (((1-activs[0])*x*x1).sum(dim=1)/norm).mean().item(); ns = 1 - no
    #     neffects.append([no, ns])



    #     x2 = (1-activs[1])*x1 + activs[1]*(x1+weights[1])
    #     eo = ((1-activs[1])*x1/x2).mean().item(); eb = 1 - eo
    #     effects.append([eo, eb])

    #     norm = x2.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = (((1-activs[1])*x1*x2).sum(dim=1)/norm).mean().item(); nb = 1 - no
    #     neffects.append([no, nb])
        
        
    #     # Apply regular layer 1
    #     x3 = F.relu(F.linear(x2, bweights[0], bweights[1]))
    #     # Transforms layer 1
    #     a = activs[2] # [4,]
    #     x4 = a[0]*x3 + a[1]*x3*weights[2] + a[2]*x3*weights[3] + a[3]*torch.matmul(x3, weights[4])
    #     eo = (a[0]*x3/x4).mean().item(); em1 = (a[1]*x3*weights[2]/x4).mean().item(); em2 = (a[2]*x3*weights[3]/x4).mean().item(); em3 = 1 - eo - em1 - em2
    #     effects.append([eo, em1, em2, em3])

    #     norm = x4.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = ((a[0]*x3*x4).sum(dim=1)/norm).mean().item(); nm1=((a[1]*x3*weights[2]*x4).sum(dim=1)/norm).mean().item();   
    #     nm2 = ((a[2]*x3*weights[3]*x4).sum(dim=1)/norm).mean().item(); nm3 = 1 - no -nm1 - nm2
    #     neffects.append([no, nm1, nm2, nm3])


    #     a = activs[3]
    #     x5 = a[0]*x4 + a[1]*(x4+weights[5]) + a[2]*(x4+weights[6])
    #     eo = (a[0]*x4/x5).mean().item(); eb1 = (a[1]*(x4+weights[5])/x5).mean().item(); eb2 = 1 - eo - eb1
    #     effects.append([eo, eb1, eb2])

    #     norm = x5.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = ((a[0]*x4*x5).sum(dim=1)/norm).mean().item(); nb1=((a[1]*(x4+weights[5])*x5).sum(dim=1)/norm).mean().item();   
    #     nb2 = 1 - no -nb1
    #     neffects.append([no, nb1, nb2])




    #     # Apply regular layer 2
    #     x6 = F.relu(F.linear(x5, bweights[2], bweights[3]))
    #     # Transforms layer 2
    #     a = activs[4] # [4,]
    #     x7 = a[0]*x6 + a[1]*x6*weights[7] + a[2]*x6*weights[8] + a[3]*torch.matmul(x6, weights[9])
    #     eo = (a[0]*x6/x7).mean().item(); em1 = (a[1]*x6*weights[7]/x7).mean().item(); em2 = (a[2]*x6*weights[8]/x7).mean().item(); em3 = 1 - eo - em1 - em2
    #     effects.append([eo, em1, em2, em3])

    #     norm = x7.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = ((a[0]*x6*x7).sum(dim=1)/norm).mean().item(); nm1=((a[1]*x6*weights[7]*x7).sum(dim=1)/norm).mean().item();   
    #     nm2 = ((a[2]*x6*weights[8]*x7).sum(dim=1)/norm).mean().item(); nm3 = 1 - no -nm1 - nm2
    #     neffects.append([no, nm1, nm2, nm3])



    #     a = activs[5]
    #     x8 = a[0]*x7 + a[1]*(x7+weights[10]) + a[2]*(x7+weights[11])
    #     eo = (a[0]*x7/x8).mean().item(); eb1 = (a[1]*(x7+weights[10])/x8).mean().item(); eb2 = 1 - eo - eb1
    #     effects.append([eo, eb1, eb2])

    #     norm = x8.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = ((a[0]*x7*x8).sum(dim=1)/norm).mean().item(); nb1=(((a[1]*(x7+weights[10]))*x8).sum(dim=1)/norm).mean().item();   
    #     nb2 = 1 - no -nb1
    #     neffects.append([no, nb1, nb2])



    #     # Apply original output layer
    #     x9 = F.linear(x8, bweights[4], bweights[5])
    #     # Output transforms
    #     x10 = (1-activs[6])*x9 + activs[6]*x9*weights[12]
    #     eo = ((1-activs[6])*x9/x10).mean().item(); em = 1-eo
    #     effects.append([eo, em])

    #     norm = x10.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = (((1-activs[6])*x9*x10).sum(dim=1)/norm).mean().item(); nm = 1 - no
    #     neffects.append([no, nm])


    #     x11 = (1-activs[7])*x10 + activs[7]*(x10+weights[13])
    #     eo = ((1-activs[7])*x10/x11).mean().item(); eb = 1-eo
    #     effects.append([eo, eb])

    #     norm = x11.norm(p=2, dim=1)**2
    #     # dot product/norm = contribution. Take mean of contributions over inputs (marginalization)
    #     no = (((1-activs[7])*x10*x11).sum(dim=1)/norm).mean().item(); nb = 1 - no
    #     neffects.append([no, nb])

    #     return x, effects, neffects

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]

        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift
        ALL_EFFECTS = []


        parts = [(1-activs[0])*x, activs[0]*x*weights[0]]
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        ALL_EFFECTS.append(effects)
        

        parts = [(1-activs[1])*x, activs[1]*(x+weights[1])]
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        ALL_EFFECTS.append(effects)
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1]))
        # Transforms layer 1
        a = activs[2] # [4,]



        #x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4])
        parts = [ a[0]*x, a[1]*x*weights[2], a[2]*x*weights[3], a[3]*torch.matmul(x, weights[4])]
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        ALL_EFFECTS.append(effects)




        a = activs[3]
        #x = a[0]*x + a[1]*(x+weights[5]) + a[2]*(x+weights[6])
        parts = [a[0]*x, a[1]*(x+weights[5]), a[2]*(x+weights[6])]
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        ALL_EFFECTS.append(effects)

        

        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3]))
        # Transforms layer 2
        a = activs[4] # [4,]
        #x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9])
        parts = [a[0]*x, a[1]*x*weights[7], a[2]*x*weights[8], a[3]*torch.matmul(x, weights[9])]
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        ALL_EFFECTS.append(effects)


        a = activs[5]
        x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        x = (1-activs[6])*x + activs[6]*x*weights[12]
        x = (1-activs[7])*x + activs[7]*(x+weights[13])
        return x


    def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]

        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift

        x = (1-activs[0])*x + activs[0]*x*weights[0] 
        x = (1-activs[1])*x + activs[1]*(x+weights[1])
        
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1]))
        # Transforms layer 1
        a = activs[2] # [4,]
        x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4])
        a = activs[3]
        x = a[0]*x + a[1]*(x+weights[5]) + a[2]*(x+weights[6])

        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3]))
        # Transforms layer 2
        a = activs[4] # [4,]
        x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9])
        a = activs[5]
        x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        x = (1-activs[6])*x + activs[6]*x*weights[12]
        x = (1-activs[7])*x + activs[7]*(x+weights[13])
        return x
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


class FreeTransformSineNetworkSVD(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, relu=False, use_grad_mask=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.free_arch = free_arch
        self.transform_type = transform
        self.relu = nn.ReLU()
        self.z = identity if not relu else partial(relu_fn, f=self.relu)
        self.use_grad_mask = use_grad_mask
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        layers = 3

        # there are 3 layers (counting the output layer too) so we need 


        tparams = []
        alfas = []
        if self.use_grad_mask:
            grad_masks = []
            out_dims = [1, 40, 40, 1]
            for lid in range(layers+1):
                outdim = out_dims[lid]
                mask = nn.Parameter(torch.zeros(outdim))
                mask.requires_grad=True
                grad_masks.append(mask)
            self.grad_masks = nn.ParameterList(grad_masks)
            self.pid_to_lid = []
            self.param_types = [] 

        for layer in range(layers+1):
            indim = 1 if layer == 0 or layer==layers else 40
            if indim == 1:
                tparams.append( nn.Parameter(torch.ones(1)) ) # single scalar multiplication -- with *
                tparams.append( nn.Parameter(torch.zeros(1)) ) # single constant shift -- +
                alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this
                alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this 
                if self.use_grad_mask:
                    self.param_types += [ParamType.Scalar, ParamType.Scalar]
                    self.pid_to_lid += [layer, layer]
                    # if layer == layers:
                    #     self.param_types += [ParamType.Scalar, ParamType.Scalar]
                    #     self.pid_to_lid += [layer-1, layer-1]
                    # else:
                    #     self.param_types += [None, None]
                    #     self.pid_to_lid += [None, None]
            else:
                # For multiplication
                tparams.append( nn.Parameter(torch.ones(1)) ) # single scalar multiplication -- with *
                tparams.append( nn.Parameter(torch.ones(indim)) ) # vector multiplication -- with *
                tparams.append( nn.Parameter(torch.ones(indim, indim)) )
                nn.init.eye_(tparams[-1])  

                # For shifting
                tparams.append( nn.Parameter(torch.zeros(1)) ) # single constant shift -- +
                tparams.append( nn.Parameter(torch.zeros(indim)) ) # vector shift   -- +   

                # add alfas
                if layer == 1 or layer == 2:
                    alfas.append( nn.Parameter(torch.zeros(7)) ) # for 3 different k approximators
                else:
                    alfas.append( nn.Parameter(torch.zeros(4)) ) # unnormalized activation strengths for the original + transformations
                alfas.append( nn.Parameter(torch.zeros(3)) ) # unnormalized activation strengths for the original + transformations 
                if self.use_grad_mask:
                    self.param_types += [ParamType.Scalar, ParamType.Vector, ParamType.Matrix, ParamType.Scalar, ParamType.Vector]
                    self.pid_to_lid += [layer for _ in range(5)]




        for j in range(2):
            # bound = 1/(indim**0.5)
            for k in [5, 10, 15]:
                U = nn.Parameter(torch.ones(40, k))
                sigma = nn.Parameter( torch.ones(k) )
                V = nn.Parameter(torch.ones(40, k))

                nn.init.eye_(U); nn.init.eye_(V)
                # torch.nn.init.uniform_(U, -math.sqrt(bound/k), math.sqrt(bound/k))
                # torch.nn.init.uniform_(V, -math.sqrt(bound/k), math.sqrt(bound/k))
                tparams.append(U); tparams.append(sigma); tparams.append(V)

                if self.use_grad_mask:
                    self.param_types += [ParamType.SVD_U, ParamType.SVD_S, ParamType.SVD_V]
                    self.pid_to_lid += [j+1 for _ in range(3)]



        self.transform = nn.ParameterList(tparams)
        self.alfas = nn.ParameterList(alfas)
        
        
        
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))


    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]
        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-18:-9]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)



        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift

        x = (1-activs[0])*x + activs[0]*x*weights[0] 
        x = (1-activs[1])*x + activs[1]*(x+weights[1])
        
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1])) # dim is now 40
        # Transforms layer 1
        a = activs[2] # [4,]
        x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4]) +\
            a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        a = activs[3]
        x = a[0]*x + a[1]*(x+weights[5]) + a[2]*(x+weights[6])


        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-9:]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)


        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3])) # dim is still 40 
        # Transforms layer 2
        a = activs[4] # [4,]
        x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9]) +\
            a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        a = activs[5]
        x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        x = (1-activs[6])*x + activs[6]*x*weights[12]
        x = (1-activs[7])*x + activs[7]*(x+weights[13])

        return x




    def transform_forward_measure_effects(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]
        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-18:-9]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)

        ALL_EFFECTS = []
        MAGNITUDES = []
        PARTS = []



        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift

        #  x = (1-activs[0])*x + activs[0]*x*weights[0] 
        parts = [(1-activs[1])*(1-activs[0])*x, (1-activs[1])*activs[0]*x*weights[0]]
        subx = sum(parts)/(1-activs[1])
        parts.append(activs[1]*(subx+weights[1]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)**2
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        # x = (1-activs[1])*x + activs[1]*(x+weights[1])
        # parts = [(1-activs[1])*x, activs[1]*(x+weights[1])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)
        
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1])) # dim is now 40
        # Transforms layer 1
        a = activs[2] # [4,]
        # x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[3][0]*a[0]*x, activs[3][0]*a[1]*x*weights[2], activs[3][0]*a[2]*x*weights[3], activs[3][0]*a[3]*torch.matmul(x, weights[4]),
            activs[3][0]*a[4]*torch.matmul(x, SM1), activs[3][0]*a[5]*torch.matmul(x, SM2), activs[3][0]*a[6]*torch.matmul(x, SM3)]
        subx = sum(parts)/activs[3][0]
        a = activs[3]
        #x = a[0]*x, a[1]*(x+weights[5]), a[2]*(x+weights[6])
        parts.append(a[1]*(subx+weights[5]))
        parts.append(a[2]*(subx+weights[6]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        
        
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)


        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-9:]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)


        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3])) # dim is still 40 
        # Transforms layer 2
        a = activs[4] # [4,]
        # x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[5][0]*a[0]*x, activs[5][0]*a[1]*x*weights[7], activs[5][0]*a[2]*x*weights[8], activs[5][0]*a[3]*torch.matmul(x, weights[9]),
                activs[5][0]*a[4]*torch.matmul(x, SM1), activs[5][0]*a[5]*torch.matmul(x, SM2), activs[5][0]*a[6]*torch.matmul(x, SM3)] 
        subx = sum(parts)/activs[5][0]
        a = activs[5]
        parts.append(a[1]*(subx+weights[10]))
        parts.append(a[2]*(subx+weights[11]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])


        
        
        # x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])
        # parts = [a[0]*x, a[1]*(x+weights[10]), a[2]*(x+weights[11])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        #x = (1-activs[6])*x + activs[6]*x*weights[12]
        parts = [(1-activs[7])*(1-activs[6])*x, (1-activs[7])*activs[6]*x*weights[12]]
        subx = sum(parts)/((1-activs[7]))
        parts.append(activs[7]*(subx+weights[13]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        #x = (1-activs[7])*x + activs[7]*(x+weights[13])
        # parts = [(1-activs[7])*x, activs[7]*(x+weights[13])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        return x, PARTS



    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()









class CompositionalSineNetworkSVD(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, relu=False, use_grad_mask=False, n_components=2, simple=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.free_arch = free_arch
        self.transform_type = transform
        self.relu = nn.ReLU()
        self.z = identity if not relu else partial(relu_fn, f=self.relu)
        self.use_grad_mask = use_grad_mask
        self.n_components = n_components
        self.simple = simple
        print("SIMPLE:", self.simple)
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        layers = 3

        # there are 3 layers (counting the output layer too) so we need 


        tparams = []
        alfas = []


        for n in range(n_components):
            if self.use_grad_mask:
                grad_masks = []
                out_dims = [1, 40, 40, 1]
                for lid in range(layers+1):
                    outdim = out_dims[lid]
                    mask = nn.Parameter(torch.zeros(outdim))
                    mask.requires_grad=True
                    grad_masks.append(mask)
                self.grad_masks = nn.ParameterList(grad_masks)
                self.pid_to_lid = []
                self.param_types = [] 

            for layer in range(layers+1):
                indim = 1 if layer == 0 or layer==layers else 40
                if indim == 1:
                    tparams.append( nn.Parameter(torch.rand(1)) ) # single scalar multiplication -- with *
                    tparams.append( nn.Parameter(torch.rand(1)) ) # single constant shift -- +
                    alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this
                    alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this 
                    if self.use_grad_mask:
                        self.param_types += [ParamType.Scalar, ParamType.Scalar]
                        self.pid_to_lid += [layer, layer]
                        # if layer == layers:
                        #     self.param_types += [ParamType.Scalar, ParamType.Scalar]
                        #     self.pid_to_lid += [layer-1, layer-1]
                        # else:
                        #     self.param_types += [None, None]
                        #     self.pid_to_lid += [None, None]
                else:
                    # For multiplication
                    tparams.append( nn.Parameter(torch.rand(1)) ) # single scalar multiplication -- with *
                    tparams.append( nn.Parameter(torch.rand(indim)) ) # vector multiplication -- with *
                    tparams.append( nn.Parameter(torch.rand(indim, indim)) )
                    alfas.append( nn.Parameter(torch.zeros(4)) )


                    # For shifting
                    tparams.append( nn.Parameter(torch.rand(1)) ) # single constant shift -- +
                    tparams.append( nn.Parameter(torch.rand(indim)) ) # vector shift   -- +   
                    alfas.append( nn.Parameter(torch.zeros(3)) ) # unnormalized activation strengths for the original + transformations 
                    
                    
                    if self.use_grad_mask:
                        self.param_types += [ParamType.Scalar, ParamType.Vector, ParamType.Matrix, ParamType.Scalar, ParamType.Vector]
                        self.pid_to_lid += [layer for _ in range(5)]



            if not simple:
                for j in range(2):
                    # bound = 1/(indim**0.5)
                    for k in [5, 10, 15]:
                        U = nn.Parameter(torch.ones(40, k))
                        sigma = nn.Parameter( torch.ones(k) )
                        V = nn.Parameter(torch.ones(40, k))

                        nn.init.eye_(U); nn.init.eye_(V)
                        # torch.nn.init.uniform_(U, -math.sqrt(bound/k), math.sqrt(bound/k))
                        # torch.nn.init.uniform_(V, -math.sqrt(bound/k), math.sqrt(bound/k))
                        tparams.append(U); tparams.append(sigma); tparams.append(V)

                        if self.use_grad_mask:
                            self.param_types += [ParamType.SVD_U, ParamType.SVD_S, ParamType.SVD_V]
                            self.pid_to_lid += [j+1 for _ in range(3)]



        self.transform = nn.ParameterList(tparams)
        self.alfas = nn.ParameterList(alfas)
        
        
        
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))


    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, return_outputs=False, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()

        outputs = [x for _ in range(self.n_components)]
        ntp = len(weights)//self.n_components; na = len(alfas)//self.n_components
        # print(ntp, self.n_components)
        # import sys; sys.exit()

        for n in range(self.n_components):
            calfas = alfas[n*na:(n+1)*na]
            cweights = weights[n*ntp:(n+1)*ntp]

            activs = [torch.sigmoid(a) for a in calfas[:2]] + [s(a) for a in calfas[2:-2]] + [torch.sigmoid(a) for a in calfas[-2:]]
            if not self.simple:
                [U1,S1,V1, U2,S2,V2, U3,S3,V3] = cweights[-18:-9]
                SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
                SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
                SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)



            # Input transform
            # 0: original
            # 1: single scalar multiplication
            # 2: constant shift

            outputs[n] = (1-activs[0])*outputs[n] + activs[0]*outputs[n]*cweights[0] 
            outputs[n] = (1-activs[1])*outputs[n] + activs[1]*(outputs[n]+cweights[1])
            
            
            # Apply regular layer 1
            outputs[n] = F.relu(F.linear(outputs[n], bweights[0], bweights[1])) # dim is now 40
            # Transforms layer 1
            a = activs[2] # [4,]
            if not self.simple:
                outputs[n] = a[0]*outputs[n] + a[1]*outputs[n]*cweights[2] + a[2]*outputs[n]*cweights[3] + a[3]*torch.matmul(outputs[n], cweights[4]) +\
                    a[4]*torch.matmul(outputs[n], SM1) + a[5]*torch.matmul(outputs[n], SM2) + a[6]*torch.matmul(outputs[n], SM3)
                sub=0
            else:
                # original + scalar multiply + vector multiply + matrix multiply
                outputs[n] = a[0]*outputs[n] + a[1]*outputs[n]*cweights[2] + a[2]*outputs[n]*cweights[3] + a[3]*torch.matmul(outputs[n],cweights[4])
                sub = 0


            a = activs[3]
            outputs[n] = a[0]*outputs[n] + a[1]*(outputs[n]+cweights[5-sub]) + a[2]*(outputs[n]+cweights[6-sub])

            if not self.simple:
                [U1,S1,V1, U2,S2,V2, U3,S3,V3] = cweights[-9:]
                SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
                SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
                SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)


            # Apply regular layer 2
            outputs[n] = F.relu(F.linear(outputs[n], bweights[2], bweights[3])) # dim is still 40 
            # Transforms layer 2
            a = activs[4] # [4,]
            if not self.simple:
                outputs[n] = a[0]*outputs[n] + a[1]*outputs[n]*cweights[7] + a[2]*outputs[n]*cweights[8] + a[3]*torch.matmul(outputs[n], cweights[9]) +\
                    a[4]*torch.matmul(outputs[n], SM1) + a[5]*torch.matmul(outputs[n], SM2) + a[6]*torch.matmul(outputs[n], SM3)
            else:
                outputs[n] = a[0]*outputs[n] + a[1]*outputs[n]*cweights[7-sub] + a[2]*outputs[n]*cweights[8-sub] + a[3]*torch.matmul(outputs[n], cweights[9-sub])
                sub = 0

            a = activs[5]
            outputs[n] = a[0]*outputs[n] + a[1]*(outputs[n]+cweights[10-sub]) + a[2]*(outputs[n]+cweights[11-sub])

            # Apply original output layer
            outputs[n] = F.linear(outputs[n], bweights[4], bweights[5])
            # Output transforms
            outputs[n] = (1-activs[6])*outputs[n] + activs[6]*outputs[n]*cweights[12-sub]
            outputs[n] = (1-activs[7])*outputs[n] + activs[7]*(outputs[n]+cweights[13-sub])

        if return_outputs:
            return outputs
            
        preds = outputs[0]
        for i in range(1, len(outputs)):
            preds = preds + outputs[i]
        
        return preds




    def transform_forward_measure_effects(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]
        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-18:-9]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)

        ALL_EFFECTS = []
        MAGNITUDES = []
        PARTS = []



        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift

        #  x = (1-activs[0])*x + activs[0]*x*weights[0] 
        parts = [(1-activs[1])*(1-activs[0])*x, (1-activs[1])*activs[0]*x*weights[0]]
        subx = sum(parts)/(1-activs[1])
        parts.append(activs[1]*(subx+weights[1]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)**2
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        # x = (1-activs[1])*x + activs[1]*(x+weights[1])
        # parts = [(1-activs[1])*x, activs[1]*(x+weights[1])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)
        
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1])) # dim is now 40
        # Transforms layer 1
        a = activs[2] # [4,]
        # x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[3][0]*a[0]*x, activs[3][0]*a[1]*x*weights[2], activs[3][0]*a[2]*x*weights[3], activs[3][0]*a[3]*torch.matmul(x, weights[4]),
            activs[3][0]*a[4]*torch.matmul(x, SM1), activs[3][0]*a[5]*torch.matmul(x, SM2), activs[3][0]*a[6]*torch.matmul(x, SM3)]
        subx = sum(parts)/activs[3][0]
        a = activs[3]
        #x = a[0]*x, a[1]*(x+weights[5]), a[2]*(x+weights[6])
        parts.append(a[1]*(subx+weights[5]))
        parts.append(a[2]*(subx+weights[6]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        
        
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)


        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-9:]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)


        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3])) # dim is still 40 
        # Transforms layer 2
        a = activs[4] # [4,]
        # x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[5][0]*a[0]*x, activs[5][0]*a[1]*x*weights[7], activs[5][0]*a[2]*x*weights[8], activs[5][0]*a[3]*torch.matmul(x, weights[9]),
                activs[5][0]*a[4]*torch.matmul(x, SM1), activs[5][0]*a[5]*torch.matmul(x, SM2), activs[5][0]*a[6]*torch.matmul(x, SM3)] 
        subx = sum(parts)/activs[5][0]
        a = activs[5]
        parts.append(a[1]*(subx+weights[10]))
        parts.append(a[2]*(subx+weights[11]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])


        
        
        # x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])
        # parts = [a[0]*x, a[1]*(x+weights[10]), a[2]*(x+weights[11])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        #x = (1-activs[6])*x + activs[6]*x*weights[12]
        parts = [(1-activs[7])*(1-activs[6])*x, (1-activs[7])*activs[6]*x*weights[12]]
        subx = sum(parts)/((1-activs[7]))
        parts.append(activs[7]*(subx+weights[13]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        #x = (1-activs[7])*x + activs[7]*(x+weights[13])
        # parts = [(1-activs[7])*x, activs[7]*(x+weights[13])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        return x, PARTS



    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()










class SimpleCompositionalSineNetworkSVD(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, 
                relu=False, use_grad_mask=False, n_components=2, no_shift=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.free_arch = free_arch
        self.transform_type = transform
        self.relu = nn.ReLU()
        self.z = identity if not relu else partial(relu_fn, f=self.relu)
        self.use_grad_mask = use_grad_mask
        self.n_components = n_components
        self.no_shift = no_shift
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        layers = 3

        # there are 3 layers (counting the output layer too) so we need 


        tparams = []
        alfas = []


        for n in range(n_components):

            for layer in range(layers+1):
                indim = 1 if layer == 0 or layer==layers else 40
                if indim == 1:
                    tparams.append( nn.Parameter(torch.rand(1)) ) # single scalar multiplication -- with *
                    alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this
                    if not self.no_shift or layer == 0:
                        tparams.append( nn.Parameter(torch.rand(1)) ) # single constant shift -- +
                        alfas.append( nn.Parameter(torch.zeros(1)) ) # sigmoid on this 
                    else:
                        print("NO SHIFTING")
                # else:
                #     # For multiplication
                #     tparams.append( nn.Parameter(torch.ones(1)) ) # single scalar multiplication -- with *
                #     tparams.append( nn.Parameter(torch.ones(indim)) ) # vector multiplication -- with *
                #     #tparams.append( nn.Parameter(torch.ones(indim, indim)) )
                #     #nn.init.eye_(tparams[-1])  

                #     # For shifting
                #     #tparams.append( nn.Parameter(torch.zeros(1)) ) # single constant shift -- +
                #     #tparams.append( nn.Parameter(torch.zeros(indim)) ) # vector shift   -- +   

                #     # add alfas
                #     if layer == 1 or layer == 2:
                #         alfas.append( nn.Parameter(torch.zeros(7)) ) # for 3 different k approximators
                #     else:
                #         alfas.append( nn.Parameter(torch.zeros(4)) ) # unnormalized activation strengths for the original + transformations
                #     alfas.append( nn.Parameter(torch.zeros(3)) ) # unnormalized activation strengths for the original + transformations 
                #     if self.use_grad_mask:
                #         self.param_types += [ParamType.Scalar, ParamType.Vector, ParamType.Matrix, ParamType.Scalar, ParamType.Vector]
                #         self.pid_to_lid += [layer for _ in range(5)]




            # for j in range(2):
            #     # bound = 1/(indim**0.5)
            #     for k in [5, 10, 15]:
            #         U = nn.Parameter(torch.ones(40, k))
            #         sigma = nn.Parameter( torch.ones(k) )
            #         V = nn.Parameter(torch.ones(40, k))

            #         nn.init.eye_(U); nn.init.eye_(V)
            #         # torch.nn.init.uniform_(U, -math.sqrt(bound/k), math.sqrt(bound/k))
            #         # torch.nn.init.uniform_(V, -math.sqrt(bound/k), math.sqrt(bound/k))
            #         tparams.append(U); tparams.append(sigma); tparams.append(V)

            #         if self.use_grad_mask:
            #             self.param_types += [ParamType.SVD_U, ParamType.SVD_S, ParamType.SVD_V]
            #             self.pid_to_lid += [j+1 for _ in range(3)]



        self.transform = nn.ParameterList(tparams)
        self.alfas = nn.ParameterList(alfas)
        
        
        
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        x = F.relu(F.linear(x, weights[0], weights[1])) 
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        # x = F.relu(F.linear(x, weights[6], weights[7])) ### <--- remove here
        # x = F.linear(x, weights[8], weights[9])
        # x = F.linear(x, weights[10], weights[11])
        # x = F.linear(x, weights[12], weights[13])
        return x

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, return_outputs=False, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()

        outputs = [x for _ in range(self.n_components)]
        ntp = len(weights)//self.n_components
        na = len(alfas)//self.n_components
        # print(ntp, self.n_components)
        # import sys; sys.exit()

        for n in range(self.n_components):
            calfas = alfas[n*na:(n+1)*na]
            cweights = weights[n*ntp:(n+1)*ntp]

            activs = [torch.sigmoid(a) for a in calfas] 




            # Input transform
            # 0: original
            # 1: single scalar multiplication
            # 2: constant shift

            outputs[n] = (1-activs[0])*outputs[n] + activs[0]*outputs[n]*cweights[0] 
            outputs[n] = (1-activs[1])*outputs[n] + activs[1]*(outputs[n]+cweights[1])
            
            # Apply regular layer 1
            outputs[n] = F.relu(F.linear(outputs[n], bweights[0], bweights[1])) # dim is now 40
            # Apply regular layer 2
            outputs[n] = F.relu(F.linear(outputs[n], bweights[2], bweights[3])) # dim is still 40 
    
            # Apply original output layer
            outputs[n] = F.linear(outputs[n], bweights[4], bweights[5])
            # Output transforms
            outputs[n] = (1-activs[2])*outputs[n] + activs[2]*outputs[n]*cweights[2]
            if not self.no_shift:
                outputs[n] = (1-activs[3])*outputs[n] + activs[3]*(outputs[n]+cweights[3])

        if return_outputs:
            return outputs
            
        preds = outputs[0]
        for i in range(1, len(outputs)):
            preds = preds + outputs[i]
        
        return preds




    def transform_forward_measure_effects(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        # Multiplication:
        # 0: original
        # 1: single scalar multiplication
        # 2: vector multiplication  
        # 3: matrix multiplication 

        # Shifting
        # 0: original
        # 1: constant shift
        # 2: vector shift 

        s = nn.Softmax()
        activs = [torch.sigmoid(a) for a in alfas[:2]] + [s(a) for a in alfas[2:-2]] + [torch.sigmoid(a) for a in alfas[-2:]]
        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-18:-9]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)

        ALL_EFFECTS = []
        MAGNITUDES = []
        PARTS = []



        # Input transform
        # 0: original
        # 1: single scalar multiplication
        # 2: constant shift

        #  x = (1-activs[0])*x + activs[0]*x*weights[0] 
        parts = [(1-activs[1])*(1-activs[0])*x, (1-activs[1])*activs[0]*x*weights[0]]
        subx = sum(parts)/(1-activs[1])
        parts.append(activs[1]*(subx+weights[1]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)**2
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        # x = (1-activs[1])*x + activs[1]*(x+weights[1])
        # parts = [(1-activs[1])*x, activs[1]*(x+weights[1])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)
        
        
        # Apply regular layer 1
        x = F.relu(F.linear(x, bweights[0], bweights[1])) # dim is now 40
        # Transforms layer 1
        a = activs[2] # [4,]
        # x = a[0]*x + a[1]*x*weights[2] + a[2]*x*weights[3] + a[3]*torch.matmul(x, weights[4]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[3][0]*a[0]*x, activs[3][0]*a[1]*x*weights[2], activs[3][0]*a[2]*x*weights[3], activs[3][0]*a[3]*torch.matmul(x, weights[4]),
            activs[3][0]*a[4]*torch.matmul(x, SM1), activs[3][0]*a[5]*torch.matmul(x, SM2), activs[3][0]*a[6]*torch.matmul(x, SM3)]
        subx = sum(parts)/activs[3][0]
        a = activs[3]
        #x = a[0]*x, a[1]*(x+weights[5]), a[2]*(x+weights[6])
        parts.append(a[1]*(subx+weights[5]))
        parts.append(a[2]*(subx+weights[6]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        
        
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)


        [U1,S1,V1, U2,S2,V2, U3,S3,V3] = weights[-9:]
        SM1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
        SM2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
        SM3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)


        # Apply regular layer 2
        x = F.relu(F.linear(x, bweights[2], bweights[3])) # dim is still 40 
        # Transforms layer 2
        a = activs[4] # [4,]
        # x = a[0]*x + a[1]*x*weights[7] + a[2]*x*weights[8] + a[3]*torch.matmul(x, weights[9]) +\
        #     a[4]*torch.matmul(x, SM1) + a[5]*torch.matmul(x, SM2) + a[6]*torch.matmul(x, SM3)
        parts = [activs[5][0]*a[0]*x, activs[5][0]*a[1]*x*weights[7], activs[5][0]*a[2]*x*weights[8], activs[5][0]*a[3]*torch.matmul(x, weights[9]),
                activs[5][0]*a[4]*torch.matmul(x, SM1), activs[5][0]*a[5]*torch.matmul(x, SM2), activs[5][0]*a[6]*torch.matmul(x, SM3)] 
        subx = sum(parts)/activs[5][0]
        a = activs[5]
        parts.append(a[1]*(subx+weights[10]))
        parts.append(a[2]*(subx+weights[11]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])


        
        
        # x = a[0]*x + a[1]*(x+weights[10]) + a[2]*(x+weights[11])
        # parts = [a[0]*x, a[1]*(x+weights[10]), a[2]*(x+weights[11])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        # Apply original output layer
        x = F.linear(x, bweights[4], bweights[5])
        # Output transforms
        #x = (1-activs[6])*x + activs[6]*x*weights[12]
        parts = [(1-activs[7])*(1-activs[6])*x, (1-activs[7])*activs[6]*x*weights[12]]
        subx = sum(parts)/((1-activs[7]))
        parts.append(activs[7]*(subx+weights[13]))
        x = sum(parts)
        norm = x.norm(p=2, dim=1)
        effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        abs_effects = [abs(e) for e in effects]
        sum_abs_effects = sum(abs_effects)
        effects = [e/sum_abs_effects for e in effects]
        ALL_EFFECTS.append(effects)
        MAGNITUDES.append([p.norm(p=2, dim=1) for p in parts])
        PARTS.append([(p/x).mean(dim=1) for p in parts])

        #x = (1-activs[7])*x + activs[7]*(x+weights[13])
        # parts = [(1-activs[7])*x, activs[7]*(x+weights[13])]
        # x = sum(parts)
        # norm = x.norm(p=2, dim=1)
        # effects = [(((x - part)**2).sum(dim=1)/norm).detach().numpy() for part in parts]
        # ALL_EFFECTS.append(effects)

        return x, PARTS



    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()






































class SineTNet(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, use_grad_mask=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.use_grad_mask = use_grad_mask
        
        self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(in_dim, 40)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU())]))
        })
        # Output layer
        self.model.update({"out": nn.Linear(40, out_dim)})
        layers = 3

        # there are 3 layers (counting the output layer too) so we need 


        tparams = []
        alfas = []
        if self.use_grad_mask:
            grad_masks = []
            out_dims = [40, 40, 1]
            for lid in range(layers):
                outdim = out_dims[lid]
                mask = nn.Parameter(torch.zeros(outdim))
                mask.requires_grad=True
                grad_masks.append(mask)
            self.grad_masks = nn.ParameterList(grad_masks)
            self.pid_to_lid = [0,0,1,1,2,2]
            self.param_types = [ParamType.Matrix, ParamType.Vector, ParamType.Matrix, ParamType.Vector, ParamType.Matrix, ParamType.Vector] # the base params

        for layer in range(layers):
            indim = 1 if layer == layers-1 else 40
            tparams.append( nn.Parameter(torch.eye(indim)) )


        self.transform = nn.ParameterList(tparams)
        self.alfas = nn.ParameterList(alfas)        
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))


    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, **kwargs):
        # hidden 1
        x = F.linear(x, bweights[0], bweights[1]) # dim is now 40
        x = torch.matmul(x, weights[0])
        x = F.relu(x)

        # hidden 2
        x = F.linear(x, bweights[2], bweights[3]) # dim is now 40
        x = torch.matmul(x, weights[1])
        x = F.relu(x)

        # output
        x = F.linear(x, bweights[4], bweights[5]) # dim is now 40
        x = torch.matmul(x, weights[2])
        return x
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        
        x = F.relu(F.linear(x, weights[0], weights[1]))
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()






# class SVDTransformSineNetwork(nn.Module):
#     """
#     Base-learner neural network for the sine wave regression task.

#     ...

#     Attributes
#     ----------
#     model : nn.ModuleDict
#         Complete sequential specification of the model
#     relu : nn.ReLU
#         ReLU function to use after w1 and w2
        
#     Methods
#     ----------
#     forward(x)
#         Perform a feed-forward pass using inputs x
    
#     forward_weights(x, weights)
#         Perform a feedforward pass on inputs x making using
#         @weights instead of the object's weights (w1, w2, w3)
    
#     get_flat_params()
#         Returns all model parameters in a flat tensor
        
#     copy_flat_params(cI)
#         Set the model parameters equal to cI
        
#     transfer_params(learner_w_grad, cI)
#         Transfer batch normalizations statistics from another learner to this one
#         and set the parameters to cI
        
#     freeze_layers()
#         Freeze all hidden layers
    
#     reset_batch_stats()
#         Reset batch normalization stats
#     """

#     def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", free_arch=False, relu=False, **kwargs):
#         """Initializes the model
        
#         Parameters
#         ----------
#         criterion : nn.loss_fn
#             Loss function to use
#         in_dim : int
#             Dimensionality of the input
#         out_dim : int
#             Dimensionality of the output
#         zero_bias : bool, optional
#             Whether to initialize biases of linear layers with zeros
#             (default is Uniform(-sqrt(k), +sqrt(k)), where 
#             k = 1/num_in_features)
#         **kwargs : dict, optional
#             Trash can for additional arguments. Keep this for constructor call uniformity
#         """
        
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.free_arch = free_arch
#         self.transform_type = transform
#         self.relu = nn.ReLU()
#         self.z = identity if not relu else partial(relu_fn, f=self.relu)
        

#         dims = [1, 1, 40, 40, 1, 1]
#         # base learner params
#         params = []
#         # [w0, bias-const0, weight, weight-scalar, bias-const, bias-vector, bias-scalar, weight, scalar, 
#         #  U-k5, sigma-k5, V-k5, U-k10, sigma-k10, V-k10, bias-const, bias-vector, bias-scalar,
#         #  weight, weight-scalar, bias-const, weight, bias-const]

#         # multipliers
#         alfas = [] # [sigmoid-bias, softmax-weights (3), sigmoid-bias, final-layer added?]
        
#         alfas.append(nn.Parameter(torch.zeros(1)))
#         for idx in range(len(dims)-1):
#             indim = dims[idx]; outdim = dims[idx+1]
#             bound = 1/math.sqrt(indim)
            
#             weight = nn.Parameter(torch.ones(outdim, indim))
#             if not (idx == 0 or idx ==4):
#                 torch.nn.init.uniform_(weight, -bound, bound)     
#             params.append(weight); 
#             if not (indim == 1 and outdim == 1):
#                 scalar = nn.Parameter(torch.ones(outdim,1))
#                 params.append(scalar)

#             # If indim or outdim =1, we cannot compress (single number cant be compressed)
#             if indim != 1 and outdim != 1:
#                 # for these hidden dimensions, we are actually compressing
#                 k_range = (1, (indim*outdim)//(indim+outdim+1) )
#                 for k in [5, 10]:
#                     U = nn.Parameter(torch.ones(outdim, k))
#                     sigma = nn.Parameter( torch.ones(k) )
#                     V = nn.Parameter(torch.ones(outdim, k))

#                     torch.nn.init.uniform_(U, -math.sqrt(bound/k), math.sqrt(bound/k))
#                     torch.nn.init.uniform_(V, -math.sqrt(bound/k), math.sqrt(bound/k))
#                     params.append(U); params.append(sigma); params.append(V)
                

#                 # Interpolation coeffieicnts for the 3 paths
#                 alfas.append( nn.Parameter(torch.zeros(3)) )
                
            
#             # Every layer has a bias constant
#             bias_const = nn.Parameter(torch.ones(1).squeeze())
#             if not (idx == 0 or idx ==4):
#                 torch.nn.init.uniform_(bias_const, -bound, bound)
#             else:
#                 torch.nn.init.zeros_(bias_const)
#             params.append(bias_const)
#             # and possibly a bias vector (if outdim > 1) 
#             if outdim > 1:
#                 bias_vect = nn.Parameter(torch.ones(outdim))
#                 bias_scalar = nn.Parameter(torch.ones(1).squeeze())
#                 torch.nn.init.uniform_(bias_vect, -bound, bound)
#                 # Interpolation coefficients for different biases
#                 alfas.append( nn.Parameter(torch.zeros(1)) )
#                 params.append(bias_vect); params.append(bias_scalar)

#         alfas.append( nn.Parameter(torch.zeros(1)) )
                
        
#         self.transform = nn.ParameterList(params)
#         self.alfas = nn.ParameterList(alfas)
#         self.model = nn.ParameterList([])
#         self.criterion = criterion
        
#         if zero_bias:
#             for m in self.model.modules():
#                 if isinstance(m, nn.Linear):
#                     m.bias = nn.Parameter(torch.zeros(m.bias.size()))


#     def forward(self, x):
#         """Feedforward pass of the network

#         Take inputs x, and compute the network's output using its weights
#         w1, w2, and w3

#         Parameters
#         ----------
#         x : torch.Tensor
#             Real-valued input tensor with shape (Batch size, 1)

#         Returns
#         ----------
#         tensor
#             Predictions with shape (Batch size, 1) of the network on inputs x 
#         """

#         features = self.model.features(x)
#         out = self.model.out(features)
#         return out

#     def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
#         # [weight, weight-scalar, bias-const, bias-vector, bias-scalar, weight, scalar, 
#         #  U-k5, sigma-k5, V-k5, U-k10, sigma-k10, V-k10, bias-const, bias-vector, bias-scalar,
#         #  weight, weight-scalar, bias-const, weight, bias-const]
#         [W0, bias0, W1, W_scale1, bias1, bias_vect1, bias_scale1, W2, W_scale2, 
#          U_k5, sigma_k5, V_k5, U_k10, sigma_k10, V_k10, bias2, bias_vect2, bias_scale2,
#          W3, W_scale3, bias3, W4, bias4] = weights

#         # [sigmoid-bias, softmax-weights (3), sigmoid-bias, final-layer added?]
#         [a0, a_bias1, a_vect2, a_bias3, a_final] = alfas

        
#         # Transform input
#         activ = torch.sigmoid(a0)
#         x = (1-activ)*x + activ*(x*W0 + bias0)
#         # First layer + bias
#         x = F.linear(x, weight=W1*W_scale1, bias=None) 
#         #print(f"1:{x}")
#         activ = torch.sigmoid(a_bias1)
#         x = F.relu(x + (1-activ)*bias1 + activ*bias_vect1*bias_scale1)
#         #print(f"2:{x}")

#         # Second layer
#         activs = F.softmax(a_vect2, dim=0)
#         # generate weight matrices using SVD
#         W2_k5 = torch.mm(torch.mm(U_k5, torch.diag(sigma_k5)), V_k5.T)
#         W2_k10 = torch.mm(torch.mm(U_k10, torch.diag(sigma_k10)), V_k10.T)

#         x1 = F.linear(x, W2*W_scale2, bias=None)
#         #print(f"3:{x1}")
#         x2 = F.linear(x, W2_k5, bias=None)
#         #print(W2_k5.size())
#         #print(U_k5)
#         #print()
#         #print(torch.diag(sigma_k5))
#         #print()
#         #print(V_k5.T)
#         #print()

#         #print(f"4:{x2}")
#         x3 = F.linear(x, W2_k10, bias=None)
#         #print(f"5:{x3}")


#         x = activs[0]*x1 + activs[1]*x2 + activs[2] * x3
#         #print(f"6:{x}")
#         activ = torch.sigmoid(a_bias3)
#         x = F.relu(x + (1-activ)*bias2 + activ*bias_vect2*bias_scale2)
#         #print(f"7:{x}")

#         # To output
#         x = F.linear(x, weight=W3*W_scale3, bias=bias3)
#         #print(f"8:{x}")
#         activ = torch.sigmoid(a_final)
#         x = (1-activ)*x + activ*(W4*x + bias4)
#         #print(f"9:{x}")
#         #import sys; sys.exit()

#         return x
    
#     def forward_weights(self, x, weights):
#         """Feedforward pass using provided weights
        
#         Take input x, and compute the output of the network with user-defined
#         weights
        
#         Parameters
#         ----------
#         x : torch.Tensor
#             Real-valued input tensor with shape (Batch size, 1)
#         weights : list
#             List of tensors representing the weights of a custom SineNetwork.
#             Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
#         Returns
#         ----------
#         tensor
#             Predictions with shape (Batch size, 1) of the implicitly defined network 
#             on inputs x 
#         """
        
#         x = F.relu(F.linear(x, weights[0], weights[1]))
#         x = F.relu(F.linear(x, weights[2], weights[3]))
#         x = F.linear(x, weights[4], weights[5])
#         return x
    
#     def forward_get_features(self, x):
#         features = []
#         x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
#         features.append(x.clone().cpu().detach().numpy())
#         x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
#         features.append(x.clone().cpu().detach().numpy())
#         x = self.model.out(x)
#         features.append(x.clone().cpu().detach().numpy())
#         return x, features

#     def forward_weights_get_features(self, x, weights):
#         """Manual feedforward pass of the network with provided weights

#         Parameters
#         ----------
#         x : torch.Tensor
#             Real-valued input tensor 
#         weights : list
#             List of torch.Tensor weight variables

#         Returns
#         ----------
#         tensor
#             The output of the block
#         """
#         features = []
#         x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
#         features.append(x.clone().cpu().detach().numpy())
#         x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
#         features.append(x.clone().cpu().detach().numpy())
#         x = F.linear(x, weights[4], weights[5])
#         features.append(x.clone().cpu().detach().numpy())
#         return x, features

#     def get_flat_params(self):
#         """Returns parameters in flat format
        
#         Flattens the current weights and returns them
        
#         Returns
#         ----------
#         tensor
#             Weight tensor containing all model weights
#         """
#         return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

#     def copy_flat_params(self, cI):
#         """Copy parameters to model 
        
#         Set the model parameters to be equal to cI
        
#         Parameters
#         ----------
#         cI : torch.Tensor
#             Flat tensor with the same number of elements as weights in the network
#         """
        
#         idx = 0
#         for p in self.model.parameters():
#             plen = p.view(-1).size(0)
#             p.data.copy_(cI[idx: idx+plen].view_as(p))
#             idx += plen

#     def transfer_params(self, learner_w_grad, cI):
#         """Transfer model parameters
        
#         Transfer parameters from cI to this network, while maintaining mean 
#         and variance of Batch Normalization 
        
#         Parameters
#         ----------
#         learner_w_grad : nn.Module
#             Base-learner network which records gradients
#         cI : torch.Tensor
#             Flat tensor of weights to copy to this network's parameters
#         """
        
#         # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
#         #  are going to be replaced by cI
#         self.load_state_dict(learner_w_grad.state_dict())
#         #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
#         idx = 0
#         for m in self.model.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
#                 wlen = m._parameters['weight'].view(-1).size(0)
#                 m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
#                 idx += wlen
#                 if m._parameters['bias'] is not None:
#                     blen = m._parameters['bias'].view(-1).size(0)
#                     m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
#                     idx += blen
    
#     def freeze_layers(self):
#         """Freeze all hidden layers
#         """
        
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.model.out.weight.requires_grad=True
#         self.model.out.bias.requires_grad=True
    
#     def reset_batch_stats(self):
#         """Resets the Batch Normalization statistics
#         """
        
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.reset_running_stats()









class SparseSineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, trans_net=False, **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.criterion = criterion
        self.trans_net = trans_net

        if not self.trans_net:
            self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
                ('lin1', nn.Linear(in_dim, 40)),
                ('relu1', nn.ReLU()),
                ('lin2', nn.Linear(40, 40)),
                ('relu2', nn.ReLU()),
                ("out", nn.Linear(40, out_dim))
                ]))
            })

            self.alfas = nn.ModuleDict({'activations': nn.Sequential(OrderedDict([
                ('a1', nn.Linear(in_dim, 40)),
                ('a2', nn.Linear(40, 40)),
                ('a3', nn.Linear(40, out_dim))]))
                })
            limit = (1, 4)
        else:
            self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
                ('tinput', nn.Linear(in_dim, in_dim)),
                ('lin1', nn.Linear(in_dim, 40)),
                ('relu1', nn.ReLU()),
                ('tlin1', nn.Linear(40,40)),
                ('lin2', nn.Linear(40, 40)),
                ('relu2', nn.ReLU()),
                ('tlin2', nn.Linear(40,40)),
                ('out', nn.Linear(40, out_dim)),
                ('tlin3', nn.Linear(1,1,bias=True)),
                ]))
            })
            self.alfas = nn.ModuleDict({'activations': nn.Sequential(OrderedDict([
                ('a1', nn.Linear(in_dim, in_dim)),
                ('a2', nn.Linear(in_dim, 40)),
                ('a3', nn.Linear(40,40)),
                ('a4', nn.Linear(40, 40)),
                ('a5', nn.Linear(40,40)),
                ('a6', nn.Linear(40, out_dim)),
                ('a7', nn.Linear(1,1,bias=True)),]))

            })
            limit = (1, 8)

            nn.init.eye_(self.model.features.tinput.weight)
            nn.init.eye_(self.model.features.tlin1.weight)
            nn.init.eye_(self.model.features.tlin2.weight)
            nn.init.eye_(self.model.features.tlin3.weight)


        for i in range(*limit):
            # Initialize to leave original inputs unaffected
            s = f"nn.init.zeros_(self.alfas.activations.a{i}.weight)"
            eval(s)

        
        # Initialize normal weights 2x as large because sigmoid of alfas is 0.5 at start
        with torch.no_grad():
            self.model.features.lin1.weight.data *= 2
            self.model.features.lin2.weight.data *= 2
            self.model.features.out.weight.data *= 2
            if self.trans_net:
                self.model.features.tinput.weight.data *= 2
                self.model.features.tlin1.weight.data *= 2
                self.model.features.tlin2.weight.data *= 2
                self.model.features.tlin3.weight.data *= 2

        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))
            for m in self.alfas.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        out = self.model.out(features)
        return out

    def transform_forward(self, x, bweights=None, weights=None, **kwargs):
        if not self.trans_net:
            x = F.relu(F.linear(x, bweights[0] * torch.sigmoid(weights[0]), bweights[1] * torch.sigmoid(weights[1])))
            x = F.relu(F.linear(x, bweights[2] * torch.sigmoid(weights[2]), bweights[3] * torch.sigmoid(weights[3])))
            x = F.linear(x, bweights[4] * torch.sigmoid(weights[4]), bweights[5] * torch.sigmoid(weights[5]))
        else:
            x = F.linear(x, bweights[0] * torch.sigmoid(weights[0]), bweights[1] * torch.sigmoid(weights[1])) # transform 1
            x = F.relu(F.linear(x, bweights[2] * torch.sigmoid(weights[2]), bweights[3] * torch.sigmoid(weights[3]))) # lin 1
            
            x = F.linear(x, bweights[4] * torch.sigmoid(weights[4]), bweights[5] * torch.sigmoid(weights[5])) # trans 2
            x = F.relu(F.linear(x, bweights[6] * torch.sigmoid(weights[6]), bweights[7] * torch.sigmoid(weights[7]))) # lin 2
            
            x = F.linear(x, bweights[8] * torch.sigmoid(weights[8]), bweights[9] * torch.sigmoid(weights[9])) # out
            x = F.linear(x, bweights[10] * torch.sigmoid(weights[10]), bweights[11] * torch.sigmoid(weights[11])) # out
            x = F.linear(x, bweights[12] * torch.sigmoid(weights[12]), bweights[13] * torch.sigmoid(weights[13])) # out

        return x
    
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


class Architecture:
    FULL = 0
    PARTIAL = 1

class FixedTransformSineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, transform="interp", arch="full", **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.transform_type = transform
        archmap = {
            "full": Architecture.FULL,
            "partial": Architecture.PARTIAL
        }
        self.arch = archmap[arch]
        print("Using architecture:", arch)

        if self.arch == Architecture.FULL:
            self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
                ('tinput', nn.Linear(in_dim, in_dim)),
                ('lin1', nn.Linear(in_dim, 40)),
                ('relu1', nn.ReLU()),
                ('tlin1', nn.Linear(40,40)),
                ('lin2', nn.Linear(40, 40)),
                ('relu2', nn.ReLU()),
                ('tlin2', nn.Linear(40,40)),
                ('lin3', nn.Linear(40, out_dim)),
                ('tlin3', nn.Linear(1,1,bias=True)),
                ]))
            })
            
            nn.init.eye_(self.model.features.tinput.weight)
            nn.init.eye_(self.model.features.tlin1.weight)
            nn.init.eye_(self.model.features.tlin2.weight)
            nn.init.eye_(self.model.features.tlin3.weight)
        else:
            self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
                ('tinput', nn.Linear(in_dim, in_dim)),
                ('lin1', nn.Linear(in_dim, 40)),
                ('relu1', nn.ReLU()),
                ('lin2', nn.Linear(40, 40)),
                ('relu2', nn.ReLU()),
                ('lin3', nn.Linear(40, out_dim)),
                ('tlin3', nn.Linear(1,1,bias=False)),
                ]))
            })
            
            nn.init.eye_(self.model.features.tinput.weight)
            nn.init.eye_(self.model.features.tlin3.weight)


        self.criterion = criterion
        print("We are using the FTSN")
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    if not m.bias is None:
                        m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model.features(x)
        return features

    def transform_forward(self, x, weights=None, **kwargs):
        # Transform the input
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(F.linear(x, weights[2], weights[3]))
        x = F.linear(x, weights[4], weights[5])
        x = F.relu(F.linear(x, weights[6], weights[7]))
        x = F.linear(x, weights[8], weights[9])
        x = F.linear(x, weights[10], weights[11])
        x = F.linear(x, weights[12], weights[13])
        return x



    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        if self.arch == Architecture.FULL:
            x = F.linear(x, weights[0], weights[1]) #t1
            x = F.relu(F.linear(x, weights[2], weights[3])) # normal
            x = F.linear(x, weights[4], weights[5]) # t1
            x = F.relu(F.linear(x, weights[6], weights[7])) # normal
            x = F.linear(x, weights[8], weights[9]) # normal
            x = F.linear(x, weights[10], weights[11]) # t2 
            x = F.linear(x, weights[12], weights[13])
        else:
            x = x + weights[1] #F.linear(x, weights[0], weights[1]) -- just add bias (shift)
            x = F.relu(F.linear(x, weights[2], weights[3]))
            x = F.relu(F.linear(x, weights[4], weights[5]))
            x = F.linear(x, weights[6], weights[7])
            x = F.linear(x, weights[8], bias=None)
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()


class SineNetwork(nn.Module):
    """
    Base-learner neural network for the sine wave regression task.

    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Complete sequential specification of the model
    relu : nn.ReLU
        ReLU function to use after w1 and w2
        
    Methods
    ----------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    
    get_flat_params()
        Returns all model parameters in a flat tensor
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
        
    freeze_layers()
        Freeze all hidden layers
    
    reset_batch_stats()
        Reset batch normalization stats
    """

    def __init__(self, criterion, in_dim=1, out_dim=1, zero_bias=True, use_tanh=False, hdims=[40,40], **kwargs):
        """Initializes the model
        
        Parameters
        ----------
        criterion : nn.loss_fn
            Loss function to use
        in_dim : int
            Dimensionality of the input
        out_dim : int
            Dimensionality of the output
        zero_bias : bool, optional
            Whether to initialize biases of linear layers with zeros
            (default is Uniform(-sqrt(k), +sqrt(k)), where 
            k = 1/num_in_features)
        **kwargs : dict, optional
            Trash can for additional arguments. Keep this for constructor call uniformity
        """
        
        super().__init__()
        self.relu = nn.ReLU()
        self.act_fn = nn.ReLU() if not use_tanh else nn.Tanh()
        self.use_tanh = use_tanh

        ls = [nn.Linear(in_dim, hdims[0])] + [nn.Linear(hdims[i], hdims[i+1]) for i in range(len(hdims)-1)] 
        
        self.model = nn.Sequential(*ls)
        self.out = nn.Linear(hdims[-1], 1)

        print(self.model, self.out)

        # self.model = nn.ModuleDict({'features': nn.Sequential(OrderedDict([
        #     # ('lin0', nn.Linear(in_dim, in_dim)),
        #     ('lin1', nn.Linear(in_dim, 40)),
        #     ('relu1', self.act_fn),
        #     ('lin2', nn.Linear(40, 40)),
        #     ('relu2', self.act_fn),
        #     # ('lin3', nn.Linear(40, 40)), ####### <----- remove this and below
        #     # ('relu3', nn.ReLU()),
        #     # ('lin4', nn.Linear(40, 40)),
        #     # ('relu4', nn.ReLU()),
        #     # ('lin5', nn.Linear(40, 1)),
        #     ]))
        # })
        
        # Output layer
        #self.model.update({"out": nn.Linear(40, out_dim)}) # should be 40,1
        self.criterion = criterion
        
        if zero_bias:
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(torch.zeros(m.bias.size()))

    def forward(self, x):
        """Feedforward pass of the network

        Take inputs x, and compute the network's output using its weights
        w1, w2, and w3

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)

        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the network on inputs x 
        """

        features = self.model(x)
        out = self.out(features)
        return out
    
    def forward_weights(self, x, weights):
        """Feedforward pass using provided weights
        
        Take input x, and compute the output of the network with user-defined
        weights
        
        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor with shape (Batch size, 1)
        weights : list
            List of tensors representing the weights of a custom SineNetwork.
            Format: [layer 1 kernel, layer 1 bias, layer 2 kernel,..., layer 3 bias]
        
        Returns
        ----------
        tensor
            Predictions with shape (Batch size, 1) of the implicitly defined network 
            on inputs x 
        """
        for i in range((len(weights)-2)//2):
            x = self.act_fn(F.linear(x, weights[2*i], weights[2*i+1]))

        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def forward_get_features(self, x):
        features = []
        x = self.model.features.relu1(self.model.features.lin1(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.features.relu2(self.model.features.lin2(x)) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        x = F.relu(F.linear(x, weights[0], weights[1])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.relu(F.linear(x, weights[2], weights[3])) # ReLU + Linear
        features.append(x.clone().cpu().detach().numpy())
        x = F.linear(x, weights[4], weights[5])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out.weight.requires_grad=True
        self.model.out.bias.requires_grad=True
    
    def reset_batch_stats(self):
        """Resets the Batch Normalization statistics
        """
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                
class ConvBlock(nn.Module):
    """
    Initialize the convolutional block consisting of:
     - 64 convolutional kernels of size 3x3
     - Batch normalization 
     - ReLU nonlinearity
     - 2x2 MaxPooling layer
     
    ...

    Attributes
    ----------
    cl : nn.Conv2d
        Convolutional layer
    bn : nn.BatchNorm2d
        Batch normalization layer
    relu : nn.ReLU
        ReLU function
    mp : nn.MaxPool2d
        Max pooling layer
    running_mean : torch.Tensor
        Running mean of the batch normalization layer
    running_var : torch.Tensor
        Running variance of the batch normalization layer
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    """
    
    def __init__(self, dev, indim=3, pool=True, out_channels=64):
        """Initialize the convolutional block
        
        Parameters
        ----------
        indim : int, optional
            Number of input channels (default=3)
        """
        
        super().__init__()
        self.dev = dev
        self.out_channels = out_channels
        self.cl = nn.Conv2d(in_channels=indim, out_channels=self.out_channels,
                            kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels, momentum=1) #momentum=1 is crucial! (only statistics for current batch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.pool = pool
        
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """

        x = self.cl(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.mp(x)
        return x
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(self.out_channels).to(self.dev)
        running_var = torch.ones(self.out_channels).to(self.dev)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
        if self.pool:                   
            x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x
    
    def reset_batch_stats(self):
        """Reset Batch Normalization stats
        """
        
        self.bn.reset_running_stats()
        

class ConvolutionalNetwork(nn.Module):
    """
    Super class for the Conv4 and BoostedConv4 networks.
    
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Feature embedding module of the network (all hidden layers + output layer)
    criterion : loss_fn
        Loss function to use
    in_features : int
        Number of dimensions of embedded inputs
        
    Methods
    ----------
    get_flat_params()
        Returns flattened parameters
        
    copy_flat_params(cI)
        Set the model parameters equal to cI
        
    transfer_params(learner_w_grad, cI)
        Transfer batch normalizations statistics from another learner to this one
        and set the parameters to cI
    
    reset_batch_stats()
        Reset batch normalization stats
    """
            
    def __init__(self, train_classes, eval_classes, criterion, dev):
        """Initialize the conv network

        Parameters
        ----------
        num_classes : int
            Number of classes, which determines the output size
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.in_features = 3*3*64
        self.criterion = criterion
        
        # Feature embedding module
        self.model = nn.ModuleDict({"features": nn.Sequential(OrderedDict([
            ("conv_block1", ConvBlock(dev=dev, indim=3)),
            ("conv_block2", ConvBlock(dev=dev, indim=64)),
            ("conv_block3", ConvBlock(dev=dev, indim=64)),
            ("conv_block4", ConvBlock(dev=dev, indim=64)),
            ("flatten", nn.Flatten())]))
        })
        
        
    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
                    
    def reset_batch_stats(self):
        """Reset BN stats
        
        Resets the Batch Normalization statistics
        
        """
        
        for m in self.model.modules():
            if isinstance(m, ConvBlock):
                m.reset_batch_stats()        
        
class Conv4(ConvolutionalNetwork):
    """
    Convolutional neural network consisting of four ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss()):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__(train_classes=train_classes, eval_classes=eval_classes, 
                         criterion=criterion, dev=dev)
        self.in_features = 3*3*64
        self.dev = dev
        # Add output layer `out'
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

        # Set bias weights to 0 of linear layers
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.bias = nn.Parameter(torch.zeros(m.bias.size()))
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """
        
        features = self.model.features(x)
        out = self.model.out(features)
        return out
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        x = self.model.features.conv_block1.forward_weights(x, weights[0:4])
        x = self.model.features.conv_block2.forward_weights(x, weights[4:8])
        x = self.model.features.conv_block3.forward_weights(x, weights[8:12])
        x = self.model.features.conv_block4.forward_weights(x, weights[12:16])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[16], weights[17])
        return x
    
    def load_state_dict(self, state):
        """Overwritten load_state function
        
        Before loading the state, check whether the dimensions of the output
        layer are matching. If not, create a layer of the output size in the provided state.  

        """
        
        out_key = "model.out.weight"
        out_classes = state[out_key].size()[0]
        if out_classes != self.model.out.weight.size()[0]:
            self.model.out = nn.Linear(in_features=self.in_features,
                                       out_features=out_classes).to(self.dev)
        super().load_state_dict(state)

    def freeze_layers(self):
        """Freeze all hidden layers
        """
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))


class ConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, out_channels=64, no_output_layer=False, rgb=True, img_size=84, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.out_channels = out_channels
        self.rgb = rgb
        print("RGB images:", self.rgb)

        if self.rgb:
            rnd_input = torch.rand((1,3,img_size[0],img_size[1]))
        else:
            rnd_input = torch.rand((1,1,img_size[0],img_size[1]))

        d = OrderedDict([])
        for i in range(self.num_blocks):
            if i == 0:
                if self.rgb:
                    indim = 3
                else:
                    indim = 1
            else:
                indim = self.out_channels
            #indim = 3 if i == 0 and self.rgb else self.out_channels
            pool = i < 4 
            d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim, out_channels=self.out_channels)})
        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        print("In-features:", self.in_features)
        if not no_output_layer:
            self.model.update({"out": nn.Linear(in_features=self.in_features,
                                                out_features=self.train_classes).to(dev)})

    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights, flat=False):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        if flat:
            return x
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))



class TransformConvBlock(nn.Module):
    """
    Initialize the convolutional block consisting of:
     - 64 convolutional kernels of size 3x3
     - Batch normalization 
     - ReLU nonlinearity
     - 2x2 MaxPooling layer
     
    ...

    Attributes
    ----------
    cl : nn.Conv2d
        Convolutional layer
    bn : nn.BatchNorm2d
        Batch normalization layer
    relu : nn.ReLU
        ReLU function
    mp : nn.MaxPool2d
        Max pooling layer
    running_mean : torch.Tensor
        Running mean of the batch normalization layer
    running_var : torch.Tensor
        Running variance of the batch normalization layer
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
    """
    
    def __init__(self, dev, indim=3, pool=True):
        """Initialize the convolutional block
        
        Parameters
        ----------
        indim : int, optional
            Number of input channels (default=3)
        """
        
        super().__init__()
        self.dev = dev
        self.cl = nn.Conv2d(in_channels=indim, out_channels=64,
                            kernel_size=3, padding=1)        
        self.bn = nn.BatchNorm2d(num_features=64, momentum=1) #momentum=1 is crucial! (only statistics for current batch)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.pool = pool
        
    def forward(self, x):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            The output of the block
        """

        x = self.cl(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.mp(x)
        return x
    
    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        # Apply conv2d
        x = F.conv2d(x, weights[0], weights[1], padding=1) 

        # Manual batch normalization followed by ReLU
        running_mean =  torch.zeros(64).to(self.dev)
        running_var = torch.ones(64).to(self.dev)
        x = F.batch_norm(x, running_mean, running_var, 
                         weights[2], weights[3], momentum=1, training=True)
        if self.pool:                   
            x = F.max_pool2d(F.relu(x), kernel_size=2)
        return x
    
    def transform_forward(self, x, bweights=None, weights=None, activation=None, channel_scale=False, **kwargs):
        


        if not type(activation) == type([]):
            # Unpack scalars and shift - 
            if channel_scale:
                scale, shift = weights[0], weights[1]
            else:
                # change size of scalars from [1] to [C_out, 1, 1, 1]
                scale, shift = weights[0].view(self.cl.weight.shape[:1]+(1,1,1)), weights[1]
            # Apply conv2d
            x = F.conv2d(x, weight=bweights[0]*(1-activation+activation*scale), bias=bweights[1]+activation*shift, padding=1)
            # Manual batch normalization followed by ReLU
            running_mean =  torch.zeros(64).to(self.dev)
            running_var = torch.ones(64).to(self.dev)
            x = F.batch_norm(x, running_mean, running_var, 
                            bweights[2], bweights[3], momentum=1, training=True)
            
        else:
            if channel_scale:
                scale, shift = weights[0], weights[1]
            else:
                # change size of scalars from [1] to [C_out, 1, 1, 1]
                scale, shift = (weights[0]*activation[4]).view(self.cl.weight.shape[:1]+(1,1,1)), weights[1]
            
            # if not channel_scale:
            #     activation[0] = activation[0].view(self.cl.weight.shape[:1]+(1,1,1))
            # print(bweights[0].shape, activation[0].shape, scale.shape, activation[4].shape)
            # print(bweights[1].shape, activation[1].shape, shift.shape, activation[5].shape)
            # print(bweights[2].shape, activation[2].shape)
            # print(bweights[3].shape, activation[3].shape)

            x = F.conv2d(x, weight=bweights[0]*activation[0]*scale, bias=bweights[1]*activation[1]+shift*activation[5], padding=1)

            # Manual batch normalization followed by ReLU
            running_mean =  torch.zeros(64).to(self.dev)
            running_var = torch.ones(64).to(self.dev)
            x = F.batch_norm(x, running_mean, running_var, 
                            bweights[2]*activation[2], bweights[3]*activation[3], momentum=1, training=True)
        if self.pool:                   
            x = F.max_pool2d(F.relu(x), kernel_size=2)

        return x
        



    
    def reset_batch_stats(self):
        """Reset Batch Normalization stats
        """
        
        self.bn.reset_running_stats()


class TransformConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, channel_scale=False, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """

        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.channel_scale = channel_scale
        

        rnd_input = torch.rand((1,3,84,84))

        d = OrderedDict([])
        t = [] # list for transform parameters
        for i in range(self.num_blocks):
            indim = 3 if i == 0 else 64
            pool = i < 4 
            outdim = 64 # we need this number of transform scalars and biases

            d.update({'conv_block%i'%i: TransformConvBlock(dev=dev, pool=pool, indim=indim)})

            if self.channel_scale:
                scale_shape = [outdim, indim, 1, 1]
            else:
                scale_shape = [outdim]

            t.append( nn.Parameter(torch.ones(*scale_shape, device=dev)) ) # transform scalar
            t.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # transform bias

        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})
        
        self.transform = nn.ParameterList(t)
        # one alfa per pair (scalar, shift constant) because only shifting or only scaling has no effect
        # meaning they have to be used simultaneously in order to have an effect on the output 
        self.alfas = [nn.Parameter(torch.zeros(1, device=dev).squeeze()) for _ in range(len(self.transform)//2)]
        self.alfas = nn.ParameterList(self.alfas)

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        assert not (bweights is None or weights is None or alfas is None),\
            "Make sure all input arguments are provided for transform_forward of TransformConvX"
         
        activs = [torch.sigmoid(a) for a in alfas]
        for i in range(self.num_blocks):
            x = self.model.features[i].transform_forward(x, bweights=bweights[i*4:i*4+4], 
                                        weights=weights[i*2:i*2+2], activation=activs[i],
                                        channel_scale=self.channel_scale)

        x = self.model.features.flatten(x)
        x = F.linear(x, bweights[-2], bweights[-1])
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))


class BlockTransformConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, channel_scale=False, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """

        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.channel_scale = channel_scale
        

        rnd_input = torch.rand((1,3,84,84))

        d = OrderedDict([])
        t = OrderedDict([]) # list for transform parameters
        for i in range(self.num_blocks):
            indim = 3 if i == 0 else 64
            pool = i < 4 
            outdim = 64 # we need this number of transform scalars and biases

            d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim)})
            t.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=False, indim=outdim)})

        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})
        
        self.transform = nn.ModuleDict({"features": nn.Sequential(t)})
        # one alfa per pair (scalar, shift constant) because only shifting or only scaling has no effect
        # meaning they have to be used simultaneously in order to have an effect on the output 
        self.alfas = [nn.Parameter(torch.zeros(1, device=dev).squeeze()) for _ in range(len(self.transform))]
        self.alfas = nn.ParameterList(self.alfas)

        self.in_features = self.get_infeatures(rnd_input).size()[1]
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def transform_forward(self, x, bweights=None, weights=None, alfas=None, **kwargs):
        assert not (bweights is None or weights is None or alfas is None),\
            "Make sure all input arguments are provided for transform_forward of TransformConvX"
         
        activs = [torch.sigmoid(a) for a in alfas]
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, bweights[i*4:i*4+4])
            x = (1-activs[0])*x + activs[0]*self.transform.features[i].forward_weights(x, weights[i*4:i*4+4])

        x = self.model.features.flatten(x)
        x = F.linear(x, bweights[-2], bweights[-1])
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))



class SparseConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, trans_net=False, channel_scale=False):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        print("Created SparseConvX mannn")
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.trans_net = trans_net
        self.channel_scale = channel_scale

        rnd_input = torch.rand((1,3,84,84))

        if not self.trans_net:
            d = OrderedDict([])
            t = OrderedDict([])
            for i in range(self.num_blocks):
                indim = 3 if i == 0 else 64
                pool = i < 4 
                d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim)})
                t.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim)})

            d.update({'flatten': nn.Flatten()})
            self.model = nn.ModuleDict({"features": nn.Sequential(d)})

            self.in_features = self.get_infeatures(rnd_input).size()[1]
            self.model.update({"out": nn.Linear(in_features=self.in_features,
                                                out_features=self.train_classes).to(dev)})

            t.update({'out': nn.Linear(in_features=self.in_features,
                                    out_features=self.train_classes).to(dev)})

            self.alfas = nn.ModuleDict(t)

            # Increase initialization by factor 2 to counter initialization of alfas
            for p in self.model.parameters():
                p.data *= 2
            
            # Initialize alfas at 0
            for p in self.alfas.parameters():
                nn.init.zeros_(p)

        else:
            d = OrderedDict([])
            t = [] # list for transform parameters
            alfas = []
            for i in range(self.num_blocks):
                indim = 3 if i == 0 else 64
                pool = i < 4 
                outdim = 64 # we need this number of transform scalars and biases
                block = TransformConvBlock(dev=dev, pool=pool, indim=indim)

                d.update({"conv_block%i"%i: block})

                #############
                # Base conv block
                #############
                # Conv weight and bias
                # t.append( nn.Parameter( block.cl.weight.clone().detach()) ) # transform scalar
                # t.append( nn.Parameter( block.cl.bias.clone().detach()) ) # transform bias
                # # BatchNorm params
                # t.append( nn.Parameter(torch.ones(outdim, device=dev)) ) # scale
                # t.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # shift

                if self.channel_scale:
                    scale_shape = [outdim, indim, 1, 1]
                else:
                    scale_shape = [outdim]


                #############
                # Transform params for conv block
                #############
                # Conv weight and bias
                t.append( nn.Parameter(torch.ones(*scale_shape, device=dev)) ) # transform scalar
                t.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # transform bias
                # # BatchNorm params
                # t.append( nn.Parameter(torch.ones(outdim, device=dev)) ) # scale
                # t.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # shift


                #############
                # Alfas for base conv block
                #############
                # Conv weight and bias
                alfas.append( nn.Parameter(torch.zeros(*block.cl.weight.shape, device=dev)) ) # transform scalar
                alfas.append( nn.Parameter(torch.zeros(*block.cl.bias.shape, device=dev)) ) # transform bias
                alfas.append( nn.Parameter(torch.ones(outdim, device=dev)) ) # scale
                alfas.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # shift

                #############
                # Alfas for base conv block
                #############
                # Alfas for transforms
                alfas.append( nn.Parameter(torch.zeros(*scale_shape, device=dev)) ) # transform scalar
                alfas.append( nn.Parameter(torch.zeros(outdim, device=dev)) ) # transform bias


            d.update({'flatten': nn.Flatten()})
            self.model = nn.ModuleDict({"features": nn.Sequential(d)})
            
            
            self.transform = nn.ParameterList(t)
            

            self.in_features = self.get_infeatures(rnd_input).size()[1]
            self.model.update({"out": nn.Linear(in_features=self.in_features,
                                                out_features=self.train_classes).to(dev)})
            
            alfas.append( nn.Parameter(torch.zeros(*self.model.out.weight.shape, device=dev)) )
            alfas.append( nn.Parameter(torch.zeros(*self.model.out.bias.shape, device=dev)) )
            self.alfas = nn.ParameterList(alfas)

            self.num_base_params = self.num_blocks * 4 + 2 # every block has 4 params, add 2 for final linear


            for p in self.model.parameters():
                p.data *= 2
            for p in self.transform.parameters():
                p.data *= 2


    def get_infeatures(self,x):
        with torch.no_grad():
            for i in range(self.num_blocks):
                x = self.model.features[i](x)
            x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def transform_forward(self, x, weights, bweights, **kwargs):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        if not self.trans_net:
            for i in range(self.num_blocks):
                wls = [bweights[k]*torch.sigmoid(weights[k]) for k in range(i*4, i*4+4)]
                x = self.model.features[i].forward_weights(x, wls)

            x = self.model.features.flatten(x)
            x = F.linear(x, bweights[-2]*torch.sigmoid(weights[-2]), bweights[-1]*torch.sigmoid(weights[-1]))
        else:
            base_weights, trans_weights = bweights[:self.num_base_params], bweights[self.num_base_params:]
            activs = [torch.sigmoid(x) for x in weights]
            for i in range(self.num_blocks):
                x = self.model.features[i].transform_forward(x, bweights=base_weights[i*4:i*4+4], 
                                            weights=trans_weights[i*2:i*2+2], activation=activs[i*6:i*6+6],
                                            channel_scale=self.channel_scale)
            x = self.model.features.flatten(x)
            x = F.linear(x, base_weights[-2]*activs[-2], base_weights[-1]*activs[-1])

        return x

    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))













class PrototypeMatrix(nn.Module):
    """
    Special output layer that learns class representations.
    
    ...
    
    Attributes
    ----------
    prototypes : nn.Parameter
        Parameter-wrapped prototype matrix which contains a column for each class
        
    Methods
    ----------
    forward(x)
        Compute matrix product of x with the prototype matrix     
    """
    
    def __init__(self, in_features, num_classes, dev):
        """Initialize the prototype matrix randomly
        """
        
        super().__init__()
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.rand([in_features, num_classes], device=dev))
    
    def forward(self, x):
        """Apply the prototype matrix to input x
        
        Parameters
        ----------
        x : torch.Tensor
            The input to which we apply our prototype matrix
        
        Returns
        ----------
        tensor
            Result of applying the prototype matrix to input x
        """

        return torch.matmul(x, self.prototypes)
        
class BoostedConv4(ConvolutionalNetwork):
    """
    Convolutional neural network with special output layer.
    This output layer maintains class representations and uses 
    cosine similarity to make predictions. 
    
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    criterion : loss_fn
        Loss function to minimize
     
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers

    Code includes snippets from https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py
    """
    
    def __init__(self, train_classes, eval_classes, dev, criterion=nn.CrossEntropyLoss()):
        """Initialize the conv network

        Parameters
        ----------
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        dev : str
            String identifier of the device to use
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__(train_classes=train_classes, eval_classes=eval_classes, criterion=criterion, dev=dev)
        self.in_features = 3*3*64 #Assuming original image size 84
        self.criterion = criterion
        self.dev = dev

        # Shape is now [batch_size, in_features]
        self.model.update({"out": nn.Linear(in_features=self.in_features, 
                                            out_features=self.train_classes, 
                                            bias=False).to(self.dev)})
        
        WeightNorm.apply(self.model.out, 'weight', dim=0)
        self.scale_factor = 2

        #self.model.update({"out": PrototypeMatrix(in_features=self.in_features, 
        #                                          num_classes=self.train_classes, dev=dev)})

   
    def forward(self, x, test_x=None, test_y=None ,train=True):
        """Feedforward pass of the network

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 

        Returns
        ----------
        tensor
            Scores for every class
        """

        features = self.model.features(x)
        x_norm = torch.norm(features, p=2, dim =1).unsqueeze(1).expand_as(features)
        x_normalized = features.div(x_norm + 0.00001)
        cos_dist = self.model.out(x_normalized) 
        scores = self.scale_factor * cos_dist 
        return scores

        # features = self.model.features(x)
        # # Compute row-wise L2 norm to create a matrix of unit row vectors
        # norm = torch.div(features, torch.reshape(torch.norm(features, dim=1), [-1,1]))

        # # Compute class outputs 
        # class_outputs = self.model.out(norm)
        # return class_outputs

    def load_state_dict(self, state):
        """Overwritten load_state function
        
        Before loading the state, check whether the dimensions of the output
        layer are matching. If not, create a layer of the output size in the provided state.  

        """
        
        out_key = "model.out.weight_g"
        out_classes = state[out_key].size()[0]
        if out_classes != self.model.out.out_features:
            self.model.out = nn.Linear(in_features=self.in_features, 
                                       out_features=out_classes, 
                                       bias=False).to(self.dev)
            WeightNorm.apply(self.model.out, 'weight', dim=0)

        super().load_state_dict(state)


    def freeze_layers(self):
        """Freeze all hidden layers
        """

        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features, 
                                       out_features=self.eval_classes, 
                                       bias=False).to(self.dev)
        WeightNorm.apply(self.model.out, 'weight', dim=0)

        # self.model.out = PrototypeMatrix(in_features=self.in_features, 
        #                                  num_classes=self.eval_classes, dev=self.dev)



class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, padding, dev):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.dev = dev
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=stride,
                               padding=padding,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        #print("bnsize:", self.bn1.running_mean.size()[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=1,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.skip = stride > 1
        if self.skip:
            self.conv3 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=3, 
                               stride=stride,
                               padding=padding,
                               bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=out_channels, momentum=1)


    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn2(z)

        y = x
        if self.skip:
            y = self.conv3(y)
            y = self.bn3(y)
        return self.relu(y + z)


    def forward_weights(self, x, weights):
        # ResNet doesn't use bias in conv2d layers
        z = F.conv2d(input=x, weight=weights[0], bias=None, 
                     stride=self.stride,padding=self.padding)
        
        z = F.batch_norm(z, torch.zeros(self.bn1.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn1.running_var.size()).to(self.dev), 
                         weights[1], weights[2], momentum=1, training=True)

        z = F.relu(z)

        z = F.conv2d(input=z, weight=weights[3], bias=None, 
                     stride=1, padding=self.padding)

        z = F.batch_norm(z, torch.zeros(self.bn2.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn2.running_var.size()).to(self.dev), weights[4], 
                         weights[5], momentum=1, training=True)

        y = x
        if self.skip:
            y = F.conv2d(input=y, weight=weights[6], bias=None, 
                     stride=self.stride,padding=self.padding)

            y = F.batch_norm(input=y, running_mean=torch.zeros(self.bn3.running_mean.size()).to(self.dev), 
                             running_var=torch.ones(self.bn3.running_var.size()).to(self.dev), weight=weights[7], 
                             bias=weights[8], momentum=1, training=True)

        return F.relu(y + z)

class ResNet(nn.Module):

    def __init__(self, num_blocks, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.num_blocks = num_blocks
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.dev = dev
        self.criterion = criterion
        

        print("ResNet constructor called with num_blocks",num_blocks)
        if num_blocks == 10:
            layers = [1,1,1,1]
            filters = [64,128,256,512]
        elif num_blocks == 12:
            layers = [1,1,1,1]
            filters = [64,128,256,512]
        elif num_blocks == 18:
            layers = [2,2,2,2]
            filters = [64,128,256,512]
        elif num_blocks == 34:
            layers = [3,4,6,3]
            filters = [64,128,256,512]
        else:
            print("Did not recognize the ResNet. It must be resnet10,18,or 34")
            import sys; sys.exit()

        self.num_resunits = sum(layers)

        
        self.conv =  nn.Conv2d(in_channels=3, kernel_size=7, 
                                out_channels=64,
                                stride=2,
                                padding=3,
                                bias=False)
        self.bn =  nn.BatchNorm2d(num_features=64,momentum=1)
        self.relu = nn.ReLU()
        #self.maxpool2d =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.globalpool2d = nn.AvgPool2d(7)
        self.flatten = nn.Flatten()
        d = OrderedDict([])

        inpsize = 56        
        c = 0
        prev_filter = 64
        for idx, (layer, filter) in enumerate(zip(layers, filters)):
            stride = 1
            if idx == 0:
                indim = 64
            else:
                indim = filters[idx-1]
                

            for i in range(layer):
                if i > 0:
                    indim = filter
                if stride == 2:
                    inpsize //= 2
                if prev_filter != filter:
                    stride = 2
                else:
                    stride = 1
                prev_filter = filter


                outsize = int(math.ceil(float(inpsize) / float(stride)))
                if inpsize % stride == 0:
                    padding = math.ceil(max((3 - stride),0)/2)
                else:
                    padding = math.ceil(max(3 - (inpsize % stride),0)/2)


                #padding = math.ceil((3 - stride * 3) * (1 - stride)/2)
                # print("filter:", filter, "input size:", inpsize, "padding:", padding)
                # print("indim:", indim, "stride", stride)
                d.update({'res_block%i'%c: ResidualBlock(in_channels=indim, 
                                                         out_channels=filter,
                                                         stride=stride,
                                                         padding=padding,
                                                         dev=dev)})
                c+=1
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})


        rnd_input = torch.rand((1,3,224,224))
        self.in_features = self.get_infeatures(rnd_input).size()[1]
        print("Dimensionality of the embedding:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

        print([x.size() for x in self.model.parameters()])


    def get_infeatures(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        # x = F.avg_pool2d(x, kernel_size=3)
        x = F.avg_pool2d(x, kernel_size=7)
        x = self.flatten(x)
        return x


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        x = F.avg_pool2d(x, kernel_size=7)
        #x = F.avg_pool2d(x, kernel_size=3)
        x = self.flatten(x)
        x = self.model.out(x)
        return x

    def forward_get_features(self, x):
        features = []
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        features.append(self.flatten(x).clone().cpu().detach().numpy())
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
            features.append(self.flatten(x).clone().cpu().detach().numpy())
        x = F.avg_pool2d(x, kernel_size=7)
        x = self.flatten(x)
        x = self.model.out(x)
        features.append(self.flatten(x).clone().cpu().detach().numpy())
        return x, features

    def forward_weights(self, x, weights):
        z = F.conv2d(input=x, weight=weights[0], bias=None, 
                     stride=2,padding=3)

        z = F.batch_norm(z, torch.zeros(self.bn.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn.running_var.size()).to(self.dev), 
                         weights[1], weights[2], momentum=1, training=True)

        z = F.relu(z)
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)

        lb = 3
        for i in range(self.num_resunits):
            if self.model.features[i].skip:
                incr = 9
            else:
                incr = 6
            z = self.model.features[i].forward_weights(z, weights[lb:lb+incr])
            lb += incr

        z = F.avg_pool2d(z, kernel_size=7)
        #z = F.avg_pool2d(z, kernel_size=3)
        z = self.flatten(z)
        z = F.linear(z, weight=weights[-2], bias=weights[-1])
        return z

    def forward_weights_get_features(self, x, weights):
        features = []
        z = F.conv2d(input=x, weight=weights[0], bias=None, 
                     stride=2,padding=3)

        z = F.batch_norm(z, torch.zeros(self.bn.running_mean.size()).to(self.dev), 
                         torch.ones(self.bn.running_var.size()).to(self.dev), 
                         weights[1], weights[2], momentum=1, training=True)

        z = F.relu(z)
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)
        features.append(self.flatten(z).clone().cpu().detach().numpy())
        lb = 3
        for i in range(self.num_resunits):
            if self.model.features[i].skip:
                incr = 9
            else:
                incr = 6
            z = self.model.features[i].forward_weights(z, weights[lb:lb+incr])
            features.append(self.flatten(z).clone().cpu().detach().numpy())
            lb += incr

        z = F.avg_pool2d(z, kernel_size=7)
        z = self.flatten(z)
        z = F.linear(z, weight=weights[-2], bias=weights[-1])
        features.append(self.flatten(z).clone().cpu().detach().numpy())
        return z, features

    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))

    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)






















class SVDConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, linear_transform=False, 
                max_pool_before_transform=False, old=False, discrete_ops=False, trans_before_relu=False, out_channels=64, 
                use_grad_mask=False, bn_before_trans=False, bn_after_trans=False, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.linear_transform = linear_transform
        self.max_pool_before_transform = max_pool_before_transform
        self.old = old
        self.discrete_ops = discrete_ops
        self.trans_before_relu = trans_before_relu
        self.out_channels = out_channels
        self.use_grad_mask = use_grad_mask
        self.bn_before_trans = bn_before_trans
        self.bn_after_trans = bn_after_trans

        if self.use_grad_mask:
            grad_masks = []
            out_dims = [3, 64, 64, 64, 64]
            if self.linear_transform:
                out_dims += [5]
            for lid in range(len(out_dims)):
                outdim = out_dims[lid]
                mask = nn.Parameter(torch.zeros(outdim))
                grad_masks.append(mask)
            self.grad_masks = nn.ParameterList(grad_masks)
            self.pid_to_lid = []
            self.param_types = [] 


        print("linear transform:", self.linear_transform)
        print("max_pool_before_transform:", self.max_pool_before_transform)

        rnd_input = torch.rand((1,3,84,84))

        d = OrderedDict([])
        t = []
        alfas = []
        if self.discrete_ops:
            distributions = [] # distributions (p(active), p(not-active)) for every transformation
        for i in range(self.num_blocks + 1):
            indim = 3 if i == 0 else self.out_channels
            outdim = 3 if i == 0 else self.out_channels
            pool = i < 4 
            if i < self.num_blocks:
                d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim, out_channels=self.out_channels)})


            # 1x1 conv
            #bound1x1 = 1/(indim)
            conv1x1 = torch.zeros(outdim, indim, 1, 1)
            for dim in range(outdim):
                conv1x1[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
            conv1x1 = nn.Parameter( conv1x1 )
            t.append(conv1x1)
            #nn.init.uniform_(conv1x1, -bound1x1, +bound1x1)

            # 3x3 conv
            #bound3x3 = 1/(indim*9)
            conv3x3 = torch.zeros(outdim, indim, 3, 3)
            for dim in range(outdim):
                conv3x3[dim,dim,0,0] = 1
            conv3x3 = nn.Parameter(conv3x3)
            t.append(conv3x3)
            #nn.init.uniform_(conv3x3, -bound3x3, +bound3x3)

            # 3x3 conv SVD
            U = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) # shape (outdim, indim, 3, 1)
            V = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) 
            for dim in range(indim):
                U.data[dim,dim,0,0] = 1
                V.data[dim,dim,0,0] = 1

            S = nn.Parameter( torch.ones(outdim, indim, 1) )# (outdim, indim, 1)
            t.append(U)
            t.append(S)
            t.append(V)

            alfas.append( nn.Parameter(torch.zeros(4)) )
            if self.discrete_ops:
                for _ in range(3):
                    distributions.append( nn.Parameter(torch.zeros(2)) )

            if self.use_grad_mask:
                self.param_types += [ParamType.ConvTensor, ParamType.ConvTensor, ParamType.ConvSVD_U, ParamType.ConvSVD_S, ParamType.ConvSVD_V]
                self.pid_to_lid += [i for _ in range(5)]

            # These operate on the weights of the original conv weights
            if i != 0:
                # MTL scale/shift -- cant apply yet when i == 0
                inm = 3 if i == 1 else self.out_channels
                outm = self.out_channels 
                mtl_scale = nn.Parameter(torch.ones(outm, inm, 1, 1))
                # Simple scale/shift
                simple_scale = nn.Parameter( torch.ones(outm) )
                t.append(mtl_scale)
                t.append(simple_scale)
                alfas.append( nn.Parameter(torch.zeros(3)) )
                if self.discrete_ops:
                    for _ in range(2):
                        distributions.append( nn.Parameter(torch.zeros(2)) )
                
                if self.use_grad_mask:
                    self.param_types += [ParamType.MTLScale, ParamType.SimpleScale]
                    self.pid_to_lid += [i for _ in range(2)]
            
            bias_const = nn.Parameter(torch.zeros(1).squeeze())
            bias_vect = nn.Parameter(torch.zeros(outdim)) 
            t.append(bias_const)
            t.append(bias_vect)

            alfas.append( nn.Parameter(torch.zeros(3)) )
            if self.discrete_ops:
                for _ in range(2):
                    distributions.append( nn.Parameter(torch.zeros(2)) )

            if self.use_grad_mask:
                self.param_types += [ParamType.Scalar, ParamType.Vector]
                self.pid_to_lid += [i for _ in range(2)]

        # transform params = 
        # if layer == 0: [conv1x1, conv3x3, U1,S1,V1, bias_const, bias_vect]
        # else: [conv1x1, conv3x3, U1,S1,V1, mtl_scale, simple_scale, bias_const, bias_vect]

        # alfas = [aconv(4), ascale(3), abias(3)]

        # bweights = [conv_w, conv_b, batch_norm_w, batch_norm_b]




        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})
        self.in_features = self.get_infeatures(rnd_input).size()[1]

        if not self.old:
            for k in [1,2,4]:
                U = torch.zeros(self.train_classes, k)
                V = torch.zeros(self.in_features, k)
                S = torch.ones(k)
                nn.init.eye_(U); nn.init.eye_(V)
                t.append(nn.Parameter(U))
                t.append(nn.Parameter(S))
                t.append(nn.Parameter(V))

            alfas.append(nn.Parameter(torch.zeros(4)))


        # 5x5 transform after final layer
        if self.linear_transform:
            weight = nn.Parameter( torch.eye(self.train_classes) )
            bias = nn.Parameter( torch.zeros(self.train_classes) )
            t.append(weight)
            t.append(bias)
            alfas.append( nn.Parameter(torch.zeros(1)) )
            if self.discrete_ops:
                distributions.append( nn.Parameter(torch.zeros(2)) )

            if self.use_grad_mask:
                self.param_types += [ParamType.Matrix, ParamType.Scalar]
                self.pid_to_lid += [5 for _ in range(2)]
        
        if self.bn_before_trans or self.bn_after_trans:
            bn_weight = nn.Parameter(torch.ones(3))
            bn_bias = nn.Parameter(torch.zeros(3))
            t.append(bn_weight)
            t.append(bn_bias)
            for j in range(4):
                bn_weight = nn.Parameter(torch.ones(64))
                bn_bias = nn.Parameter(torch.zeros(64))
                t.append(bn_weight)
                t.append(bn_bias)



        self.transform = nn.ParameterList(t)
        self.alfas = nn.ParameterList(alfas)
        if self.discrete_ops:
            self.distributions = nn.ParameterList(distributions)
            self.num_distributions = len(self.distributions)


        print("In-features:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})


        print("self.discrete_ops:", self.discrete_ops)
        print("self.trans_before_relu:", self.trans_before_relu)
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("Number of parameters:", num_params)
        self.ignore_linear_transform = False



    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def transform_forward(self, x, bweights=None, weights=None, alfas=None, binary_masks=None, **kwargs):
        assert not (binary_masks is None and self.discrete_ops), "please feed in binary masks when using --discrete ops"

        [conv1x1, conv3x3, U,S,V, bias_const, bias_vect] = weights[:7]
        conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
        [conv_alfas, bias_alfas] = alfas[:2]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)


        if self.bn_before_trans:
            running_mean =  torch.zeros(3).to(self.dev)
            running_var = torch.ones(3).to(self.dev)
            bn_weight, bn_bias = weights[-10:-8]
            x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)

        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        # Transform input
        x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
        x2 = F.conv2d(x_pad, weight=conv3x3, bias=None)
        x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

        #print(conv_svd.size(), conv3x3.size(), conv1x1.size())
        #print(x1.size(), x2.size(), x3.size(), x.size())

        if self.discrete_ops:
            masks = binary_masks[:5]
            x = conv_alfas[0]*x + masks[0][0]*conv_alfas[1]*x1 + masks[1][0]*conv_alfas[2]*x2 + masks[2][0]*conv_alfas[3]*x3
            #print(x.size(), bias_const.size(), bias_vect.size())
            x = bias_alfas[0]*x + masks[3][0]**bias_alfas[1]*(x+bias_const) + masks[4][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
        else:
            x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
            #print(x.size(), bias_const.size(), bias_vect.size())
            x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))

        if self.bn_after_trans:
            running_mean =  torch.zeros(3).to(self.dev)
            running_var = torch.ones(3).to(self.dev)
            bn_weight, bn_bias = weights[-10:-8]
            x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)

        for lid in range(self.num_blocks):
            conv, conv_bias,bn_weight,bn_bias = bweights[lid*4: lid*4+4] # (conv_weight, conv_bias, running_mean, running_std)
            [conv1x1, conv3x3, U,S,V, mtl_scale, simple_scale, bias_const, bias_vect] = weights[7+lid*9:7+lid*9+9]

            conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
            simple_scale = simple_scale.view(conv3x3.shape[:1]+(1,1,1))

            [conv_alfas, mtl_alfas, bias_alfas] = alfas[2+lid*3:2+lid*3+3]
            conv_alfas = F.softmax(conv_alfas, dim=0)
            mtl_alfas = F.softmax(mtl_alfas, dim=0)
            bias_alfas = F.softmax(bias_alfas, dim=0)

            if self.discrete_ops:
                masks = binary_masks[5+(lid)*7:5+(lid+1)*7]
                conv_weights = mtl_alfas[0]*conv + masks[0][0]*mtl_alfas[1]*conv*mtl_scale + masks[1][0]*mtl_alfas[2]*conv*simple_scale
            else:
                conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

            x = F.conv2d(x, weight=conv_weights, bias=conv_bias, padding=1)

            # Batch norm
            running_mean =  torch.zeros(self.out_channels).to(self.dev)
            running_var = torch.ones(self.out_channels).to(self.dev)
            x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)
            
            if self.trans_before_relu:
                x_pad = F.pad(x, (0,2,0,2), mode='constant')
                # Transform input
                x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
                x2 = F.conv2d(x_pad, weight=conv3x3, bias=None)
                x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

                if self.discrete_ops:
                    x = conv_alfas[0]*x + masks[2][0]*conv_alfas[1]*x1 + masks[3][0]*conv_alfas[2]*x2 + masks[4][0]*conv_alfas[3]*x3
                    x = bias_alfas[0]*x + masks[5][0]*bias_alfas[1]*(x+bias_const) + masks[6][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
                else:
                    x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
                    x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
            
            
            x = F.relu(x)

            if self.max_pool_before_transform:
                x = F.max_pool2d(x, kernel_size=2)

            if not self.trans_before_relu:
                if self.bn_before_trans:
                    running_mean =  torch.zeros(self.out_channels).to(self.dev)
                    running_var = torch.ones(self.out_channels).to(self.dev)
                    if -8 + (lid+1)*2 == 0:
                        bn_weight, bn_bias = weights[-10 + (lid+1)*2:]
                    else:
                        bn_weight, bn_bias = weights[-10 + (lid+1)*2: -8 + (lid+1)*2]
                    x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)


                x_pad = F.pad(x, (0,2,0,2), mode='constant')
                # Transform input
                x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
                x2 = F.conv2d(x_pad, weight=conv3x3, bias=None)
                x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

                if self.discrete_ops:
                    x = conv_alfas[0]*x + masks[2][0]*conv_alfas[1]*x1 + masks[3][0]*conv_alfas[2]*x2 + masks[4][0]*conv_alfas[3]*x3
                    x = bias_alfas[0]*x + masks[5][0]*bias_alfas[1]*(x+bias_const) + masks[6][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
                else:
                    x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
                    x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
                

            if not self.max_pool_before_transform:
                x = F.max_pool2d(x, kernel_size=2)
        
            if self.bn_after_trans:
                running_mean =  torch.zeros(self.out_channels).to(self.dev)
                running_var = torch.ones(self.out_channels).to(self.dev)
                if -8 + (lid+1)*2 == 0:
                    bn_weight, bn_bias = weights[-10 + (lid+1)*2:]
                else:
                    bn_weight, bn_bias = weights[-10 + (lid+1)*2: -8 + (lid+1)*2]
                x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)

        x = self.model.features.flatten(x)
        weight, bias = bweights[-2:]

        if not self.old:
            if not self.linear_transform:
                [U1,S1,V1,U2,S2,V2,U3,S3,V3] = weights[-9:]
                activs = alfas[-1]
            else:
                [U1,S1,V1,U2,S2,V2,U3,S3,V3] = weights[-11:-2]
                activs = alfas[-2]

            activs = F.softmax(activs, dim=0)
            W1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
            W2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
            W3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)

            x = activs[0]*F.linear(x, weight, bias=None) + activs[1]*F.linear(x, W1, bias=None) + activs[2]*F.linear(x, W2, bias=None) +\
                activs[3]*F.linear(x, W3, bias=None) + bias
        else:
            x = F.linear(x, weight, bias=bias)


        if self.linear_transform and not self.ignore_linear_transform:
            #print('hereeeeeeeeee')
            if self.bn_before_trans or self.bn_after_trans:
                weight, bias = weights[-12:-10]
            else:
                weight, bias = weights[-2:]
            alfa = torch.sigmoid(alfas[-1])
            if self.discrete_ops:
                mask = binary_masks[-1]
                x = (1-alfa)*x + mask[0]*alfa*F.linear(x, weight, bias)
            else:
                x = (1-alfa)*x + alfa*F.linear(x, weight, bias)

        return x

    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))









class SimpleSVDConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, linear_transform=False, 
                max_pool_before_transform=False, old=False, discrete_ops=False, trans_before_relu=False, out_channels=64, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.linear_transform = linear_transform
        self.max_pool_before_transform = max_pool_before_transform
        self.old = old
        self.discrete_ops = discrete_ops
        self.trans_before_relu = trans_before_relu
        self.out_channels = out_channels

        print("linear transform:", self.linear_transform)
        print("max_pool_before_transform:", self.max_pool_before_transform)
        print("USING SIMPLE SVDCONVNET")

        rnd_input = torch.rand((1,3,84,84))

        d = OrderedDict([])
        t = []
        alfas = []
        if self.discrete_ops:
            distributions = [] # distributions (p(active), p(not-active)) for every transformation
        for i in range(self.num_blocks + 1):
            indim = 3 if i == 0 else self.out_channels
            outdim = 3 if i == 0 else self.out_channels
            pool = i < 4 
            if i < self.num_blocks:
                d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim, out_channels=self.out_channels)})


            # 1x1 conv
            #bound1x1 = 1/(indim)
            conv1x1 = torch.zeros(outdim, indim, 1, 1)
            for dim in range(outdim):
                conv1x1[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
            conv1x1 = nn.Parameter( conv1x1 )
            t.append(conv1x1)
            #nn.init.uniform_(conv1x1, -bound1x1, +bound1x1)

            # 3x3 conv
            #bound3x3 = 1/(indim*9)
            # conv3x3 = torch.zeros(outdim, indim, 3, 3)
            # for dim in range(outdim):
            #     conv3x3[dim,dim,0,0] = 1
            # conv3x3 = nn.Parameter(conv3x3)
            # t.append(conv3x3)
            #nn.init.uniform_(conv3x3, -bound3x3, +bound3x3)

            # 3x3 conv SVD
            U = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) # shape (outdim, indim, 1, 1)
            V = nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(indim)]) for _ in range(outdim)]) ) 
            for dim in range(indim):
                U.data[dim,dim,0,0] = 1
                V.data[dim,dim,0,0] = 1

            S = nn.Parameter( torch.ones(outdim, indim, 1) )# (outdim, indim, 1)
            t.append(U)
            t.append(S)
            t.append(V)

            alfas.append( nn.Parameter(torch.zeros(3)) )
            if self.discrete_ops:
                for _ in range(3):
                    distributions.append( nn.Parameter(torch.zeros(2)) )

            # These operate on the weights of the original conv weights
            if i != 0:
                # MTL scale/shift -- cant apply yet when i == 0
                inm = 3 if i == 1 else self.out_channels
                outm = self.out_channels 
                mtl_scale = nn.Parameter(torch.ones(outm, inm, 1, 1))
                # Simple scale/shift
                simple_scale = nn.Parameter( torch.ones(outm) )
                t.append(mtl_scale)
                t.append(simple_scale)
                alfas.append( nn.Parameter(torch.zeros(3)) )
                if self.discrete_ops:
                    for _ in range(2):
                        distributions.append( nn.Parameter(torch.zeros(2)) )
            
            bias_const = nn.Parameter(torch.zeros(1).squeeze())
            bias_vect = nn.Parameter(torch.zeros(outdim)) 
            t.append(bias_const)
            t.append(bias_vect)

            alfas.append( nn.Parameter(torch.zeros(3)) )
            if self.discrete_ops:
                for _ in range(2):
                    distributions.append( nn.Parameter(torch.zeros(2)) )

        # transform params = 
        # if layer == 0: [conv1x1, conv3x3, U1,S1,V1, bias_const, bias_vect]
        # else: [conv1x1, conv3x3, U1,S1,V1, mtl_scale, simple_scale, bias_const, bias_vect]

        # alfas = [aconv(4), ascale(3), abias(3)]

        # bweights = [conv_w, conv_b, batch_norm_w, batch_norm_b]




        d.update({'flatten': nn.Flatten()})
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})
        self.in_features = self.get_infeatures(rnd_input).size()[1]

        if not self.old:
            for k in [1,2,4]:
                U = torch.zeros(self.train_classes, k)
                V = torch.zeros(self.in_features, k)
                S = torch.ones(k)
                nn.init.eye_(U); nn.init.eye_(V)
                t.append(nn.Parameter(U))
                t.append(nn.Parameter(S))
                t.append(nn.Parameter(V))

            alfas.append(nn.Parameter(torch.zeros(4)))


        # 5x5 transform after final layer
        if self.linear_transform:
            weight = nn.Parameter( torch.eye(self.train_classes) )
            bias = nn.Parameter( torch.zeros(self.train_classes) )
            t.append(weight)
            t.append(bias)
            alfas.append( nn.Parameter(torch.zeros(1)) )
            if self.discrete_ops:
                distributions.append( nn.Parameter(torch.zeros(2)) )

        self.transform = nn.ParameterList(t)
        self.alfas = nn.ParameterList(alfas)
        if self.discrete_ops:
            self.distributions = nn.ParameterList(distributions)
            self.num_distributions = len(self.distributions)


        print("In-features:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})


        print("self.discrete_ops:", self.discrete_ops)
        print("self.trans_before_relu:", self.trans_before_relu)



    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def transform_forward(self, x, bweights=None, weights=None, alfas=None, binary_masks=None, **kwargs):
        assert not (binary_masks is None and self.discrete_ops), "please feed in binary masks when using --discrete ops"

        [conv1x1, U,S,V, bias_const, bias_vect] = weights[:6]
        conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
        [conv_alfas, bias_alfas] = alfas[:2]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)


        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        # Transform input
        x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
        x2 = F.conv2d(x_pad, weight=conv_svd, bias=None)

        #print(conv_svd.size(), conv3x3.size(), conv1x1.size())
        #print(x1.size(), x2.size(), x3.size(), x.size())

        if self.discrete_ops:
            masks = binary_masks[:4]
            x = conv_alfas[0]*x + masks[0][0]*conv_alfas[1]*x1 + masks[1][0]*conv_alfas[2]*x2
            #print(x.size(), bias_const.size(), bias_vect.size())
            x = bias_alfas[0]*x + masks[2][0]**bias_alfas[1]*(x+bias_const) + masks[3][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
        else:
            x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 
            #print(x.size(), bias_const.size(), bias_vect.size())
            x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))


        for lid in range(self.num_blocks):
            conv, conv_bias,bn_weight,bn_bias = bweights[lid*4: lid*4+4] # (conv_weight, conv_bias, running_mean, running_std)
            [conv1x1, U,S,V, mtl_scale, simple_scale, bias_const, bias_vect] = weights[6+lid*8:6+lid*8+8]

            conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
            simple_scale = simple_scale.view(conv.shape[:1]+(1,1,1))

            [conv_alfas, mtl_alfas, bias_alfas] = alfas[2+lid*3:2+lid*3+3]
            conv_alfas = F.softmax(conv_alfas, dim=0)
            mtl_alfas = F.softmax(mtl_alfas, dim=0)
            bias_alfas = F.softmax(bias_alfas, dim=0)

            if self.discrete_ops:
                masks = binary_masks[4+(lid)*6:5+(lid+1)*6]
                conv_weights = mtl_alfas[0]*conv + masks[0][0]*mtl_alfas[1]*conv*mtl_scale + masks[1][0]*mtl_alfas[2]*conv*simple_scale
            else:
                conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

            x = F.conv2d(x, weight=conv_weights, bias=conv_bias, padding=1)

            # Batch norm
            running_mean =  torch.zeros(self.out_channels).to(self.dev)
            running_var = torch.ones(self.out_channels).to(self.dev)
            x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)
            
            if self.trans_before_relu:
                x_pad = F.pad(x, (0,2,0,2), mode='constant')
                # Transform input
                x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
                x2 = F.conv2d(x_pad, weight=conv_svd, bias=None)

                if self.discrete_ops:
                    x = conv_alfas[0]*x + masks[2][0]*conv_alfas[1]*x1 + masks[3][0]*conv_alfas[2]*x2
                    x = bias_alfas[0]*x + masks[4][0]*bias_alfas[1]*(x+bias_const) + masks[5][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
                else:
                    x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2
                    x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
            
            
            x = F.relu(x)

            if self.max_pool_before_transform:
                x = F.max_pool2d(x, kernel_size=2)

            if not self.trans_before_relu:
                x_pad = F.pad(x, (0,2,0,2), mode='constant')
                # Transform input
                x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
                x2 = F.conv2d(x_pad, weight=conv_svd, bias=None)

                if self.discrete_ops:
                    x = conv_alfas[0]*x + masks[2][0]*conv_alfas[1]*x1 + masks[3][0]*conv_alfas[2]*x2
                    x = bias_alfas[0]*x + masks[4][0]*bias_alfas[1]*(x+bias_const) + masks[5][0]*bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
                else:
                    x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2
                    x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))

            if not self.max_pool_before_transform:
                x = F.max_pool2d(x, kernel_size=2)
        
        x = self.model.features.flatten(x)
        weight, bias = bweights[-2:]

        if not self.old:
            if not self.linear_transform:
                [U1,S1,V1,U2,S2,V2,U3,S3,V3] = weights[-9:]
                activs = alfas[-1]
            else:
                [U1,S1,V1,U2,S2,V2,U3,S3,V3] = weights[-11:-2]
                activs = alfas[-2]

            activs = F.softmax(activs, dim=0)
            W1 = torch.mm(torch.mm(U1, torch.diag(S1)), V1.T)
            W2 = torch.mm(torch.mm(U2, torch.diag(S2)), V2.T)
            W3 = torch.mm(torch.mm(U3, torch.diag(S3)), V3.T)

            x = activs[0]*F.linear(x, weight, bias=None) + activs[1]*F.linear(x, W1, bias=None) + activs[2]*F.linear(x, W2, bias=None) +\
                activs[3]*F.linear(x, W3, bias=None) + bias
        else:
            x = F.linear(x, weight, bias=bias)


        if self.linear_transform:
            weight, bias = weights[-2:]
            alfa = torch.sigmoid(alfas[-1])
            if self.discrete_ops:
                mask = binary_masks[-1]
                x = (1-alfa)*x + mask[0]*alfa*F.linear(x, weight, bias)
            else:
                x = (1-alfa)*x + alfa*F.linear(x, weight, bias)

        return x

    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))













class TNetConvX(nn.Module):
    """
    Convolutional neural network consisting of X ConvBlock layers.
     
    ...

    Attributes
    ----------
    model : nn.ModuleDict
        Full sequential specification of the model
    in_features : int
        Number of input features to the final output layer
    train_state : state_dict
        State of the model for training
        
    Methods
    -------
    forward(x)
        Perform a feed-forward pass using inputs x
    
    forward_weights(x, weights)
        Perform a feedforward pass on inputs x making using
        @weights instead of the object's weights (w1, w2, w3)
        
    freeze_layers()
        Freeze all hidden layers
    """
    
    def __init__(self, dev, train_classes, eval_classes, criterion=nn.CrossEntropyLoss(), num_blocks=4, 
                 out_channels=64, use_grad_mask=False, warpgrad=False, transform_out_channels=None, use_bias=False, **kwargs):
        """Initialize the conv network

        Parameters
        ----------
        dev : str
            Device to put the model on
        train_classes : int
            Number of training classes
        eval_classes : int
            Number of evaluation classes
        criterion : loss_fn
            Loss function to use (default = cross entropy)
        """
        
        super().__init__()
        self.num_blocks = num_blocks
        self.dev = dev
        self.in_features = 3*3*64
        self.criterion = criterion
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.out_channels = out_channels
        self.use_grad_mask = use_grad_mask
        self.warpgrad = warpgrad
        self.transform_out_channels = transform_out_channels
        self.use_bias = use_bias

        if self.use_bias:
            print("USING BIAS")

        if self.transform_out_channels is None:
            self.transform_out_channels = self.out_channels


        if self.warpgrad:
            ksize = 3
        else:
            ksize = 1
            self.use_bias = False
        assert not (self.warpgrad and self.use_grad_mask), "warpgrad not compatible with grad masks"

        print("USING T-NET architecture")
        rnd_input = torch.rand((1,3,84,84))

        if self.use_grad_mask:
            grad_masks = []
            out_dims = [self.out_channels, self.out_channels, self.out_channels, self.out_channels]
            for lid in range(len(out_dims)):
                outdim = out_dims[lid]
                mask = nn.Parameter(torch.zeros(outdim))
                grad_masks.append(mask)
            self.grad_masks = nn.ParameterList(grad_masks)
            self.pid_to_lid = []
            self.param_types = []

            for _ in range(4):
                self.param_types += [ParamType.ConvTensor, ParamType.Vector, ParamType.Vector, ParamType.Vector]
                self.pid_to_lid += [_ for k in range(4)]
            self.param_types += [ParamType.Matrix, ParamType.Vector]
            self.pid_to_lid += [None, None]

        d = OrderedDict([])
        t = []
        alfas = []
        for i in range(self.num_blocks):
            indim = 3 if i == 0 else self.out_channels
            if self.warpgrad:
                if i > 1:
                    indim = self.transform_out_channels

            pool = i < 4 
            if i < self.num_blocks:
                d.update({'conv_block%i'%i: ConvBlock(dev=dev, pool=pool, indim=indim, out_channels=self.out_channels)})

            # they didnt use warp layer for layer 1
            if warpgrad and i == 0:
                continue
            if self.warpgrad and (i == self.num_blocks - 1):
                conv1x1 = torch.zeros(self.out_channels, self.out_channels, ksize, ksize)
                # if not self.warpgrad:
                #     for dim in range(self.out_channels):
                #         conv1x1[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
            else:
                conv1x1 = torch.zeros(self.transform_out_channels, self.out_channels, ksize, ksize)
                if self.transform_out_channels == self.out_channels and not self.warpgrad:
                    for dim in range(self.transform_out_channels):
                        conv1x1[dim,dim,0,0] = 1 # initialize the filters with a 1 in the top-left corner and zeros elsewhere
            
            t.append( nn.Parameter( conv1x1 ) )
            if self.use_bias:
                if self.warpgrad and (i == self.num_blocks - 1):
                    bias = torch.zeros(self.out_channels)
                else:
                    bias = torch.zeros(self.transform_out_channels)
                t.append( nn.Parameter(bias) )
            
            # if self.warpgrad:
            #     # batch norm after warp layers
            #     scale = torch.ones(self.out_channels)
            #     shift = torch.zeros(self.out_channels)
            #     t.append(nn.Parameter(scale))
            #     t.append(nn.Parameter(shift))


        if not self.warpgrad:
            t.append( nn.Parameter(torch.eye(self.train_classes)) )
        d.update({'flatten': nn.Flatten()})
        self.transform = nn.ParameterList(t)
        self.alfas = nn.ParameterList([])
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})
        self.in_features = self.transform_forward(rnd_input, list(self.model.parameters()), list(self.transform.parameters()), get_shape=True)[1]
        print("In-features:", self.in_features)
        self.model.update({"out": nn.Linear(in_features=self.in_features,
                                            out_features=self.train_classes).to(dev)})

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("Number of parameters:", num_params)
        


    def get_infeatures(self,x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        print(x.size())
        x = self.model.features.flatten(x)
        return x

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        return x

    def forward_weights(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        return x
    
    def transform_forward(self, x, bweights=None, weights=None, get_shape=False, **kwargs):
        if not self.use_bias:
            const = 1
            bias_transform = None
        else:
            const = 2

        for lid in range(self.num_blocks):
            conv, conv_bias,bn_weight,bn_bias = bweights[lid*4: lid*4+4] # (conv_weight, conv_bias, running_mean, running_std)
            if not self.warpgrad:
                conv1x1_transform = weights[lid]
            elif lid > 0 and self.warpgrad:
                conv1x1_transform = weights[(lid-1)*const]
                if self.use_bias:
                    bias_transform = weights[(lid-1)*const + 1]
                #bn_scale, bn_shift = weights[(lid-1)*4+2], weights[(lid-1)*4+3]


            x = F.conv2d(x, weight=conv, bias=conv_bias, padding=1)
            if not self.warpgrad:
                x = F.conv2d(x, weight=conv1x1_transform, bias=None) # no padding because its 1x1 conv  

            # Batch norm
            if not get_shape:
                running_mean =  torch.zeros(self.out_channels).to(self.dev)
                running_var = torch.ones(self.out_channels).to(self.dev)
            else:
                running_mean =  torch.zeros(self.out_channels)
                running_var = torch.ones(self.out_channels)

            x = F.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, momentum=1, training=True)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2)

            if self.warpgrad and lid > 0:
                x = F.conv2d(x, weight=conv1x1_transform, bias=bias_transform, padding=1)
            
            #x = F.batch_norm(x, running_mean, running_var, bn_scale, bn_shift, momentum=1, training=True)

        x = self.model.features.flatten(x)
        if get_shape:
            return x.size()
        weight, bias = bweights[-2:]
        x = F.linear(x, weight, bias=bias)
        if not self.warpgrad:
            x = F.linear(x, weights[-1], bias=None) # transform
        return x

    def forward_get_features(self, x):
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i](x)
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = self.model.out(x)
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def forward_weights_get_features(self, x, weights):
        """Manual feedforward pass of the network with provided weights

        Parameters
        ----------
        x : torch.Tensor
            Real-valued input tensor 
        weights : list
            List of torch.Tensor weight variables

        Returns
        ----------
        tensor
            The output of the block
        """
        features = []
        for i in range(self.num_blocks):
            x = self.model.features[i].forward_weights(x, weights[i*4:i*4+4])
            features.append(self.model.features.flatten(x).clone().cpu().detach().numpy())
        x = self.model.features.flatten(x)
        x = F.linear(x, weights[-2], weights[-1])
        features.append(x.clone().cpu().detach().numpy())
        return x, features

    def get_flat_params(self):
        """Returns parameters in flat format
        
        Flattens the current weights and returns them
        
        Returns
        ----------
        tensor
            Weight tensor containing all model weights
        """
        return torch.cat([p.view(-1) for p in self.parameters()], 0)

    def copy_flat_params(self, cI):
        """Copy parameters to model 
        
        Set the model parameters to be equal to cI
        
        Parameters
        ----------
        cI : torch.Tensor
            Flat tensor with the same number of elements as weights in the network
        """
        
        idx = 0
        for p in self.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen
    
    def load_params(self, state_dict):
        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]
        self.load_state_dict(state_dict, strict=False)

    def transfer_params(self, learner_w_grad, cI):
        """Transfer model parameters
        
        Transfer parameters from cI to this network, while maintaining mean 
        and variance of Batch Normalization 
        
        Parameters
        ----------
        learner_w_grad : nn.Module
            Base-learner network which records gradients
        cI : torch.Tensor
            Flat tensor of weights to copy to this network's parameters
        """
        
        # Use load_state_dict only to copy the running mean/var in batchnorm, the values of the parameters
        #  are going to be replaced by cI
        self.load_state_dict(learner_w_grad.state_dict())
        #  replace nn.Parameters with tensors from cI (NOT nn.Parameters anymore).
        idx = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen
    
    def freeze_layers(self, freeze):
        """Freeze all hidden layers
        """
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.out = nn.Linear(in_features=self.in_features,
                                   out_features=self.eval_classes).to(self.dev)
        self.model.out.bias = nn.Parameter(torch.zeros(self.model.out.bias.size(), device=self.dev))



class GeneralLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, criterion, output_size=1, input_size=1, dev="cpu", final_linear=True, hyper=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.criterion = criterion
        self.hyper = hyper
        self.dev = dev
        self.final_linear = final_linear
        self.output_size = output_size
        print("FINAL LINEAR:", self.final_linear)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(dev)
        if self.final_linear:
            self.output = nn.Linear(hidden_size, output_size).to(dev)
            if self.hyper:
                self.weight_mask_net = nn.Sequential(nn.Linear(hidden_size+output_size, output_size).to(dev), nn.Sigmoid())
                self.bias_mask_net = nn.Sequential(nn.Linear(hidden_size+output_size, output_size).to(dev), nn.Sigmoid())
        else:
            self.output = nn.LSTM(input_size=hidden_size, hidden_size=output_size, num_layers=1).to(dev)

        
        # write condition image. if true -> retrieve input size from cnn

        
    
    def forward(self, x, prevh=None, prevc=None, prevh_final=None, prevc_final=None):
        # x has shape [seq length, batch size, input features]
        # ht has shape [num layers, batch size, hidden dim]
        if prevh is None and prevc is None:
            x, (ht, ct) = self.lstm(x)
            if not self.final_linear:
                if prevh_final is None and prevc_final is None:
                    _, (ht_final, ct_final) = self.output(ht[-1,:,:].unsqueeze(0)) # shape [1, batch_size, outdim]
                else:
                    _, (ht_final, ct_final) = self.output(ht[-1,:,:].unsqueeze(0), (prevh_final, prevc_final)) # shape [1, batch_size, outdim]
                
                return x, (ht,ct), (ht_final, ct_final)
        else:
            if prevh is None:
                prevh = torch.zeros([self.num_layers, x.size(1), self.hidden_size], device=self.dev) # x.size(1): batch size
            if prevc is None:
                prevc = torch.zeros([self.num_layers, x.size(1), self.hidden_size], device=self.dev) # x.size(1): batch size
            
            x, (ht, ct) = self.lstm(x, (prevh, prevc))

            # ht,ct have size [num_layers, batch size, hidden_dim]
            if not self.final_linear:
                
                if prevh_final is None and prevc_final is None:
                    _, (ht_final, ct_final) = self.output(ht[-1,:,:].unsqueeze(0)) # shape [1, batch_size, outdim]
                else:
                    if prevc_final is None:
                        prevc_final = torch.zeros([1, x.size(1), self.output_size], device=self.dev) # x.size(1): batch size
                    _, (ht_final, ct_final) = self.output(ht[-1,:,:].unsqueeze(0), (prevh_final, prevc_final)) # shape [1, batch_size, outdim]
                
                return x, (ht,ct), (ht_final, ct_final)
        
        return x, (ht,ct) # [seq len, batch size, out dim]
    
    def predict(self, x, weight=None, bias=None):
        if weight is None or bias is None:
            # ompute output for all hidden states (present in x)
            # x now has shape [batch size, hidden dim]
            return self.output(x)
        else:
            return F.linear(x, weight=weight, bias=bias)














def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockLight(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockLight, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # transformation parameters
        self.trans1x1s = nn.ParameterList([nn.Parameter(torch.zeros(planes, planes, 1, 1)) for _ in range(1)])
        self.trans3x3s = nn.ParameterList([nn.Parameter(torch.zeros(planes, planes, 3, 3)) for _ in range(1)])
        self.transUs = nn.ParameterList([nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)])) for _ in range(1)])
        self.transVs = nn.ParameterList([nn.Parameter(torch.stack([torch.stack([torch.zeros(3,1) for _ in range(planes)]) for _ in range(planes)])) for _ in range(1)])
        self.transS = nn.ParameterList([nn.Parameter( torch.ones(planes, planes, 1) ) for _ in range(1)])

        # element-wise multiplication transformations on the weights of the 3 conv layers
        self.transMTLs = nn.ParameterList([nn.Parameter(torch.ones(planes, planes, 1, 1)) for _ in range(1)])
        self.transSIMPLEs = nn.ParameterList([ nn.Parameter(torch.ones(planes)) for _ in range(1)])

        # addition operations
        self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(1)])
        self.transVECT = nn.ParameterList([nn.Parameter(torch.zeros(planes)) for _ in range(1)]) 

        # alfas
        self.alfasCONV = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(1)])
        self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])
        self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])

        # Identity tensor initialization for 1x1, 3x3, SVD
        for b in [self.trans1x1s, self.trans3x3s, self.transUs, self.transVs]:
            for p in b:
                for dim in range(planes):
                    p.data[dim,dim,0,0] = 1


    def get_alfas(self):
        for block in [self.alfasCONV, self.alfasWEIGHT, self.alfasBIAS]:
            for param in block:
                yield param

    def transform_params(self):
        for block in [self.trans1x1s, self.trans3x3s, self.transUs, self.transS, self.transVs, self.transMTLs, self.transSIMPLEs, self.transCONST, self.transVECT]:
            for param in block:
                yield param
    
    def base_params(self):
        for block in [self.conv1, self.bn1, self.conv2, self.bn2, self.downsample]:
            if block is None:
                continue
            for param in block.parameters():
                yield param

    def _forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # transform weights
        conv1x1, conv3x3, U,S,V, mtl_scale, simple_scale, bias_const, bias_vect = self.trans1x1s[0], self.trans3x3s[0], self.transUs[0], self.transS[0], self.transVs[0], self.transMTLs[0], self.transSIMPLEs[0], self.transCONST[0], self.transVECT[0]
        conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
        simple_scale = simple_scale.view(conv3x3.shape[:1]+(1,1,1))

        # alfas
        conv_alfas, mtl_alfas, bias_alfas = self.alfasCONV[0], self.alfasWEIGHT[0], self.alfasBIAS[0]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        mtl_alfas = F.softmax(mtl_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)

        # base-weights
        conv, conv_bias = self.conv2.weight, self.conv2.bias
        conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

        # compute x with weight transforms

        out = F.conv2d(out, weight=conv_weights, bias=conv_bias, padding=self.conv2.padding, stride=self.conv2.stride)
        # conv transformations 
        out_pad = F.pad(out, (0,2,0,2), mode='constant')
        # Transform input
        out1 = F.conv2d(out, weight=conv1x1, bias=None) # no padding required cuz k=1
        out2 = F.conv2d(out_pad, weight=conv3x3, bias=None)
        out3 = F.conv2d(out_pad, weight=conv_svd, bias=None)

        out = conv_alfas[0]*out + conv_alfas[1]*out1 + conv_alfas[2]*out2 + conv_alfas[3]*out3
        out = bias_alfas[0]*out + bias_alfas[1]*(out+bias_const) + bias_alfas[2]*(out+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,out.size(-2),out.size(-1)))

        out = self.bn2(out)
        out = self.relu(out + residual)
        return out


class ResNetSAPLight(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), layers=[4, 4, 4], **kwargs):
        super(ResNetSAPLight, self).__init__()

        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes

        # define blocks and conv2d that we will use
        self.Conv2d = nn.Conv2d
        block = BasicBlockLight

        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transformation parameters
        self.trans1x1s = nn.ParameterList([nn.Parameter(torch.zeros(3, 3, 1, 1)), nn.Parameter(torch.zeros(iChannels, iChannels, 1, 1))])
        self.trans3x3s = nn.ParameterList([nn.Parameter(torch.zeros(3, 3, 3, 3)), nn.Parameter(torch.zeros(iChannels, iChannels, 3, 3))])
        # svd operations
        self.transUs = nn.ParameterList([
            nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(3)]) for _ in range(3)]) ),# shape (outdim, indim, 1, 1)
            nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(iChannels)]) for _ in range(iChannels)]) ),
        ])

        self.transVs = nn.ParameterList([
            nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(3)]) for _ in range(3)]) ),# shape (outdim, indim, 1, 1)
            nn.Parameter( torch.stack([torch.stack([torch.zeros(3,1) for _ in range(iChannels)]) for _ in range(iChannels)]) ),
        ])

        self.transS = nn.ParameterList([
            nn.Parameter( torch.ones(3, 3, 1)),
            nn.Parameter( torch.ones(iChannels, iChannels, 1)),
        ])

        # element-wise multiplication transformations on the weights of the 3 conv layers
        self.transMTLs = nn.ParameterList([nn.Parameter(torch.ones(iChannels, 3, 1, 1))])
        self.transSIMPLEs = nn.ParameterList([nn.Parameter(torch.ones(iChannels))])

        # addition operations
        self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(2)])
        self.transVECT = nn.ParameterList([nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(iChannels))]) 

        # alfas
        self.alfasCONV = nn.ParameterList([nn.Parameter(torch.zeros(4)) for _ in range(2)])
        self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])
        self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(2)])

        # Identity tensor initialization for 1x1, 3x3, SVD
        for b in [self.trans1x1s, self.trans3x3s, self.transUs, self.transVs]:
            for p in b:
                dim = p.size(0)
                for d in range(dim):
                   p.data[d,d,0,0] = 1
        

        rnd_input = torch.rand(1,3,80,80)
        rnd_output = self._forward(rnd_input)
        self.outdim = rnd_output.size(1) 

        self.linear = nn.Linear(rnd_output.size(1), self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))

        self.lineartransform = nn.Linear(self.num_classes, self.num_classes)
        self.lineartransform.weight.data = torch.eye(self.num_classes)
        self.lineartransform.bias.data = torch.zeros(*list(self.linear.bias.size()))
        print("Random output size:", rnd_output.size())


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def base_params(self):
        for block in [self.conv1, self.bn1]:
            for param in block.parameters():
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3,]:
            for block in l:
                for param in block.base_params():
                    yield param

    def transform_params(self):
        for block in [self.trans1x1s, self.trans3x3s, self.transUs, self.transS, self.transVs, self.transMTLs, self.transSIMPLEs, self.transCONST, self.transVECT]:
            for param in block:
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3]:
            for block in l:
                for param in block.transform_params():
                    yield param
        
        for block in [self.linear, self.lineartransform]:
            for param in block.parameters():
                yield param


    def get_alfas(self):
        for a in [self.alfasCONV, self.alfasBIAS, self.alfasWEIGHT]:
            for param in a:
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3]:
            for block in l:
                for param in block.get_alfas():
                    yield param
        

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        ###################################################
        #  input transform
        ###################################################
        conv_svd = torch.matmul(torch.matmul(self.transUs[0], torch.diag_embed(self.transS[0])), self.transVs[0].transpose(-2, -1))
        # transform input x 
        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        x1 = F.conv2d(x, weight=self.trans1x1s[0], bias=None) # no padding required cuz k=1
        x2 = F.conv2d(x_pad, weight=self.trans3x3s[0], bias=None)
        x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

        conv_alfas = F.softmax(self.alfasCONV[0], dim=0)
        bias_alfas = F.softmax(self.alfasBIAS[0], dim=0)
        x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
        x = bias_alfas[0]*x + bias_alfas[1]*(x+self.transCONST[0]) + bias_alfas[2]*(x+self.transVECT[0].unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
        ###################################################
        #  End input transform
        ###################################################
        

        # transform weights
        conv1x1, conv3x3, U,S,V, mtl_scale, simple_scale, bias_const, bias_vect = self.trans1x1s[1], self.trans3x3s[1], self.transUs[1], self.transS[1], self.transVs[1], self.transMTLs[0], self.transSIMPLEs[0], self.transCONST[1], self.transVECT[1]
        conv_svd = torch.matmul(torch.matmul(U, torch.diag_embed(S)), V.transpose(-2, -1))
        simple_scale = simple_scale.view(conv3x3.shape[:1]+(1,1,1))

        # alfas
        conv_alfas, mtl_alfas, bias_alfas = self.alfasCONV[1], self.alfasWEIGHT[0], self.alfasBIAS[1]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        mtl_alfas = F.softmax(mtl_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)

        # base-weights
        conv, conv_bias = self.conv1.weight, self.conv1.bias
        conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

        # compute x with weight transforms
        x = F.conv2d(x, weight=conv_weights, bias=conv_bias, padding=self.conv1.padding, stride=self.conv1.padding)
        x = self.bn1(x)
        x = self.relu(x)


        # PostConv Transform
        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        # Transform input
        x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1
        x2 = F.conv2d(x_pad, weight=conv3x3, bias=None)
        x3 = F.conv2d(x_pad, weight=conv_svd, bias=None)

        x = conv_alfas[0]*x + conv_alfas[1]*x1 + conv_alfas[2]*x2 + conv_alfas[3]*x3
        x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))

        # Done with transforms. Now every layer will handle their transforms individually
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        x = self.lineartransform(x)
        return x
        




class BasicBlockExtraLight(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockExtraLight, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # transformation parameters
        self.trans1x1s = nn.ParameterList([nn.Parameter(torch.zeros(planes, planes, 1, 1)) for _ in range(1)])

        # element-wise multiplication transformations on the weights of the 3 conv layers
        self.transMTLs = nn.ParameterList([nn.Parameter(torch.ones(planes, planes, 1, 1)) for _ in range(1)])
        self.transSIMPLEs = nn.ParameterList([ nn.Parameter(torch.ones(planes)) for _ in range(1)])

        # addition operations
        self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(1)])
        self.transVECT = nn.ParameterList([nn.Parameter(torch.zeros(planes)) for _ in range(1)]) 

        # alfas
        self.alfasCONV = nn.ParameterList([nn.Parameter(torch.zeros(2)) for _ in range(1)])
        self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])
        self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])

        # Identity tensor initialization for 1x1, 3x3, SVD
        for b in [self.trans1x1s]:
            for p in b:
                for dim in range(planes):
                    p.data[dim,dim,0,0] = 1


    def get_alfas(self):
        for block in [self.alfasCONV, self.alfasWEIGHT, self.alfasBIAS]:
            for param in block:
                yield param

    def transform_params(self):
        for block in [self.trans1x1s, self.transMTLs, self.transSIMPLEs, self.transCONST, self.transVECT]:
            for param in block:
                yield param
    
    def base_params(self):
        for block in [self.conv1, self.bn1, self.conv2, self.bn2, self.downsample]:
            if block is None:
                continue
            for param in block.parameters():
                yield param

    def _forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # base-weights
        conv, conv_bias = self.conv2.weight, self.conv2.bias
        

        # transform weights
        conv1x1, mtl_scale, simple_scale, bias_const, bias_vect = self.trans1x1s[0], self.transMTLs[0], self.transSIMPLEs[0], self.transCONST[0], self.transVECT[0]
        simple_scale = simple_scale.view(conv.shape[:1]+(1,1,1))

        # alfas
        conv_alfas, mtl_alfas, bias_alfas = self.alfasCONV[0], self.alfasWEIGHT[0], self.alfasBIAS[0]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        mtl_alfas = F.softmax(mtl_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)

        conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

        # compute x with weight transforms

        out = F.conv2d(out, weight=conv_weights, bias=conv_bias, padding=self.conv2.padding, stride=self.conv2.stride)
        # conv transformations 
        # Transform input
        out1 = F.conv2d(out, weight=conv1x1, bias=None) # no padding required cuz k=1

        out = conv_alfas[0]*out + conv_alfas[1]*out1
        out = bias_alfas[0]*out + bias_alfas[1]*(out+bias_const) + bias_alfas[2]*(out+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,out.size(-2),out.size(-1)))

        out = self.bn2(out)
        out = self.relu(out + residual)
        return out


class ResNetSAPExtraLight(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), layers=[4, 4, 4], **kwargs):
        super(ResNetSAPExtraLight, self).__init__()

        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes

        # define blocks and conv2d that we will use
        self.Conv2d = nn.Conv2d
        block = BasicBlockExtraLight

        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transformation parameters
        self.trans1x1s = nn.ParameterList([nn.Parameter(torch.zeros(3, 3, 1, 1)), nn.Parameter(torch.zeros(iChannels, iChannels, 1, 1))])

        # element-wise multiplication transformations on the weights of the 3 conv layers
        self.transMTLs = nn.ParameterList([nn.Parameter(torch.ones(iChannels, 3, 1, 1))])
        self.transSIMPLEs = nn.ParameterList([nn.Parameter(torch.ones(iChannels))])

        # addition operations
        self.transCONST= nn.ParameterList([nn.Parameter(torch.zeros(1).squeeze()) for _ in range(2)])
        self.transVECT = nn.ParameterList([nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(iChannels))]) 

        # alfas
        self.alfasCONV = nn.ParameterList([nn.Parameter(torch.zeros(2)) for _ in range(2)])
        self.alfasWEIGHT = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(1)])
        self.alfasBIAS = nn.ParameterList([nn.Parameter(torch.zeros(3)) for _ in range(2)])

        # Identity tensor initialization for 1x1, 3x3, SVD
        for b in [self.trans1x1s]:
            for p in b:
                dim = p.size(0)
                for d in range(dim):
                   p.data[d,d,0,0] = 1
        

        rnd_input = torch.rand(1,3,80,80)
        rnd_output = self._forward(rnd_input)
        self.outdim = rnd_output.size(1) 

        self.linear = nn.Linear(rnd_output.size(1), self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))

        self.lineartransform = nn.Linear(self.num_classes, self.num_classes)
        self.lineartransform.weight.data = torch.eye(self.num_classes)
        self.lineartransform.bias.data = torch.zeros(*list(self.linear.bias.size()))
        print("Random output size:", rnd_output.size())


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def base_params(self):
        for block in [self.conv1, self.bn1]:
            for param in block.parameters():
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3,]:
            for block in l:
                for param in block.base_params():
                    yield param

    def transform_params(self):
        for block in [self.trans1x1s, self.transMTLs, self.transSIMPLEs, self.transCONST, self.transVECT]:
            for param in block:
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3]:
            for block in l:
                for param in block.transform_params():
                    yield param
        
        for block in [self.linear, self.lineartransform]:
            for param in block.parameters():
                yield param


    def get_alfas(self):
        for a in [self.alfasCONV, self.alfasBIAS, self.alfasWEIGHT]:
            for param in a:
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3]:
            for block in l:
                for param in block.get_alfas():
                    yield param
        

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        ###################################################
        #  input transform
        ###################################################
        # transform input x 
        x_pad = F.pad(x, (0,2,0,2), mode='constant')
        x1 = F.conv2d(x, weight=self.trans1x1s[0], bias=None) # no padding required cuz k=1

        conv_alfas = F.softmax(self.alfasCONV[0], dim=0)
        bias_alfas = F.softmax(self.alfasBIAS[0], dim=0)
        x = conv_alfas[0]*x + conv_alfas[1]*x1
        x = bias_alfas[0]*x + bias_alfas[1]*(x+self.transCONST[0]) + bias_alfas[2]*(x+self.transVECT[0].unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))
        ###################################################
        #  End input transform
        ###################################################
        
        # base-weights
        conv, conv_bias = self.conv1.weight, self.conv1.bias
        

        # transform weights
        conv1x1, mtl_scale, simple_scale, bias_const, bias_vect = self.trans1x1s[1], self.transMTLs[0], self.transSIMPLEs[0], self.transCONST[1], self.transVECT[1]
        simple_scale = simple_scale.view(conv.shape[:1]+(1,1,1))

        # alfas
        conv_alfas, mtl_alfas, bias_alfas = self.alfasCONV[1], self.alfasWEIGHT[0], self.alfasBIAS[1]
        conv_alfas = F.softmax(conv_alfas, dim=0)
        mtl_alfas = F.softmax(mtl_alfas, dim=0)
        bias_alfas = F.softmax(bias_alfas, dim=0)

        conv_weights = mtl_alfas[0]*conv + mtl_alfas[1]*conv*mtl_scale + mtl_alfas[2]*conv*simple_scale

        # compute x with weight transforms
        x = F.conv2d(x, weight=conv_weights, bias=conv_bias, padding=self.conv1.padding, stride=self.conv1.padding)
        x = self.bn1(x)
        x = self.relu(x)


        # PostConv Transform
        # Transform input
        x1 = F.conv2d(x, weight=conv1x1, bias=None) # no padding required cuz k=1

        x = conv_alfas[0]*x + conv_alfas[1]*x1
        x = bias_alfas[0]*x + bias_alfas[1]*(x+bias_const) + bias_alfas[2]*(x+bias_vect.unsqueeze(1).unsqueeze(2).repeat(1,x.size(-2),x.size(-1)))

        # Done with transforms. Now every layer will handle their transforms individually
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        x = self.lineartransform(x)
        return x
        

class BasicBlockPlain(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockPlain, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride


    def get_alfas(self):
        return 
        yield

    def transform_params(self):
        return
        yield
    
    def base_params(self):
        for block in [self.conv1, self.bn1, self.conv2, self.bn2, self.downsample]:
            if block is None:
                continue
            for param in block.parameters():
                yield param

    def _forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetPlain(nn.Module):

    def __init__(self, eval_classes, dev, criterion=nn.CrossEntropyLoss(), layers=[4, 4, 4], **kwargs):
        super(ResNetPlain, self).__init__()

        self.dev = dev
        self.criterion = criterion
        self.num_classes = eval_classes

        # define blocks and conv2d that we will use
        self.Conv2d = nn.Conv2d
        block = BasicBlockPlain

        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        rnd_input = torch.rand(1,3,80,80)
        rnd_output = self._forward(rnd_input)
        self.outdim = rnd_output.size(1) 

        self.linear = nn.Linear(rnd_output.size(1), self.num_classes)
        self.linear.bias.data = torch.zeros(*list(self.linear.bias.size()))
        print("Random output size:", rnd_output.size())


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def base_params(self):
        return
        yield

    def transform_params(self):
        for block in [self.conv1, self.bn1]:
            for param in block.parameters():
                yield param

        # yield base params from all layers
        for l in [self.layer1, self.layer2, self.layer3,]:
            for block in l:
                for param in block.base_params():
                    yield param
        
        for param in self.linear.parameters():
            yield param


    def get_alfas(self):
        return
        yield
        

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
        