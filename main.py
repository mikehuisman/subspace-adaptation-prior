"""
Script to run experiments with a single algorithm of choice.
The design allows for user input and flexibility. 

Command line options are:
------------------------
runs : int, optional
    Number of experiments to perform (using different random seeds)
    (default = 1)
N : int, optional
    Number of classes per task
k : int
    Number of examples in the support sets of tasks
k_test : int
    Number of examples in query sets of meta-validation and meta-test tasks
T : int
    Number of optimization steps to perform on a given task
train_batch_size : int, optional
    Size of minibatches to sample from META-TRAIN tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
test_batch_size : int, optional
    Size of minibatches to sample from META-[VAL/TEST] tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
logfile : str
    File name to write results in (does not have to exist, but the containing dir does)
seed : int, optional
    Random seed to use
cpu : boolean, optional
    Whether to use cpu

Usage:
---------------
python main.py --arg=value --arg2=value2 ...
"""

import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import random
import torchmeta
import matplotlib.pyplot as plt
import pickle

try:
    import GPUtil
except ImportError:
    from pip._internal import main as pip
    pip(['install', 'GPUtil'])
    import GPUtil

try:
    import psutil
except ImportError:
    from pip._internal import main as pip
    pip(['install', 'psutil'])
    import psutil


from datasets.mini import MiniImagenet
from datasets.cub import CUB
from datasets.tiered import TieredImagenet
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter, RandomHorizontalFlip, ToPILImage, Grayscale
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm #Progress bars
from networks import SineNetwork, Conv4, BoostedConv4, ConvX, ResNet, LinearNet,\
                     TransformSineNetwork, FixedTransformSineNetwork, SparseSineNetwork,\
                     TransformConvX, SparseConvX, FreeTransformSineNetwork, BlockTransformConvX,\
                     FreeTransformSineNetworkSVD, SVDConvX, TNetConvX,\
                     SimpleSVDConvX, SineTNet, CompositionalSineNetworkSVD,\
                     SimpleCompositionalSineNetworkSVD, GeneralLSTM, ResNetSAPLight, ResNetSAPExtraLight,\
                     ResNetPlain #SVDTransformSineNetwork,\
from resnet import ResNet12, ResNet12Real
from dropblock import ResNetDrop
from algorithms.train_from_scratch import TrainFromScratch
from algorithms.finetuning import FineTuning
from algorithms.turtle import Turtle
from algorithms.maml import MAML
from algorithms.sap import SAP
from algorithms.boil import BOIL
from algorithms.sapresnet import SAPResNet
from algorithms.psap import PSAP
from algorithms.turtle_loss import TurtleLoss 
from algorithms.modules.utils import get_init_score_and_operator
#from sine_loader import SineLoader
from sine_loader2 import SineLoader
from sinecomp_loader import SineCompLoader
from toy_loader import ProblemLoader
from image_loader import ImageLoader
from linear_loader import LinearLoader
from polynomial_loader import PolynomialLoader
from misc import BANNER, NAMETAG
from configs import TFS_CONF, FT_CONF, CFT_CONF, LSTM_CONF,\
                    MAML_CONF, MOSO_CONF, TURTLE_CONF, LSTM_CONF2,\
                    REPTILE_CONF, SPFT_CONF, BOIL_CONF
from batch_loader import BatchDataset, cycle, Data
from custom_min import CustomMiniImagenet

FLAGS = argparse.ArgumentParser()

# Required arguments
FLAGS.add_argument("--problem", choices=["sine", "min", "cub", "linear", "tiered", "sinecomp", "mincustom", "toy", "poly", "omniglot"], required=True,
                   help="Which problem to address?")

FLAGS.add_argument("--k", type=int, required=True,
                   help="Number examples per task set during meta-validation and meta-testing."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_train", type=int, default=None,
                   help="Number examples per task set during meta-training."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_test", type=int, required=True,
                   help="Number examples per class in query set")

FLAGS.add_argument("--model", choices=["tfs", "finetuning",  
                    "maml", "moso", "lstm2", "turtle", "tloss", 
                    "sap", "psap", "boil",
                    "sapresnet"], required=True,
                   help="Which model to use?")

# Optional arguments
FLAGS.add_argument("--N", type=int, default=None,
                   help="Number of classes (only applicable when doing classification)")   

FLAGS.add_argument("--meta_batch_size", type=int, default=1,
                   help="Number of tasks to compute outer-update")   

FLAGS.add_argument("--val_after", type=int, default=None,
                   help="After how many episodes should we perform meta-validation?")

FLAGS.add_argument("--decouple", type=int, default=None,
                   help="After how many train tasks switch from meta-mode to base-mode?")

FLAGS.add_argument("--lr", type=float, default=None,
                   help="Learning rate for (meta-)optimizer")

FLAGS.add_argument("--cpe", type=float, default=0.5,
                   help="#Times best weights get reconsidered per episode (only for baselines)")

FLAGS.add_argument("--T", type=int, default=None,
                   help="Number of weight updates per training set")

FLAGS.add_argument("--T_val", type=int, default=None,
                   help="Number of weight updates at validation time")

FLAGS.add_argument("--T_test", type=int, default=None,
                   help="Number of weight updates at test time")

FLAGS.add_argument("--history", choices=["none", "grads", "updates"], default="none",
                   help="Historical information to use (only applicable for TURTLE): none/grads/updates")

FLAGS.add_argument("--beta", type=float, default=None,
                   help="Beta value to use (only applies when model=TURTLE)")

FLAGS.add_argument("--train_batch_size", type=int, default=None,
                   help="Size of minibatches for training "+\
                         "only applies for flat batch models")

FLAGS.add_argument("--test_batch_size", type=int, default=None,
                   help="Size of minibatches for testing (default = None) "+\
                   "only applies for flat-batch models")

FLAGS.add_argument("--activation", type=str, choices=["relu", "tanh", "sigmoid"],
                   default=None, help="Activation function to use for TURTLE/MOSO")

FLAGS.add_argument("--runs", type=int, default=30, 
                   help="Number of runs to perform")

FLAGS.add_argument("--devid", type=int, default=None, 
                   help="CUDA device identifier")

FLAGS.add_argument("--second_order", action="store_true", default=False,
                   help="Use second-order gradient information for TURTLE")

FLAGS.add_argument("--batching_eps", action="store_true", default=False,
                   help="Batching from episodic data")

FLAGS.add_argument("--input_type", choices=["raw_grads", "raw_loss_grads", 
                   "proc_grads", "proc_loss_grads", "maml"], default=None, 
                   help="Input type to the network (only for MOSO and TURTLE"+\
                   " choices = raw_grads, raw_loss_grads, proc_grads, proc_loss_grads, maml")

FLAGS.add_argument("--layer_wise", action="store_true", default=False,
                   help="Whether TURTLE should use multiple meta-learner networks: one for every layer in the base-learner")

FLAGS.add_argument("--param_lr", action="store_true", default=False,
                   help="Whether TURTLE should learn a learning rate per parameter")

FLAGS.add_argument("--base_lr", type=float, default=None,
                   help="Inner level learning rate")

FLAGS.add_argument("--train_base_lr", type=float, default=None,
                   help="Inner level learning rate for meta-training")

FLAGS.add_argument("--train_iters", type=int, default=None,
                    help="Number of meta-training iterations")

FLAGS.add_argument("--model_spec", type=str, default=None,
                   help="Store results in file ./results/problem/k<k>test<k_test>/<model_spec>/")

FLAGS.add_argument("--layers", type=str, default=None,
                   help="Neurons per hidden/output layer split by comma (e.g., '10,10,1')")

FLAGS.add_argument("--cross_eval", default=False, action="store_true",
                   help="Evaluate on tasks from different dataset (cub if problem=min, else min)")

FLAGS.add_argument("--backbone", type=str, default=None,
                    help="Backbone to use (format: convX)")

FLAGS.add_argument("--seed", type=int, default=1337,
                   help="Random seed to use")

FLAGS.add_argument("--single_run", action="store_true", default=False,
                   help="Whether the script is run independently of others for paralellization. This only affects the storage technique.")

FLAGS.add_argument("--no_annealing", action="store_true", default=False,
                   help="Whether to not anneal the meta learning rate for reptile")

FLAGS.add_argument("--random", action="store_true", default=False,
                   help="Use random new layer for MAML in eval phase")

FLAGS.add_argument("--cpu", action="store_true",
                   help="Use CPU instead of GPU")

FLAGS.add_argument("--time_input", action="store_true", default=False,
                   help="Add a timestamp as input to TURTLE")                   

FLAGS.add_argument("--validate", action="store_true", default=False,
                   help="Validate performance on meta-validation tasks")


FLAGS.add_argument("--no_freeze", action="store_true", default=False,
                   help="Whether to freeze the weights in the finetuning model of earlier layers")

FLAGS.add_argument("--eval_on_train", action="store_true", default=False,
                    help="Whether to also evaluate performance on training tasks")

FLAGS.add_argument("--test_adam", action="store_true", default=False,
                   help="Optimize weights with Adam, LR = 0.001 at test time.")

FLAGS.add_argument("--test_opt", choices=["adam", "sgd"], default=None,
                   help="Optimizer to use at meta-validation or meta-test time for the finetuning model")

FLAGS.add_argument("--test_lr", type=float, default=None, help="LR to use at meta-val/test time for finetuning")

FLAGS.add_argument("--special", action="store_true", default=False,
                   help="Train MAML on 64 classes")

FLAGS.add_argument("--log_norm", action="store_true", default=None,
                   help="Log grad norms")

FLAGS.add_argument("--log_test_norm", action="store_true", default=None,
                   help="Log grad norms of final model")
                   
FLAGS.add_argument("--var_updates", action="store_true", default=False,
                   help="Use variable number of updates")

FLAGS.add_argument("--gamma", type=str, default=None, help="Gamma value for mamlft")

FLAGS.add_argument("--transform", type=str, choices=["interp", "scale"], default="interp", help="Type of transform integration for SAP")

FLAGS.add_argument("--reg", type=str, choices=["num_params", "l2", "null", "l1", "entropy", "we"], default="null", help="Type of regularization method for transforms in SAP")

FLAGS.add_argument("--pretrain", action="store_true", default=False,
                   help="pretrain sap")

FLAGS.add_argument("--freeze_init", action="store_true", default=False,
                   help="Freeze initialization SAP")

FLAGS.add_argument("--freeze_transform", action="store_true", default=False,
                   help="Freeze initialization SAP")

FLAGS.add_argument("--trans_net", action="store_true", default=False,
                   help="use FixedTransformSineNetwork")

FLAGS.add_argument("--exp1", action="store_true", default=False,
                   help="Experiment 1 for SAP")

FLAGS.add_argument("--learn_alfas", action="store_true", default=False,
                   help="Learnable alfas")

FLAGS.add_argument("--free_arch", default=False, action="store_true",
                   help="More degrees of freedom in the architecture (not just preset shiftscale)")

FLAGS.add_argument("--solid", type=int, default=float("inf"),
                   help="Activate/deactivate after X number of meta-training iterations of being above/below threshold")

FLAGS.add_argument("--arch", choices=["full", "partial"], default=None,
                   help="Architecture type for FixedTransNet")

FLAGS.add_argument("--relu", default=False, action="store_true",
                   help="ReLU after transformed outputs (not final layer)")

FLAGS.add_argument("--channel_scale", default=False, action="store_true",
                   help="For SAP on image tasks: learn a scalar per input channel")

FLAGS.add_argument("--free_net", action="store_true", default=False,
                   help="use FreeTransformSineNetwork")

FLAGS.add_argument("--block_transform", action="store_true", default=False,
                   help="use BlockTransformConvX")


FLAGS.add_argument("--unfreeze_init", action="store_true", default=False,
                   help="unfreeze_init for sap")

FLAGS.add_argument("--boil", action="store_true", default=False,
                   help="use BOIL for SAP")

FLAGS.add_argument("--svd", action="store_true", default=False,
                   help="use SVD for SAP")

FLAGS.add_argument("--grad_clip", type=int, default=None,
                   help="gradient clipping")


FLAGS.add_argument("--linear_transform", action="store_true", default=False,
                   help="linear transform")

FLAGS.add_argument("--max_pool_before_transform", action="store_true", default=False,
                   help="max pool before doing transform")

FLAGS.add_argument("--old", action="store_true", default=False,
                   help="no linear SVD in final layer of ConvSVD")

FLAGS.add_argument("--discrete_ops", action="store_true", default=False,
                   help="Use discrete transform operations in forward-passes")

FLAGS.add_argument("--trans_before_relu", action="store_true", default=False,
                   help="Do transforms before relu")

FLAGS.add_argument("--anneal_temp", action="store_true", default=False,
                   help="Anneal temperature")

FLAGS.add_argument("--soft", action="store_true", default=False,
                   help="Use soft masks at validation time")

FLAGS.add_argument("--out_channels", type=int, default=None,
                   help="out_channels to use in ConvX")

FLAGS.add_argument("--transform_out_channels", type=int, default=None,
                   help="out_channels for transforms in TNet/MTNet/WarpGrad")


FLAGS.add_argument("--warm_start", type=int, default=0,
                   help="Don't update alfas for so many iterations")


FLAGS.add_argument("--sign_sgd", action="store_true", default=False,
                   help="Use SGD")

FLAGS.add_argument("--train_only_sign_sgd", action="store_true", default=False,
                   help="Use SGD only during meta-training time")


FLAGS.add_argument("--tnet", action="store_true", default=False,
                   help="T-Net")

FLAGS.add_argument("--avg_grad", action="store_true", default=False,
                   help="avg grads in meta-batch")

FLAGS.add_argument("--simple", action="store_true", default=False,
                   help="avoid 3x3 convs")


FLAGS.add_argument("--swap_base_trans", action="store_true", default=False,
                   help="finetune base params instead of trans params")


FLAGS.add_argument("--use_grad_mask", action="store_true", default=False,
                   help="Gradient masking as in MT-Net")


FLAGS.add_argument("--model_path", type=str, default=None,
                   help="Path to where the models are stored")


FLAGS.add_argument("--test_k_test", type=int, default=None,
                   help="How many query examples at test time for sine loader")


FLAGS.add_argument("--measure_distances", action="store_true", default=False,
                   help="Measure inter-class and intra-class distances during adaptation at test time")


FLAGS.add_argument("--arch_steps", type=int, default=None,
                   help="How many steps to update the architecture during tasks-specific adaptation (reptile +SAP only)")


FLAGS.add_argument("--bn_before_trans", action="store_true", default=False,
                   help="BN b4 trans")

FLAGS.add_argument("--bn_after_trans", action="store_true", default=False,
                   help="BN after trans")
                   
FLAGS.add_argument("--train_all_dense", action="store_true", default=False,
                   help="Train both final dense layers (only applicable if linear_transform is true)")

FLAGS.add_argument("--warpgrad", action="store_true", default=False,
                   help="use warpgrad")

FLAGS.add_argument("--train_curve", action="store_true", default=False,
                   help="save training curve")


FLAGS.add_argument("--finite_diff", action="store_true", default=False,
                   help="finite diff for MAML")

FLAGS.add_argument("--debug", action="store_true", default=False,
                   help="quit after single iter")

FLAGS.add_argument("--simplecomp", action="store_true", default=False,
                   help="use simple comp architecture")


FLAGS.add_argument("--use_bias", action="store_true", default=False,
                   help="use bias in warpgrad")


FLAGS.add_argument("--noshift", action="store_true", default=False,
                   help="no final shift transform operator for SImpleComp")

FLAGS.add_argument("--special_opt", choices=["fdtrunc", "osass"], required=False, default=None, 
                   help="Special optimizer to use for MAML?")

FLAGS.add_argument("--max_freq_train", type=float, default=2.5,
                   help="max freq for validation/testing of comp sine")

FLAGS.add_argument("--max_freq_eval", type=float, default=2.5,
                   help="max freq for validation/testing of compsine")

FLAGS.add_argument("--warpgrad_optimizer", action="store_true", default=False,
                   help="Use Warpgrad optimizer with SAP")

FLAGS.add_argument("--dotraining", action="store_true", default=False,
                   help="Do training, even when model_path is specified")

FLAGS.add_argument("--top", type=int, default=0,
                   help="Use only the top transofmation operators")

FLAGS.add_argument("--xrange", type=float, default=5,
                   help="xrange used for training the sinecomp")

FLAGS.add_argument("--min_freq", type=float, default=0,
                   help="minimum frequency of sinecomp")

FLAGS.add_argument("--uniform", action="store_true", default=False,
                   help="uniformly spaces samples sine comp")

FLAGS.add_argument("--comp_pretrain", action="store_true", default=False,
                   help="pretrain on sinecomp task")

FLAGS.add_argument("--only_cross", action="store_true", default=False,
                   help="Only perform cross-eval")


FLAGS.add_argument("--vary", type=str, default="[True,True,True,True]",
                   help="vary (ampl, freq, phase, outshift)")

FLAGS.add_argument("--order", type=int, default=None,
                   help="The order of the polynomials to use")

FLAGS.add_argument("--hidden_size", type=int, default=3,
                   help="dimensionality of hidden size simplelstm")

FLAGS.add_argument("--num_layers", type=int, default=1,
                   help="number of layers in simplelstm")

FLAGS.add_argument("--zero_test",  default=False, action="store_true",
                   help="zero all previous targets in input sequence simplelstm")

FLAGS.add_argument("--loss_type", type=str, default="post", choices=["post","multi"],
                   help="multi-step loss or post-adaptation loss")

FLAGS.add_argument("--avg_cell",  default=False, action="store_true",
                   help="average cells state")


FLAGS.add_argument("--pretrained_path",  default=None,
                   help="path to pre-trained weights of ResNet12")

FLAGS.add_argument("--freeze_pretrained",  default=False, action="store_true",
                   help="Freeze pretrained weights")

FLAGS.add_argument("--num_aug", default=0, type=int, help="Number of augmented examples per class in")


FLAGS.add_argument("--nn", default=False, action="store_true", help="Nearest neighbor classifier for DropBLock Resnet")

FLAGS.add_argument("--use_logits", default=False, action="store_true", help="Nearest neighbor classifier for DropBLock Resnet")

FLAGS.add_argument("--no_training", default=False, action="store_true", help="Wheter not to train")

FLAGS.add_argument("--adapt", default=False, action="store_true", help="Wheter adapt the embedding during training for dropblock resnet")

FLAGS.add_argument("--simple_linear", default=False, action="store_true", help="Simple linear output head")


FLAGS.add_argument("--enable_sap", default=False, action="store_true", help="Simple linear output head")

FLAGS.add_argument("--enable_conv", default="True", type=str, help="eable conv transforms for resnet-12 SAP")

FLAGS.add_argument("--use_tanh", default=False, action="store_true", help="Use Tanh in sinenetwork")

FLAGS.add_argument("--hdims", default="[40,40]", type=str, help="Hidden dims for sine network")

FLAGS.add_argument("--final_linear", type=str, default="True",
                   help="Use final linear for simpleLSTM")


FLAGS.add_argument("--img_size", type=int, default=None,
                   help="img size")

FLAGS.add_argument("--rgb",  default="True", type=str,
                   help="RGB images")

FLAGS.add_argument("--zero_supp",  default="True", type=str,
                   help="whether to offset targets in support set")

FLAGS.add_argument("--hyper",  default="False", type=str,
                   help="use hyperLSTM")


RESULT_DIR = "./results/"

def create_dir(dirname):
    """
    Create directory <dirname> if not exists
    """
    
    if not os.path.exists(dirname):
        print(f"[*] Creating directory: {dirname}")
        try:
            os.mkdir(dirname)
        except FileExistsError:
            # Dir created by other parallel process so continue
            pass

def print_conf(conf):
    """Print the given configuration
    
    Parameters
    -----------
    conf : dictionary
        Dictionary filled with (argument names, values) 
    """
    
    print(f"[*] Configuration dump:")
    for k in conf.keys():
        print(f"\t{k} : {conf[k]}")

def set_batch_size(conf, args, arg_str):
    value = getattr(args, arg_str)
    # If value for argument provided, set it in configuration
    if not value is None:
        conf[arg_str] = value
    else:
        try:
            # Else, try to fetch it from the configuration
            setattr(args, arg_str, conf[arg_str]) 
            args.train_batch_size = conf["train_batch_size"]
        except:
            # In last case (nothing provided in arguments or config), 
            # set batch size to N*k
            num = args.k
            if not args.N is None:
                num *= args.N
            setattr(args, arg_str, num)
            conf[arg_str] = num             

def overwrite_conf(conf, args, arg_str):
    # If value provided in arguments, overwrite the config with it
    value = getattr(args, arg_str)
    if not value is None:
        conf[arg_str] = value
    else:
        # Try to fetch argument from config, if it isnt there, then the model
        # doesn't need it
        try:
            setattr(args, arg_str, conf[arg_str])
        except:
            return
        
def setup(args):
    """Process arguments and create configurations
        
    Process the parsed arguments in order to create corerct model configurations
    depending on the specified user-input. Load the standard configuration for a 
    given algorithm first, and overwrite with explicitly provided input by the user.

    Parameters
    ----------
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    args : cmd arguments
        The processed command-line arguments
    conf : dictionary
        Dictionary defining the meta-learning algorithm and base-learner
    data_loader
        Data loader object, responsible for loading data
    """
    
    if args.k_train is None:
        args.k_train = args.k
    if args.model == "simplelstm" and args.zero_supp.lower() == "true":
       assert args.T > 1, "T=1 means that the LSTM hasn't received targets yet so it can't learn..."

    
    # Mapping from model names to configurations
    mod_to_conf = {
        "tfs": (TrainFromScratch, TFS_CONF),
        "finetuning": (FineTuning, FT_CONF),
        "centroidft": (FineTuning, CFT_CONF), 
        "maml": (MAML, MAML_CONF),
        "sap": (SAP, MAML_CONF),
        "psap": (PSAP, MAML_CONF),
        "turtle": (Turtle, TURTLE_CONF),
        "boil": (BOIL, BOIL_CONF), 
        "tloss": (TurtleLoss, TURTLE_CONF), 
        "sapresnet": (SAPResNet, MAML_CONF), 
    }

    baselines = {"tfs", "finetuning", "centroidft"}#, "spft"}
    
    if not args.backbone is None:
        assert not ("resnet12" in args.backbone and args.model=="sap"), "Resnet12 can only be used with sapresnet"

    # Get model constructor and config for the specified algorithm
    model_constr, conf = mod_to_conf[args.model]

    # Set batch sizes
    set_batch_size(conf, args, "train_batch_size")
    set_batch_size(conf, args, "test_batch_size")
        
    # Set values of T, lr, and input type
    overwrite_conf(conf, args, "T")
    overwrite_conf(conf, args, "lr")
    overwrite_conf(conf, args, "input_type")
    overwrite_conf(conf, args, "beta")
    overwrite_conf(conf, args, "meta_batch_size")
    overwrite_conf(conf, args, "time_input")
    conf["no_annealing"] = args.no_annealing
    conf["test_adam"] = args.test_adam
    conf["var_updates"] = args.var_updates
    if not args.gamma is None:
        if '[' in args.gamma:
            args.gamma = [float(x) for x in args.gamma.strip()[1:-1].split(',')]
            print("Parsed gamma to:", args.gamma)
            assert args.reg == "we", "List of gammas is only supported for weight-entropy regularizer 'we'"
        else:
            assert not args.reg == "we", "weight-entropy regularization requires a list of gammas"
            try:
                args.gamma = float(args.gamma)
            except:
                print("Could not parse gamma to a float")
                import sys; sys.exit()

    conf["gamma"] = args.gamma
    conf["transform"] = args.transform
    conf["reg"] = args.reg
    conf["sine_constr"] = SineNetwork
    conf["pretrain"] = args.pretrain
    conf["trans_net"] = args.trans_net
    conf["solid"] = args.solid
    conf["free_arch"] = args.free_arch
    conf["relu"] = args.relu
    conf["image"] = (not "sine" in args.problem and not "toy" in args.problem) and not "linear" in args.problem and not "poly" in args.problem
    conf["channel_scale"] = args.channel_scale
    conf["unfreeze_init"] = args.unfreeze_init
    conf["boil"] = args.boil
    conf["linear_transform"] = args.linear_transform
    conf["max_pool_before_transform"] = args.max_pool_before_transform
    conf["old"] = args.old
    conf["discrete_ops"] = args.discrete_ops
    conf["trans_before_relu"] = args.trans_before_relu
    conf["anneal_temp"] = args.anneal_temp
    conf["soft"] = args.soft
    conf["warm_start"] = args.warm_start
    conf["sign_sgd"] = args.sign_sgd
    conf["train_only_sign_sgd"] = args.train_only_sign_sgd
    conf["tnet"] = args.tnet
    conf["warpgrad"] = args.warpgrad
    conf["avg_grad"] = args.avg_grad
    conf["swap_base_trans"] = args.swap_base_trans
    conf["use_grad_mask"] = args.use_grad_mask
    conf["finite_diff"] = args.finite_diff
    conf["special_opt"] = args.special_opt
    conf["warpgrad_optimizer"] = args.warpgrad_optimizer
    conf["comp_pretrain"] = args.comp_pretrain
    conf["max_freq_train"] = args.max_freq_train
    conf["xrange"] = args.xrange
    conf["num_layers"] = args.num_layers
    conf["hidden_size"] = args.hidden_size
    conf["avg_cell"] = args.avg_cell
    conf["loss_type"] = args.loss_type
    conf["freeze_pretrained"] = args.freeze_pretrained
    conf["nn"] = args.nn
    conf["use_logits"] = args.use_logits
    conf["adapt"] = args.adapt
    conf["simple_linear"] = args.simple_linear
    conf["enable_sap"] = args.enable_sap
    conf["final_linear"] = args.final_linear.lower() == "true"
    conf["zero_supp"] = args.zero_supp.lower() == "true"
    conf["hyper"] = args.hyper.lower() == "true"

    if conf["hyper"]:
        assert conf["final_linear"], "if hyper is True, final_linear must also be True"

    args.rgb = args.rgb.lower() == "true" 

    if args.model == "sapresnet":
        assert not args.second_order, "sapresnet only works first-order"
        #assert args.backbone == "resnet12", "sapresnet only works with resnet12"


    assert not (args.finite_diff and args.second_order), "finite_diff incompatible with second_order"

    assert not (args.train_all_dense and not args.linear_transform), "train_all_dense only applicable if linear_transform=True"
    conf["train_all_dense"] = args.train_all_dense

    if not args.test_opt is None or not args.test_lr is None:
        assert args.model == "finetuning", "test_opt and test_lr arguments only suited for finetuning model"
        conf["test_opt"] = args.test_opt
        conf["test_lr"] = args.test_lr

    if args.test_k_test is None:
        args.test_k_test = args.k_test
    
    # Parse the 'layers' argument
    if not args.layers is None:
        try:
            layers = [int(x) for x in args.layers.split(',')]
        except:
            raise ValueError(f"Error while parsing layers argument {args.layers}")
        conf["layers"] = layers
    
    # Make sure argument 'val_after' is specified when 'validate'=True
    if args.validate:
        assert not args.val_after is None,\
                    "Please specify val_after (number of episodes after which to perform validation)"
    
    # If using multi-step maml, perform gradient clipping with -10, +10
    if not conf["T"] is None:
        if conf["T"] > 1 and (args.model=="maml" or args.model=="turtle"):# or args.model=="reptile"):
            conf["grad_clip"] = 10
        elif args.model == "lstm" or args.model == "lstm2":
            conf["grad_clip"] = 0.25 # it does norm clipping
        else:
            conf["grad_clip"] = None
    if args.grad_clip is not None:
        conf["grad_clip"] = args.grad_clip
    
    # If MOSO or TURTLE is selected, set the activation function
    if args.activation:
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "sigmoid": nn.Sigmoid()
        }
        conf["act"] = act_dict[args.activation]
    
    # Set the number of reconsiderations of best weights during meta-training episodes,
    # and the device to run the algorithms on 
    conf["cpe"] = args.cpe
    conf["dev"] = args.dev
    conf["second_order"] = args.second_order
    conf["history"] = args.history
    conf["layer_wise"] = args.layer_wise
    conf["param_lr"] = args.param_lr
    conf["decouple"] = args.decouple
    conf["batching_eps"] = args.batching_eps
    conf["freeze"] = not args.no_freeze
    conf["special"] = args.special
    conf["random"] = args.random
    conf["freeze_init"] = args.freeze_init
    conf["freeze_transform"] = args.freeze_transform
    conf["exp1"] = args.exp1
    conf["learn_alfas"] = args.learn_alfas
    conf["arch"] = args.arch
    conf["free_net"] = args.free_net
    conf["svd"] = args.svd
    conf["train_curve"] = args.train_curve
    conf["zero_test"] = args.zero_test
    conf["N"] = args.N

    if args.problem == "sine":
        assert args.N == 1, "sine wave regression -> N should be 1"

    if not args.log_norm is None:
        conf["log_norm"] = True

    if args.T_test is None:
        conf["T_test"] = conf["T"]
        args.T_test = conf["T"]
    else:
        conf["T_test"] = args.T_test
    
    if args.T_val is None:
        conf["T_val"] = conf["T"]
        args.T_val = conf["T"]
    else:
        conf["T_val"] = args.T_val

    if not args.arch_steps is None:
        assert args.model == "reptsap", "arch_steps only supported when model=reptsap"
        conf["arch_steps"] = args.arch_steps
    else:
        args.arch_steps = max(args.T, args.T_test, args.T_val)


    if not args.base_lr is None:
        conf["base_lr"] = args.base_lr

    if not args.train_base_lr is None:
        conf["train_base_lr"] = args.train_base_lr
    else:
        if args.test_lr is not None:
            conf["test_lr"] = args.test_lr
            conf["train_base_lr"] = args.test_lr
        else:
            try:
                conf["train_base_lr"] = conf["base_lr"]
            except:
                pass

    assert not (args.input_type == "maml" and args.history != "none"), "input type 'maml' and history != none are not compatible"
    assert not (conf["T"] == 1 and args.history != "none"), "Historical information cannot be used when T == 1" 

    # Different data set loader to test domain shift robustness
    cross_loader = None
    
    # Pick appropriate base-learner model for the chosen problem [sine/image]
    # and create corresponding data loader obejct
    if args.problem == "linear":
        data_loader = LinearLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        conf["baselearner_fn"] = LinearNet
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
        }
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "sine":
        data_loader = SineLoader(k=args.k, k_test=args.k_test, test_k_test=args.test_k_test, seed=args.seed)
        if not "sap" in args.model:
            assert not (args.free_net),"cant use both free_net and trans_net"
            if args.trans_net:
                conf["baselearner_fn"] = FixedTransformSineNetwork
            else:
                if args.model == "simplelstm":
                    conf["baselearner_fn"] = GeneralLSTM
                else:
                    conf["baselearner_fn"] = SineNetwork if not args.trans_net else FixedTransformSineNetwork
        else:
            if "psap" in args.model:
                conf["baselearner_fn"] = SparseSineNetwork
            else:
                if args.free_net:
                    print("using free transform net")
                    conf["baselearner_fn"] = FreeTransformSineNetwork
                elif args.svd:
                    print("Using SVD net")
                    conf["baselearner_fn"] = FreeTransformSineNetworkSVD
                elif args.tnet:
                    conf["baselearner_fn"] = SineTNet
                else:
                    conf["baselearner_fn"] = TransformSineNetwork
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        if args.model == "simplelstm":
            conf["baselearner_args"]["input_size"] = 2

        conf["baselearner_args"]["hdims"] = [int(x) for x in args.hdims[1:-1].split(",")]
        print([int(x) for x in args.hdims[1:-1].split(",")])

        conf["baselearner_args"]["use_tanh"] = args.use_tanh
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        if args.train_iters is None:
            args.train_iters = 70000
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "toy":
        data_loader = ProblemLoader(k=args.k, k_test=args.k_test, vary=[True if x=="True" else False for x in args.vary[1:-1].split(',')], seed=args.seed)

        if not "sap" in args.model:
            assert not (args.free_net),"cant use both free_net and trans_net"
            conf["baselearner_fn"] = SineNetwork
        else:
            if not args.tnet and not args.use_grad_mask:
                conf["baselearner_fn"] = CompositionalSineNetworkSVD
                assert args.simple, "provide the simple argument"
            else:
                conf["baselearner_fn"] = SineTNet

        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev, "n_components":1}
        if args.noshift:
            conf["baselearner_args"]["no_shift"] = True
        conf["baselearner_args"]["simple"] = args.simple
        
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        if args.train_iters is None:
            args.train_iters = 70000
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "poly":
        assert not args.order is None, "order should be provided when using polyloader" 
        data_loader = PolynomialLoader(order=args.order, k=args.k, k_test=args.k_test, seed=args.seed)

        if not "sap" in args.model:
            assert not (args.free_net),"cant use both free_net and trans_net"
            conf["baselearner_fn"] = SineNetwork
        else:
            if not args.tnet and not args.use_grad_mask:
                conf["baselearner_fn"] = CompositionalSineNetworkSVD
                assert args.simple, "provide the simple argument"
            else:
                conf["baselearner_fn"] = SineTNet

        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev, "n_components":1}
        if args.noshift:
            conf["baselearner_args"]["no_shift"] = True
        conf["baselearner_args"]["simple"] = args.simple
        
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        if args.train_iters is None:
            args.train_iters = 70000
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "sinecomp":
        data_loader = SineCompLoader(k=args.k, k_test=args.k_test, test_k_test=args.test_k_test, seed=args.seed, 
                                     max_freq_eval=args.max_freq_eval, max_freq_train=args.max_freq_train, x_range=args.xrange, 
                                     min_freq=args.min_freq, uniform=args.uniform)
        if not "sap" in args.model:
            assert not (args.free_net),"cant use both free_net and trans_net"
            conf["baselearner_fn"] = SineNetwork
        else:
            if not args.tnet and not args.use_grad_mask:
                if not args.simplecomp:
                    conf["baselearner_fn"] = CompositionalSineNetworkSVD
                else:
                    conf["baselearner_fn"] = SimpleCompositionalSineNetworkSVD
            else:
                conf["baselearner_fn"] = SineTNet

        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev, "n_components":2}
        if args.noshift:
            conf["baselearner_args"]["no_shift"] = True
        conf["baselearner_args"]["simple"] = args.simple
        
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        if args.train_iters is None:
            args.train_iters = 70000
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    else:
        assert not args.N is None, "Please provide the number of classes N per set"
        

        if "lstm" in args.model:
            conf["lstm_constructor"] = GeneralLSTM
        
        normalize = False
        # Image problem
        if args.backbone is None:
            if args.model == "centroidft":
                conf["baselearner_fn"] = BoostedConv4
                lowerstr = "Bconv4"
            else:    
                if "psap" in args.model:
                    conf["baselearner_fn"] = SparseConvX
                elif "sap" in args.model:
                    if args.block_transform:
                        conf["baselearner_fn"] = BlockTransformConvX
                    elif args.svd:
                        if args.simple:
                            conf["baselearner_fn"] = SimpleSVDConvX
                        else:    
                            conf["baselearner_fn"] = SVDConvX
                    elif args.tnet or args.warpgrad:
                        conf["baselearner_fn"] = TNetConvX
                    else:
                        conf["baselearner_fn"] = TransformConvX
                else:
                    conf["baselearner_fn"] = ConvX 
                lowerstr = "conv4"
            if not args.img_size is None:
                img_size = (args.img_size, args.img_size)
            else:
                img_size = (84,84) if not args.problem == "omniglot" else (28,28)
        else:
            lowerstr = args.backbone.lower()    
            args.backbone = lowerstr    
            if lowerstr == "resnet12":
                modelstr = "resnet"
                constr = ResNet12
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNet12
            elif lowerstr == "dropblock":
                modelstr = "dropblock"
                constr = ResNetDrop
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNetDrop
            elif lowerstr == "resnetlight12":
                modelstr = "resnetlight"
                constr = ResNetSAPLight
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNetSAPLight
            elif lowerstr == "resnetextralight12":
                print("Working with ResnetExtraLight")
                modelstr = "resnetextralight"
                constr = ResNetSAPExtraLight
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNetSAPExtraLight
            elif lowerstr == "resnetplain12":
                print("Working with ResnetPlain")
                modelstr = "resnetplain"
                constr = ResNetPlain
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84)
                conf["baselearner_fn"] = ResNetPlain
            elif lowerstr == "resnetreal12":
                print("Working with ResnetPlain")
                modelstr = "resnetreal"
                constr = ResNet12Real
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size=(84,84) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNet12Real
            elif "resnet" in lowerstr:
                modelstr = "resnet"
                constr = ResNet
                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size = (224,224) if not args.problem == "omniglot" else (28,28)
                conf["baselearner_fn"] = ResNet
            elif "conv" in lowerstr:
                modelstr = "conv"
                if "psap" in args.model:
                    conf["baselearner_fn"] = SparseConvX
                elif "sap" in args.model:
                    if args.block_transform:
                        conf["baselearner_fn"] = BlockTransformConvX
                    elif args.tnet or args.warpgrad:
                        conf["baselearner_fn"] = TNetConvX
                    else:
                        conf["baselearner_fn"] = TransformConvX
                else:
                    conf["baselearner_fn"] = ConvX 

                if not args.img_size is None:
                    img_size = (args.img_size, args.img_size)
                else:
                    img_size = (84,84) if not args.problem == "omniglot" else (28,28)
            else:
                raise ValueError("Could not parse the provided backbone argument")
            
            if lowerstr != "dropblock":
                num_blocks = int(lowerstr.split(modelstr)[1])
            else:
                num_blocks = None
            print(f"Using backbone: {modelstr}{num_blocks}")
            if args.backbone != "dropblock" and num_blocks > 4:
                normalize = True
                print("normalizing")

        if lowerstr == "dropblock":
            mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
            std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

            normalize = Normalize(mean=mean, std=std)

            train_transform = Compose([ToTensor()])
                
            # Compose([
            #     RandomCrop(84, padding=8),
            #     ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            #     RandomHorizontalFlip(),
            #     lambda x: np.asarray(x),
            #     ToTensor(),
            #     normalize
            # ])

            test_transform = Compose([ToTensor()])

            
        elif normalize and not args.backbone.lower() == "resnetlight12" and not args.backbone.lower() == "resnetextralight12" and not args.backbone.lower() == "resnetplain12":
            train_transform =  test_transform = Compose([Resize(size=img_size), ToTensor(), 
                        Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            # transform = Compose([Resize(size=img_size), ToTensor(), 
            #                      Normalize(np.array([0.485, 0.456, 0.406]),
            #                                np.array([0.229, 0.224, 0.225]))])
            #mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]
            print("Normalizing using MTL values")
        elif normalize and (args.backbone.lower() == "resnetlight12" or args.backbone.lower() == "resnetextralight12" or args.backbone.lower() == "resnetplain12"):
            image_size = 80
            train_transform = test_transform = Compose([
                Resize(92),
                CenterCrop(image_size),
                ToTensor(),
                Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                          np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            print("Using same transform as in MTL")
        else:
            if args.rgb:
                transform = test_transform = train_transform = Compose([Resize(size=img_size), ToTensor()])
            else:
                transform = test_transform = train_transform = Compose([Resize(size=img_size), Grayscale(), ToTensor()])

        if args.train_iters is None:
            if args.k >= 5:
                train_iters = 40000
            else:
                train_iters = 60000
        else:
            train_iters = args.train_iters

        eval_iters = 600
        args.eval_iters = 600
        args.train_iters = train_iters
        conf["train_iters"] = args.train_iters


        if args.problem == "mincustom":
            ds = MiniImagenet
            cds = CUB
            dataset_specifier = Data.MIN
        elif "min" in args.problem:
            ds = MiniImagenet#datasets.MiniImagenet
            cds = CUB
            dataset_specifier = Data.MIN
        elif "cub" in args.problem:
            ds = CUB
            cds = MiniImagenet
            dataset_specifier = Data.CUB
        elif "tiered" in args.problem:
            ds = TieredImagenet
            cds = CUB
            dataset_specifier = Data.Tiered


        val_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                        meta_val=True, meta_test=False, meta_split="val",
                        transform=test_transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        val_loader = ClassSplitter(val_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
        val_loader = BatchMetaDataLoader(val_loader, batch_size=1, num_workers=2, shuffle=True)


        test_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                        meta_val=False, meta_test=True, meta_split="test",
                        transform=test_transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        test_loader = ClassSplitter(test_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k)
        test_loader = BatchMetaDataLoader(test_loader, batch_size=1, num_workers=2, shuffle=True)


        cross_loader = None
        if args.cross_eval:
            cross_loader = cds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                               meta_val=False, meta_test=True, meta_split="test",
                               transform=test_transform,
                               target_transform=Compose([Categorical(args.N)]),
                               download=True)
            cross_loader = ClassSplitter(cross_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
            cross_loader = BatchMetaDataLoader(cross_loader, batch_size=1, num_workers=2, shuffle=True)


        train_class_per_problem = {
            "min": 64,
            "cub": 140
        }

        problem_to_root = {
            "min": "./data/miniimagenet/",
            "cub": "./data/cub/",
            "tiered": "./data/tieredimagenet/",
            "omniglot": "./data/omniglot/"
        }

        if args.problem == "mincustom":
            ds = CustomMiniImagenet

        if args.model in baselines:
            if not args.model == "tfs":
                train_classes = train_class_per_problem[args.problem.lower()]
            else:
                train_classes = args.N # TFS does not train, so this enforces the model to have the correct output dim. directly

            train_loader = BatchDataset(root_dir=problem_to_root[args.problem],
                                        dataset_spec=dataset_specifier, transform=transform)
            train_loader = iter(cycle(DataLoader(train_loader, batch_size=conf["train_batch_size"], shuffle=True, num_workers=4)))
            args.batchmode = True
            print("Using custom made BatchDataset")
        else:
            if args.special:
                train_classes = 64
                train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                          meta_val=False, meta_test=False, meta_split="train",
                          transform=train_transform,
                          target_transform=Compose([Categorical(64)]),
                          download=True)
                train_loader = ClassSplitter(train_loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
                train_loader = iter(cycle(BatchMetaDataLoader(train_loader, batch_size=1, num_workers=4, shuffle=True)))
            else:
                train_classes = args.N
                
                if args.model == "spft":
                    train_classes = train_class_per_problem[args.problem.lower()]
                    train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                          meta_val=False, meta_test=False, meta_split="train",
                          transform=transform,
                          target_transform=None,
                          download=True)
                    
                    print(train_loader.dataset._labels, len(train_loader.dataset._labels))
                    class_map = dict()
                    for cid, label in enumerate(train_loader.dataset._labels):
                        assert label not in class_map, "Duplicate label found"
                        class_map[label] = cid 
                    args.labels = class_map
                else:
                    train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                            meta_val=False, meta_test=False, meta_split="train",
                            transform=train_transform,
                            target_transform=Compose([Categorical(args.N)]),
                            download=True)
                    print("using this data loader")
                    if not args.gamma is None and not args.gamma == 0:
                        train_loader2 = BatchDataset(root_dir=problem_to_root[args.problem],
                                            dataset_spec=dataset_specifier, transform=transform)

                        train_loader2 = iter(cycle(DataLoader(train_loader2, batch_size=conf["train_batch_size"], shuffle=True, num_workers=4)))
            
                train_loader = ClassSplitter(train_loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
                train_loader = BatchMetaDataLoader(train_loader, batch_size=1, num_workers=4, shuffle=True)
            args.batchmode = False
            
        conf["baselearner_args"] = {
            "train_classes": train_classes,
            "eval_classes": args.N, 
            "criterion": nn.CrossEntropyLoss(),
            "dev":args.dev,
            "rgb": True if args.problem  != "omniglot" or args.rgb else False,
            "img_size": img_size,
        }

        if args.warpgrad and args.use_bias:
            conf["baselearner_args"]["use_bias"] = True
        if args.out_channels is not None:
            conf["baselearner_args"]["out_channels"] = args.out_channels 
        
        if args.transform_out_channels is not None:
            assert args.tnet or args.warpgrad, "transform out channels can only be used with TNet/MTNet/WarpGrad"
            conf["baselearner_args"]["transform_out_channels"] = args.transform_out_channels 


        if not args.backbone is None:
            conf["baselearner_args"]["num_blocks"] = num_blocks
        
        if args.bn_before_trans:
            conf["baselearner_args"]["bn_before_trans"] = True
        if args.bn_after_trans:
            conf["baselearner_args"]["bn_after_trans"] = True
        
        conf["baselearner_args"]["enable_conv"] = args.enable_conv.lower() == "true"
        
        
        args.backbone = lowerstr
        
    # Print the configuration for confirmation
    print_conf(conf)
    

    if args.problem == "linear" or "sine" in args.problem or "toy" in args.problem or "poly" in args.problem:
        episodic = True
        args.batchmode = False
        if args.model in baselines:
            episodic = False
            args.batchmode = True
        
        print(args.train_batch_size)
        args.data_loader = data_loader
        val_loader = data_loader.generator(episodic=episodic, batch_size=args.test_batch_size, mode="val")
        test_loader = data_loader.generator(episodic=episodic, batch_size=args.test_batch_size, mode="test")
        train_loader = data_loader.generator(episodic=episodic, batch_size=args.train_batch_size, mode="train")
        print("train batch size:", args.train_batch_size)
        print("test batch size:", args.test_batch_size)
        args.linear = True
        args.sine = True
        args.eval_iters = 1000
    else:
        args.linear = False
        args.sine = False



    
    args.resdir = RESULT_DIR
    bstr = args.backbone if not args.backbone is None else ""
    # Ensure that ./results directory exists
    create_dir(args.resdir)
    args.resdir += args.problem + '/'
    # Ensure ./results/<problem> exists
    create_dir(args.resdir)
    if args.N:
        args.resdir += 'N' + str(args.N) + 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    else:
        args.resdir += 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    # Ensure ./results/<problem>/k<k>test<k_test> exists
    create_dir(args.resdir)
    if args.model_spec is None:
        args.resdir += args.model + '/'
    else:
        args.resdir += args.model_spec + '/'
    # Ensure ./results/<problem>/k<k>test<k_test>/<model>/ exists
    create_dir(args.resdir)

    # If args.single_run is true, we should store the results in a directory runs
    if args.single_run or args.runs < 30:
        args.resdir += f"{bstr}-runs/"
        create_dir(args.resdir)
        args.resdir += f"run{args.seed}-" 

    if not args.only_cross:
        test_loaders = [test_loader]
        filenames = [args.resdir+f"{args.backbone}-test_scores.csv"]
        loss_filenames = [args.resdir+f"{args.backbone}-test_losses-T{conf['T_test']}.csv"]
    else:
        test_loaders = []
        filenames = []
        loss_filenames = []


    with open(f"{args.resdir}config.pkl", "wb") as f:
        pickle.dump(conf, f)
        print(f"Stored config in {args.resdir}config.pkl")

    if args.eval_on_train:
        train_classes = args.N

        loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                        meta_val=False, meta_test=False, meta_split="train",
                        transform=transform,
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        loader = ClassSplitter(loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
        loader = BatchMetaDataLoader(loader, batch_size=1, num_workers=4, shuffle=True)
        test_loaders.append(loader)
        filenames.append(args.resdir+f"{args.backbone}-train_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-train_losses-T{conf['T_test']}.csv")
    
    if args.cross_eval:
        test_loaders.append(cross_loader)
        filenames.append(args.resdir+f"{args.backbone}-cross_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-cross_losses-T{conf['T_test']}.csv")        

    if args.gamma is None or args.model == "sap" or args.model == "psap" or args.model=="sapresnet":
        return args, conf, train_loader, val_loader, test_loaders, [filenames, loss_filenames], model_constr
    else:
        return args, conf, [train_loader, train_loader2], val_loader, test_loaders, [filenames, loss_filenames], model_constr


def validate(model, data_loader, best_score, best_state, conf, args, val_losses=None):
    """Perform meta-validation
        
    Create meta-validation data generator obejct, and perform meta-validation.
    Update the best_loss and best_state if the current loss is lower than the
    previous best one. 

    Parameters
    ----------
    model : Algorithm
        The chosen meta-learning model
    data_loader : DataLoader
        Data container which can produce a data generator
    best_score : float
        Best validation performance obtained so far
    best_state : nn.StateDict
        State of the meta-learner which gave rise to best_loss
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    best_loss
        Best obtained loss value so far during meta-validation
    best_state
        Best state of the meta-learner so far
    score 
        Performance score on this validation run
    """
    
    print("[*] Validating performance...")
    scores = []
    c = 0
    if not args.sine:
        for epoch in val_loader:
            (train_x, train_y), (test_x, test_y) = epoch['train'], epoch['test']

            if args.num_aug > 0 :
                # only augment support and normalize query images
                train_x, train_y, test_x, test_y = train_x[0], train_y[0].repeat(args.num_aug), test_x[0], test_y[0]
                mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
                std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

                
                if args.num_aug > 1:
                    transform = Compose([
                        RandomCrop(84, padding=8),
                        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        RandomHorizontalFlip(),
                        # lambda x: np.asarray(x),
                        #ToTensor(),
                        Normalize(mean=mean, std=std)
                    ])
                else:
                    transform = Compose([
                        Normalize(mean=mean, std=std)
                    ])

                norm_transform = Compose([Normalize(mean=mean, std=std)])
                
                augmented_train_x = []
                for _ in range(args.num_aug):
                    newx = transform(train_x)
                    augmented_train_x.append(newx)
                train_x = torch.cat(augmented_train_x)

 
                test_x = norm_transform(test_x) #train_x = transform(train_x); 

                acc, loss_history = model.evaluate(train_x = train_x, 
                                train_y = train_y, 
                                test_x = test_x, 
                                test_y = test_y)
            else:

                acc, loss_history = model.evaluate(train_x = train_x[0], 
                                    train_y = train_y[0], 
                                    test_x = test_x[0], 
                                    test_y = test_y[0])
            scores.append(acc)
            if not val_losses is None:
                val_losses.append(loss_history[-1])
            c+=1
            if c == args.eval_iters:
                break
            if args.debug:
                break
    else:
        loader = args.data_loader.generator(mode="val", batch_size=args.test_batch_size, episodic=True, reset_ptr=True)
        for epoch in loader:
            train_x, train_y, test_x, test_y = epoch
            acc, loss_history = model.evaluate(train_x = train_x, 
                                train_y = train_y, 
                                test_x = test_x, 
                                test_y = test_y)
            if not val_losses is None:
                val_losses.append(loss_history[-1])
            scores.append(acc)
            if args.debug:
                break

    score = np.mean(scores)
    # Compute min/max (using model.operator) of new score and best score 
    tmp_score = model.operator(score, best_score)
    # There was an improvement, so store info
    if tmp_score != best_score and not math.isnan(tmp_score):
        best_score = score
        best_state = model.dump_state()
    print("validation loss:", score)
    return best_score, best_state, score
        
def body(args, conf, train_loader, val_loader, test_loaders, files, model_constr):
    """Create and apply the meta-learning algorithm to the chosen data
    
    Backbone of all experiments. Responsible for:
    1. Creating the user-specified model
    2. Performing meta-training
    3. Performing meta-validation
    4. Performing meta-testing
    5. Logging and writing results to output channels
    
    Parameters
    -----------
    args : arguments
        Parsed command-line arguments
    conf : dictionary
        Configuration dictionary with all model arguments required for construction
    data_loader : DataLoader
        Data loder object which acts as access point to the problem data
    model_const : constructor fn
        Constructor function for the meta-learning algorithm to use
    
    """
        
    # Write learning curve to file "curves<val_after>.csv"    
    curvesfile = args.resdir+f"{args.backbone}-curves"+str(args.val_after)+".csv"

    overall_best_score = get_init_score_and_operator(conf["baselearner_args"]["criterion"])[0]
    overall_best_state = None
    print("overall best score:", overall_best_score)
    
    seeds = [random.randint(0, 100000) for _ in range(args.runs)]
    print("Actual seed:", seeds)

    if not args.gamma is None and not "sap" in args.model:
        train_loader, train_loader2 = train_loader

    time_curves = []
    for run in range(args.runs):
        time_curve = []
        stime = time.time()
        print("\n\n"+"-"*40)
        print(f"[*] Starting run {run}")
        # Set torch seed to ensure same base-learner initialization across techniques
        torch.manual_seed(seeds[run])
        model = model_constr(**conf)
        
        if args.model == "spft":
            model.setmap(args.labels)
        
        if "sap" in args.model and not args.pretrained_path is None:
            if args.backbone.lower() == "dropblock":
                weights = torch.load(args.pretrained_path)["model"]
                model.baselearner.load_state_dict(weights, strict=False)
                model.baselearner = model.baselearner.to(model.dev)
                print("Loaded pre-trained weights")
            elif not "real" in args.backbone:
                weights = dict(torch.load(args.pretrained_path)['params'])
                model.baselearner.load_state_dict(weights, strict=False)
                model.baselearner = model.baselearner.to(model.dev)
                print("Loaded pre-trained weights")
            else:
                sd = torch.load(args.pretrained_path)["model_sd"]
                for key in list(sd.keys()):
                    new_key = "".join(key.split("encoder.")[-1])
                    if new_key == key: # classifier components not copied
                        print(f"removed {new_key}")
                        del sd[key]
                        continue
                    sd[new_key] = deepcopy(sd[key])
                    del sd[key]
                model.baselearner.load_state_dict(sd, strict=False)
                model.baselearner = model.baselearner.to(model.dev)
                print("Loaded pre-trained weights".upper())
        
        # num_params = sum([p.numel() for p in model.baselearner.parameters()])
        # print("Number of parameters:", num_params)
        # import sys; sys.exit()

        if model.operator == max:
            logstr = "accuracy"
        else:
            logstr = "loss"
    

        vtime = time.time()
        val_losses = []

        if args.model_path is None or args.dotraining:

            if args.model_path:
                print("loading model, HERE")
                try:
                    model.top = args.top
                    model.read_file(args.model_path+f"model-{run}.pkl")
                    print("Loaded model from:", args.model_path+f"model-{run}.pkl")
                except Exception as e:
                    print(e)
                    import sys; sys.exit()


            # Start with validation to ensure non-trainable model get 
            # validated at least once
            if args.validate:
                best_score, best_state = model.init_score, None
                best_score, best_state, score = validate(model, val_loader, 
                                                        best_score, best_state, 
                                                        conf, args, val_losses=val_losses)
                print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
                # Stores all validation performances over time (learning curve) 
                learning_curve = [score]
                
            
            if model.trainable:
                dcounter = [1,0] if conf["decouple"] else [0]

                print('\n[*] Training...')
                ttime = time.time()
                for el in dcounter:
                    c = 0
                    history = []
                    if args.sine:
                        train_loader = args.data_loader.generator(episodic=True, batch_size=args.train_batch_size, reset_ptr=True, mode="train")
                    custom = "custom" in args.problem
                    for eid, epoch in enumerate(train_loader):
                        if args.no_training:
                            break

                        if not args.gamma is None and not "sap" in args.model:
                            joint_x, joint_y = train_loader2.__next__()

                        #task_time = time.time()
                        # Unpack the episode. If the model is non-episodic in nature, test_x and 
                        # test_y will be None
                        if args.batchmode:
                            if args.linear:
                                train_x, train_y, _, _ = epoch
                                model.train(train_x=train_x, train_y=train_y, test_x=None, test_y=None)
                            else:
                                train_x, train_y = epoch
                                model.train(train_x=train_x, train_y=train_y.view(-1), test_x=None, test_y=None)
                        elif custom:
                            (train_x, train_y, global_train_y), (test_x, test_y, global_test_y) = epoch['train'], epoch['test']
                            #print(global_train_y, global_train_y.size())
                            model.train(train_x=train_x[0], train_y=train_y[0], test_x=test_x[0], test_y=test_y[0], 
                                        global_train_y=global_train_y[0], global_test_y=global_test_y[0])
                            #import sys; sys.exit()
                        else:
                            if args.linear:
                                (train_x, train_y, test_x, test_y) = epoch
                                model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
                            else:
                                (train_x, train_y), (test_x, test_y) = epoch['train'], epoch['test']
                                if args.num_aug > 0:

                                    train_x, train_y, test_x, test_y = train_x[0], train_y[0].repeat(args.num_aug), test_x[0], test_y[0].repeat(args.num_aug)

                                    mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
                                    std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

                                    
                                    if args.num_aug > 1:
                                        transform = Compose([
                                            RandomCrop(84, padding=8),
                                            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                            RandomHorizontalFlip(),
                                            # lambda x: np.asarray(x),
                                            #ToTensor(),
                                            Normalize(mean=mean, std=std)
                                        ])
                                    else:
                                        transform = Compose([
                                            Normalize(mean=mean, std=std)
                                        ])


                                    augmented_train_x = []
                                    for _ in range(args.num_aug):
                                        newx = transform(train_x)
                                        augmented_train_x.append(newx)
                                    train_x = torch.cat(augmented_train_x)


                                    augmented_test_x = []
                                    for _ in range(args.num_aug):
                                        newx = transform(test_x)
                                        augmented_test_x.append(newx)
                                    test_x = torch.cat(augmented_test_x)
                                    
                                    model.train(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
                                else:
                                    # Perform update using selected batch
                                    if not args.gamma is None and not "min" in args.problem and not "cub" in args.problem and not "tiered" in args.problem:
                                        model.train(train_x=train_x[0], train_y=train_y[0], test_x=test_x[0], test_y=test_y[0], joint_x=joint_x, joint_y=joint_y)
                                    else:
                                        model.train(train_x=train_x[0], train_y=train_y[0], test_x=test_x[0], test_y=test_y[0])
                        #print(time.time() - task_time, "seconds")
                        #task_time = time.time()

                        if eid % 1000 == 0 and args.linear and not args.sine:
                            model_params = np.array(model._get_params())
                            history.append(model_params)
                            print(model_params)
                            print(np.mean(model.train_losses[-1000:]))
                        
                        if args.linear and eid%10000==0 and not args.sine:
                            hist = np.array(history)
                            x = hist[:,0]
                            y = hist[:,1]
                            plt.figure()
                            plt.xlim((0,6))
                            plt.ylim((0,6))
                            plt.plot(x,y, color='green')
                            plt.scatter(x[-1], y[-1], color='blue', label='End')
                            plt.scatter(x[0], y[0], color='red', label='Start')
                            tasks = [(1,1), (1,2), (2,2), (2,1), (3,4), (4,4), (4,3), (3,3)]
                            plt.scatter([t[0] for t in tasks], [t[1] for t in tasks], color='black', label='optimal') 
                            plt.legend()
                            plt.show()     

                        # Perform meta-validation
                        if args.validate and (eid + 1) % args.val_after == 0 and el != 1:
                            print(f"{time.time() - ttime} seconds for training")
                            time_curve.append(time.time() - ttime)
                            vtime = time.time()
                            best_score, best_state, score = validate(model, val_loader, 
                                                                    best_score, best_state, 
                                                                    conf, args, val_losses=val_losses)
                            print(f"[*] Done validating, cost: {time.time()-vtime} seconds")
                            # Store validation performance for the learning curve
                            # note that score is more informative than best_score 
                            learning_curve.append(score)
                            ttime = time.time()

                        c+=1
                        if c == args.train_iters:
                            print("c reached", args.train_iters)
                            break
                        if args.debug:
                            break

            if args.linear and not args.sine:
                hist = np.array(history)
                x = hist[:,0]
                y = hist[:,1]
                plt.figure()
                plt.xlim((0,6))
                plt.ylim((0,6))
                plt.plot(x,y, color='green')
                plt.scatter(x[-1], y[-1], color='blue', label='End')
                plt.scatter(x[0], y[0], color='red', label='Start')
                tasks = [(1,1), (1,2), (2,2), (2,1), (3,4), (4,4), (4,3), (3,3)]
                plt.scatter([t[0] for t in tasks], [t[1] for t in tasks], color='black', label='optimal') 
                plt.legend()
                plt.show()       


            if args.validate:
                # Load best found state during meta-validation
                model.load_state(best_state, reset=False)
                save_path = args.resdir+f"model-{run}.pkl"
                print(f"[*] Writing best model state to {save_path}")
                model.store_file(save_path)
                if args.log_test_norm:
                    model.log_test_norm = True
        else:
            # Load the pre-trained model and skip training. Just go directly to testing
            try:
                model.read_file(args.model_path+f"model-{run}.pkl", reset=False)
                print("Loaded model from:", args.model_path+f"model-{run}.pkl")
            except Exception as e:
                print(e)
                import sys; sys.exit()

        
        if args.measure_distances:
            model.measure_distances = True

        generators = test_loaders
        filenames, loss_filenames = files 
        print(generators, filenames, loss_filenames)
        
        time_curves.append(time_curve)

        # Set seed and next test seed to ensure test diversity
        set_seed(args.test_seed)
        args.test_seed = random.randint(0,100000)
        if not args.linear or args.sine:
            for idx, (eval_gen, filename) in enumerate(zip(generators, filenames)):
                accstring = "" if not "cross" in filename else "cross_"

                accfile = args.resdir+f"{args.backbone}-{accstring}alltestperfs.csv"
                accmode = "w+" if run ==0 else "a"

                test_accuracies = []
                print('\n[*] Evaluating test performance...')
                

                loss_info = []
                c = 0
                if args.sine:
                    eval_gen = args.data_loader.generator(episodic=True, batch_size=args.test_batch_size, mode="test", reset_ptr=True)
                for eid, epoch in enumerate(eval_gen):
                    if args.sine:
                        train_x, train_y, test_x, test_y  = epoch
                        acc, loss_history = model.evaluate(
                                train_x = train_x, 
                                train_y = train_y, 
                                test_x = test_x, 
                                test_y = test_y, 
                                val=False #real test! no validation anymore
                        )
                    else:
                        (train_x, train_y), (test_x, test_y)  = epoch['train'], epoch['test'] 

                        if args.num_aug > 0:
                            # only augment support and normalize query images
                            train_x, train_y, test_x, test_y = train_x[0], train_y[0].repeat(args.num_aug), test_x[0], test_y[0]
                            mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
                            std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

                            
                            if args.num_aug > 1:
                                transform = Compose([
                                    RandomCrop(84, padding=8),
                                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    RandomHorizontalFlip(),
                                    # lambda x: np.asarray(x),
                                    #ToTensor(),
                                    Normalize(mean=mean, std=std)
                                ])
                            else:
                                transform = Compose([
                                    Normalize(mean=mean, std=std)
                                ])

                            norm_transform = Compose([Normalize(mean=mean, std=std)])
                            
                            augmented_train_x = []
                            for _ in range(args.num_aug):
                                newx = transform(train_x)
                                augmented_train_x.append(newx)
                            train_x = torch.cat(augmented_train_x)
                            test_x = norm_transform(test_x) #train_x = transform(train_x); 

                            acc, loss_history = model.evaluate(train_x = train_x, 
                                            train_y = train_y, 
                                            test_x = test_x, 
                                            test_y = test_y)
                        else:
                            acc, loss_history = model.evaluate(
                                    train_x = train_x[0], 
                                    train_y = train_y[0], 
                                    test_x = test_x[0], 
                                    test_y = test_y[0], 
                                    val=False #real test! no validation anymore
                            )
                    test_accuracies.append(acc)
                    c+=1
                    loss_info.append(loss_history)
                    if not args.sine:
                        if c >= args.eval_iters:
                            break    
                        if args.debug:
                            break  
                

                if args.measure_distances:
                    fn = args.model_path+f"measured_distances_support-{run}.npy"
                    fnq = args.model_path+f"measured_distances_query-{run}.npy"
                    np.save(fn, np.array(model.supp_dists))
                    np.save(fnq, np.array(model.query_dists))


                # Create files and headers if they do not exist or if we started a new run (and want to overwrite previous results)
                if not os.path.exists(filename) or run == 0:
                    with open(filename, "w+", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["run",f"mean_{logstr}",f"median_{logstr}"])
                
                # loss file name
                lfname = loss_filenames[idx]
                flat_losses = [item for sublist in loss_info for item in sublist]
                if not os.path.exists(lfname) or run == 0:
                    open_mode = "w+"
                else:
                    open_mode = "a"
                # Write learning curve to file
                with open(lfname, open_mode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(flat_loss) for flat_loss in flat_losses])

                print(test_accuracies)
                r, mean, median = str(run), str(np.mean(test_accuracies)),\
                                str(np.median(test_accuracies))
                with open(filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([r, mean, median])
                
                with open(accfile, accmode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(acc_p) for acc_p in test_accuracies])
                
                valc_file = args.resdir+f"{args.backbone}-valoss_curve.csv"
                try:
                    if not os.path.exists(valc_file) or run == 0:
                        open_mode = "w+"
                    else:
                        open_mode = "a"
                    val_losses = [str(x) for x in val_losses]

                    with open(valc_file, open_mode, newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(val_losses)
                except Exception as e:
                    print("Error saving val losses to file")
                    print(e)
                    pass


                gpu_file = args.resdir+f"{args.backbone}-gpumem.csv"
                try:
                    if not os.path.exists(gpu_file) or run == 0:
                        open_mode = "w+"
                    else:
                        open_mode = "a"
                    gpu_usage = [str(x) for x in model.gpu_usage]

                    with open(gpu_file, open_mode, newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(gpu_usage)
                except Exception as e:
                    print("Error saving gpu usage to file")
                    print(e)
                    pass


                if args.train_curve:
                    for s in ["losses", "scores"]:
                        trloss_file = args.resdir+f"{args.backbone}-train_{s}.csv"
                        try:
                            if not os.path.exists(trloss_file) or run == 0:
                                open_mode = "w+"
                            else:
                                open_mode = "a"
                            if "loss" in s:
                                train_losses = [str(x) for x in model.train_losses]
                            else:
                                train_losses = [str(x) for x in model.train_scores]

                            with open(trloss_file, open_mode, newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(train_losses)
                        except:
                            print("Error saving train losses to file")
                            pass
                

                # Loss info 
                # linfo = np.array(loss_info)
                # cols = linfo.shape[1]
                # mean_initloss = str(np.mean(linfo[:,0]))
                # median_initloss = str(np.median(linfo[:,0]))
                # mean_multiloss = str(np.mean(np.mean(linfo[:,1:cols-1], axis=1)))
                # median_multiloss = str(np.median(np.mean(linfo[:,1:cols-1], axis=1)))
                # mean_finloss = str(np.mean(linfo[:,cols-1]))
                # median_finloss = str(np.median(linfo[:,cols-1]))


                # loss file name
                # with open(lfname, "a", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow([r, mean_initloss, median_initloss, mean_multiloss, 
                #                      median_multiloss, mean_finloss, median_finloss])

                print(f"Run {run} done, mean {logstr}: {mean}, median {logstr}: {median}")
                print(f"Time used: {time.time() - stime}")
                print("-"*40)
                if args.sine:
                    break            
            
            if not args.model_path is None and not args.dotraining:
                print("Done training")
                import sys; sys.exit()

            if args.validate:
                print(learning_curve)
                # Determine writing mode depending on whether learning curve file already exists
                # and the current run
                if not os.path.exists(curvesfile) or run == 0:
                    open_mode = "w+"
                else:
                    open_mode = "a"
                # Write learning curve to file
                with open(curvesfile, open_mode, newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([str(score) for score in learning_curve])
                
                # Check if the best score is better than the overall best score
                # if so, update best score and state across runs. 
                # It is better if tmp_best != best
                tmp_best_score = model.operator(best_score, overall_best_score)
                if tmp_best_score != overall_best_score and not math.isnan(tmp_best_score):
                    print(f"[*] Updated best model configuration across runs")
                    overall_best_score = best_score
                    overall_best_state = deepcopy(best_state)

            if args.log_test_norm:
                np.save(f"{args.model_spec}-{args.backbone}-losses-{run}.npy", model.test_losses)
                np.save(f"{args.model_spec}-{args.backbone}-norms-{run}.npy", model.test_norms)
                np.save(f"{args.model_spec}-{args.backbone}-perfs-{run}.npy", model.test_perfs)
                np.save(f"{args.model_spec}-{args.backbone}-dists-{run}.npy", model.distances)
                np.save(f"{args.model_spec}-{args.backbone}-angles-{run}.npy", model.angles)
                np.save(f"{args.model_spec}-{args.backbone}-gangles-{run}.npy", model.gangles)
                np.save(f"{args.model_spec}-{args.backbone}-gdistances-{run}.npy", model.gdistances)

            if args.model == "sap":
                alfa_hist = np.array(model.alfa_history)
                np.save(f"{args.resdir}alfa_history-{run}.npy", alfa_hist)
                print(f"Stored alfa history in {args.resdir}alfa_history-{run}.npy")

                if args.discrete_ops:
                    distribution_hist = np.array(model.distribution_history)
                    np.save(f"{args.resdir}distr_history-{run}.npy", distribution_hist)
                    print(f"Stored distribution history in {args.resdir}distr_history-{run}.npy")
    
    save_time = args.resdir+"time.npy"
    np.save(save_time, np.array(time_curves))



    # At the end of all runs, write the best found configuration to file
    if args.validate:            
        save_path = args.resdir+f"model.pkl"
        print(f"[*] Writing best model state to {save_path}")
        model.load_state(overall_best_state, reset=False)
        model.store_file(save_path)

def set_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == "__main__":
    # Parse command line arguments
    args, unparsed = FLAGS.parse_known_args()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")
    
    # Set device to cpu if --cpu was specified
    if args.cpu:
        args.dev="cpu"
    
    # If cpu argument wasn't given, check access to CUDA GPU
    # defualt device is cuda:1, if that raises an exception
    # cuda:0 is used
    if not args.cpu:
        print("Current device:", torch.cuda.current_device())
        print("Available devices:", torch.cuda.device_count())
        if not args.devid is None:
            torch.cuda.set_device(args.devid)
            args.dev = f"cuda:{args.devid}"
            print("Using cuda device: ", args.dev)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable.")
            try:
                torch.cuda.set_device(1)
                args.dev="cuda:1"
            except:
                torch.cuda.set_device(0)
                args.dev="cuda:0"

    # Let there be reproducibility!
    set_seed(args.seed)
    print("Chosen seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.test_seed = random.randint(0,100000)

    # Let there be recognizability!
    print(BANNER)
    print(NAMETAG)

    # Let there be structure!
    pargs, conf, train_loader, val_loader, test_loaders, files, model_constr = setup(args)


    # Let there be beauty!
    body(pargs, conf, train_loader, val_loader, test_loaders, files, model_constr)