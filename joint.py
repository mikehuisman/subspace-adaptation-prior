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
import torchmeta.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import json

from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm #Progress bars
from networks import SineNetwork, Conv4, BoostedConv4, ConvX, ResNet, LinearNet
from algorithms.metalearner_lstm import LSTMMetaLearner  
from algorithms.train_from_scratch import TrainFromScratch
from algorithms.finetuning import FineTuning
from algorithms.moso import MOSO
from algorithms.turtle import Turtle
from algorithms.reptile import Reptile
from algorithms.maml import MAML
from algorithms.ownlstm import LSTM
from algorithms.modules.utils import get_init_score_and_operator, set_weights, accuracy, put_on_device
from sine_loader import SineLoader
from image_loader import ImageLoader
from linear_loader import LinearLoader
from misc import BANNER, NAMETAG
from configs import TFS_CONF, FT_CONF, CFT_CONF, LSTM_CONF,\
                    MAML_CONF, MOSO_CONF, TURTLE_CONF, LSTM_CONF2,\
                    REPTILE_CONF
from batch_loader import BatchDataset, cycle, Data

FLAGS = argparse.ArgumentParser()

# Required arguments
FLAGS.add_argument("--problem", choices=["sine", "min", "cub", "linear"], required=True,
                   help="Which problem to address?")

FLAGS.add_argument("--k", type=int, required=True,
                   help="Number examples per task set during meta-validation and meta-testing."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_train", type=int, default=None,
                   help="Number examples per task set during meta-training."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_test", type=int, required=True,
                   help="Number examples per class in query set")

FLAGS.add_argument("--model", choices=["tfs", "finetuning", "centroidft", 
                   "lstm", "maml", "moso", "lstm2", "turtle", "reptile"], required=True,
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

FLAGS.add_argument("--beta1", type=float, default=0.9,
                   help="beta1 for (meta-)optimizer")

FLAGS.add_argument("--beta2", type=float, default=0.999,
                   help="beta2 for (meta-)optimizer")

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

FLAGS.add_argument("--train_iters", type=int, default=None,
                    help="Number of meta-training iterations")

FLAGS.add_argument("--model_spec", type=str, default=None,
                   help="Store results in file ./results/problem/k<k>test<k_test>/<model_spec>/")

FLAGS.add_argument("--model_spec_save", type=str, default=None,
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

FLAGS.add_argument("--cpu", action="store_true",
                   help="Use CPU instead of GPU")

FLAGS.add_argument("--test_opt", choices=["adam", "sgd"], default=None,
                   help="Optimizer to use at meta-validation or meta-test time for the finetuning model")

FLAGS.add_argument("--time_input", action="store_true", default=False,
                   help="Add a timestamp as input to TURTLE")                   

FLAGS.add_argument("--validate", action="store_true", default=False,
                   help="Validate performance on meta-validation tasks")

FLAGS.add_argument("--train_base_lr", type=float, default=None,
                   help="Inner level learning rate for meta-training")

FLAGS.add_argument("--test_lr", type=float, default=None, help="LR to use at meta-val/test time for finetuning")

FLAGS.add_argument("--no_freeze", action="store_true", default=False,
                   help="Whether to freeze the weights in the finetuning model of earlier layers")

FLAGS.add_argument("--eval_on_train", action="store_true", default=False,
                    help="Whether to also evaluate performance on training tasks")

FLAGS.add_argument("--test_adam", action="store_true", default=False,
                   help="Optimize weights with Adam, LR = 0.001 at test time.")

FLAGS.add_argument("--normal_ft", action="store_true", default=False, help="Normal joint optimization training")


FLAGS.add_argument('--split', required=True, choices=["all", "train", "val", "test"])

FLAGS.add_argument('--eval_problem', default=None, choices=['min', 'cub'])


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
    
    if args.normal_ft:
        assert args.model == "finetuning", "Incompatible argument. model should be finetuning because normal_ft is true"

    if args.k_train is None:
        args.k_train = args.k

    # Mapping from model names to configurations
    mod_to_conf = {
        "tfs": (TrainFromScratch, TFS_CONF),
        "finetuning": (FineTuning, FT_CONF),
        "centroidft": (FineTuning, CFT_CONF), 
        "lstm": (LSTMMetaLearner, LSTM_CONF),
        "lstm2": (LSTM, LSTM_CONF2),
        "maml": (MAML, MAML_CONF),
        "moso": (MOSO, MOSO_CONF),
        "turtle": (Turtle, TURTLE_CONF),
        "reptile": (Reptile, REPTILE_CONF)
    }

    baselines = {"tfs", "finetuning", "centroidft"}
    
    # Get model constructor and config for the specified algorithm
    real_model_constr, real_conf = mod_to_conf[args.model]
    model_constr, conf = mod_to_conf["finetuning"]

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

    set_batch_size(real_conf, args, "train_batch_size")
    set_batch_size(real_conf, args, "test_batch_size")
        
    # Set values of T, lr, and input type
    overwrite_conf(real_conf, args, "T")
    overwrite_conf(real_conf, args, "lr")
    overwrite_conf(real_conf, args, "input_type")
    overwrite_conf(real_conf, args, "beta")
    overwrite_conf(real_conf, args, "meta_batch_size")
    overwrite_conf(real_conf, args, "time_input")

    conf["no_annealing"] = args.no_annealing
    conf["test_adam"] = args.test_adam
    real_conf["no_annealing"] = args.no_annealing
    real_conf["test_adam"] = args.test_adam
    
    # Parse the 'layers' argument
    if not args.layers is None:
        try:
            layers = [int(x) for x in args.layers.split(',')]
        except:
            raise ValueError(f"Error while parsing layers argument {args.layers}")
        conf["layers"] = layers
        real_conf["layers"] = layers
    
    # Make sure argument 'val_after' is specified when 'validate'=True
    if args.validate:
        assert not args.val_after is None,\
                    "Please specify val_after (number of episodes after which to perform validation)"
    
    # If using multi-step maml, perform gradient clipping with -10, +10
    if not conf["T"] is None:
        if conf["T"] > 1 and (args.model=="maml" or args.model=="turtle"):# or args.model=="reptile"):
            conf["grad_clip"] = 10
            real_conf["grad_clip"] = 10
        elif args.model == "lstm" or args.model == "lstm2":
            conf["grad_clip"] = 0.25 # it does norm clipping
            real_conf["grad_clip"] = 0.25 # it does norm clipping
        else:
            conf["grad_clip"] = None
            real_conf["grad_clip"] = None
    
    # If MOSO or TURTLE is selected, set the activation function
    if args.activation:
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "sigmoid": nn.Sigmoid()
        }
        conf["act"] = act_dict[args.activation]
        real_conf["act"] = act_dict[args.activation]
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
    conf["beta1"] = args.beta1
    conf["beta2"] = args.beta2
    conf["test_opt"] = "adam"

    real_conf["cpe"] = args.cpe
    real_conf["dev"] = args.dev
    real_conf["second_order"] = args.second_order
    real_conf["history"] = args.history
    real_conf["layer_wise"] = args.layer_wise
    real_conf["param_lr"] = args.param_lr
    real_conf["decouple"] = args.decouple
    real_conf["batching_eps"] = args.batching_eps
    real_conf["freeze"] = not args.no_freeze
    real_conf["beta1"] = args.beta1
    real_conf["beta2"] = args.beta2
    real_conf["test_opt"] = "adam"
    
    args.test_opt = "adam"

    if not args.test_lr is None:
        assert args.model == "finetuning", "test_opt and test_lr arguments only suited for finetuning model"
        conf["test_opt"] = args.test_opt
        conf["test_lr"] = args.test_lr

    if args.T_test is None:
        conf["T_test"] = conf["T"]
        real_conf["T_test"] = conf["T"]
    else:
        conf["T_test"] = args.T_test
        real_conf["T_test"] = args.T_test
    
    if args.T_val is None:
        conf["T_val"] = conf["T"]
        real_conf["T_val"] = conf["T"]
    else:
        conf["T_val"] = args.T_val
        real_conf["T_val"] = args.T_val

    if not args.base_lr is None:
        conf["base_lr"] = args.base_lr
        real_conf["base_lr"] = args.base_lr


    if not args.train_base_lr is None:
        conf["train_base_lr"] = args.train_base_lr
        real_conf["train_base_lr"] = args.train_base_lr
    else:
        if args.test_lr is not None:
            conf["test_lr"] = args.test_lr
            conf["train_base_lr"] = args.test_lr
            real_conf["test_lr"] = args.test_lr
            real_conf["train_base_lr"] = args.test_lr
        else:
            try:
                conf["train_base_lr"] = conf["base_lr"]
                real_conf["train_base_lr"] = conf["base_lr"]
            except:
                print("exception")
                pass

    assert not (args.input_type == "maml" and args.history != "none"), "input type 'maml' and history != none are not compatible"
    assert not (conf["T"] == 1 and args.history != "none"), "Historical information cannot be used when T == 1" 

    normalize = False
    # Image problem
    if args.backbone is None:
        if args.model == "centroidft":
            conf["baselearner_fn"] = BoostedConv4
            real_conf["baselearner_fn"] = BoostedConv4
            lowerstr = "Bconv4"
        else:    
            conf["baselearner_fn"] = ConvX
            real_conf["baselearner_fn"] = ConvX
            lowerstr = "conv4"
        img_size = (84,84)
    else:
        lowerstr = args.backbone.lower()    
        args.backbone = lowerstr        
        if "resnet" in lowerstr:
            modelstr = "resnet"
            constr = ResNet
            img_size = (224,224)
        elif "conv" in lowerstr:
            modelstr = "conv"
            constr = ConvX
            img_size = (84,84)
        else:
            raise ValueError("Could not parse the provided backbone argument")
        
        num_blocks = int(lowerstr.split(modelstr)[1])
        print(f"Using backbone: {modelstr}{num_blocks}")
        conf["baselearner_fn"] = constr
        real_conf["baselearner_fn"] = constr
        if num_blocks > 4:
            normalize = True

    if normalize:
        transform = Compose([Resize(size=img_size), ToTensor(), 
                                Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                        np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
    else:
        transform = Compose([Resize(size=img_size), ToTensor()])

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



    if args.eval_problem is None:
        args.eval_problem = args.problem

    if "min" in args.eval_problem:
        ds = datasets.MiniImagenet
        cds = datasets.CUB
        dataset_specifier = Data.MIN
    elif "cub" in args.eval_problem:
        ds = datasets.CUB
        cds = datasets.MiniImagenet
        dataset_specifier = Data.CUB


    problem_to_root = {
        "min": "./data/miniimagenet/",
        "cub": "./data/cub/"
    }

    
    loader = BatchDataset(root_dir=problem_to_root[args.eval_problem],
                                dataset_spec=dataset_specifier, transform=transform, split=args.split)
    classes = loader.num_classes
    length = len(loader)
    if args.split == "all":
        args.classes_per_split = loader.classes_per_split
    test_size = length // 6; train_size = length - test_size 
    train_set, test_set = torch.utils.data.random_split(loader, [train_size, test_size])

    train_loader = iter(cycle(DataLoader(train_set, batch_size=conf["train_batch_size"], shuffle=True, num_workers=2)))
    test_loader = iter(cycle(DataLoader(test_set, batch_size=conf["train_batch_size"], shuffle=True, num_workers=2)))
    args.batchmode = True
    print("Using custom made BatchDataset")

        
    conf["baselearner_args"] = {
        "train_classes": classes,
        "eval_classes": classes, 
        "criterion": nn.CrossEntropyLoss(),
        "dev":args.dev
    }

    real_conf["baselearner_args"] = {
        "train_classes": classes,
        "eval_classes": classes, 
        "criterion": nn.CrossEntropyLoss(),
        "dev":args.dev
    }

    if not args.backbone is None:
        conf["baselearner_args"]["num_blocks"] = num_blocks
        real_conf["baselearner_args"]["num_blocks"] = num_blocks
    
    args.backbone = lowerstr
        
    # Print the configuration for confirmation
    print_conf(conf)

    
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


    args.resdir += f"{bstr}-runs/"
    create_dir(args.resdir)


    filenames = [args.resdir+f"{args.backbone}-test_scores.csv"]
    loss_filenames = [args.resdir+f"{args.backbone}-test_losses-T{conf['T_test']}.csv"]    

    return args, conf, train_loader, test_loader, [filenames, loss_filenames], model_constr, real_model_constr, real_conf
        

def get_save_paths(resdir):
    fn = "test_scores"
    files = [resdir+x for x in os.listdir(resdir) if fn in x]

    print("obtained resdir:", resdir)

    if len(files) == 0 :
        print("Could not find results file.")
        import sys; sys.exit()
    elif len(files) > 1:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@ WARNING: MORE THAN 1 RESULT FILES FOUND. MERGING THEM.")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    prefixes = ["/".join(x.split('/')[:-1])+'/'+x.split('/')[-1].split('-')[0] for x in files]
    seeds = set([x.split('/')[-1].split('-')[0] for x in files])

    print(files, prefixes)
    dfs = []
    lens = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
        lens.append(len(df))
        
    df = pd.concat(dfs)
    perfs = df["mean_accuracy"] 
    q1 = perfs.quantile(0.25)
    q3 = perfs.quantile(0.75)
    iqr = q3 - q1

    #sub = (perfs >= q1 - 1.5*iqr) & (perfs <= q3 + 1.5*iqr)
    sub = perfs > -1
    models_to_use = np.where(sub)[0]

    if len(models_to_use) == len(perfs):
        print("@@@@@@@- USING ALL MODELS")

    if len(seeds) > 1:
        print("Multiple seeds detected, loading best model for each")
        models_to_load = np.array(prefixes)[models_to_use]
        print("Will load:", [p+'-model.pkl' for p in prefixes])
        return [p+'-model.pkl' for p in prefixes]


    # model paths that should be locked and loaded
    mfiles = []
    # partition counter
    pc = 0
    # global counter, model id
    for gc, mid in enumerate(models_to_use):
        if mid > sum(lens[:pc+1]) - 1:
            pc += 1
        name = prefixes[pc]+f"-model-{mid-sum(lens[:pc])}.pkl"
        mfiles.append(name)
        print("Will load", name)
    
    return mfiles

def make_sd(model, weights, ignore="model.out"):
    sd = dict()
    for pid, (name,_) in enumerate(model.named_parameters()):
        if not ignore in name:
            sd[name] = weights[pid]
    return sd


def body(args, conf, real_conf, train_loader, test_loader, model_constr, real_model_constr):
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

    if args.eval_problem != args.problem:
        problem_specifier = f"{args.problem}2{args.eval_problem}"
    else:
        problem_specifier = args.problem

    print("Problem:", problem_specifier)

    create_dir("joint")
    create_dir(f"joint/{problem_specifier}")
    create_dir(f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/") 
    create_dir(f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/") 
    create_dir(f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/") 


    # Set seed and next test seed to ensure test diversity
    set_seed(args.test_seed)

    print(model_constr, real_model_constr)

    save_paths = get_save_paths(args.resdir) #[args.resdir+x for x in os.listdir(args.resdir) if "model-" in x]
    real_models = []; ft_models = []
    for sid, sp in enumerate(save_paths):
        real_models.append(real_model_constr(**real_conf))
        real_models[sid].to("cpu")
        torch.cuda.empty_cache()
        ft_models.append(model_constr(**conf))
        ft_models[sid].to("cpu")
        torch.cuda.empty_cache()

    for mid, model in enumerate(real_models):
        print("Loading model from", save_paths[mid])
        model.read_file(save_paths[mid], ignore_output=True)
        # Reptile and MAML only load initialization, so we have to clone them into the self.baselearner of the ft models
        if not model_constr == real_model_constr:
            sd = make_sd(ft_models[mid].baselearner, real_models[mid].initialization) 
            ft_models[mid].baselearner.load_state_dict(sd, strict=False)
        else:
            # we do finetuning so we can directly load the save_path
            ft_models[mid].read_file(save_paths[mid], ignore_output=True)

        ft_models[mid].baselearner.train()
        if not (args.model == "finetuning" and args.normal_ft):
            ft_models[mid].baselearner.freeze_layers(True)
        else:
            print("NOT FREEZING HIDDEN LAYERS")
        ft_models[mid].init_optimizer()
    del real_models


    use_alldata = args.split == "all"
    test_interval = 3

    if use_alldata:
        print("CLASSES:", args.classes_per_split)
        train = ["train" for _ in range(args.classes_per_split[0])] 
        val = ["val" for _ in range(sum(args.classes_per_split[:2]) - args.classes_per_split[0])] 
        test = ["test" for _ in range(sum(args.classes_per_split[:]) - sum(args.classes_per_split[:2]))] 
        split_map = train + val + test

        result_dict = [{"train":dict(), "val":dict(), "test":dict()} for _ in range(len(ft_models))]
        for mid in range(len(ft_models)):
            for split in result_dict[mid].keys():
                result_dict[mid][split]["correct"] = [0]
                result_dict[mid][split]["seen"] = [0] 

    AVG_ACC = [[] for _ in range(len(ft_models))]
    STDS = [[] for _ in range(len(ft_models))]
    for eid, epoch in enumerate(train_loader):
        for model in ft_models:
            model.baselearner.train()
            model.to(args.dev)
            train_x, train_y = epoch
            model.train(train_x=train_x, train_y=train_y.view(-1), test_x=None, test_y=None)
            model.to("cpu")
            torch.cuda.empty_cache()

        if (eid + 1) >= 100:
            test_interval = 100

        if (eid + 1) % test_interval == 0: 
            TEST_ACCURACIES = [[] for _ in range(len(ft_models))]
            for vid, vepoch in enumerate(test_loader):
                for mid, model in enumerate(ft_models):
                    model.baselearner.eval()
                    model.to(args.dev)
                    test_x, test_y = vepoch
                    test_x, test_y = put_on_device(args.dev, [test_x, test_y.view(-1)])
                    preds = torch.argmax(model.baselearner(test_x), dim=1)
                    acc = accuracy(preds, test_y)
                    TEST_ACCURACIES[mid].append(acc)
                    if use_alldata:
                        for pred,target in zip(preds, test_y):
                            split = split_map[target.item()]
                            result_dict[mid][split]["seen"][-1] += 1
                            if pred.item() == target.item(): result_dict[mid][split]["correct"][-1] += 1
                    model.to("cpu")     
                    torch.cuda.empty_cache()               

                if (vid + 1) * args.train_batch_size >= 10000:  
                    for mid, model in enumerate(ft_models):
                        mean_acc, std = np.mean(TEST_ACCURACIES[mid]), np.std(TEST_ACCURACIES[mid])
                        print(f"Mean acc, std: {mean_acc:.3f}, {std:.3f}")
                        AVG_ACC[mid].append(mean_acc); STDS[mid].append(std)
                        if use_alldata:
                            for split in result_dict[mid].keys():
                                # append a new counter to the "correct" and "seen" lists
                                result_dict[mid][split]["correct"].append(0)
                                result_dict[mid][split]["seen"].append(0)
                    break 

        if (eid + 1) * args.train_batch_size >= 100000:
            break

    # Store full data
    sp = f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/{args.split}-{args.model_spec_save}-full.np"
    np.save(sp, np.array(AVG_ACC))

    AVG_ACC = np.array(AVG_ACC).mean(axis=0)
    STDS = np.array(STDS).mean(axis=0)
    errors = 1.96 * STDS/len(ft_models)**0.5

    if use_alldata:
        results_per_split = dict()
        for split in ["train", "val", "test"]:
            avg_accuracies = []
            acc_stds = []
            acc_95ci = []
            breakflag=False
            for t in range(len(result_dict[0][split]["seen"])):
                # Holds accuracy scores for all models at given time step t
                accuracies = []
                for mid in range(len(ft_models)):
                    if result_dict[mid][split]["seen"][t] == 0:
                        breakflag = True
                        break
                    else:
                        acc = result_dict[mid][split]["correct"][t] / result_dict[mid][split]["seen"][t]
                    accuracies.append(acc)
                if breakflag:
                    break
                accuracies = np.array(accuracies)
                avg_acc = accuracies.mean()
                std_acc = accuracies.std()
                error = 1.96 * std_acc / len(ft_models)**0.5
                avg_accuracies.append(avg_acc)
                acc_stds.append(std_acc)
                acc_95ci.append(error)
            results_per_split[split] = dict()
            results_per_split[split]["acc"] = deepcopy(avg_accuracies)
            results_per_split[split]["std"] = deepcopy(acc_stds)
            results_per_split[split]["95ci"] = deepcopy(acc_95ci)

        save_path = f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/{args.split}-{args.model_spec_save}.json"
        print("Writing dictionary to file:", save_path)
        with open(save_path, "w+") as f:
            json.dump(results_per_split, f)
        save_path2 = f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/{args.split}-{args.model_spec_save}-fulldata.json"
        with open(save_path2, "w+") as f:
            json.dump(result_dict, f)
        print(f"Average accuracy: {AVG_ACC}")
        import sys; sys.exit()

    save_path = f"joint/{problem_specifier}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/{args.split}-{args.model_spec_save}.txt"

    with open(save_path, "w+") as f:
        f.writelines([",".join([str(x) for x in AVG_ACC])+"\n", ",".join([str(x) for x in STDS])+"\n", ",".join([str(x) for x in errors])+"\n" ])
        
    print(f"Average accuracy: {AVG_ACC}")

# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/7
def set_device():
    """Automatically chooses the right device to run pytorch on

    Returns:
        str: device identifier which is best suited (most free GPU, or CPU in case GPUs are unavailable)
    """
    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = int(np.argmax(memory_available))
        print("trying gpu_id", gpu_id)
        torch.cuda.set_device(gpu_id)
        dev = torch.cuda.current_device()
    else:
        dev = "cpu"
    return dev

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
            args.dev = torch.cuda.current_device()
            
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
    pargs, conf, train_loader, test_loader, _, model_constr, real_model_constr, real_conf = setup(args)


    # Let there be beauty!
    body(pargs, conf, real_conf, train_loader, test_loader, model_constr, real_model_constr)