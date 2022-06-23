import os
import csv
import numpy as np
import torch
import pickle
import random
from functools import partial

DIRNAME = "./data/sine/"
FILES = ["train.csv", "val.csv", "test.csv"]
SIZES = [70000, 1000, 2000] # Number of functions per split (train/val/test)
TOTAL = sum(SIZES)

class PolynomialLoader:
    """
    Data loader for sine wave regression

    ...

    Attributes
    -------
    ptr : dict
        Mapping of operation mode to episode index -> [train/val/test]->[episode index]
    k : int
        Number of examples in all support sets
    k_test : int
        Number of examples in query sets
    functions : dict
        Dictionary of functions -> [train/val/test] -> list of (amplitude,phase) pairs
    episodic_fn_data : dict
        Episode container [train/val/test]-> list of episodes (train_x, train_y, test_x, test_y)
    flat_fn_data : dict
        Container used for sampling flat batches (without task structure)
        [train/val/test]->[x/y]->all inputs/labels of that mode

    Methods
    -------
    _load_data()
        Prepare and load data into the loader object
    _sample_batch(self, size, mode)
        Sample a flat batch of data (no explicit task structure)
    _sample_episode(mode)
        Sample an episode consisting of a support and query set
    _draw_props()
        Generates a random amplitude and phase
    _draw_fn(return_props=False)
        Generates an actual sine function
    _get_fn(amplitude, phase)
        Returns a sine function with the given amplitude and phase
        -- Not used at the moment
    generator(mode, batch_size)
        Return a generator object that iterates over episodes
    """
    
    def __init__(self, order, k, k_test, seed=1337, **kwargs):
        """
        initialize random seed used to generate sine wave data

        Parameters
        -------
        k : int
            Sizes of support sets (and size of query set during meta-training time)
        k_test : int
            Sizes of query sets
        seed : int, optional
            Randoms seed to use
        **kwargs : dict, optional
            Trash can for optional arguments that are ignored (but has to stay for function call uniformity)
        """

        random.seed(seed)
        np.random.seed(seed)

        self.k = k
        self.k_test = k_test
        self.order = order
        assert self.order >= 0, "Order should be at least 1"

    def _draw_props(self, **kwargs):
        """Generate random amplitude and phase

        Select amplitude and phase uniformly at random.
        Interval for amplitude : [0.1, 5.0]
        Interval for phase : [0, 3.14...(pi)]

        Returns
        ----------
        amplitude
            Amplitude of the sine function
        Phase
            Phase of the sine function
        """

        coefficients = np.random.uniform(low=-1.0, high=+1.0, size=(self.order+1))  
        return coefficients
    
    def _draw_fn(self, return_props=False, train=True):
        """Generate random sine function

        Randomly generate sine function fn that takes as input a real-valued x
        and returns y=fn(x) 
        The function has the form fn(x) = phase * np.sin(x + phase)

        Parameters
        ----------
        return_props : bool, optional
            Whether to return the amplitude and phase

        Returns
        ----------
        function
            The generated sine function
        amplitude (optional)
            Amplitude of the function
        phase (optional)
            Phase of the function
        """
        
        coefficients = self._draw_props()
        
        def fn(x):
            powers = np.power(x, np.arange(self.order+1)) # matrix of dim [batch_size, order+1] such that every row consists of x**powers
            return np.dot(powers, coefficients)
        
        if return_props:
            return fn, coefficients
        
        return fn

    def _get_fn(self, coefficients):
        """Construct sine function 

        Use the provided amplitude and phase to return the corresponding
        sine function

        Parameters
        ----------
        amplitude : float
            Amplitude of the function
        phase : float
            Phase of the function

        Returns
        ----------
        function
            The sine function with user-defined amplitude and phase
        """
        
        def fn(x):
            powers = np.power(x, np.arange(self.order+1)) # matrix of dim [batch_size, order+1] such that every row consists of x**powers
            return np.dot(powers, coefficients)

        return fn
    
    def _generate_data(self, k, k_test, fn, tensor=True):
        """Generate input, output pairs for a given sine function

        Return input and output vectors x, y. Every y_i = fn(x_i) 

        Parameters
        ----------
        k : int
            Number of (x,y) pairs to generate
        k_test : int
            Number of examples in query set
        fn : function
            Sine function to use for data point generation
        tensor : bool, optional
            Whether to return x and y as torch.Tensor objects
            (default is np.array with dtype=float32)

        Returns
        ----------
        train_x
            Inputs of support set, randomly sampled from [-5,5]
        train_y
            Outputs of support set 
        test_x
            Inputs of query set drawn at random from [-5,5]
        test_y
            Outputs of query set
        """
        x = np.linspace(-5.0, 5.0, k+k_test).reshape(-1, 1).astype('float32')
        y = fn(x).reshape(-1, 1).astype('float32') 
        train_x, train_y, test_x, test_y = x[:k], y[:k], x[k:], y[k:]

        if tensor:
            return torch.from_numpy(train_x), torch.from_numpy(train_y),\
                   torch.from_numpy(test_x), torch.from_numpy(test_y)
        return train_x, train_y, test_x, test_y 


    def _sample_episode(self, return_props, **kwargs):
        """Sample a single episode
        
        Look up and return the current episode for the given mode

        Parameters
        ----------
        mode : str
            "train"/"val"/"test": mode of operation
        **kwargs : dict
            Trashcan for additional args

        Returns 
        ----------
        train_x
            Inputs of support sets
        train_y
            Outputs of support set 
        test_x
            Inputs of query set
        test_y
            Outputs of query set
        """
        if not return_props:
            fn = self._draw_fn(return_props=False)
        else:
            fn, amplitude, freq, phase, outshift = self._draw_fn(return_props=True)

        train_x, train_y, test_x, test_y  = self._generate_data(self.k, self.k_test, fn)

        if not return_props:
            return train_x, train_y, test_x, test_y

        return train_x, train_y, test_x, test_y, [fn, amplitude, phase, freq, outshift]

    def generator(self, episodic, batch_size, mode, return_props=False, **kwargs):
        """Data generator
        
        Iterate over all tasks (if episodic), or for a fixed number of episodes (if episodic=False)
        and yield batches of data at every step.

        Parameters
        ----------
        episodic : boolean
            Whether to return a task (train_x, train_y, test_x, test_y) or a flat batch (x, y)
        mode : str
            "train"/"val"/"test": mode of operation
        batch_size : int
            Size of flat batch to draw
        reset_ptr : boolean, optional
            Whether to reset the episode pointer for the given mode
        **kwargs : dict
            Other optional keyword arguments to keep flexibility with other data loaders which use other 
            args like N (number of classes)
        
        Returns 
        ----------
        generator
            Yields episodes = (train_x, train_y, test_x, test_y)
        """

        if mode == "train":
            iters = 70000
            amode = 0
        elif mode == "val":
            iters = 1000
            amode = 1
        else:
            iters = 2000
            amode = 1

        # If episodic set number of iterations to the number of tasks
        if episodic:
            print(f"\n[*] Creating episodic generator for '{mode}' mode")
            gen_fn = partial(self._sample_episode, return_props=return_props)
        else:
            print("Sine loader has to be set to episodic mode")
            import sys; sys.exit()

        print(f"[*] Generator set to perform {iters} iterations")
        for _ in range(iters):
            yield gen_fn(mode=amode, size=batch_size)
