import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

from torch.utils.data import DataLoader

class FFNN:
    '''Feedforward neural network.

    Not intended for big models!
    '''

    def __init__(self, layers, neurons, f_act=nn.ReLU(), loss=F.mse_loss, model=None, max_iter=200, holdout=0.1, batch_size=256, tol=1e-4, n_iter_no_change=10, opt=optim.Adam, seed=None):
        '''
        Parameters
        ----------
        layers : int
            The number of layers in the ffnn. Includes the input and output layer. i.e. (in:4, 10), (10, out:1) is 2 layers.
        neurons : List[int] or int
            The number of neurons per layer. If int, then each layer uses the same number of neurons. If list, then each element specifies the layer and the layers input is ignored.
        f_act : List[class torch.nn] or class torch.nn, optional
            The activation function per layer. If not list, then each layer uses the same activation functions. If list, then each element specifies the activation function of the layer and the layers input is ignored. 
        loss : class torch.nn.functional, optional
            The loss function, need to be created out of torch.nn.function functions. 
        model : class torch.nn or str, optional
            A fully custom model. If a str is given, then it is assumed that a trained model is at that location. If this is given then layers, neurons, and f_act is ignored.
        max_iter : int, optional
            The maximum number of epochs to train for, increase this to as high as you can affort computationally.
        holdout : float, optional
            The percentage of thet training set to use as the holdout set. Holdout set is used to determine convergence.
        batch_size : int, optional
            Batch size, usually a power of 2. Bigger values are more stable in training. Smaller values needed for larger data sets so it fits in memory.
        tol : float, optional
            When loss doesn't improve by tol for n_iter_no_change, then terminate.
        n_iter_no_change : int, optional
            The number of iterations that the loss of the holdout set doesn't improve by before terminating.
        opt : class torch.optim, optional
            The optimiser to use.
        seed : int, optional
            The seed for torch.
        '''

        if seed is not None:
            torch.manual_seed(seed)

        if type(model) is str:
            self.load(model)
        else:
            self.model = model

        if model is None:
            # neurons
            if type(neurons) is list: # custom
                self.neurons = neurons
            else: # create neuron list
                self.neurons = [neurons for _ in range(layers-1)]
            # f_act
            if type(f_act) is list: # custom
                self.f_act = f_act
            else: # create list based on other settings
                self.f_act = [f_act for _ in range(len(self.neurons))]

            # check that the neuron and activation layers are the same length
            if not ((len(self.neurons) == len(self.f_act)-1) or (len(self.neurons) == len(self.f_act))):
                raise ValueError('length of neurons and f_act is not compatible, need len(neurons) == len(f_act)-1 or len(neurons) == len(f_act) (last f_act is linear)')

        self.loss = loss
        self.max_iter = max_iter
        self.holdout = holdout
        self.batch_size = batch_size
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.opt = opt

    def _make_model(self, in_neurons, out_neurons):
        '''Make model.
        '''

        # make neuron blocks
        neurons = [in_neurons, *self.neurons, out_neurons]
        n_blocks = [nn.Linear(neurons[ind], neurons[ind+1]) for ind in range(len(neurons)-1)]
        
        # make model by folding n_blocks and f_blocks
        if len(n_blocks) == len(self.f_act): # f_act last layer
            model = []
            for n, f in zip(n_blocks, self.f_act):
                model.append(n)
                model.append(f)
        elif len(n_blocks)-1 == len(self.f_act): # linear f_act last
            model = [n_blocks[0]]
            for n, f in zip(n_blocks[1:], self.f_act):
                model.append(f)
                model.append(n)

        self.model = nn.Sequential(*model)

    def _converge(self, history):
        '''Determines if the model has converged. Return True when not converged.
        '''
        # too short to check
        if len(history) < self.n_iter_no_change:
            return True

        # check the last n_iter_no_changes values
        vals = np.array(history[-self.n_iter_no_change:])
        # all decreasing
        if not ((vals[1:] - vals[:-1]) <= 0).all(): 
            return True
        # all less than tolerance
        if not (np.abs(vals[1:] - vals[:-1]) < self.tol).all():
            return True
        return False

    def train(self, X, y, path=None, seed=None):
        '''Train model.

        Parameters
        ----------
        X : ndarray
            The features matrix.
        y : ndarray
            The outputs in the same row order as X.
        path : str, optional
            Path to save the model to. Needs to be a folder. All previous files with same name will be overwritten. If folder doesn't exist, it will be created. If no path given, best holdout model will not be saved. 
        seed : int, optional
            The seed to use when generating the holdout set. 
        '''
        
        # reset history every time you train
        self.history = {'train':[], 'holdout':[]}

        # reshape to get rid of warnings
        if len(y.shape) == 1:
            y.shape = (y.shape[0], -1)

        # make holdout set
        if self.holdout == 0: # use all data as training set
            train_X, train_y = X, y
            holdout_X, holdout_y = X, y
            holdout_X = torch.Tensor(holdout_X).float()
            holdout_y = torch.Tensor(holdout_y).float()
        else:
            holdout_num = int(X.shape[0]*self.holdout)
            inds = list(range(X.shape[0]))
            random.seed(a=seed)
            random.shuffle(inds)
            # train
            train_inds = inds[holdout_num:]
            train_X, train_y = X[train_inds], y[train_inds]
            train_y = y[train_inds]
            # holdout
            holdout_inds = inds[:holdout_num]
            holdout_X, holdout_y = X[holdout_inds], y[holdout_inds]
            holdout_X = torch.Tensor(holdout_X).float()
            holdout_y = torch.Tensor(holdout_y).float()

        # make model
        if self.model is None:
            self._make_model(train_X.shape[1], train_y.shape[1])
        # make optimiser
        opt = self.opt(self.model.parameters()) 

        # set up training data
        train_data = list(zip(train_X, train_y))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # train
        self.term = 'convergence reached'
        epoch = 0
        while self._converge(self.history['holdout']):
            # exit anyway if max_iter is reached
            if epoch > self.max_iter:
                self.term = 'max_iter reached'
                break
            epoch += 1

            # train batch
            self.model.train() # put in train mode
            for x_batch, y_batch in train_dataloader:
                # something about doubles and floats? 
                x_batch = x_batch.float()
                y_batch = y_batch.float()

                def closure():
                    '''some optimisers need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model.
                    https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
                    '''

                    opt.zero_grad()

                    # predict and compute loss
                    y_pred = self.model(x_batch)
                    loss = self.loss(y_pred, y_batch) 
                    self.history['train'].append(loss.item())

                    # backprop
                    loss.backward() 
                    return loss

                opt.step(closure)

            # compute loss of holdout
            self.model.eval() # put in test mode
            with torch.no_grad():
                loss_holdout = self.loss(self.model(holdout_X), holdout_y).item()
                # save smallest loss 
                if (path is not None) and (len(self.history['holdout']) > 0) and (loss_holdout < np.min(self.history['holdout'])):
                    self.save(path)
                self.history['holdout'].append(loss_holdout)
            
    def predict(self, X):
        '''Predict results. 
         
        Parameters
        ----------
        X : ndarray
            The features matrix to predict results on. 
        
        Returns
        -------
        y : ndarray
            THe predicted results. 
        '''

        self.model.eval() # put in test mode
        with torch.no_grad():
            return self.model(torch.Tensor(X).float()).detach().numpy()

    def save(self, path, model='ffnn'):
        '''Save the model.

        Parameters
        ----------
        path : str
            Path to save the model to. Needs to be a folder. All previous files with same name will be overwritten. If folder doesn't exist, it will be created.
        '''

        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.model, f'{path}/{model}')
        np.save(f'{path}/{model}_train_hist', self.history['train'])
        np.save(f'{path}/{model}_holdout_hist', self.history['holdout'])

    def load(self, path, model='ffnn'):
        '''Load a saved model.

        Parameters
        ----------
        path : str
            Path to load the model from. Needs to be a folder.
        '''

        self.model = torch.load(f'{path}/{model}')
        self.history = {
            'train':np.load(f'{path}/{model}_train_hist.npy'), 
            'holdout':np.load(f'{path}/{model}_holdout_hist.npy')}
