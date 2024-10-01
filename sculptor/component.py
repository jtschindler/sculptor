
import os
import pickle
import numpy as np
from sculptor.parameter import FitParameter

class FitComponent(object):

    def __init__(self, name, function, parameters=None, param_names=None, param_mapping=None):

        self.name = name
        self.function = function
        self.function_n_args = self.function.__code__.co_argcount
        self.function_args = list(
            self.function.__code__.co_varnames[1:self.function_n_args])

        self.param_mapping = param_mapping

    def __str__(self):
        return f'{self.name}: {self.function}'

    def info(self):
        print('[INFO] Component name: {}'.format(self.name))
        print('[INFO] Function name: {}'.format(self.function.__name__))
        print('[INFO] Function arguments: {}'.format(self.function_args))

    def eval(self, x, params):

        return self.function(x, *params)

    def create_params(self):

        parameters = {}
        self.param_mapping = {}
        for name in self.function_args:
            parameters.update({self.name+'_'+name:
                                   FitParameter(self.name+'_'+name, 0)})
            self.param_mapping.update({name: self.name+'_'+name})

        return parameters


    def save(self, save_dir, save_prefix=''):

        # Check if the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the object
        save_name = '{}.pkl'.format(self.name)
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(self, f)