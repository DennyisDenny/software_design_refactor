import os
import numpy as np
# import torch
import torch.optim as optim


class Optimizer:
    def __init__(self, opt, model):
        self.__filtered_parameters = []
        self.__params_num = []
        self.__optimizer = None
        self.__model = model
        self.__opt = opt

    def filter_parameter(self):
        for p in filter(lambda p: p.requires_grad, self.__model.parameters()):
            self.__filtered_parameters.append(p)
            self.__params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(self.__params_num))
        
    def set_optimizer(self):
        
        self.filter_parameter()
        
        if self.__opt.optim == 'adam':
            self.__optimizer = optim.Adam(self.__filtered_parameters)
        else:
            self.__optimizer = optim.Adadelta(self.__filtered_parameters, lr=self.__opt.lr, rho=self.__opt.rho, eps=self.__opt.eps)
        
        print(f"Optimizer: {self.__optimizer}")
        
    def get_optimizer(self):
        
        self.set_optimizer()
        
        return self.__optimizer