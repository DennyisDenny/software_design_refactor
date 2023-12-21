import os
import torch
from utils import Averager

class Loss:
    def __init__(self, opt):
        self.__opt = opt
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_criterion(self):
        if 'CTC' in self.__opt.Prediction:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(self.__device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(self.__device)  # ignore [GO] token = ignore index 0
        return criterion
    
    def get_loss_avg(self):
        loss_avg = Averager()
        return loss_avg