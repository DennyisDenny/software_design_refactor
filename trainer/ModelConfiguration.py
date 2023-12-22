import os
import torch
import torch.nn as nn
import torch.nn.init as init
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from Parameter import Parameter

class ModelConfiguration:
    def __init__(self, opt):
        self.__opt = opt
        self.__model = None
        self.__converter = None
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __choose_converter(self):
        if 'CTC' in self.__opt.Prediction:
            self.__converter = CTCLabelConverter(self.__opt.character)
        else:
            self.__converter = AttnLabelConverter(self.__opt.character)
        
        self.__opt.num_class = len(self.__converter.character)
        
        self.__input_channel_config()
    
    def __input_channel_config(self):
        if self.__opt.rgb:
            self.__opt.input_channel = 3
            
    def get_opt(self):
        return self.__opt
    
    def get_converter(self):
        return self.__converter
    
    def set_model_prediction(self):
        if self.__opt.new_prediction:
            self.__model.Prediction = nn.Linear(self.__model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
    
    def model_load_state_dict(self, pretrained_dict):
        if self.__opt.FT:
            self.__model.load_state_dict(pretrained_dict, strict=False)
        else:
            self.__model.load_state_dict(pretrained_dict)
            
    def set_init_constant(self, param):
        if 'bias' in name:
            init.constant_(param, 0.0)
        elif 'weight' in name:
            init.kaiming_normal_(param)
            
    def __load_pre_trained_model(self):
        if self.__opt.saved_model != '':
            pretrained_dict = torch.load(self.__opt.saved_model)
            self.set_model_prediction()
            self.__model = torch.nn.DataParallel(self.__model).to(self.__device)
            print(f'loading pretrained model from {self.__opt.saved_model}')
            
            self.model_load_state_dict(pretrained_dict)

            if self.__opt.new_prediction:
                self.__model.module.Prediction = nn.Linear(self.__model.module.SequenceModeling_output, self.__opt.num_class)  
                for name, param in self.__model.module.Prediction.named_parameters():
                   self.set_init_constant(param)
                self.__model = self.__model.to(self.__device) 
        else:
            # weight initialization
            for name, param in self.__model.named_parameters():
                if 'localization_fc2' in name:
                    print(f'Skip {name} as it is already initialized')
                    continue
                try:
                    self.set_init_constant(param)
                except Exception as e:  # for batchnorm.
                    if 'weight' in name:
                        param.data.fill_(1)
                    continue
            self.__model = torch.nn.DataParallel(self.__model).to(self.__device)
            
    def load_model(self):
        self.__choose_converter()
        self.__model = Model(self.__opt)
        print('model input parameters', self.__opt.imgH, self.__opt.imgW, self.__opt.num_fiducial, self.__opt.input_channel, self.__opt.output_channel,
            self.__opt.hidden_size, self.__opt.num_class, self.__opt.batch_max_length, self.__opt.Transformation, self.__opt.FeatureExtraction,
            self.__opt.SequenceModeling, self.__opt.Prediction)
        
        self.__load_pre_trained_model()
        
        self.train()
        
        return self.__model
        
    def train(self):
        self.__model.train()
        print(f"Model: {self.__model}")
        parameter = Parameter(self.__model)
        parameter.count_parameters()