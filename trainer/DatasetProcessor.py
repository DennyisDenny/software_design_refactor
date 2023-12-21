import os
from Log import ValidDatasetLog
import torch
import torch.utils.data
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset 

class DatasetProcessor:
    def __init__(self, opt):
        self.__opt = opt
        
    def get_opt(self, opt):
        return self.__opt
    
    def load_dataset(self):
        pass
    
class TrainDatasetProcessor(DatasetProcessor):
    def __init__(self, opt):
        super().__init__(opt)
        
    def __select_data(self):
        self.__opt.select_data = self.__opt.select_data.split('-')
    
    def __set_data_ratio(self):
        self.__opt.batch_ratio = self.__opt.batch_ratio.split('-')
        
    def load_dataset(self):
        self.check_data_filtering()
        self.__select_data()
        self.__set_data_ratio()
        train_dataset = Batch_Balanced_Dataset(self.__opt)
        return train_dataset
    
class ValidDatasetProcessor(DatasetProcessor):
    def __init__(self, opt):
        super().__init__(opt)
        self.__log = ValidDatasetLog(opt.experiment_name)
        
    def load_dataset(self):
        AlignCollate_valid = AlignCollate(imgH=self.__opt.imgH, imgW=self.__opt.imgW, keep_ratio_with_pad=self.__opt.PAD, contrast_adjust=self.__opt.contrast_adjust)
        valid_dataset, valid_dataset_log = hierarchical_dataset(root=self.__opt.valid_data, opt=self.__opt)
        self.__log.write_log(valid_dataset_log)
        return AlignCollate_valid, valid_dataset
    
    def create_valid_loader(self):
        AlignCollate_valid, valid_dataset = self.load_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=min(32, self.__opt.batch_size),
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(self.__opt.workers), prefetch_factor=512,
            collate_fn=AlignCollate_valid, pin_memory=True)
        return valid_loader
        