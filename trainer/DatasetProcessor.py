import os
from Logger import ValidDatasetLogger
import torch
import torch.utils.data
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset 

class DatasetProcessor:
    def __init__(self, opt):
        self._opt = opt
        
    def get_opt(self):
        return self._opt
    
    def load_dataset(self):
        pass
    
class TrainDatasetProcessor(DatasetProcessor):
    def __init__(self, opt):
        super().__init__(opt)
    
    def check_data_filtering(self):
        if not self._opt.data_filtering_off:
            print('Filtering the images containing characters which are not in opt.character')
            print('Filtering the images whose label is longer than opt.batch_max_length')
        
    def __select_data(self):
        self._opt.select_data = self._opt.select_data.split('-')
    
    def __set_data_ratio(self):
        self._opt.batch_ratio = self._opt.batch_ratio.split('-')
        
    def load_dataset(self):
        self.check_data_filtering()
        self.__select_data()
        self.__set_data_ratio()
        train_dataset = Batch_Balanced_Dataset(self._opt)
        return train_dataset
    
class ValidDatasetProcessor(DatasetProcessor):
    def __init__(self, opt):
        super().__init__(opt)
        self._log = ValidDatasetLogger(self._opt.experiment_name)
        
    def load_dataset(self):
        
        AlignCollate_valid = AlignCollate(imgH=self._opt.imgH, imgW=self._opt.imgW, keep_ratio_with_pad=self._opt.PAD, contrast_adjust=self._opt.contrast_adjust)
        valid_dataset, valid_dataset_log = hierarchical_dataset(root=self._opt.valid_data, opt=self._opt)
        self._log.write_log(valid_dataset_log)
        return AlignCollate_valid, valid_dataset
    
    def create_valid_loader(self):
        AlignCollate_valid, valid_dataset = self.load_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=min(32, self._opt.batch_size),
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(self._opt.workers), prefetch_factor=512,
            collate_fn=AlignCollate_valid, pin_memory=True)
        return valid_loader
        