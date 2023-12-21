import os
import random
import time
from utils import Averager
from test import validation
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from ValidDatasetProcessor import ValidDatasetProcessor
from ModelConfiguration import ModelConfiguration
from LayerControl import LayerControl
from Optimizer import Optimizer
from Loss import Loss

class Trainer:
    def __init__(self, opt):
        self.__opt = opt
        self.__model = None
        self.__current_iter = 0
        self.__best_accuracy = -1
        self.__best_norm_ED = -1
        self.__scaler = GradScaler()
        self.__converter = None
        self.__preds = None
        self.__labels = None
        self.loss_avg = Averager()
        self.__valid_loader = None
        self.__optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _init_training_components(self):
        # load valid loader
        valid_dataset_processor = ValidDatasetProcessor(self.__opt)
        self.__valid_loader = valid_dataset_processor.create_valid_loader()
        self.__opt = valid_dataset_processor.get_opt()

        # load model
        model_config = ModelConfiguration(self.__opt)
        self.__model = model_config.load_model()
        self.__converter = model_config.get_converter()
        self.__opt = model_config.get_opt()
        
        #freeze some layer
        layer_control = LayerControl(self.__opt)
        self.__model = layer_control.freeze_layer(self.__model)
        
        # load optimizer
        optimizer_obj = Optimizer(self.__opt, self.__model)
        self.__optimizer = optimizer_obj.get_optimizer()
        
    def start_training(self, show_number = 2, amp = False):
        self._init_training_components()
        loss_obj = Loss(self.__opt)
        criterion = loss_obj.set_criterion()
        
        start_time = time.time()
        # t1 = time.time()
        validation_log = ValidationLog(self.__opt)
        while(True):           
            # train part
            # self.__optimizer.zero_grad(set_to_none = True)
            
            # if amp:
            #     with autocast():
            #         cost = self.__predict(criterion)
            #     self.__mix_precision_training(cost, optimizer)
            # else:
            #     cost = self.__predict(criterion)
            #     self.__single_precision_training(cost, optimizer)
            # self.loss_avg.add(cost)
            self._train_one_iteration(amp, criterion, show_number, start_time)
            
            if (self.__current_iter % self.__opt.valInterval == 0) and (self.__current_iter!=0):
                print('training time: ', time.time() - t1)
                t1=time.time()
                elapsed_time = time.time() - start_time  
                            
                self.__model.eval()
                
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, self.__preds, confidence_score, self.__labels,\
                    infer_time, length_of_data = validation(self.__model, criterion, self.__valid_loader, self.__converter, self.__opt, self.device)
                
                self.__model.train()
                
                # training loss and validation loss
                loss_log = f'[{self.__current_iter}/{self.__opt.num_iter}] Train loss: {self.loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                self.loss_avg.reset()
                
                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
                
                self.keep_best_accuracy_model(current_accuracy, current_norm_ED)
                best_model_log = f'{"Best_accuracy":17s}: {self.__best_accuracy:0.3f}, {"Best_norm_ED":17s}: {self.__best_norm_ED:0.4f}'
                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                
                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                            
                start = random.randint(0,len(self.__labels) - show_number )    
                for gt, pred, confidence in zip(self.__labels[start:start+show_number], self.__preds[start:start+show_number], confidence_score[start:start+show_number]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print('validation time: ', time.time()-t1)
                t1=time.time()
                
                validation_log.write_log(loss_model_log, predicted_result_log)
                
            # save model per 1e+4 iter.
            if (self.__current_iter + 1) % 1e+4 == 0:
                torch.save(
                    self.__model.state_dict(), f'./saved_models/{self.__opt.experiment_name}/iter_{self.__current_iter + 1}.pth')

            if self.__current_iter == self.__opt.num_iter:
                print('end the training')
                sys.exit()
            self.__current_iter += 1
            
    def _train_one_iteration(self, amp, criterion, show_number, start_time):
        t1 = time.time()
        # train part
        self.__optimizer.zero_grad(set_to_none = True)
        
        if amp:
            with autocast():
                cost = self.__predict(criterion)
            self.__mix_precision_training(cost, optimizer)
        else:
            cost = self.__predict(criterion)
            self.__single_precision_training(cost, optimizer)
        self.loss_avg.add(cost)
        
        if (self.__current_iter % self.__opt.valInterval == 0) and (self.__current_iter!=0):
            self._validate_and_log(t1, criterion, start_time, show_number)
        
        # save model per 1e+4 iter.
        if (self.__current_iter + 1) % 1e+4 == 0:
            torch.save(
                self.__model.state_dict(), f'./saved_models/{self.__opt.experiment_name}/iter_{self.__current_iter + 1}.pth')

        if self.__current_iter == self.__opt.num_iter:
            print('end the training')
            sys.exit()
        self.__current_iter += 1
        
    def _validate_and_log(self, t1, criterion, start_time, show_number):
        print('training time: ', time.time()-t1)
        t1=time.time()
        elapsed_time = time.time() - start_time
        
        self.__model.eval()
        
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, self.__preds, confidence_score, self.__labels,\
            infer_time, length_of_data = validation(self.__model, criterion, self.__valid_loader, self.__converter, self.__opt, self.device)
        
        self.__model.train()
        
        # training loss and validation loss
        loss_log = f'[{self.__current_iter}/{self.__opt.num_iter}] Train loss: {self.loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
        self.loss_avg.reset()
        
        current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'
        
        self.keep_best_accuracy_model(current_accuracy, current_norm_ED)
        best_model_log = f'{"Best_accuracy":17s}: {self.__best_accuracy:0.3f}, {"Best_norm_ED":17s}: {self.__best_norm_ED:0.4f}'
        loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
        print(loss_model_log)
        
        # show some predicted results
        dashed_line = '-' * 80
        head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
        predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
        
        start = random.randint(0,len(self.__labels) - show_number )    
        for gt, pred, confidence in zip(self.__labels[start:start+show_number], self.__preds[start:start+show_number], confidence_score[start:start+show_number]):
            if 'Attn' in self.__opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

            predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
        predicted_result_log += f'{dashed_line}'
        print(predicted_result_log)
        log.write(predicted_result_log + '\n')
        print('validation time: ', time.time()-t1)
        t1=time.time()
        
        validation_log.write_log(loss_model_log, predicted_result_log)
        
            
    def __CTC_in_Prediction(self, image, text, length, criterion):
        self.__preds = model(image, text).log_softmax(2)
        preds_size = torch.IntTensor([self.__preds.size(1)] * batch_size)
        self.__preds = self.__preds.permute(1, 0, 2)
        torch.backends.cudnn.enabled = False
        cost = criterion(self.__preds, text.to(self.device), preds_size.to(self.device), length.to(self.device))
        torch.backends.cudnn.enabled = True
        return cost
    
    def __CTC_no_in_Prediction(self, image, text, criterion):
        self.__preds = model(image, text[:, :-1])  # align with Attention.forward
        target = text[:, 1:]  # without [GO] Symbol
        cost = criterion(self.__preds.view(-1, self.__preds.shape[-1]), target.contiguous().view(-1))
        return cost
    
    def __predict(self, criterion):
        # load train dataset
        train_data_obj = TrainDatasetProcessor(self.__opt)
        train_dataset = train_data_obj.load_dataset()
        self.__opt = train_data_obj.get_opt()
        
        image_tensor, self.__labels = train_dataset.get_batch()
        image = image_tensors.to(self.device)
        
        text, length = self.__converter.encode(self.__labels, batch_max_length=self.__opt.batch_max_length)
        batch_size = image.size(0)
        
        if 'CTC' in self.__opt.Prediction:
            cost = self.__CTC_in_Prediction(image, text, length, criterion)
        else:
            cost = self.__CTC_no_in_Prediction(image, text, criterion)
        return cost
    
    def __mix_precision_training(self, cost, optimizer):
        scaler.scale(cost).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__opt.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
    def __single_precision_training(self, cost, optimizer):
        cost.backward()
        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), self.__opt.grad_clip) 
        optimizer.step()
    
    # keep best accuracy model (on valid dataset)
    def keep_best_accuracy_model(self, current_accuracy, current_norm_ED):
        if current_accuracy > self.__best_accuracy:
            update_best_accuracy(current_accuracy)
        if current_norm_ED > self.__best_norm_ED:
            update_best_norm_ED(current_norm_ED)
            
    def update_best_accuracy(self, current_accuracy):
        self.__best_accuracy = current_accuracy
        torch.save(self.__model.state_dict(), f'./saved_models/{self.__opt.experiment_name}/best_accuracy.pth')
        
    def update_best_norm_ED(self, current_norm_ED):
        self.__best_norm_ED = current_norm_ED
        torch.save(self.__model.state_dict(), f'./saved_models/{self.__opt.experiment_name}/best_norm_ED.pth')
