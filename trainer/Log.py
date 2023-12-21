import os

class Log:
    def __init__(self, experiment_name):
        self.__experiment_name = experiment_name
    
    def write_log(self):
        pass
    
class ValidDatasetLog(Log):
    def __init__(self):
        super().__init__(experiment_name)

    def write_log(self, valid_dataset_log):
        log = open(f'./saved_models/{self.__experiment_name}/log_dataset.txt', 'a', encoding="utf8")
        log.write(valid_dataset_log)
        print('-' * 80)
        log.write('-' * 80 + '\n')
        log.close()
        
class ValidationLog(Log):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        
    def write_log(self, loss_model_log, predicted_result_log):
        with open(f'./save_models/{self.__experiment_name}/log_train.txt', 'a', encoding='utf8') as log:
            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            log.write(loss_model_log + '\n')
            log.write(predicted_result_log + '\n')