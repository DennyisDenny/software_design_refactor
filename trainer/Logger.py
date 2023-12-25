import os

class Logger:
    def __init__(self, experiment_name):
        self._experiment_name = experiment_name
    
    def write_log(self):
        pass
    
class ValidDatasetLogger(Logger):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)

    def write_log(self, valid_dataset_log):
        log = open(f'./saved_models/{self._experiment_name}/log_dataset.txt', 'a', encoding="utf8")
        log.write(valid_dataset_log)
        print('-' * 80)
        log.write('-' * 80 + '\n')
        log.close()
        
class ValidationLogger(Logger):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        print("Experiment Name:", self._experiment_name)
        
    def write_log(self, loss_model_log, predicted_result_log):
        log = open(f'./saved_models/{self._experiment_name}/log_train.txt', 'a', encoding="utf8")
        log.write(loss_model_log + '\n')
        log.write(predicted_result_log + '\n')
        log.close()