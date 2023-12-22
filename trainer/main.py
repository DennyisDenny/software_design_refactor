import os
import torch.backends.cudnn as cudnn
import yaml
from Trainer import Trainer
from utils import AttrDict
import pandas as pd
import time as time

cudnn.benchmark = True
cudnn.deterministic = False

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt



if __name__ == '__main__':
    opt = get_config("./config_files/en_filtered_config.yaml")
    start_time = time.time()
    # print(start_time)    
    pid = os.getpid()
    print(f"Current process PID: {pid}")
    trainer = Trainer(opt)
    # train(opt, amp=False)
    trainer.start_training()
    print(f"Execution time: {time.time() - start_time} second")