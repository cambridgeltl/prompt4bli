import os
import sys


n_shot = 5

for Model in ["huggyllama/llama-13b","huggyllama/llama-7b","google/mt5-large","google/mt5-base", "facebook/xglm-1.7B", "facebook/xglm-564M"]:
    for size_train in ["5k","1k"]:

        # --best_template
        DATA_ROOT = "/media/data/T2TData/"
        SAVE_ROOT = "/media/data/T2TDataTrainDev/" # save aligend WEs 
        os.system('python ./src/prepare_training_data.py --model_name {} --train_size {} --n_shot {} --data_dir {} --save_dir {} --best_template'.format(Model, size_train, n_shot, DATA_ROOT, SAVE_ROOT))