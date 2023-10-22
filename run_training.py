import os

n_shot = 5
Model = "facebook/xglm-1.7B"  # Only 4 smaller-scale LLMs are adopted for fine-tuning in our preliminary experiments: mT5-base, mT5-large, XGLM-564M, and XGLM-1.7B. 
size_train = "5k"  # or "1K" 
batch_size_train = 16  # 16 for XGLM-1.7B, 32 for XGLM-564M and mT5-base/large
learning_rate = 5e-9  # 2e-6 for mT5-base, 1e-6 for mT5-large, 5e-8 for XGLM-564M, 5e-9 for XGLM-1.7B in the 5K setup, 2e-8 for XGLM-1.7B in the 1K setup 
random_seed = 0

SAVE_ROOT = "/media/data/T2TModel/" # save LLM
DATA_ROOT = "/media/data/T2TDataTrainDev/" 
os.system('python ./src/train.py --model_name {} --train_size {} --batch_train {} --n_shot {} --lr {} --seed {} --data_dir {} --save_dir {}'.format(Model, size_train, batch_size_train, n_shot, learning_rate, random_seed, DATA_ROOT, SAVE_ROOT))

