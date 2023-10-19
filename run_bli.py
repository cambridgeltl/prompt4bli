import os
import sys

# XLING
lang_pairs = [('de', 'fi'),
 ('de', 'fr'),
 ('de', 'hr'),
 ('de', 'it'),
 ('de', 'ru'),
 ('de', 'tr'),
 ('en', 'de'),
 ('en', 'fi'),
 ('en', 'fr'),
 ('en', 'hr'),
 ('en', 'it'),
 ('en', 'ru'),
 ('en', 'tr'),
 ('fi', 'fr'),
 ('fi', 'hr'),
 ('fi', 'it'),
 ('fi', 'ru'),
 ('hr', 'fr'),
 ('hr', 'it'),
 ('hr', 'ru'),
 ('it', 'fr'),
 ('ru', 'fr'),
 ('ru', 'it'),
 ('tr', 'fi'),
 ('tr', 'fr'),
 ('tr', 'hr'),
 ('tr', 'it'),
 ('tr', 'ru')]


# PanLex-BLI
# lang_pairs = [('bg', 'ca'),
# ('ca','hu'),
# ('hu','bg'),
# ('ca','bg'),
# ('hu','ca'),
# ('bg','hu')]

Model = "huggyllama/llama-13b"  # a list of model ids for off-the-shelf LLMs available in our paper's Table 7; for models after BLI-oriented fine-tuning, use the local dir of the model. 
size_train = "5k" # or "1k"
n_shot = 5

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()
    # --best_template
    DATA_ROOT = "/media/data/T2TData/"
    SAVE_ROOT = "/media/data/T2TModel/" # save aligend WEs 
    os.system('python ./src/main.py --l1 {} --l2 {} --model_name {} --train_size {} --n_shot {} --data_dir {} --save_dir {} --best_template'.format(lang1, lang2, Model, size_train, n_shot, DATA_ROOT, SAVE_ROOT))
