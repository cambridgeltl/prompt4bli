import os
import sys

lang_pairs = [('bg', 'ca'),
 ('ca','hu'),
 ('hu','bg'),
 ('ca','bg'),
 ('hu','ca'),
 ('bg','hu')]

for (lang1, lang2) in lang_pairs:
    print(lang1, lang2)
    sys.stdout.flush()

    size_train = "1k"
    ROOT_EMB_SRC = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang1)
    ROOT_EMB_TRG = "/media/data/WESPLX/fasttext.cc.{}.300.vocab_200K.vec".format(lang2)
    ROOT_TRAIN_DICT = "/media/data/panlex-bli/lexicons/all/{}-{}/{}-{}.train.1000.cc.trans".format(lang1, lang2, lang1, lang2)
    ROOT_TEST_DICT = "/media/data/panlex-bli/lexicons/all/{}-{}/{}-{}.test.2000.cc.trans".format(lang1, lang2, lang1, lang2)
    SAVE_ROOT = "/media/data/T2TData/" # save dir

    os.system('python ./src/extract_bli_data.py --l1 {} --l2 {} --train_size {} --emb_src_dir {} --emb_tgt_dir {} --train_dict_dir {} --test_dict_dir {} --save_dir {}'.format(lang1, lang2, size_train, ROOT_EMB_SRC, ROOT_EMB_TRG, ROOT_TRAIN_DICT, ROOT_TEST_DICT, SAVE_ROOT))
