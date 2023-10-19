import argparse
import pickle as pkl
import os
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Vocabularies')
    parser.add_argument("--lang", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--emb_dir", type=str, default="./",
                    help="emb_dir")
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")

    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    args.str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}

    ####Defining Directories
    DIR_EMB = args.emb_dir

    ####LOAD WORD EMBS
    voc_lang, embs_lang = load_embs(DIR_EMB)#,topk=10000)
    print("LANG INPUT WORD VECTOR SPACE OF SIZE:", embs_lang.shape)
    print("Static WEs Loaded")

    #Finally Save All The Dicts
    save_dict = voc_lang
    save_pkl = args.save_dir + "{}_vocabulary.pkl".format(args.lang)
    with open(save_pkl, 'wb') as f:
        pkl.dump(save_dict, f)
    
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
