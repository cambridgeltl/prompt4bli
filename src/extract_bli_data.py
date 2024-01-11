import sys
import argparse
import pickle as pkl
import os
import numpy as np
from nltk.corpus import wordnet as wn
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
random = np.random
random.seed(1234)
torch.manual_seed(1234)


def normed_input(x):
    y = x/(np.linalg.norm(x,axis=1,keepdims=True) + 1e-9)
    return y

def my_sorted(x):    
    y = sorted(x,key=lambda t: t[1]) 
    y = [t[0] for t in y]
    return y

def return_definition_wn(word):
    
    try:
        synsets = wn.synsets(word)
        res = synsets[0].definition()
    except:
        res = None
    #sense2freq = {}
    #for s in synsets:
    #    freq = 0  
    #    for lemma in s.lemmas():
    #        freq += lemma.count()
    #        sense2freq[str(s.offset())+"-"+s.pos()] = freq
    #offset,pos = sorted(sense2freq.items(),key=lambda x:x[1])[-1][0].split("-")
    #return wn.synset_from_pos_and_offset(pos, int(offset)).definition()
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Neighbours')

    parser.add_argument("--l1", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--l2", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--norm_input", action="store_true", default=True,
                    help="True if unit-norm word embeddings")
    parser.add_argument("--train_size", type=str, default="1k",
                    help="train dict size")
    parser.add_argument("--emb_src_dir", type=str, default="./",
                    help="emb_src_dir")
    parser.add_argument("--emb_tgt_dir", type=str, default="./",
                    help="emb_tgt_dir")
    parser.add_argument("--train_dict_dir", type=str, default="./",
                    help="train_dict_dir")
    parser.add_argument("--test_dict_dir", type=str, default="./",
                    help="test_dict_dir")    
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")
    parser.add_argument("--random", action="store_true", default=False,
                    help="Randomly select in-context examples (for ablation study).")

    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")
    print(args_dict)
    args.str2lang = {"hr":"croatian", "en":"english","fi":"finnish","fr":"french","de":"german","it":"italian","ru":"russian","tr":"turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}


####Defining Directories
    DIR_EMB_SRC = args.emb_src_dir
    DIR_EMB_TRG = args.emb_tgt_dir
    DIR_TEST_DICT = args.test_dict_dir
    DIR_TRAIN_DICT = args.train_dict_dir



####LOAD WORD EMBS
    voc_l1, embs_l1 = load_embs(DIR_EMB_SRC)#,topk=10000)
    print("L1 INPUT WORD VECTOR SPACE OF SIZE:", embs_l1.shape)

    args.class_num_l1 = len(voc_l1)
    print("L1 Contain", args.class_num_l1, " Words")

    voc_l2, embs_l2 = load_embs(DIR_EMB_TRG)#,topk=10000)
    print("L2 INPUT WORD VECTOR SPACE OF SIZE:", embs_l2.shape)

    args.class_num_l2 = len(voc_l2)
    print("L2 Contain", args.class_num_l2, " Words")

    if args.norm_input:
        embs_l1 = normed_input(embs_l1)
        embs_l2 = normed_input(embs_l2)

    wvs_l1 = torch.from_numpy(embs_l1.copy())
    wvs_l2 = torch.from_numpy(embs_l2.copy())
    print("Static WEs Loaded")

#### LOAD TRAIN PARALLEL DATA
    file = open(DIR_TRAIN_DICT,'r')
    l1_dic = []
    l2_dic = []
    for line in file.readlines():
        pair = line[:-1].split('\t')
        l1_dic.append(pair[0].lower())
        l2_dic.append(pair[1].lower())
    file.close()
    l1_idx_sup = []
    l2_idx_sup = []

    train_pairs = set()

    for i in range(len(l1_dic)):
        l1_tok = voc_l1.get(l1_dic[i])
        l2_tok = voc_l2.get(l2_dic[i])
        if (l1_tok is not None) and (l2_tok is not None):
            l1_idx_sup.append(l1_tok)
            l2_idx_sup.append(l2_tok)
            train_pairs.add((l1_tok,l2_tok))

    print("Sup Set Size: ", len(l1_idx_sup), len(l2_idx_sup), len(l1_dic), len(l2_dic))
    assert len(l1_idx_sup) == len(l1_dic)
    assert len(l2_idx_sup) == len(l2_dic)
    assert len(l1_idx_sup) == len(l2_idx_sup)
    print("Sup L1 Word Frequency Ranking: ", 'min ',min(l1_idx_sup), ' max ', max(l1_idx_sup), ' average ', float(sum(l1_idx_sup))/len(l1_idx_sup))
    print("Sup L2 Word Frequency Ranking: ", 'min ',min(l2_idx_sup), ' max ', max(l2_idx_sup), ' average ', float(sum(l2_idx_sup))/len(l2_idx_sup))
    sys.stdout.flush() 


    id2w_l1 = {}
    for k,v in enumerate(l1_dic):
        id2w_l1[k] = v

    id2w_l2 = {}
    for k,v in enumerate(l2_dic):
        id2w_l2[k] = v

    s2t_train_dict = {}
    t2s_train_dict = {}
    for i,w in enumerate(l1_dic):
        if w in s2t_train_dict:
            if (l2_dic[i],l2_idx_sup[i]) not in s2t_train_dict[w]:
                s2t_train_dict[w].append((l2_dic[i],l2_idx_sup[i]))
        else:
            s2t_train_dict[w] = []
            s2t_train_dict[w].append((l2_dic[i],l2_idx_sup[i]))

    for i,w in enumerate(l2_dic):
        if w in t2s_train_dict:
            if (l1_dic[i],l1_idx_sup[i]) not in t2s_train_dict[w]:
                t2s_train_dict[w].append((l1_dic[i],l1_idx_sup[i]))            
        else:   
            t2s_train_dict[w] = []
            t2s_train_dict[w].append((l1_dic[i],l1_idx_sup[i]))

    for k,v in s2t_train_dict.items():
        s2t_train_dict[k] =  my_sorted(v)

    for k,v in t2s_train_dict.items():
        t2s_train_dict[k] = my_sorted(v)

    wvs_l1_train = torch.index_select(wvs_l1,0,torch.tensor(l1_idx_sup))
    wvs_l2_train = torch.index_select(wvs_l2,0,torch.tensor(l2_idx_sup))


#### LOAD TEST PARALLEL DATA
    file = open(DIR_TEST_DICT,'r')
    l1_dic_test = []
    l2_dic_test = []
    for line in file.readlines():
        pair = line[:-1].split('\t')
        l1_dic_test.append(pair[0].lower())
        l2_dic_test.append(pair[1].lower())
    file.close()
    l1_idx_test = []
    l2_idx_test = []

    test_pairs = set()

    for i in range(len(l1_dic_test)):
        l1_tok = voc_l1.get(l1_dic_test[i])
        l2_tok = voc_l2.get(l2_dic_test[i])
        if (l1_tok is not None) and (l2_tok is not None):
            l1_idx_test.append(l1_tok)
            l2_idx_test.append(l2_tok)
            test_pairs.add((l1_tok,l2_tok))

    print("Test Set Size: ", len(l1_idx_test), len(l2_idx_test), len(l1_dic_test), len(l2_dic_test))
    assert len(l1_idx_test) == len(l1_dic_test)
    assert len(l2_idx_test) == len(l2_dic_test)
    assert len(l1_idx_test) == len(l2_idx_test)
    print("Test L1 Word Frequency Ranking: ", 'min ',min(l1_idx_test), ' max ', max(l1_idx_test), ' average ', float(sum(l1_idx_test))/len(l1_idx_test))
    print("Test L2 Word Frequency Ranking: ", 'min ',min(l2_idx_test), ' max ', max(l2_idx_test), ' average ', float(sum(l2_idx_test))/len(l2_idx_test))
    sys.stdout.flush()


    s2t_test_dict = {}
    t2s_test_dict = {}
    for i,w in enumerate(l1_dic_test):
        if w in s2t_test_dict:
            if (l2_dic_test[i],l2_idx_test[i]) not in s2t_test_dict[w]:
                s2t_test_dict[w].append((l2_dic_test[i],l2_idx_test[i]))
        else:
            s2t_test_dict[w] = []
            s2t_test_dict[w].append((l2_dic_test[i],l2_idx_test[i]))

    for i,w in enumerate(l2_dic_test):
        if w in t2s_test_dict:
            if (l1_dic_test[i],l1_idx_test[i]) not in t2s_test_dict[w]:
                t2s_test_dict[w].append((l1_dic_test[i],l1_idx_test[i]))
        else:
            t2s_test_dict[w] = []
            t2s_test_dict[w].append((l1_dic_test[i],l1_idx_test[i]))

    for k,v in s2t_test_dict.items():
        s2t_test_dict[k] =  my_sorted(v)

    for k,v in t2s_test_dict.items():
        t2s_test_dict[k] = my_sorted(v)

    wvs_l1_test = torch.index_select(wvs_l1,0,torch.tensor(l1_idx_test))
    wvs_l2_test = torch.index_select(wvs_l2,0,torch.tensor(l2_idx_test))


    s2t_train_dict_in_context_prompt = {}
    t2s_train_dict_in_context_prompt = {}

    s2t_test_dict_in_context_prompt = {}
    t2s_test_dict_in_context_prompt = {}
   
    len_ = 15
    if args.random:
        idxs_s2t_train = torch.topk(torch.randn(wvs_l1_train.size(0), wvs_l1_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()
    else:
        idxs_s2t_train = torch.topk(wvs_l1_train @ wvs_l1_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()
    for i,w in enumerate(l1_dic):
        if w not in s2t_train_dict_in_context_prompt:
            s2t_train_dict_in_context_prompt[w] = []
            j = 0
            left_seen = set()
            left_seen.add(w)
            while len(s2t_train_dict_in_context_prompt[w])<len_:
                left = id2w_l1[idxs_s2t_train[i][j]]
                right = s2t_train_dict[left][0]
                if left not in left_seen:
                    s2t_train_dict_in_context_prompt[w].append((left,right))
                    left_seen.add(left)
                j+=1


    if args.random:                 
        idxs_t2s_train = torch.topk(torch.randn(wvs_l2_train.size(0), wvs_l2_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()
    else:
        idxs_t2s_train = torch.topk(wvs_l2_train @ wvs_l2_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()   
    for i,w in enumerate(l2_dic):
        if w not in t2s_train_dict_in_context_prompt:
            t2s_train_dict_in_context_prompt[w] = []
            j = 0
            left_seen = set()
            left_seen.add(w)
            while len(t2s_train_dict_in_context_prompt[w])<len_:
                left = id2w_l2[idxs_t2s_train[i][j]]
                right = t2s_train_dict[left][0]
                if left not in left_seen:
                    t2s_train_dict_in_context_prompt[w].append((left,right))
                    left_seen.add(left)
                j+=1

    if args.random: 
        idxs_s2t_test = torch.topk(torch.randn(wvs_l1_test.size(0), wvs_l1_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()        
    else:
        idxs_s2t_test = torch.topk(wvs_l1_test @ wvs_l1_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()
    for i,w in enumerate(l1_dic_test):
        if w not in s2t_test_dict_in_context_prompt:
            s2t_test_dict_in_context_prompt[w] = []
            j = 0
            left_seen = set()
            left_seen.add(w)
            while len(s2t_test_dict_in_context_prompt[w])<len_:
                left = id2w_l1[idxs_s2t_test[i][j]]
                right = s2t_train_dict[left][0]
                if left not in left_seen:
                    s2t_test_dict_in_context_prompt[w].append((left,right))
                    left_seen.add(left)
                j+=1

    if args.random: 
        idxs_t2s_test = torch.topk(torch.randn(wvs_l2_test.size(0), wvs_l2_train.size(0)), len_*6, dim=1, largest=True, sorted=True).indices.tolist()
    else:
        idxs_t2s_test = torch.topk(wvs_l2_test @ wvs_l2_train.T, len_*6, dim=1, largest=True, sorted=True).indices.tolist()   
    for i,w in enumerate(l2_dic_test):
        if w not in t2s_test_dict_in_context_prompt:
            t2s_test_dict_in_context_prompt[w] = []
            j = 0
            left_seen = set()
            left_seen.add(w)
            while len(t2s_test_dict_in_context_prompt[w])<len_:
                left = id2w_l2[idxs_t2s_test[i][j]]
                right = t2s_train_dict[left][0]
                if left not in left_seen:
                    t2s_test_dict_in_context_prompt[w].append((left,right))
                    left_seen.add(left)
                j+=1

    s2t_train_dict_word_def = None
    t2s_train_dict_word_def = None
    s2t_test_dict_word_def = None
    t2s_test_dict_word_def = None


    if (args.l1 == "en") or (args.l2 == "en"):
        s2t_train_dict_word_def = {}
        s2t_test_dict_word_def = {}

        for i,w in enumerate(l1_dic):
            if w not in s2t_train_dict_word_def:
            	s2t_train_dict_word_def[w] = return_definition_wn(w)

        for i,w in enumerate(l1_dic_test):
            if w not in s2t_test_dict_word_def:
                s2t_test_dict_word_def[w] = return_definition_wn(w)


        t2s_train_dict_word_def = {}
        t2s_test_dict_word_def = {}
        for i,w in enumerate(l2_dic):
            if w not in t2s_train_dict_word_def:
                t2s_train_dict_word_def[w] = return_definition_wn(t2s_train_dict[w][0])

        for i,w in enumerate(l2_dic_test):
            if w not in t2s_test_dict_word_def:
            	t2s_test_dict_word_def[w] = return_definition_wn(t2s_test_dict[w][0])


    #Finally Save All The Dicts

    save_dict = {}
    save_dict["s2t_train_dict"] = s2t_train_dict
    save_dict["t2s_train_dict"] = t2s_train_dict
    save_dict["s2t_test_dict"] = s2t_test_dict
    save_dict["t2s_test_dict"] = t2s_test_dict
    save_dict["s2t_train_dict_in_context_prompt"] = s2t_train_dict_in_context_prompt
    save_dict["t2s_train_dict_in_context_prompt"] = t2s_train_dict_in_context_prompt
    save_dict["s2t_test_dict_in_context_prompt"] = s2t_test_dict_in_context_prompt
    save_dict["t2s_test_dict_in_context_prompt"] = t2s_test_dict_in_context_prompt
    save_dict["s2t_train_dict_word_def"] = s2t_train_dict_word_def
    save_dict["t2s_train_dict_word_def"] = t2s_train_dict_word_def
    save_dict["s2t_test_dict_word_def"] = s2t_test_dict_word_def
    save_dict["t2s_test_dict_word_def"] = t2s_test_dict_word_def


    save_pkl = args.save_dir + "{}2{}_{}.pkl".format(args.l1,args.l2,args.train_size)

    with open(save_pkl, 'wb') as f:
        pkl.dump(save_dict, f)    
    
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
