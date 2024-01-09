import sys
import argparse
import pickle as pkl
import os
import numpy as np
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TRANSFORMERS_CACHE'] = '/media/cache/'
from tqdm import tqdm
from model_wrapper import Model_Wrapper

def inference(model, tokenizer, prompt_dict, args, target_voc, max_len=5, num_seq=5):
    # Input: prompt_dict
    #            key: source word
    #            value: prompt (input to transformer models)
    # Output: return_dict
    #             key: source word
    #             value: predicted target word
    return_dict = {}
    source_words, prompts = zip(*prompt_dict.items())
    predictions = []
    if "mbart" in args.model_name:
        add_spec_toks = False
    else:
        add_spec_toks = True 
    if ("<extra_id_0>" in prompts[0]) or ("<mask>" in prompts[0]):
        span_mask = True
    else:
        span_mask = False   
    if span_mask or ("t5" in args.model_name) or ("t0" in args.model_name):
        max_len = 5

    if (("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name)) and (args.batch_eval == 1):
        dynamic_max_len = True
    else:
        dynamic_max_len = False

    for i in tqdm(np.arange(0, len(source_words), args.batch_eval)):

        # i ~ i+args.batch_eval
        TXT = prompts[i:i+args.batch_eval] 
        input_encs = tokenizer.batch_encode_plus(TXT,padding=True,add_special_tokens=add_spec_toks,return_tensors="pt")   
        if dynamic_max_len: 
            max_len = 5 + input_encs.input_ids.size(1) 
        input_encs_cuda = {k:v.cuda() for k,v in input_encs.items()}     
        outputs = model.generate(**input_encs_cuda,num_beams=num_seq, num_return_sequences=num_seq,max_length=max_len)
        output_sents = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        output_grouped = [output_sents[i*num_seq:(i+1)*num_seq] for i in range((len(output_sents)//num_seq))]  
        predictions.extend(output_grouped)

    # Extract Answers From Predicted Sequences
    assert len(source_words) == len(predictions)
    for i in tqdm(range(len(source_words))):
        candidates = predictions[i]   
        for seq in candidates:
            if "mt5" in args.model_name:
                txt_body = seq.split("<extra_id_0>")[1].split("<extra_id_1>")[0].strip()
                try: word_predict = txt_body.split()[0]
                except: word_predict = None
            elif "mbart" in args.model_name:
                try: word_predict = seq.split(prompt_dict[source_words[i]][6:-4])[1].split()[0]
                except: word_predict = None
            elif "flan-t5" in args.model_name:
                try: word_predict = seq.split()[0]
                except: word_predict = None
            elif "t0" in args.model_name:
                try: word_predict = seq.split()[0]
                except: word_predict = None
            else: # GPT models
                try: word_predict = seq.split(prompt_dict[source_words[i]])[1].split()[0]
                except: word_predict = None
            # remove the dots "..." in "word..."
            if word_predict is not None:
                word_predict = word_predict.split(".")[0]             
                if len(word_predict) == 0:
                    word_predict = None
            if word_predict is not None:
                word_predict = word_predict.lower()    
                if word_predict in target_voc:
                    return_dict[source_words[i]] = word_predict
                    break
                elif (len(word_predict) > 1) and (word_predict[-1] == ".") and (word_predict[:-1] in target_voc):
                    return_dict[source_words[i]] = word_predict[:-1]
                    break
    return return_dict


def evaluate_bli(inferred_dict, reference_dict):
    assert len(inferred_dict) <= len(reference_dict)
    num_source_words = len(reference_dict)
    num_correct_answers = 0
    for k,v in inferred_dict.items():
        if v in reference_dict[k]:
            num_correct_answers += 1
    return num_correct_answers/float(num_source_words)
 
def count_parameters(m):
    num_all = sum(p.numel() for p in m.parameters())
    num_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return (num_all, num_trainable)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--l1", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--l2", type=str, default=" ",
                    help="Language (string)")
    parser.add_argument("--train_size", type=str, default="1k",
                    help="train dict size")
    parser.add_argument("--data_dir", type=str, default="./",
                    help="data_dir")    
    parser.add_argument("--save_dir", type=str, default="./",
                    help="save_dir")
    parser.add_argument("--model_name", type=str, default="./",
                    help="mt5,mbart,byt5,flant5,xglm,mgpt...")
    parser.add_argument("--print_every", type=int, default=25,
                    help="Print every k training steps")
    parser.add_argument("--eval_every", type=int, default=50,
                    help="evaluate model every k training steps")
    parser.add_argument("--batch_eval", type=int, default=100,
                    help="batch size evaluation")
    parser.add_argument("--batch_train", type=int, default=100,
                    help="batch size train")
    parser.add_argument("--n_shot", type=int, default=0,
                    help="0,1,2,3,4,5,6,7,8,9,10")    
    parser.add_argument('--finetune',  action="store_true")
    parser.add_argument('--best_template',  action="store_true")
    parser.add_argument('--max_length', default=6, type=int)
    parser.add_argument('--template_id', default=0, type=int)
    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    model_string = args.model_name.split("/")[-1]

    save_txt_name = f"./{model_string}_{args.n_shot}_shot_{args.train_size}_{args.finetune}.txt"

    del model_string

    if ("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name):
        args.batch_eval = 1
    elif "xl" in args.model_name:
        args.batch_eval = 8

    args.str2lang = {"hr":"Croatian", "en":"English","fi":"Finnish","fr":"French","de":"German","it":"Italian","ru":"Russian","tr":"Turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}
    args.str2mbart_code = {"hr":"hr_HR", "en":"en_XX","fi":"fi_FI","fr":"fr_XX","de":"de_DE","it":"it_IT","ru":"ru_RU","tr":"tr_TR"}
 
    ### Load Data

    f_name = args.data_dir + "{}_vocabulary.pkl".format(args.l1)
    with open(f_name,"rb") as f:
        voc_l1 = pkl.load(f)
    f_name = args.data_dir + "{}_vocabulary.pkl".format(args.l2)
    with open(f_name,"rb") as f:
        voc_l2 = pkl.load(f)

    
    f_name = "{}/{}2{}_{}.pkl".format(args.data_dir, args.l1, args.l2, args.train_size)

    with open(f_name,"rb") as f:
        data = pkl.load(f)
    s2t_train_dict = data["s2t_train_dict"]
    t2s_train_dict = data["t2s_train_dict"]
    s2t_test_dict = data["s2t_test_dict"]
    t2s_test_dict = data["t2s_test_dict"]
    s2t_train_dict_in_context_prompt = data["s2t_train_dict_in_context_prompt"]
    t2s_train_dict_in_context_prompt = data["t2s_train_dict_in_context_prompt"]
    s2t_test_dict_in_context_prompt = data["s2t_test_dict_in_context_prompt"]
    t2s_test_dict_in_context_prompt = data["t2s_test_dict_in_context_prompt"]
    del data, f_name

    ### Load Model
    model_wrapper = Model_Wrapper()
    model, tokenizer = model_wrapper.load_model(
        path=args.model_name,
        max_length=args.max_length
    )

    ## Count Model Parameters
    num_all, num_trainable = count_parameters(model)
    model_line = f"Model: {args.model_name} Parameter Counts (All): {num_all} Parameter Counts (Trainable): {num_trainable} {args.n_shot} Shot {args.train_size} Seed Pairs Fine-Tune: {args.finetune} "
    print(model_line)
    sys.stdout.flush()
    with open(save_txt_name, "a") as f:
        f.write(model_line + "\n")
    del num_all, num_trainable, model_line
    
    ### Get Templates
    if args.best_template: 
        templates_s2t = get_best_template(args, args.str2lang[args.l1], args.str2lang[args.l2], args.l2)
        templates_t2s = get_best_template(args, args.str2lang[args.l2], args.str2lang[args.l1], args.l1)
    else:
        templates_s2t = get_templates(args, args.str2lang[args.l1], args.str2lang[args.l2], args.l2)
        templates_t2s = get_templates(args, args.str2lang[args.l2], args.str2lang[args.l1], args.l1)
    assert len(templates_s2t) == len(templates_t2s)
    print(len(templates_s2t)," Source to Target Templates and ",len(templates_t2s)," Target to Source Templates." )
    sys.stdout.flush()

    if "mbart" in args.model_name:
        add_spec_toks = False
    else:
        add_spec_toks = True

    for template_ids in range(len(templates_s2t)): 
        template_s2t, template_t2s = templates_s2t[template_ids], templates_t2s[template_ids]
        template_line = f"Template {template_ids} {template_s2t}"
        print(template_line)
        max_len_s2t = tokenizer([template_s2t], add_special_tokens=add_spec_toks, return_tensors="pt").input_ids.size(1) + 20
        max_len_t2s = tokenizer([template_t2s], add_special_tokens=add_spec_toks, return_tensors="pt").input_ids.size(1) + 20

        ### Apply Templates
        s2t_train_prompt = apply_template(template_s2t, s2t_train_dict, s2t_train_dict_in_context_prompt, args) 
        t2s_train_prompt = apply_template(template_t2s, t2s_train_dict, t2s_train_dict_in_context_prompt, args)
        s2t_test_prompt = apply_template(template_s2t, s2t_test_dict, s2t_test_dict_in_context_prompt, args)
        t2s_test_prompt = apply_template(template_t2s, t2s_test_dict, t2s_test_dict_in_context_prompt, args)
    
        ### S2T BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict = inference(model, tokenizer, s2t_test_prompt, args, voc_l2, max_len=max_len_s2t, num_seq=5) 
            ans_s2t = evaluate_bli(inferred_dict, s2t_test_dict)
        ans_line_s2t = f"{args.l1}->{args.l2}: {ans_s2t}"
        print(ans_line_s2t)
        sys.stdout.flush()
        ### T2S BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict = inference(model, tokenizer, t2s_test_prompt, args, voc_l1, max_len=max_len_t2s, num_seq=5)  
            ans_t2s = evaluate_bli(inferred_dict, t2s_test_dict)
        ans_line_t2s = f"{args.l2}->{args.l1}: {ans_t2s}"
        print(ans_line_t2s)
        sys.stdout.flush()
        ans = 0.5 * (ans_s2t + ans_t2s)
        avg_line = f"average: {ans}"
        with open(save_txt_name, "a") as f:
            f.write(template_line + "\n")
            f.write(ans_line_s2t + " | " + ans_line_t2s + " | " + avg_line  + "\n")
            f.write("\n")
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
    sys.stdout.flush()
