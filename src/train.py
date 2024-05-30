import sys
import copy
import argparse
import pickle as pkl
import os
import numpy as np
from util import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['TRANSFORMERS_CACHE'] = '/media/cache/'
from tqdm import tqdm
random = np.random
from model_wrapper import Model_Wrapper
from dataset import BLI_Dataset
import torch.optim as optim

def my_collate_fn(batch):
    inputs_txt, labels_txt, query_id = zip(*batch)
    if "mt5" in args.model_name:
        input_encs = tokenizer(inputs_txt, text_target=labels_txt, padding=True, return_tensors="pt")
    else: # XGLM, mGPT, LLaMA 
        input_encs = tokenizer(labels_txt, padding=True, return_tensors="pt")

        labels = copy.deepcopy(input_encs.input_ids)
        input_encs['labels'] = labels

    return  input_encs

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

    if (("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name) or ("Llama" in args.model_name)) and (args.batch_eval == 1):
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
                #mask 
                try: word_predict = txt_body.split()[0]
                except: word_predict = None
                    #print("prompt: ", prompts[i])
                    #print("txt_body: ", txt_body)
                    #print("seq: ", seq)
                #all
                #try: word_predict = txt_body.split(prompt_dict[source_words[i]])[1].split()[0]
                #except: word_predict = None
            elif "mbart" in args.model_name:
                #try: word_predict = seq.split()[0]
                #except: word_predict = None
                try: word_predict = seq.split(prompt_dict[source_words[i]][6:-4])[1].split()[0]
                except: word_predict = None
            elif "flan-t5" in args.model_name:
                #mask
                try: word_predict = seq.split()[0]
                except: word_predict = None
                #all
                #try: word_predict = seq.split(prompt_dict[source_words[i]])[1].split()[0]
                #except: word_predict = None
            elif "t0" in args.model_name:
                try: word_predict = seq.split()[0]
                except: word_predict = None
            else: # GPT models
                #all
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

def train(args, data_loader, model, optimizer, step_global=0):
    loss_avg = 0
    train_steps = 0
    model.train()
    for i, input_batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()
        for k,v in input_batch.items():
            input_batch[k] = v.cuda()
        output_batch = model(**input_batch)   
        loss_batch = output_batch.loss
        loss_batch.backward()
        optimizer.step()
        loss_avg += loss_batch.item()
        train_steps += 1
        step_global += 1
    loss_avg /= (train_steps + 1e-9)
    return loss_avg, step_global
 
def count_parameters(m):
    num_all = sum(p.numel() for p in m.parameters())
    num_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return (num_all, num_trainable)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main')
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
    parser.add_argument("--lr", type=float, default=0,
                    help="learning rate")   
    parser.add_argument("--seed", type=int, default=20000,
                    help="Print every k training steps")
    parser.add_argument('--finetune',  action="store_true")
    parser.add_argument('--best_template',  action="store_true")
    parser.add_argument('--max_length', default=6, type=int)
    parser.add_argument('--template_id', default=0, type=int)
    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print(args_dict)
    print("Entering Main")

    model_string = args.model_name.split("/")[-1]
    if "_" in model_string:
        model_string = model_string.split("_")[0]
    data_pkl = f"{args.data_dir}/{model_string}_{args.n_shot}_shot_{args.train_size}_training_dev_data.pkl"
    save_txt_name = f"./{model_string}_{args.n_shot}_shot_{args.train_size}_training_log.txt"

    if ("xglm" in args.model_name) or ("GPT" in args.model_name) or ("llama" in args.model_name) or ("Llama" in args.model_name):
        args.batch_eval = 1
    elif "xl" in args.model_name:
        args.batch_eval = 8

    ### Load Data

    f_name = args.data_dir + "de_vocabulary.pkl"
    with open(f_name,"rb") as f:
        voc_l1 = pkl.load(f)
    f_name = args.data_dir + "fr_vocabulary.pkl"
    with open(f_name,"rb") as f:
        voc_l2 = pkl.load(f)


    with open(data_pkl,"rb") as f:
        train_dev_data = pkl.load(f)


    s2t_dev_prompt = train_dev_data["s2t_dev_prompt"]
    t2s_dev_prompt = train_dev_data["t2s_dev_prompt"]
    s2t_dev_dict = train_dev_data["s2t_dev_dict"]
    t2s_dev_dict = train_dev_data["t2s_dev_dict"]    
    training_data = train_dev_data["train_data"] 

    del f_name, data_pkl, train_dev_data

    print("Training Data: ", len(training_data), " Data Points. One Examples: ", training_data[0])
    sys.stdout.flush()
    ### Random Seed   
    seed = args.seed 
    print("Using seed={}, pid={}".format(seed, os.getpid()))
    sys.stdout.flush()

    ### Load Model
    model_wrapper = Model_Wrapper()
    model, tokenizer = model_wrapper.load_model(
        path=args.model_name,
        max_length=args.max_length
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


    ### Torch Dataset

    train_set = BLI_Dataset(training_data)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_train,
        shuffle=True,
        num_workers=16,
        collate_fn=my_collate_fn
    )

    ### Optimizer
    optimizer = optim.AdamW([{'params': model.parameters()}], lr=args.lr, betas=(0.9,0.98),weight_decay=0.1)

    best_avg = 0
    for epoch in range(21): 
        print("Epoch: " + str(epoch) + "\n")

        ### Training
        if epoch > 0 :
            loss_epoch, step_global = train(args, data_loader=train_loader, model=model, optimizer=optimizer, step_global=step_global)
        else:
            loss_epoch = "Unknown"
            step_global = 0
        torch.cuda.empty_cache()
        ### S2T BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict = inference(model, tokenizer, s2t_dev_prompt, args, voc_l2, max_len=5, num_seq=5) 
            ans_s2t = evaluate_bli(inferred_dict, s2t_dev_dict)
        ans_line_s2t = f"de->fr: {ans_s2t}"
        print(ans_line_s2t)
        sys.stdout.flush()
        torch.cuda.empty_cache()
        ### T2S BLI Evaluation
        model.eval()
        with torch.no_grad():
            inferred_dict = inference(model, tokenizer, t2s_dev_prompt, args, voc_l1, max_len=5, num_seq=5)  
            ans_t2s = evaluate_bli(inferred_dict, t2s_dev_dict)
        ans_line_t2s = f"fr->de: {ans_t2s}"
        print(ans_line_t2s)
        sys.stdout.flush()
        torch.cuda.empty_cache()
        ans = 0.5 * (ans_s2t + ans_t2s)
        avg_line = f"average: {ans}"
        if ans > best_avg:
            print("Save Model")
            save_dir = args.save_dir + f"{model_string}_{args.n_shot}_shot_{args.train_size}/" 
            os.system(f"rm -rf {save_dir}")
            os.makedirs(save_dir)
            model_wrapper.save_model(save_dir)
            best_avg = ans
            # Save Model
            
        with open(save_txt_name, "a") as f:
            f.write("Epoch: " + str(epoch) + " Average Loss: " + str(loss_epoch) + "\n")       
            f.write(ans_line_s2t + " | " + ans_line_t2s + " | " + avg_line  + "\n")
            f.write("\n")
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
    sys.stdout.flush()
