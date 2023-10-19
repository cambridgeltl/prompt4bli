import sys
import argparse
import pickle as pkl
from util import *
# import os
# os.environ['TRANSFORMERS_CACHE'] = '/media/cache/'


def generate_data_list(prompt, bli_dict, args):
    return_list = []
    assert len(prompt) == len(bli_dict)
    for k,v in prompt.items():
        input_ = v
        output_ws = bli_dict[k]
        for target_w in output_ws:
            if ("mt5" in args.model_name) or ("<extra_id_0>" in input_): # span-masking models
                output_ = "<extra_id_0> " + target_w.strip()
            else: # GPT-style models
                output_ = input_.strip() + " " + target_w.strip()
            return_list.append((input_,output_))
    return return_list


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
    parser.add_argument('--finetune',  action="store_true")
    parser.add_argument('--best_template',  action="store_true")
    parser.add_argument('--max_length', default=6, type=int)
    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")

    model_string = args.model_name.split("/")[-1]

    args.str2lang = {"hr":"Croatian", "en":"English","fi":"Finnish","fr":"French","de":"German","it":"Italian","ru":"Russian","tr":"Turkish","bg":"Bulgarian","ca":"Catalan","hu":"Hungarian"}
    args.str2mbart_code = {"hr":"hr_HR", "en":"en_XX","fi":"fi_FI","fr":"fr_XX","de":"de_DE","it":"it_IT","ru":"ru_RU","tr":"tr_TR"}
 
    ### Load Data

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


    train_dev_data = {}
    train_dev_data["train_data"] = []

    for l1, l2 in lang_pairs:
        f_name = args.data_dir + "{}_vocabulary.pkl".format(l1)
        with open(f_name,"rb") as f:
            voc_l1 = pkl.load(f)
        f_name = args.data_dir + "{}_vocabulary.pkl".format(l2)
        with open(f_name,"rb") as f:
            voc_l2 = pkl.load(f)

    
        f_name = "{}/{}2{}_{}.pkl".format(args.data_dir, l1, l2, args.train_size)

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

        ### Get Templates
        templates_s2t = get_best_template(args, args.str2lang[l1], args.str2lang[l2], l2)
        templates_t2s = get_best_template(args, args.str2lang[l2], args.str2lang[l1], l1)
        assert len(templates_s2t) == len(templates_t2s)
        print(len(templates_s2t)," Source to Target Templates and ",len(templates_t2s)," Target to Source Templates." )
        sys.stdout.flush()


        template_s2t, template_t2s = templates_s2t[0], templates_t2s[0]

        ### Apply Templates
        s2t_train_prompt = apply_template(template_s2t, s2t_train_dict, s2t_train_dict_in_context_prompt, args) 
        t2s_train_prompt = apply_template(template_t2s, t2s_train_dict, t2s_train_dict_in_context_prompt, args)
        s2t_test_prompt = apply_template(template_s2t, s2t_test_dict, s2t_test_dict_in_context_prompt, args)
        t2s_test_prompt = apply_template(template_t2s, t2s_test_dict, t2s_test_dict_in_context_prompt, args)     
 
        s2t_train_list = generate_data_list(s2t_train_prompt, s2t_train_dict, args)
        t2s_train_list = generate_data_list(t2s_train_prompt, t2s_train_dict, args)
        s2t_test_list = generate_data_list(s2t_test_prompt, s2t_test_dict, args)
        t2s_test_list = generate_data_list(t2s_test_prompt, t2s_test_dict, args)

        train_dev_data["train_data"] = train_dev_data["train_data"] + s2t_train_list + t2s_train_list 
        if (l1 == "de") and (l2 == "fr"):
           
            train_dev_data["s2t_dev_prompt"] = s2t_test_prompt
            train_dev_data["t2s_dev_prompt"] = t2s_test_prompt
            train_dev_data["s2t_dev_dict"] = s2t_test_dict
            train_dev_data["t2s_dev_dict"] = t2s_test_dict
            train_dev_data["dev_data"] = s2t_test_list + t2s_test_list
   
 
    save_pkl = f"{args.save_dir}/{model_string}_{args.n_shot}_shot_{args.train_size}_training_dev_data.pkl"

    for k, v in train_dev_data.items():
        print(k,len(v))
    with open(save_pkl, 'wb') as f:
        pkl.dump(train_dev_data, f)

 
         
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
    sys.stdout.flush()
