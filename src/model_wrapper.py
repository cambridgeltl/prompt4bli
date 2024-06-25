import torch
from transformers import (
    AutoTokenizer, AutoModel, 
    T5Tokenizer, MT5ForConditionalGeneration, 
    MBartForConditionalGeneration,
    T5ForConditionalGeneration, AutoModelForSeq2SeqLM,
    XGLMTokenizer, XGLMForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    AutoModelForCausalLM
)
# import os
# os.environ['TRANSFORMERS_CACHE'] = '/media/cache/'

class Model_Wrapper(object):
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def save_model(self, output_dir, context=False):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


    def load_model(self, path, max_length):

        if "mt5" in path:
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            self.model = MT5ForConditionalGeneration.from_pretrained(path)
        elif "mbart" in path:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = MBartForConditionalGeneration.from_pretrained(path)
        elif "byt5" in path:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        elif "flan-t5" in path:
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            self.model = T5ForConditionalGeneration.from_pretrained(path)
        elif "mt0" in path:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        elif "xglm" in path:
            self.tokenizer = XGLMTokenizer.from_pretrained(path, padding_side='left')
            self.model = XGLMForCausalLM.from_pretrained(path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id #1
        elif "mGPT" in path:
            self.tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side='left')
            self.model = GPT2LMHeadModel.from_pretrained(path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id #1
        elif 'Llama-3' in path:
            self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id #1        
        elif "llama" in path or "Llama-2" in path:
            self.tokenizer = LlamaTokenizer.from_pretrained(path, padding_side='left')
            self.model = LlamaForCausalLM.from_pretrained(path)
            # self.model = LlamaForCausalLM.from_pretrained(path,torch_dtype=torch.float16) # fp16
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id #1
        else:
            print("WARNING: UNKNOWN MODEL")
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)           

        self.model = self.model.cuda()
        return self.model, self.tokenizer
    
