import time
import codecs
import numpy as np
from itertools import chain


def apply_template(template, bli_dict, in_context_prompt, args):
    # Input: template (a string)  
    #        bli_dict
    #            key: source word
    #            value: target word
    #        in_context_prompt
    #            key: source word
    #            value: a list of (source, target) example translation pairs
    # Output: return_dict
    #             key: source word
    #             value: prompt (input to transformer models)
    return_dict = {}
    for k,v in bli_dict.items():
        if args.n_shot == 0:
            # only put the source word into the template
            prompt = template.format(k)
        else:
            # put n_shot examples and the source word into the template
            examples = in_context_prompt[k][:args.n_shot]
            #to_fill = []
            #for t in examples:
            #    to_fill.append(t[0])
            #    to_fill.append(t[1])
            to_fill = list(chain(*examples))
            to_fill.append(k)
            prompt = template.format(*to_fill)
        return_dict[k] = prompt
    return return_dict

def get_templates(args, src, tgt, tgt_code):
    if ("t5" in args.model_name) or ("t0" in args.model_name):
        span_mask = True
        mask = "<extra_id_0>"
    elif "bart" in args.model_name:
        span_mask = True
        mask = "<mask>"
    else:
        span_mask = False
        mask = ""
    n_shot = args.n_shot

    if n_shot == 0:
        templates_mask = [f"The word '{{}}' in {tgt} is: {mask}.", #0 mt5-small-Best (Zero-shot)
                        f"The word {{}} in {tgt} is: {mask}.", #1        
                        f"The word '{{}}' in {tgt} is: {mask}", #2
                        f"The word {{}} in {tgt} is {mask}", #3                        
                        f"The {src} word {{}} in {tgt} is: {mask}.", #4
                        f"The {src} word {{}} in {tgt} is {mask}.", #5 
                        f"The {src} word '{{}}' in {tgt} is: {mask}.", #6 mt5-large-Best (Zero-shot) mt5-xl-Best (Zero-shot)
                        f"The {src} word '{{}}' in {tgt} is {mask}.", #7 
                        f"The {src} word {{}} in {tgt} is: {mask}", #8
                        f"The {src} word {{}} in {tgt} is {mask}", #9 
                        f"The {src} word '{{}}' in {tgt} is: {mask}", #10 mt5-xxl-Best (Zero-shot)
                        f"The {src} word '{{}}' in {tgt} is {mask}", #11 
                        f"'{{}}' in {tgt} is: {mask}.", #12 
                        f"{{}} in {tgt} is: {mask}.", #13
                        f"'{{}}' in {tgt} is: {mask}", #14 
                        f"{{}} in {tgt} is: {mask}", #15
                        f"What is the translation of the word '{{}}' into {tgt}? {mask}.", #16
                        f"What is the translation of the word {{}} into {tgt}? {mask}.", #17
                        f"What is the translation of the {src} word '{{}}' into {tgt}? {mask}.", #18  
                        f"What is the translation of the {src} word {{}} into {tgt}? {mask}.", #19
                        f"The translation of the word '{{}}' into {tgt} is {mask}.", #20
                        f"The translation of the word {{}} into {tgt} is {mask}.", #21      
                        f"The translation of the {src} word '{{}}' into {tgt} is {mask}.", #22               
                        f"How do you say '{{}}' in {tgt}? {mask}.", #23
                        f"How do you say {{}} in {tgt}? {mask}.", #24
                        f"How do you say the {src} word '{{}}' in {tgt}? {mask}.", #25
                        f"How do you say the {src} word {{}} in {tgt}? {mask}.", #26
                        f"Translate the word '{{}}' into {tgt}: {mask}.", #27 mt5-base-Best (Zero-shot)
                        f"Translate the word {{}} into {tgt}: {mask}.", #28
                        f"Translate the word {{}} into {tgt}: {mask}", #29 
                        f"Translate {{}} into {tgt}: {mask}.", #30        
                        f"Translate the {src} word {{}} into {tgt}: {mask}.", #31 flan-t5-xxl-Best (Zero-shot)
                        f"Translate the {src} word {{}} into {tgt}: {mask}", #32
                        f"Translate from {src} to {tgt}: {{}}-> {mask}.", #33
                        f"Translate from {src} to {tgt}: {{}}-> {mask}", #34
                        f"Translate from {src} to {tgt}: {{}}=> {mask}.", #35 mt0-small-Best (Zero-shot) mt0-base-Best (Zero-shot)
                        f"Translate from {src} to {tgt}: {{}}=> {mask}"] #36
        
        templates_gpt = [f"The word '{{}}' in {tgt} is:", #0,37
                        f"The word {{}} in {tgt} is:", #1,38
                        f"The word {{}} in {tgt} is", #2,39
                        f"The {src} word {{}} in {tgt} is:", #3,40 xglm-564M-Best (0-shot) xglm-1.7B-Best (0-shot) xglm-2.9B-Best (0-shot) xglm-4.5B-Best (0-shot) xglm-7.5B-Best (0-shot) llama-7B-Best (0-shot)
                        f"The {src} word {{}} in {tgt} is", #4,41
                        f"The {src} word '{{}}' in {tgt} is:", #5,42 
                        f"The {src} word '{{}}' in {tgt} is", #6,43
                        f"'{{}}' in {tgt} is:", #7,44
                        f"{{}} in {tgt} is:", #8,45
                        f"Translate the word '{{}}' into {tgt}:", #9,46 
                        f"Translate the word {{}} into {tgt}:", #10,47
                        f"Translate from {src} to {tgt}: {{}}->", #11,48
                        f"Translate from {src} to {tgt}: {{}}=>", #12,49 llama-13b-Best (0-shot)
                        f"Translate {{}} into {tgt}:", #13,50
                        f"Translate the {src} word {{}} into {tgt}:", #14,51 mgpt-BEST (0-shot)
                        f"Translate the {src} word '{{}}' into {tgt}:", #15,52 
                        f"What is the translation of the word '{{}}' into {tgt}?", #16,53
                        f"What is the translation of the word {{}} into {tgt}?", #17,54
                        f"The translation of the word '{{}}' into {tgt} is", #18,55
                        f"The translation of the word {{}} into {tgt} is", #19,56
                        f"The translation of the {src} word '{{}}' into {tgt} is", #20,57
                        f"The translation of the {src} word {{}} into {tgt} is", #21,58
                        f"How do you say '{{}}' in {tgt}?", #22,59 flan-t5-large-Best (Zero-shot)
                        f"How do you say {{}} in {tgt}?", #23,60 flan-t5-base-Best (Zero-shot)
                        f"How do you say '{{}}' in {tgt}:", #24,61 flan-t5-small-Best (Zero-shot), flan-t5-xl-Best (Zero-shot)
                        f"How do you say {{}} in {tgt}:", #25,62 
                        f"How do you say the {src} word '{{}}' in {tgt}?", #26,63 mt0-xl-Best (Zero-shot) mt0-xxl-Best (Zero-shot)
                        f"How do you say the {src} word {{}} in {tgt}?", #27,64
                        f"Q: What is the {tgt} translation of {{}} A:"] #28,65 mt0-large-Best (Zero-shot)

    else:
        templates_mask = [f"Translate from {src} to {tgt}: " + f"{{}}->{{}} "*n_shot + f"{{}}-> {mask}.", #0
                        f"Translate from {src} to {tgt}: " + f"{{}}->{{}}, "*n_shot + f"{{}}-> {mask}.", #1 
                        f"Translate from {src} to {tgt}: " + f"{{}}->{{}} "*n_shot + f"{{}}-> {mask}",   #2
                        f"Translate from {src} to {tgt}: " + f"{{}}->{{}}, "*n_shot + f"{{}}-> {mask}",  #3 
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}} "*n_shot + f"{{}}=> {mask}.",  #4
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}}, "*n_shot + f"{{}}=> {mask}.", #5 
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}} "*n_shot + f"{{}}=> {mask}",   #6
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}}, "*n_shot + f"{{}}=> {mask}",  #7 
                        f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is {mask}.", #8 mt5-small-BEST (5-shot) mt5-base-BEST (5-shot)
                        f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is {mask}.", #9 mt5-large-BEST (5-shot)
                        f"The {src} word '{{}}' in {tgt} is '{{}}'. "*n_shot + f"The {src} word '{{}}' in {tgt} is '{mask}'.", #10 mt0-large-Best (5-shot)  
                        f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is {mask}."] #11 mt5-xl-BEST (5-shot) mt5-xxl-BEST (5-shot)
        
        templates_gpt = [f"Translate from {src} to {tgt}: " + f"{{}}->{{}} "*n_shot + f"{{}}->", #0,12
                        f"Translate from {src} to {tgt}: " + f"{{}}->{{}}, "*n_shot + f"{{}}->", #1,13
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}} "*n_shot + f"{{}}=>",  #2,14
                        f"Translate from {src} to {tgt}: " + f"{{}}=>{{}}, "*n_shot + f"{{}}=>", #3,15
                        f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is", #4,16  xglm-564M-Best (5-shot)  
                        f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is:", #5,17 
                        f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is", #6,18 xglm-1.7B-Best (5-shot) xglm-2.9B-Best (5-shot) xglm-4.5B-Best (5-shot) xglm-7.5B-Best (5-shot) mGPT-Best (5-shot) flan-t5-xxl-BEST (5-shot) mt0-base-Best (5-shot)  
                        f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is:", #7,19 flan-t5-large-BEST (5-shot) flan-t5-xl-BEST (5-shot) mt0-xl-Best (5-shot)  
                        f"The word {{}} in {tgt} is {{}}, "*n_shot + f"The word {{}} in {tgt} is", #8,20   
                        f"The word {{}} in {tgt} is {{}}, "*n_shot + f"The word {{}} in {tgt} is:", #9,21 
                        f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is", #10,22 
                        f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is:", #11,23 mt0-xxl-Best (5-shot)  
                        f"The word '{{}}' in {tgt} is {{}}. "*n_shot + f"The word '{{}}' in {tgt} is", #12,24 flan-t5-small-BEST (5-shot)
                        f"The word '{{}}' in {tgt} is {{}}. "*n_shot + f"The word '{{}}' in {tgt} is:", #13,25                      
                        f"The {src} word '{{}}' in {tgt} is {{}}. "*n_shot + f"The {src} word '{{}}' in {tgt} is", #14,26 llama-7B-Best (5-shot) llama-13B-Best (5-shot)
                        f"The {src} word '{{}}' in {tgt} is {{}}. "*n_shot + f"The {src} word '{{}}' in {tgt} is:", #15,27
                        f"The word '{{}}' in {tgt} is {{}}, "*n_shot + f"The word '{{}}' in {tgt} is", #16,28 flan-t5-base-BEST (5-shot)
                        f"The word '{{}}' in {tgt} is {{}}, "*n_shot + f"The word '{{}}' in {tgt} is:", #17,29              
                        f"The {src} word '{{}}' in {tgt} is {{}}, "*n_shot + f"The {src} word '{{}}' in {tgt} is", #18,30 
                        f"The {src} word '{{}}' in {tgt} is {{}}, "*n_shot + f"The {src} word '{{}}' in {tgt} is:", #19,31
                        f"The word {{}} in {tgt} is {{}}. "*n_shot + f"How do you say {{}} in {tgt}?", #20,32       
                        f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"How do you say the {src} word {{}} in {tgt}?", #21,33  
                        f"The word '{{}}' in {tgt} is {{}}. "*n_shot + f"How do you say '{{}}' in {tgt}?", #22,34  mt0-small-Best (5-shot)     
                        f"The {src} word '{{}}' in {tgt} is {{}}. "*n_shot + f"How do you say the {src} word '{{}}' in {tgt}?"] #23,35 

    if "mt5" in args.model_name:
        templates_all = templates_mask
    elif ("flan-t5" in args.model_name) or ("bart" in args.model_name) or ("mt0" in args.model_name):
        templates_all = templates_mask + templates_gpt
    else:
        templates_all = templates_gpt

    if "bart" in args.model_name:
        prefix = args.str2mbart_code[tgt_code] + " "
        suffix = "</s>"
        for i,t in enumerate(templates_all):
            templates_all[i] = prefix+t+suffix
    #print("From ",len(templates_all)," Templates Choose ",args.template_id, ": ", templates_all[args.template_id])
    #sys.stdout.flush()
    return templates_all

def get_best_template(args, src, tgt, tgt_code):
    if ("t5" in args.model_name) or ("t0" in args.model_name):
        mask = "<extra_id_0>"
    elif "bart" in args.model_name:
        mask = "<mask>"
    else:
        mask = ""
    n_shot = args.n_shot

    if n_shot == 0:
        if "mt5-small" in args.model_name:
            best_template = [f"The word '{{}}' in {tgt} is: {mask}."]
        elif "mt5-base" in args.model_name:
            best_template = [f"Translate the word '{{}}' into {tgt}: {mask}."]
        elif "mt5-large" in args.model_name:
            best_template = [f"The {src} word '{{}}' in {tgt} is: {mask}."]
        elif "mt5-xl" in args.model_name:
            best_template = [f"The {src} word '{{}}' in {tgt} is: {mask}."]
        elif "mt5-xxl" in args.model_name:    
            best_template = [f"The {src} word '{{}}' in {tgt} is: {mask}"]
        elif "mt0-small" in args.model_name:
            best_template = [f"Translate from {src} to {tgt}: {{}}=> {mask}."]
        elif "mt0-base" in args.model_name:
            best_template = [f"Translate from {src} to {tgt}: {{}}=> {mask}."]
        elif "mt0-large" in args.model_name:
            best_template = [f"Q: What is the {tgt} translation of {{}} A:"]
        elif "mt0-xl" in args.model_name:
            best_template = [f"How do you say the {src} word '{{}}' in {tgt}?"]
        elif "mt0-xxl" in args.model_name: 
            best_template = [f"How do you say the {src} word '{{}}' in {tgt}?"]
        elif "mGPT" in args.model_name:     
            best_template = [f"Translate the {src} word {{}} into {tgt}:"]
        elif "xglm-564M" in args.model_name:                 
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "xglm-1.7B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "xglm-2.9B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "xglm-4.5B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "xglm-7.5B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "llama-7b" in args.model_name:  
            best_template = [f"The {src} word {{}} in {tgt} is:"]
        elif "llama-13b" in args.model_name:  
            best_template = [f"Translate from {src} to {tgt}: {{}}=>"]
        else:
            print("UNKNOWN MODEL, PLEASE SEARCH THROUGH OUR TEMPLATE POOL FIRST VIA get_templates()")
            exit()

    else:
        if "mt5-small" in args.model_name:
            best_template = [f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is {mask}."]
        elif "mt5-base" in args.model_name:
            best_template = [f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is {mask}."]
        elif "mt5-large" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is {mask}."]
        elif "mt5-xl" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is {mask}."]
        elif "mt5-xxl" in args.model_name:    
            best_template = [f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is {mask}."]
        elif "mt0-small" in args.model_name:
            best_template = [f"The word '{{}}' in {tgt} is {{}}. "*n_shot + f"How do you say '{{}}' in {tgt}?"]
        elif "mt0-base" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "mt0-large" in args.model_name:
            best_template = [f"The {src} word '{{}}' in {tgt} is '{{}}'. "*n_shot + f"The {src} word '{{}}' in {tgt} is '{mask}'."]
        elif "mt0-xl" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is:"]
        elif "mt0-xxl" in args.model_name: 
            best_template = [f"The {src} word {{}} in {tgt} is {{}}, "*n_shot + f"The {src} word {{}} in {tgt} is:"]
        elif "mGPT" in args.model_name:     
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "xglm-564M" in args.model_name:                 
            best_template = [f"The word {{}} in {tgt} is {{}}. "*n_shot + f"The word {{}} in {tgt} is"]
        elif "xglm-1.7B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "xglm-2.9B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "xglm-4.5B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "xglm-7.5B" in args.model_name:
            best_template = [f"The {src} word {{}} in {tgt} is {{}}. "*n_shot + f"The {src} word {{}} in {tgt} is"]
        elif "llama-7b" in args.model_name:  
            best_template = [f"The {src} word '{{}}' in {tgt} is {{}}. "*n_shot + f"The {src} word '{{}}' in {tgt} is"]
        elif "llama-13b" in args.model_name:  
            best_template = [f"The {src} word '{{}}' in {tgt} is {{}}. "*n_shot + f"The {src} word '{{}}' in {tgt} is"]
        else:
            print("UNKNOWN MODEL, PLEASE SEARCH THROUGH OUR TEMPLATE POOL FIRST VIA get_templates()")
            exit()
    return best_template


def load_embs(path, topk = None, dimension = None):
  print(topk)
  print("Loading embeddings")
  vocab_dict = {}
  embeddings = []
  with codecs.open(path, encoding = 'utf8', errors = 'replace') as f:
      line = f.readline().strip().split()
      cntr = 1
      if len(line) == 2:
        vocab_size = int(line[0])
        if not dimension: 
          dimension = int(line[1])
      else: 
        if not dimension or (dimension and len(line[1:]) == dimension):
          vocab_dict[line[0].strip()] = len(vocab_dict)
          embeddings.append(np.array(line[1:], dtype=np.float32))
        if not dimension:
          dimension = len(line) - 1
      print("Vector dimensions: " + str(dimension))
      while line: 
        line = f.readline().strip().split() 
        if (not line):
          print("Loaded " + str(cntr) + " vectors.") 
          break

        if line[0].strip() == "":
          continue
        
        cntr += 1
        if cntr % 20000 == 0:
          print(cntr)

        if len(line[1:]) == dimension:
          if (line[0].strip().lower() not in vocab_dict): 
              vocab_dict[line[0].strip().lower()] = len(vocab_dict) 
              embeddings.append(np.array(line[1:], dtype=np.float32))
        else: 
          print("Error in the embeddings file, line " + str(cntr) + 
                             ": unexpected vector length (expected " + str(dimension) + 
                             " got " + str(len(np.array(line[1:]))) + " for word '" + line[0] + "'")
        
        if (topk and cntr >= topk): 
          print("Loaded " + str(cntr) + " vectors.") 
          break           

  embeddings = np.array(embeddings, dtype=np.float32)
  print(len(vocab_dict), str(embeddings.shape))
  return vocab_dict, embeddings 
