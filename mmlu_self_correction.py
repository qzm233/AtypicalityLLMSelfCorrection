import numpy as np
import torch
import torch.nn.functional as F
import re
from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import LlamaForCausalLM, LlamaTokenizer, PreTrainedTokenizerFast,AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
import pickle

task_list = ['machine_learning']
             # ,'business_ethics','high_school_computer_science','clinical_knowledge']

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

def get_response(model, tokenizer, prompts, text, correction = False):
    generated_texts = []
    
    if correction:
        text = text + prompts['critique']
        inputs = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 500,
                eos_token_id=terminators,
                # num_return_sequences=sample_num,
                # max_length=400,
                # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
                # do_sample = True,
                # temperature = 0.6,
                # top_p = 0.9,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        del outputs, inputs
        torch.cuda.empty_cache()
        
        text = generated_text + prompts['improve']
        inputs = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 500,
                eos_token_id=terminators,
                # num_return_sequences=sample_num,
                # max_length=400,
                # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
                # do_sample = True,
                # temperature = 0.6,
                # top_p = 0.9,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        del outputs, inputs
        torch.cuda.empty_cache()
    else:
        text = prompts['head'] + text + "[/INST]\n"
        inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 500,
                eos_token_id=terminators,
                # num_return_sequences=sample_num,
                # max_length=400,
                # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
                # do_sample = True,
                # temperature = 0.6,
                # top_p = 0.9,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        del outputs, inputs
        torch.cuda.empty_cache()
    return generated_texts

def extract_answer(text):
    # pattern = r"\[INST\].*?\[/INST\](.*?)(?=\[INST\]|$)"
    # matches = re.findall(pattern, text, re.DOTALL)
    # # 提取每个匹配中的数字
    # numbers = []
    # for match in matches:
    #     # print(match)
    #     number_pattern = r"####.*?(\d+)"
    #     if len(re.findall(number_pattern, match)) >=1:
    #         number_matches = re.findall(number_pattern, match)
    #         print(number_matches)
    #         numbers.extend(number_matches)
    #     else:
    #         numbers.extend([-10000])
    # print("Extracted numbers: ", numbers)

    # # print(answer)
    # answer_pattern = r"####.*?(\d+)"
    # answer_match = re.findall(answer_pattern, answer)
    # print("Answers: ", answer_match)
    return 1
    
def calculate_atypicality(model, tokenizer, text):
    # text = prompts['head'] + text + "\n" + prompts['answer']
    inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    logit_output = model(input_ids)
    logits = logit_output.logits.detach().cpu()
    labels = input_ids.cpu()
    assert logits.shape[-2] == labels.shape[-1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels[shift_labels == tokenizer.pad_token_id] = -100
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
    ll_per_sample = loss.view(shift_logits.shape[0], shift_logits.shape[1])
    nonpad_per_row = (shift_labels != -100).sum(dim=1) # number of non-padding tokens per sequence
    ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row # mean log-likelihood per sequence
    
    ll_per_sample[(shift_labels == -100)] = 0
    ll_total = ll_per_sample.sum(dim=1)
    torch.cuda.empty_cache()
    return ll_mean.cpu().numpy(), ll_total.cpu().numpy(), ll_per_sample.cpu().numpy()

def prepare_data(task_data):
    '''
    prompts = {}
    prompts['head'] = "[INST]«SYS»You are doing multichoice question answering. Provide the final answer after '####'«/SYS»\nAnswer the following question. Please give a short explanation.\nQuestion: "
    prompts['answer'] = "Answer: "
    prompts['critique'] = '\n[INST]Review your answer and find problems with your answer[/INST]\n'
    prompts['improve'] = "\n[INST]Based on the problems you found, show me the final answer.[/INST]\n"
    '''

    questions = []
    answers = []
    choices = []
    for split in task_data.keys():
        if split == 'train':
            start = 1  
            # In at least one subject, we found first train question to be unrelated to subject,
            # that's why we remove question 1. 数据集问题？
        else:
            start = 0
        for i in range(start, len(task_data[split]['input'])):
            choice_dict = {}
            
            prompt_q = task_data[split]['input'][i]
            for letter in ['A', 'B', 'C', 'D']:
                prompt_q += '\n' + letter + '. ' + task_data[split][letter][i] + ''
                choice_dict[letter] = task_data[split][letter][i]
                
            choices.append(choice_dict)
            questions.append(prompt_q)
            answers.append(task_data[split]['target'][i])
    return questions, answers, choices 

save_dir = '/root/autodl-fs/Llama-2-7b-chat-hf'
# save_dir = '/root/autodl-fs/zephyr-7b-sft-full'
# tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir, low_cpu_mem_usage=True, return_token_type_ids=False)
tokenizer = AutoTokenizer.from_pretrained(save_dir, low_cpu_mem_usage=True,return_token_type_ids=False)
# model = PreTrainedTokenizerFast.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.unk_token
model.to(device)
print("good")

prompts = {}
prompts['head'] = "[INST]«SYS»You are answering multichoice questions.«/SYS»\nAnswer the following question. Please give a short explanation and provide the final answer after '####' for each questions. \n\nQuestion: "
prompts['answer'] = "Answer: "
prompts['critique'] = '\n[INST]Review your answer and find problems with your answer[/INST]\n'
prompts['improve'] = "\n[INST]Based on the problems you found, show me the final answer.[/INST]\n"

num_correct = 1

i = 0

all_records = {}

for subject in task_list:
    record = {}
    logs = []
    targets = []
    aty_means = []
    aty_totals = []
    aty_logs = []
    task_data = load_from_disk(f"/root/autodl-tmp/ConformalLLM/data/{subject}")
    questions, answers, choices = prepare_data(task_data)
    # data_train = data['train']
    # data_test = data['test']
    for i, (question, target) in tqdm(enumerate(zip(questions,answers))):
        # if i >=5:
        #     break
        i += 1
        log = []
        # question = data['question']
        aty_mean, aty_total, aty_log = calculate_atypicality(model, tokenizer, question)
        print(aty_mean)
        initial_response = get_response(model, tokenizer, prompts, question)[0]
        log.append(initial_response)
        # print(initial_response)
        initial_answer = extract_answer(initial_response)
        # answer.append(initial_answer)
        correction_improve = initial_response
        for j in range(num_correct):
            response = get_response(model, tokenizer, prompts, correction_improve, correction = True)
            correction_critique = response[0]
            correction_improve = response[1]
            log.append(correction_critique)
            log.append(correction_improve)
            print("*****************************\n",correction_improve)
            correction_answer = extract_answer(correction_improve)
            # answer.append(correction_answer)
        aty_means.append(aty_mean)
        aty_totals.append(aty_total)
        aty_logs.append(aty_log)
        targets.append(target)
        logs.append(log)
    
    record['aty_means'] = aty_means
    record['aty_total'] = aty_totals
    record['aty_log'] = aty_logs
    record['targets'] = targets
    record['logs'] = logs
    all_records[subject] = record

    filename = 'records/mmlu_record.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(all_records, file)
    
    print(f"数据已成功写入文件 {filename}")
