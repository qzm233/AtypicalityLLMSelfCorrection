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
from openai import OpenAI

client = OpenAI(api_key="sk-fb368ecf4caf4f7686a75b97f4f2c7ed", base_url="https://api.deepseek.com")

task_list = ['formal_logic']
             # ,'business_ethics','high_school_computer_science','clinical_knowledge']

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

def get_response(model, tokenizer, text):
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

    # 提取新生成的回答
    input_text_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    new_response = generated_text[input_text_length:]
    
    del outputs, inputs
    torch.cuda.empty_cache()
    return new_response

def review(client, text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"There is an answer.\n[{text}]\nPlease review the answer and the explanation. Point out the logical or commonsense error. Do not mention anything about the choices."},
        ],
        stream=False
    )
    return response.choices[0].message.content

def extract_answer(client, text, choices):
    choice_text = ""
    for (key, value) in choices.items():
        choice_text = choice_text + f"{key}. {value}\n"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"Context: {text}\nBased on the choices\n{choice_text}extract the answer in the context. Show me just one letter, i.e A, B, C or D. If there is no explicit answer, show X."},
        ],
        stream=False
    )
    return response.choices[0].message.content
    
def calculate_atypicality(model, tokenizer, text, rev = False):
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
# tokenizer = AutoTokenizer.from_pretrained(save_dir,return_token_type_ids=False)
# model = PreTrainedTokenizerFast.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.unk_token
model.to(device)
print("good")

prompts = {}
prompts['head'] = "[INST]«SYS»You are answering multichoice questions. Give the answer after the explanation.«/SYS»\nAnswer the following question. Also provide a short explanation.\n\nQuestion: "
prompts['answer'] = "Answer: "
prompts['critique'] = '\n[INST]Review your answer and find problems with your answer[/INST]\n'
prompts['improve'] = "\n[INST]This is a review\n["

num_correct = 1

i = 0

all_records = {}

for subject in task_list:
    record = {}
    logs = []
    targets = []
    aty_means = {}
    aty_totals = {}
    aty_logs = {}
    aty_means['question'] = []
    aty_totals['question'] = []
    aty_logs['question'] = []
    aty_means['init_an'] = []
    aty_totals['init_an'] = []
    aty_logs['init_an'] = []
    aty_means['review'] = []
    aty_totals['review'] = []
    aty_logs['review'] = []
    aty_means['correct_an'] = []
    aty_totals['correct_an'] = []
    aty_logs['correct_an'] = []
    
    model_answers = []
    gather_choices = []
    data = load_from_disk(f"/root/autodl-tmp/Experiments/Atypicality/beyond-confidence-atypicality/data_zoo/data/gsm8k")
    task_data = load_from_disk(f"/root/autodl-tmp/ConformalLLM/data/{subject}")
    questions, answers, choices = prepare_data(task_data)
    # data_train = data['train']
    # data_test = data['test']
    i = 5
    for (question, target, choice) in tqdm(zip(questions,answers,choices)):
        # if i >=10:
        #     break
        i += 1
        log = []
        model_answer = []
        # question = data['question']
        aty_mean, aty_total, aty_log = calculate_atypicality(model, tokenizer, question)
        aty_means['question'].append(aty_mean)
        aty_totals['question'].append(aty_total)
        aty_logs['question'].append(aty_log)
        print("question",aty_mean)
        
        text = prompts['head'] + question + "[/INST]\n"
        initial_response = get_response(model, tokenizer, text)
        ini_aty_mean, ini_aty_total, ini_aty_log = calculate_atypicality(model, tokenizer, initial_response)
        aty_means['init_an'].append(ini_aty_mean)
        aty_totals['init_an'].append(ini_aty_total)
        aty_logs['init_an'].append(ini_aty_log)
        print("init_ans", ini_aty_mean)
        log.append(initial_response)
        initial_answer = extract_answer(client, initial_response, choice)
        
        model_answer.append(initial_answer)
        correction_response = initial_response
        for j in range(num_correct):
            review_response = review(client, correction_response)
            rev_aty_mean, rev_aty_total, rev_aty_log = calculate_atypicality(model, tokenizer, review_response)
            aty_means['review'].append(rev_aty_mean)
            aty_totals['review'].append(rev_aty_total)
            aty_logs['review'].append(rev_aty_log)
            print("review",rev_aty_mean)
            log.append(review_response)
            
            whole_response = text + correction_response
            text = whole_response + prompts['improve'] + review_response + "]\nBased on the review, updata your answer to the question.[/INST]\n"
            correction_response = get_response(model, tokenizer, text)
            cor_aty_mean, cor_aty_total, cor_aty_log = calculate_atypicality(model, tokenizer, correction_response)
            aty_means['correct_an'].append(cor_aty_mean)
            aty_totals['correct_an'].append(cor_aty_total)
            aty_logs['correct_an'].append(cor_aty_log)
            print("correct_an",cor_aty_mean)
            log.append(correction_response)
            
            correction_answer = extract_answer(client, correction_response, choice)
            model_answer.append(correction_answer)

        print(f"initial answer:{initial_answer} ||| correction answer:{correction_answer}")
        targets.append(target)
        logs.append(log)
        gather_choices.append(choice)
        model_answers.append(model_answer)
    
    record['aty_means'] = aty_means
    record['aty_total'] = aty_totals
    record['aty_log'] = aty_logs
    record['targets'] = targets
    record['logs'] = logs
    record['model_answers'] = model_answers
    record['choices'] = gather_choices
    
    all_records[subject] = record

    filename = 'records/mmlu_record.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(all_records, file)
    
    print(f"数据已成功写入文件 {filename}")
