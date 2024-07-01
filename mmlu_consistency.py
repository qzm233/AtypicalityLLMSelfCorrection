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

task_list = ['formal_logic','professional_accounting', 'public_relations','computer_security']
# ['marketing', 'business_ethics', 'high_school_computer_science', 'clinical_knowledge'.'college_chemistry']

# task_list = ['college_computer_science' , 'formal_logic', 'high_school_computer_science',
#  'computer_security', 'machine_learning',
             
#  'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
#  'college_medicine', 'professional_medicine',

#  'business_ethics', 'professional_accounting', 'public_relations',
#  'management', 'marketing'
#  ]

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"We will use the GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, using the CPU instead.")

def get_response(model, tokenizer, text, sample_num = 10):
    inputs = tokenizer(text, truncation=True, max_length=300, return_tensors="pt").to("cuda")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_length=400,
            max_new_tokens = 500,
            num_return_sequences=sample_num,
            eos_token_id=terminators,
            # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
            do_sample = True,
            temperature = 1.1,
            # top_k = 20,
            top_p = 0.90,
        )
    new_response = []
    input_text_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    for i in range(sample_num):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        if i == 0:
            print(generated_text+"\nxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        new_response.append(generated_text[input_text_length:])
    del outputs, inputs
    torch.cuda.empty_cache()

    inputs = tokenizer(text, truncation=True, max_length=300, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_length=400,
            max_new_tokens = 500,
            num_return_sequences=sample_num,
            eos_token_id=terminators,
            # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
            do_sample = True,
            temperature = 1.1,
            # top_k = 20,
            top_p = 0.90,
        )
    input_text_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    for i in range(sample_num):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        # print(generated_text[input_text_length:]+"\nxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        new_response.append(generated_text[input_text_length:])
    del outputs, inputs
    torch.cuda.empty_cache()

    inputs = tokenizer(text, truncation=True, max_length=300, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_length=400,
            max_new_tokens = 500,
            num_return_sequences=sample_num,
            eos_token_id=terminators,
            # attention_mask=inputs.get("attention_mask", None).to("cuda") if "attention_mask" in inputs else None
            do_sample = True,
            temperature = 1.1,
            # top_k = 20,
            top_p = 0.90,
        )
    input_text_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    for i in range(sample_num):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        # print(generated_text[input_text_length:]+"\nxxxxxxxxxxxxxxxxxxxxxxxxxx\n")
        new_response.append(generated_text[input_text_length:])
    del outputs, inputs
    torch.cuda.empty_cache()
    
    return new_response

def review(client, text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"There is an answer.\n[{text}]\nPlease review the answer and the explanation. Point out the logical or commonsenseerror. Do not mention anything about the choices."},
        ],
        stream=False
    )
    return response.choices[0].message.content

def extract_answer(client, texts, choices):
    choice_text = ""
    for (key, value) in choices.items():
        choice_text = choice_text + f"{key}. {value}\n"
    print(choice_text)
    responses = []
    for i in range(len(texts)):
        # print("[\n"+texts[i]+"\n*******************************\n")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Follow the instruction strictly. "},
                {"role": "user", "content": f"Choices: \n{choice_text}\nContext: {texts[i]}\nBased on the choices, extract the answer chosen in the context. Pay attention to the keywords 'the correct answer is ...', etc. \nShow me just the letter of the answer, i.e A, B, C or D."},
            ],
            stream=False
        )
        # print(response.choices[0].message.content,"]")
        responses.append(response.choices[0].message.content)
    return responses
    
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
tokenizer = AutoTokenizer.from_pretrained(save_dir, low_cpu_mem_usage=True, return_token_type_ids=False)
# tokenizer = AutoTokenizer.from_pretrained(save_dir,return_token_type_ids=False)
# model = PreTrainedTokenizerFast.from_pretrained(save_dir, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
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

filename = 'records/mmlu_consistency.pkl'
with open(filename, 'rb') as file:
    all_records = pickle.load(file)

print(f"从文件 {filename} 读取的数据：")
# all_records = {}
print(all_records.keys())

for subject in task_list:
    record = {}
    logs = []
    targets = []
    aty_means = []
    aty_totals = []
    aty_logs = []
    model_answers = []
    gather_choices = []
    error = []
    task_data = load_from_disk(f"/root/autodl-tmp/ConformalLLM/data/{subject}")
    questions, answers, choices = prepare_data(task_data)
    # data_train = data['train']
    # data_test = data['test']
    i = 0
    for (question, target, choice) in tqdm(zip(questions,answers,choices)):
        # if i >=36 or i < 34:
        #     i +=1
        #     continue
        i += 1
        log = []
        model_answer = []
        # question = data['question']
        aty_mean, aty_total, aty_log = calculate_atypicality(model, tokenizer, question)
        print(aty_mean)
        text = prompts['head'] + question + "[/INST]\n"
        try:
            initial_responses = get_response(model, tokenizer, text, sample_num = 10)
            print("hahahha")
        except MemoryError as e:
            print(f"Memory error occurred: {e}")
            error.append((i,e))
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            error.append((i,e))
            continue
        # initial_responses = get_response(model, tokenizer, text, sample_num = 15)
        log.append(initial_responses)
        # answers = extract_answer(client, initial_responses, choice)
        # model_answer.append(answers)
        # correction_response = initial_response
        # for j in range(num_correct):
        #     review_response = review(client, correction_response)
        #     log.append(review_response)
        #     whole_response = text + correction_response
        #     text = whole_response + prompts['improve'] + review_response + "]\nBased on the review, updata your answer to the question.[/INST]\n"
        #     correction_response = get_response(model, tokenizer, text)
        #     log.append(correction_response)
        #     # print("*****************************\n", correction_response)
        #     correction_answer = extract_answer(client, correction_response, choice)
        #     model_answer.append(correction_answer)
        aty_means.append(aty_mean)
        aty_totals.append(aty_total)
        aty_logs.append(aty_log)
        targets.append(target)
        logs.append(log)
        gather_choices.append(choice)
        # model_answers.append(model_answer)
    
    record['aty_means'] = aty_means
    record['aty_total'] = aty_totals
    record['aty_log'] = aty_logs
    record['targets'] = targets
    record['logs'] = logs
    # record['model_answers'] = model_answers
    record['choices'] = gather_choices
    record['error'] = error
    
    all_records[subject] = record

    filename = 'records/mmlu_consistency.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(all_records, file)
    
    print(f"数据已成功写入文件 {filename}")
