import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer

from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
from sklearn.metrics import log_loss
import torch
from datasets import Dataset
from functools import partial
import numpy as np
import os 
from scipy.special import softmax



class CFG():
    train_path = '/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train.csv'
    test_path = '/data0/huangjing/workspace/kaggle/lmsys/data/test.csv'
    model_path = "/data0/huangjing/workspace/backbone/longformer/model"
    tokenizer_path = "/data0/huangjing/workspace/backbone/longformer/tokenizer"
    
    MAX_LEN = 1024
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/longformer_4096_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_2024_06_12_1epoch"
    output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/deberta-v3-base_log_loss_05_17"

cfg = CFG()

train = pd.read_csv(cfg.train_path)
test = pd.read_csv(cfg.test_path)
model = AutoModelForSequenceClassification.from_pretrained(cfg.checkpoint_path, num_labels=3).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)

def tokenize_function(row, tokenizer):
    max_len = cfg.MAX_LEN #- 4 # We need 1 cls token and 4 [SEP] tokens
    global_attention_mask = [0] * max_len  # Initialize global attention mask

    # Tokenize each of the texts individually
    tokens_prompt = tokenizer.encode(row['prompt'], add_special_tokens=True, padding = 'max_length', max_length=max_len//4, truncation=True)
    remaining_length = max_len - len(tokens_prompt)
    
    tokens_response_a = tokenizer.encode(row['response_a'], add_special_tokens=True, padding = 'max_length', max_length=remaining_length//2, truncation=True)
    remaining_length -= len(tokens_response_a)
    
    tokens_response_b = tokenizer.encode(row['response_b'], add_special_tokens=True, padding = 'max_length', max_length=remaining_length, truncation=True)
    
    # Check if the total length exceeds MAX_LEN before adding separators
    total_length = len(tokens_prompt) + len(tokens_response_a) + len(tokens_response_b) # for 2 additional [SEP] tokens
    if total_length > max_len:
        print("error")
        # Adjust the length of tokens_response_b to fit within MAX_LEN
    #    excess_length = total_length - max_len
    #    tokens_response_b = tokens_response_b[:-excess_length]
    
    # Construct the input_ids with separators and CLS token
    #input_ids = [tokenizer.cls_token_id] + tokens_prompt + [tokenizer.sep_token_id] + tokens_response_a + [tokenizer.sep_token_id] + tokens_response_b + [tokenizer.sep_token_id]
    input_ids = tokens_prompt + tokens_response_a + tokens_response_b
    # Set global attention on the CLS token
    global_attention_mask[0] = 1
    
    # Adjust token_type_ids to mark segments
    #token_type_ids = [0] * (len(tokens_prompt) + 2) + [1] * (len(tokens_response_a) + 1) + [2] * (len(tokens_response_b) + 1)
    
    # Attention mask should be 1 for all real tokens
    attention_mask = [1] * len(input_ids)
    
    # Padding if necessary
    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        # token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        global_attention_mask[len(input_ids):] = [0] * padding_length  # Extend global attention mask for padding
    
    tokenized = {
        'input_ids': input_ids,
        # 'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'global_attention_mask': global_attention_mask[:len(input_ids)]  # Trim the global attention mask to match input_ids length
    
    }
    
    return tokenized


def process_data(df, mode='train'):
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=False)
    remove_cols = ['id', 'prompt', 'response_a', 'response_b']
    if mode == 'train':
        remove_cols += ['model_a', 'model_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
    tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)
    return tokenized_dataset

test_tokenized = process_data(test, mode='test')

input_ids = torch.tensor(test_tokenized["input_ids"]).to(cfg.device)
attention_mask = torch.tensor(test_tokenized["attention_mask"]).to(cfg.device)
global_attention_mask = torch.tensor(test_tokenized["global_attention_mask"]).to(cfg.device)
from scipy.special import softmax
model.eval()
logits = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).logits
logits = logits.cpu().detach().numpy()
preds = softmax(logits, axis=-1)

submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': preds[:, 0],#preds[:, 0],
    'winner_model_b': preds[:, 1],#preds[:, 1],
    'winner_tie': preds[:, 2] #preds[:, 2],
})
submission.to_csv('/data0/huangjing/workspace/kaggle/lmsys/submission.csv', index=False)