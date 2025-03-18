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
    model_path = "/data0/huangjing/workspace/backbone/bert-base-uncased"
    tokenizer_path = "/data0/huangjing/workspace/kaggle/lmsys/checkpoint/bert_base_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison__2024_06_04_10epoch_tokenizer"
    
    MAX_LEN = 512
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/bert_base_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison__2024_06_04_10epoch"
    output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/deberta-v3-base_log_loss_05_17"

cfg = CFG()

train = pd.read_csv(cfg.train_path)
test = pd.read_csv(cfg.test_path)
model = AutoModelForSequenceClassification.from_pretrained(cfg.checkpoint_path, num_labels=3).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

sep_token = tokenizer('[SEP]', add_special_tokens=False)['input_ids']

def tokenize_function(row, tokenizer):
    max_len = cfg.MAX_LEN - 2 # We need 2 separator tokens
    # We should clip each of the three texts individually, otherwise we might lose response_a or response_b completely
    tokens_prompt = tokenizer(row['prompt'], truncation=True, max_length=max_len//4)['input_ids'][1:]
    remaining_length = max_len - len(tokens_prompt)
    
    tokens_response_a = tokenizer(row['response_a'], truncation=True, max_length=remaining_length//2)['input_ids'][1:-1]
    remaining_length -= len(tokens_response_a)
    tokens_response_b = tokenizer(row['response_b'], truncation=True, max_length=remaining_length)['input_ids'][:-1]
    remaining_length -= len(tokens_response_b)
    input_ids = tokens_prompt + sep_token + tokens_response_a + sep_token + tokens_response_b
    token_type_ids = [0] * (len(tokens_prompt) + len(sep_token)) + [1] * (len(tokens_response_a) + len(sep_token) + len(tokens_response_b))
    attention_mask = [1] * len(input_ids)
    
    if len(input_ids) < cfg.MAX_LEN:
        pad_len = (cfg.MAX_LEN - len(input_ids))
        input_ids = input_ids + [0] * pad_len
        token_type_ids = token_type_ids + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len
    elif len(input_ids) > cfg.MAX_LEN:
        print("Error")
        
    tokenized = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
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

from scipy.special import softmax
model.eval()
logits = model(input_ids, attention_mask).logits
logits = logits.cpu().detach().numpy()
preds = softmax(logits, axis=-1)

submission = pd.DataFrame({
    'id': test['id'],
    'winner_model_a': preds[:, 0],#preds[:, 0],
    'winner_model_b': preds[:, 1],#preds[:, 1],
    'winner_tie': preds[:, 2] #preds[:, 2],
})
submission.to_csv('/data0/huangjing/workspace/kaggle/lmsys/submission.csv', index=False)