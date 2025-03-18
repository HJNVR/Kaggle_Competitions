import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer
from sklearn.metrics import log_loss
import torch
from datasets import Dataset
from functools import partial
import numpy as np
import os 
from scipy.special import softmax
from tqdm import tqdm

class CFG():
    train_path = '/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train.csv'
    test_path = '/data0/huangjing/workspace/kaggle/lmsys/data/test.csv'
    model_path = "/data0/huangjing/workspace/backbone/longformer/model" #'/data0/huangjing/workspace/backbone/deberta-v3-base'
    tokenizer = "/data0/huangjing/workspace/backbone/longformer/tokenizer"
    
    num_epoch=1
    MAX_LEN = 1024 #2048 #2048 #4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/deberta-v3-base_log_loss_data_add_train_2024_06_04_10epoch"
    #output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/deberta-v3-base_log_loss_data_add_06_04"
    checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/longformer_4096_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_2024_06_13_1epoch"
    output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/longformer_4096_log_loss_data_lmsys_train_6_24"
    
    loss_log_history_path = "/data0/huangjing/workspace/kaggle/lmsys/loss_log_history/longformer_4096_log_loss_lmsys_train_06_24_1epoch.csv"

cfg = CFG()

train = pd.read_csv(cfg.train_path).head(1000)
test = pd.read_csv(cfg.test_path)
model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=3).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

# sep_token = tokenizer('[SEP]', add_special_tokens=False)['input_ids']

def tokenize_function(row, tokenizer):
    max_len = cfg.MAX_LEN #- 4 # We need 1 cls token and 4 [SEP] tokens
    global_attention_mask = [0] * max_len  # Initialize global attention mask

    # Tokenize each of the texts individually
    tokens_prompt = tokenizer.encode(row['prompt'], add_special_tokens=True, padding = 'max_length', max_length=max_len//3, truncation=True)
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


def add_label(df):
    labels = np.zeros(len(df), dtype=np.int32)
    labels[df['winner_model_a'] == 1] = 0
    labels[df['winner_model_b'] == 1] = 1
    labels[df['winner_tie'] == 1] = 2
    df['labels'] = labels
    return df

def process_data(df, mode='train'):
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=False)
    remove_cols = ['id', 'prompt', 'response_a', 'response_b']
    if mode == 'train':
        remove_cols += ['model_a', 'model_b', 'winner_model_a', 'winner_model_b', 'winner_tie']
    tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)
    return tokenized_dataset

train = add_label(train)
train_tokenized = process_data(train, mode='train')
# test_tokenized = process_data(test, mode='test')

def split_train_val(dataset, train_fraction):
    np.random.seed(0)
    ixs = np.arange(len(dataset))
    cutoff = int(len(ixs) * train_fraction)
    np.random.shuffle(ixs)
    ixs_train = ixs[:cutoff]
    ixs_val = ixs[cutoff:]
    fit_train = dataset.select(ixs_train)
    fit_val = dataset.select(ixs_val)
    return fit_train, fit_val

fit_train, fit_val = split_train_val(train_tokenized, 0.8)

# import datasets
# import wandb
# wandb.init(project="lmsys", mode="disabled")

def compute_metrics(eval_pred):
    '''
    log_loss 多用于计算multi-nomial 分类器的每个准确度的量化
    '''
    logits, labels = eval_pred
    softmax_scores = softmax(logits, axis=-1)
    #preds = np.argmax(logits, axis=1)
    #argmax：array([0, 2, 0, ..., 0, 0, 1])
    #max: array([0.49951416, 0.35601103, 0.4122035 , ..., 0.38621908, 0.392522  ,0.5922973 ], dtype=float32)
    # preds = np.max(softmax(logits, axis=-1), axis=1)
    # log_loss 需要两个lsoftmax(preds, axis=1)做计算，并且要有一个是probability
    # ex: log_loss([1], [[1/3, 1/3, 1/3]], labels=[0,1,2])
    #return {'log_loss': log_loss(preds.tolist(), softmax(logits, axis=-1).tolist(), labels=list(set(labels)))}
    multi_class_label = []
    for label in labels:
        if label == 0:
            multi_class_label.append([1,0,0])
        elif label == 1:
            multi_class_label.append([0,1,0])
        else:
            multi_class_label.append([0,0,1])
    #return {'log_loss': log_loss(labels, logits)}
    return {'log_loss': log_loss(multi_class_label, softmax_scores)}

training_args = TrainingArguments(
    output_dir=cfg.output_dir,
    num_train_epochs=cfg.num_epoch,
    per_device_train_batch_size=2, #4, #8,
    logging_steps=500,
    eval_steps=500,
    save_steps=500,
    evaluation_strategy='steps',
    learning_rate=2e-5,
    metric_for_best_model='log_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True,
    report_to=None,
)

trainer = Trainer(
    model=model,
    # model=model_peft,
    args=training_args,
    train_dataset=fit_train,
    eval_dataset=fit_val,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=15)]
)

trainer.train(
    # resume_from_checkpoint=True
    )
metrics=trainer.evaluate()
trainer.save_model(cfg.checkpoint_path)
# with open(cfg.loss_log_history_path) as f:
    # f.write(trainer.state.log_history)
log_history = pd.DataFrame(trainer.state.log_history)
log_history.to_csv(cfg.loss_log_history_path)

import pandas as pd
import math
import matplotlib.pyplot as plt
import os
path = cfg.loss_log_history_path
root = path[:path.rfind('/')]
file_name = path[path.rfind('/')+1:path.rfind('.')] 

loss, epoch, eval_loss, eval_log_loss = [], [], [], []
for column in log_history.columns:
    if column == 'loss':
        for value in log_history[column]:
            if not math.isnan(value):
                loss.append(value)
    elif column == 'eval_loss':
        for value in log_history[column]:
            if not math.isnan(value):
                eval_loss.append(value)
    elif column == 'eval_log_loss':
        for value in log_history[column]:
            if not math.isnan(value):
                eval_log_loss.append(value)
    elif column == 'epoch':
        count = 0
        while count < len(log_history[column]):
            if count % 2 == 0:
                epoch.append(log_history[column][count])
            count += 1
loss += [loss[-1]]
# plt.plot(epoch, eval_loss, label='eval_loss')
plt.plot(epoch, loss, label='loss')
plt.plot(epoch, eval_log_loss, label='eval_log_loss')
plt.legend()
plt.savefig(f"{root}/{file_name}.png")