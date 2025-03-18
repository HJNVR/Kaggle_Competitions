import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer

from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
from sklearn.metrics import log_loss
import torch
from datasets import Dataset
from functools import partial
import numpy as np
import os 
from scipy.special import softmax
from tqdm import tqdm

class CFG():
    train_path = '/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_webgpt_comparison.csv'
    test_path = '/data0/huangjing/workspace/kaggle/lmsys/data/test.csv'
    model_path = "/data0/huangjing/workspace/backbone/bert-base-uncased" #'/data0/huangjing/workspace/backbone/deberta-v3-base'
    
    MAX_LEN = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/deberta-v3-base_log_loss_data_add_train_2024_06_04_10epoch"
    #output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/deberta-v3-base_log_loss_data_add_06_04"
    checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/bert_base_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison__2024_06_05_2epoch"
    output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/bert_base_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison_06_04"

    loss_log_history_path = "/data0/huangjing/workspace/kaggle/lmsys/loss_log_history/bert_base_log_loss_lmsys_train_hh_rlhf_additional_webgpt_comparison_06_05_log_history_2epoch.csv"

cfg = CFG()

train = pd.read_csv(cfg.train_path)
test = pd.read_csv(cfg.test_path)
model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=3).to(cfg.device)
'''
peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS,r=8,
                         lora_alpha=32,lora_dropout=0.1)
model_peft = get_peft_model(model,peft_config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(model_peft)
'''
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
    num_train_epochs=1,
    per_device_train_batch_size=8,
    logging_steps=100,
    eval_steps=100,
    save_steps=100,
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
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=15)]
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