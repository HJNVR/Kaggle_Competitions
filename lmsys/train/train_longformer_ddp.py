import pandas as pd
import torch.distributed
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from torch.nn.parallel import DistributedDataParallel as DDP
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
    
    num_epoch=10
    batch_size = 2
    lr = 2e-5
    MAX_LEN = 4096 #4096 #2048 #4096
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/deberta-v3-base_log_loss_data_add_train_2024_06_04_10epoch"
    #output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/deberta-v3-base_log_loss_data_add_06_04"
    is_resume = False
    checkpoint_path = f"/data0/huangjing/workspace/kaggle/lmsys/ddp/checkpoint/2024_06_25_lmsys_train_epoch_2_14_03_31.pth"


    output_dir = f"/data0/huangjing/workspace/kaggle/lmsys/model_train/longformer_4096_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_06_14"
    loss_log_history_path = "/data0/huangjing/workspace/kaggle/lmsys/loss_log_history/longformer_4096_log_loss_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_log_history_06_14_1epoch.csv"

cfg = CFG()

train = pd.read_csv(cfg.train_path).head(100)
test = pd.read_csv(cfg.test_path)
#model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=3).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)


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

fit_train, fit_val = split_train_val(train_tokenized, 0.9)

# import datasets
# import wandb
# wandb.init(project="lmsys", mode="disabled")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')


class CustomImageDataset(Dataset):
    def __init__(self, lmsys_data):
        self.lmsys_data = lmsys_data

    def __len__(self):
        return len(self.lmsys_data)

    def __getitem__(self, idx):
        return self.lmsys_data[idx]['labels'], self.lmsys_data[idx]['input_ids'], self.lmsys_data[idx]['attention_mask'], self.lmsys_data[idx]['global_attention_mask']   
         
class CustomLogLoss(nn.Module):
    def __init__(self):
        super(CustomLogLoss, self).__init__()

    def forward(self, logits, labels):
        # Apply softmax to logits to get probabilities
        ss_score = F.softmax(logits, dim=-1)

        # Convert labels to one-hot encoding
        multi_class_label = F.one_hot(labels, num_classes=3).float()

        # Calculate log loss using sklearn's log_loss function
        # Note: This requires detaching tensors and converting to numpy arrays
        #loss = log_loss(multi_class_label.cpu().detach().numpy(), ss_score.cpu().detach().numpy())
        loss = -torch.mean(torch.sum(multi_class_label * torch.log(ss_score), dim=1))
        #print(ss_score)
        #print(multi_class_label)
        #print(loss)
        # Convert the loss back to a tensor
        return torch.tensor(loss, requires_grad=True)

def custom_collate_fn(batch):
    # Separate the components of the dataset items
    labels = [torch.tensor(item[0]) for item in batch]
    input_ids = [torch.tensor(item[1]) for item in batch]
    attention_mask = [torch.tensor(item[2]) for item in batch]
    global_attention_mask = [torch.tensor(item[3]) for item in batch]

    # Stack each component to create batches
    labels = torch.tensor(labels)
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    global_attention_mask = torch.stack(global_attention_mask)

    return labels, input_ids, attention_mask, global_attention_mask
# Convert to dataset and dataloader
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch import nn, optim
from sklearn.metrics import accuracy_score
import argparse
parser = argparse.ArgumentParser(description='Pytorch distribute training',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu_id', type=str, default='0,1')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
device = torch.device(f"cuda:{local_rank}")
print("device: {}".format(device))
#torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
#train_sampler = DistributedSampler(dataset=train_dataset)

train_dataset = CustomImageDataset(fit_train)
val_dataset = CustomImageDataset(fit_val)
tr_sampler = DistributedSampler(dataset=train_dataset)
tr_loader  = DataLoader(dataset=train_dataset,
                            batch_size=cfg.batch_size,
                            sampler=tr_sampler,
                            drop_last=True,
                            shuffle=False,
                            collate_fn=custom_collate_fn)

val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=custom_collate_fn)

### 分布式改造，构建DDP分布式模型 ###
model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path, num_labels=3).to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=4)


if cfg.is_resume:
    from collections import OrderedDict

    # Assuming 'checkpoint' is your loaded checkpoint with 'module.' prefix
    checkpoint = torch.load(cfg.checkpoint_path)

    # Create a new OrderedDict without the 'module.' prefix
    #new_state_dict = OrderedDict()
    #for k, v in checkpoint.items():
    #    name = k[7:]  # remove `module.`
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    model.load_state_dict(checkpoint)
    #exit()
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
loss_func = CustomLogLoss()

# os.makedirs(args.output_dir, exist_ok=True)
train_log_losses = []
val_log_losses = []
val_accuracies = []
tr_accuracies = []
for epoch in tqdm(range(1, cfg.num_epoch + 1)):
    model.train()
    train_log_loss = 0
    tr_pred_record = []
    tr_real_record = []
    ### 分布式改造，DDP sampler, 基于当前的epoch为其设置随机数，避免加载到重复数据 ###
    tr_sampler.set_epoch(epoch)
    ### 分布式改造，DDP sampler, 基于当前的epoch为其设置随机数，避免加载到重复数据 ###

    for step, (labels, input_ids, attention_mask, global_attention_mask) in tqdm(enumerate(tr_loader)):
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        global_attention_mask = torch.tensor(global_attention_mask).to(device)

        logits = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).logits
        log_loss = loss_func(logits, labels.to(device))
        optimizer.zero_grad()
        log_loss.backward()
        optimizer.step()
        probs = torch.softmax(logits.detach(), dim=-1)
        preds = torch.argmax(probs, dim=-1)
         # Update records
        tr_pred_record += preds.tolist()
        tr_real_record += labels.tolist()

        # Update running loss
        train_log_loss += log_loss.item()
        if local_rank == 0 and step % 1000 == 0:
            print(f'Training | Step {step} | log_loss: {train_log_loss / (step + 1):.4f} | accuracy: {accuracy_score(tr_real_record, tr_pred_record):.4f}')
    tr_accu = accuracy_score(tr_real_record, tr_pred_record)
    tr_accuracies.append(tr_accu)
    train_log_losses.append(train_log_loss / len(tr_loader))
    print('train | local_rank: %d | epoch: %d | log_loss: %.4f | accuracy: %.4f' % (local_rank, epoch, train_log_loss / len(tr_loader), tr_accu))

    val_log_loss = 0
    val_pred_record = []
    val_real_record = []
    model.eval()
    with torch.no_grad():
        for step, (labels, input_ids, attention_mask, global_attention_mask) in tqdm(enumerate(val_loader)):
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            global_attention_mask = torch.tensor(global_attention_mask).to(device)

            logits = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask).logits
            # Calculate log loss
            val_log_loss += loss_func(logits, labels.to(device)).item()

            # Convert logits to probabilities for accuracy calculation
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            # Update records
            val_pred_record += preds.tolist()
            val_real_record += labels.tolist()
    val_accu = accuracy_score(val_real_record, val_pred_record)
    val_accuracies.append(val_accu)
    val_log_losses.append(val_log_loss / len(val_loader))
    print('val | local_rank: %d, epoch: %d | log_loss: %.4f | accuracy: %.4f' % (local_rank, epoch, val_log_loss / len(val_loader), val_accu))

    if local_rank == 0:
        from datetime import datetime
        current_time = datetime.now().strftime("%H_%M_%S")
        # save ckpt every epoch
        if val_log_loss / len(val_loader) <= 0.9:
            torch.save(model.state_dict(), os.path.join("/data0/huangjing/workspace/kaggle/lmsys/ddp/checkpoint/val_log_loss", f'2024_06_28_lmsys_train_epoch_{epoch}_{val_log_loss / len(val_loader)}_{current_time}.pth'))
        elif val_accu >= 0.5:
            torch.save(model.state_dict(), os.path.join("/data0/huangjing/workspace/kaggle/lmsys/ddp/checkpoint/val_acc", f'2024_06_28_lmsys_train_epoch_{epoch}_{val_accu}_{current_time}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join("/data0/huangjing/workspace/kaggle/lmsys/ddp/checkpoint", f'2024_06_28_lmsys_train_epoch_{epoch}_{current_time}.pth'))
print(train_log_losses)
print(val_log_losses)
print(tr_accuracies)
print(val_accuracies)
import matplotlib.pyplot as plt

# Plotting both log losses and accuracies in subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot training and validation log loss
ax1.plot(train_log_losses, label='Train Log Loss')
ax1.plot(val_log_losses, label='Validation Log Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Log Loss')
ax1.legend()
ax1.set_title('Training and Validation Log Loss')

# Plot training and validation accuracy
ax2.plot(tr_accuracies, label='Train Accuracy')
ax2.plot(val_accuracies, label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.set_title('Training and Validation Accuracy')

# Save the figure
plt.tight_layout()
plt.savefig(os.path.join('/data0/huangjing/workspace/kaggle/lmsys/ddp/loss_history',f'epoch_{epoch}.png'))

# Show the plot
plt.show()

# torchrun --nproc_per_node=2 train_longformer_ddp.py --gpu_id=1,2