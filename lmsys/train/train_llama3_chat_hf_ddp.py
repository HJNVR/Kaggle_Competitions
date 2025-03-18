import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from transformers import AutoTokenizer, LlamaForSequenceClassification, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class CFG():
    train_path = '/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train.csv'
    test_path = '/data0/huangjing/workspace/kaggle/lmsys/data/test.csv'
    model_path = "/data0/huangjing/workspace/backbone/llama-3-8b-chat-hf"
    tokenizer_path = "/data0/huangjing/workspace/backbone/longformer/tokenizer"
    num_epoch = 1
    batch_size = 1
    lr = 2e-5
    MAX_LEN = 2048 #256 #1024 #2048 #4096
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_resume = False
    checkpoint_path = "/data0/huangjing/workspace/kaggle/lmsys/ddp/checkpoint/2024_06_25_lmsys_train_epoch_2_14_03_31.pth"
    output_dir = "/data0/huangjing/workspace/kaggle/lmsys/model_train/longformer_4096_log_loss_data_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_06_14"
    loss_log_history_path = "/data0/huangjing/workspace/kaggle/lmsys/loss_log_history/longformer_4096_log_loss_lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_log_history_06_14_1epoch.csv"
    DROPOUT = 0.05
    SEED = 2024 
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5 
    NUM_LABELS = 3 
    LORA_RANK = 4
    LORA_ALPHA = 8
    LORA_MODULES = ['o_proj', 'v_proj']

cfg = CFG()

def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join(sentences)

def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    return [len(t) for t in input_ids]

class CustomLmsysDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.labels[idx], self.input_ids[idx], self.attention_masks[idx]

def custom_collate_fn(batch):
    labels = torch.stack([item[0] for item in batch])
    input_ids = torch.stack([item[1] for item in batch])
    attention_mask = torch.stack([item[2] for item in batch])

    return labels, input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description='Pytorch distribute training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_id', type=str, default='0,1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True

    train = pd.read_csv(cfg.train_path).head(100)
    train['prompt'] = train['prompt'].apply(process)
    train['response_a'] = train['response_a'].apply(process)
    train['response_b'] = train['response_b'].apply(process)
    indexes = train[(train.response_a == 'null') & (train.response_b == 'null')].index
    train.drop(indexes, inplace=True)
    train.reset_index(inplace=True, drop=True)
    train['text'] = 'User prompt: ' + train['prompt'] + '\n\nModel A :\n' + train['response_a'] + '\n\n--------\n\nModel B:\n' + train['response_b']
    train['token_count'] = get_token_lengths(train['text'], tokenizer)
    train['label'] = np.argmax(train[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)

    tokens = tokenizer(train['text'].tolist(), padding='max_length', max_length=cfg.MAX_LEN, truncation=True, return_tensors='np')
    INPUT_IDS = tokens['input_ids']
    ATTENTION_MASKS = tokens['attention_mask']
    LABELS = train[['winner_model_a', 'winner_model_b', 'winner_tie']].values

    train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_labels, val_labels = train_test_split(
        INPUT_IDS, ATTENTION_MASKS, LABELS, test_size=0.1, random_state=cfg.SEED)

    train_dataset = CustomLmsysDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = CustomLmsysDataset(val_input_ids, val_attention_masks, val_labels)

    train_sampler = DistributedSampler(dataset=train_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=cfg.batch_size, drop_last=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

    quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
    )

    #model = LlamaForSequenceClassification.from_pretrained(cfg.model_path, num_labels=cfg.NUM_LABELS, torch_dtype=torch.bfloat16).to(device)
    model = LlamaForSequenceClassification.from_pretrained(cfg.model_path, num_labels=cfg.NUM_LABELS, 
                                                           quantization_config=quantization_config, device_map=device)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.DROPOUT,
        bias='none',
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=cfg.LORA_MODULES)
    model = get_peft_model(model, lora_config)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=4)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.NUM_WARMUP_STEPS, num_training_steps=100000)

    LOSS_FN = torch.nn.CrossEntropyLoss().to(dtype=torch.float32)

    train_log_losses = []
    val_log_losses = []
    val_accuracies = []
    tr_accuracies = []

    for epoch in range(1, cfg.num_epoch + 1):
        model.train()
        train_log_loss = 0
        tr_pred_record = []
        tr_real_record = []

        train_sampler.set_epoch(epoch)
        for step, (labels, input_ids, attention_mask) in enumerate(train_loader):
            labels = labels.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask).logits
            log_loss = LOSS_FN(logits, labels.float())
            log_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            probs = torch.softmax(logits.detach(), dim=-1)
            preds = torch.argmax(probs, dim=-1)
            tr_pred_record += preds.tolist()
            tr_real_record += torch.argmax(labels, dim=-1).tolist()
            train_log_loss += log_loss.item()
            if local_rank == 0 and step % 1000 == 0:
                print(f'Training | Step {step} | log_loss: {train_log_loss / (step + 1):.4f} | accuracy: {accuracy_score(tr_real_record, tr_pred_record):.4f}')

        tr_accu = accuracy_score(tr_real_record, tr_pred_record)
        tr_accuracies.append(tr_accu)
        train_log_losses.append(train_log_loss / len(train_loader))

        model.eval()
        val_log_loss = 0
        val_pred_record = []
        val_real_record = []

        with torch.no_grad():
            for step, (labels, input_ids, attention_mask) in enumerate(val_loader):
                labels = labels.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                logits = model(input_ids, attention_mask=attention_mask).logits
                val_log_loss += LOSS_FN(logits, labels.float()).item()
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                val_pred_record += preds.tolist()
                val_real_record += torch.argmax(labels, dim=-1).tolist()

                if local_rank == 0 and step % 1000 == 0:
                    print(f'Validation | Step {step} | log_loss: {val_log_loss / (step + 1):.4f} | accuracy: {accuracy_score(val_real_record, val_pred_record):.4f}')

        val_accu = accuracy_score(val_real_record, val_pred_record)
        val_accuracies.append(val_accu)
        val_log_losses.append(val_log_loss / len(val_loader))

        if local_rank == 0:
            print(f'Epoch {epoch} | Train log_loss: {train_log_losses[-1]:.4f} | Val log_loss: {val_log_losses[-1]:.4f} | Val accuracy: {val_accuracies[-1]:.4f}')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}, #model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_log_losses': train_log_losses,
                'val_log_losses': val_log_losses,
                'tr_accuracies': tr_accuracies,
                'val_accuracies': val_accuracies
            }

            checkpoint_name = f"/data0/huangjing/workspace/kaggle/lmsys/checkpoint/llama3_ddp/checkpoint_{epoch:02d}.pth"
            torch.save(checkpoint, checkpoint_name)
            print(f'Checkpoint saved to {checkpoint_name}')

            loss_log_history = pd.DataFrame({
                'train_log_losses': train_log_losses,
                'val_log_losses': val_log_losses,
                'tr_accuracies': tr_accuracies,
                'val_accuracies': val_accuracies
            })

            loss_log_history.to_csv(cfg.loss_log_history_path, index=False)

            plt.plot(range(1, len(train_log_losses) + 1), train_log_losses, label='train_log_loss')
            plt.plot(range(1, len(val_log_losses) + 1), val_log_losses, label='val_log_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"/data0/huangjing/workspace/kaggle/lmsys/loss_log_history/llama3_ddp/loss_plot.png")
            plt.close()

if __name__ == '__main__':
    main()

    