import os
import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForSequenceClassification
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
class CFG():
    # Define your configuration parameters here
    test_path = '/data0/huangjing/workspace/kaggle/lmsys/data/test.csv'
    model_path = "/data0/huangjing/workspace/backbone/llama-3-8b-chat-hf"
    num_epoch = 1
    batch_size = 16
    lr = 2e-5
    MAX_LEN = 4096 #256 #1024 #2048 #4096
    device = torch.device("cpu")#torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    is_resume = False
    checkpoint_path = "/data0/huangjing/workspace/kaggle/lmsys/checkpoint/llama3_ddp/checkpoint_01.pth"
    DROPOUT = 0.05
    SEED = 2024 
    NUM_WARMUP_STEPS = 128
    LR_MAX = 5e-5 
    NUM_LABELS = 3 
    LORA_RANK = 4 #8
    LORA_ALPHA = 8 #16
    LORA_MODULES = ['o_proj', 'v_proj']

cfg = CFG()

def process(input_str):
    stripped_str = input_str.strip('[]')
    sentences = [s.strip('"') for s in stripped_str.split('","')]
    return ' '.join(sentences)

def get_token_lengths(texts, tokenizer):
    input_ids = tokenizer(texts.tolist(), return_tensors='np')['input_ids']
    return [len(t) for t in input_ids]

def main():
    test = pd.read_csv(cfg.test_path)
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.add_eos_token = True
    model = LlamaForSequenceClassification.from_pretrained(cfg.model_path, num_labels=cfg.NUM_LABELS)
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

    # Load checkpoint
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)  # Load on CPU if GPU not available

    # Create a new OrderedDict without the 'module.' prefix
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['model_state_dict'].items():
    #    name = k[7:]  # remove `module.`
    #    new_state_dict[name] = v
    # model.load_state_dict(new_state_dict, strict=False)
    # model.to(cfg.device)
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(cfg.device)

    # Set model to evaluation mode
    model.eval()
    test['prompt'] = test['prompt'].apply(process)
    test['response_a'] = test['response_a'].apply(process)
    test['response_b'] = test['response_b'].apply(process)
    indexes = test[(test.response_a == 'null') & (test.response_b == 'null')].index
    test.drop(indexes, inplace=True)
    test.reset_index(inplace=True, drop=True)
    test['text'] = 'User prompt: ' + test['prompt'] + '\n\nModel A :\n' + test['response_a'] + '\n\n--------\n\nModel B:\n' + test['response_b']
    test['token_count'] = get_token_lengths(test['text'], tokenizer)
    # test['label'] = np.argmax(test[['winner_model_a', 'winner_model_b', 'winner_tie']].values, axis=1)
    llama3_preds = np.empty(shape=[0, model.num_labels])  # Adjust the second dimension based on your number of labels

    for i in tqdm(range(0, len(test), cfg.batch_size)):
        # Process the data in batches
        test_sub = test.iloc[i:i+cfg.batch_size]
        tokens = tokenizer(test_sub['text'].tolist(), padding='max_length', max_length=cfg.MAX_LEN, truncation=True, return_tensors='np')
        input_ids = torch.tensor(tokens["input_ids"]).to(cfg.device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(cfg.device)
        # Get model predictions
        with torch.no_grad():  # Ensure no gradients are calculated
            logits = model(input_ids, attention_mask=attention_mask).logits
            logits = logits.cpu().detach().numpy()  # Move logits to CPU and convert to NumPy array
            batch_preds = softmax(logits, axis=-1)  # Apply softmax to get predictions

        # Concatenate the predictions
        llama3_preds = np.concatenate([llama3_preds, batch_preds])

    submission = pd.DataFrame({
        'id': test['id'],
        'winner_model_a': llama3_preds[:, 0],#preds[:, 0],
        'winner_model_b': llama3_preds[:, 1],#preds[:, 1],
        'winner_tie': llama3_preds[:, 2] #preds[:, 2],
    })
    submission.to_csv('/data0/huangjing/workspace/kaggle/lmsys/submission.csv', index=False)




if __name__ == '__main__':
    main()
