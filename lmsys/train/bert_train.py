import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

train = pd.read_csv('/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train.csv')
test = pd.read_csv('/data0/huangjing/workspace/kaggle/lmsys/data/test.csv')
train, valid = train_test_split(train, test_size=0.9, shuffle=True)

class CFG():
    # Hyperparameters
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 16 #32
    VALID_BATCH_SIZE = 16 #32
    TEST_BATCH_SIZE = 16 #32
    EPOCHS = 5
    LEARNING_RATE = 1e-05
    THRESHOLD = 0.5 # threshold for the sigmoid
    tokenizer = BertTokenizer.from_pretrained('/data0/huangjing/workspace/backbone/bert-base-uncased')
    model = BertModel.from_pretrained('/data0/huangjing/workspace/backbone/bert-base-uncased', return_dict=True)

cfg = CFG()

# # Test the tokenizer
# test_text = "We are testing BERT tokenizer."
# # generate encodings
# encodings = cfg.tokenizer.encode_plus(test_text, 
#                                   add_special_tokens = True,
#                                   max_length = 512,
#                                   truncation = True,
#                                   padding = "max_length", 
#                                   return_attention_mask = True, 
#                                   return_tensors = "pt")
# # we get a dictionary with three keys (see: https://huggingface.co/transformers/glossary.html) 
# encodings


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.prompt = list(df['prompt'])
        self.response_a = list(df['response_a'])
        self.response_b = list(df['response_b'])
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title
        }