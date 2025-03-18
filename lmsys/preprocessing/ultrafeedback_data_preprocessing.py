import pandas as pd
import re
import gzip
import json
from tqdm import tqdm

# ["prompt", "response_a", "response_b", "winner_model_a", ""winner_model_b", "winner_tie"]
#=================================================
lmsys_train_data = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/train.csv")
#=================================================
additional_train_data = pd.read_csv('/data0/huangjing/workspace/kaggle/lmsys/data/additional_33k_labelled_conversations/lmsys-33k-deduplicated.csv')
# ===================================
# https://huggingface.co/datasets/Anthropic/hh-rlhf
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/harmless-base/train.jsonl.gz", "r") as file:
    harmless_base_train = file.read().decode('utf-8')
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/harmless-base/test.jsonl.gz", "r") as file:
    harmless_base_test = file.read().decode('utf-8')
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/helpful-base/train.jsonl.gz", "r") as file:
    helpful_base_train = file.read().decode('utf-8')
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/helpful-base/test.jsonl.gz", "r") as file:
    helpful_base_test = file.read().decode('utf-8')
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/helpful-online/train.jsonl.gz", "r") as file:
    helpful_online_train = file.read().decode('utf-8')
with gzip.open("/data0/huangjing/workspace/kaggle/lmsys/data/hh-rlhf/helpful-online/test.jsonl.gz", "r") as file:
    helpful_online_test = file.read().decode('utf-8')

def get_human_prompts(dialogue):
    pattern = r'(Human: .*?)(?=\n\n|$)'
    # Find all matches
    matches = re.findall(pattern, dialogue, re.DOTALL)
    return matches

def get_assistant_prompts(dialogue):
    pattern = r'(Assistant: .*?)(?=\n\n|$)'
    # Find all matches
    matches = re.findall(pattern, dialogue, re.DOTALL)
    return matches

def hh_rlhf_preprocessing(data):
    prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
    data = data.split('\n')
    count = 0
    for pair in tqdm(data):
        try:
            pair = json.loads(pair)
        except:
            continue
        chosen = pair['chosen']
        reject = pair['rejected']
        # get human prompts
        prompt_list = get_human_prompts(chosen)
        prompt_list = [human_prompt.replace("Human: ", "")for human_prompt in prompt_list]
        prompts.append(prompt_list)
        # get response_a assistant responses
        if count % 2 == 0:
            response_a_list = get_assistant_prompts(chosen)
            response_a_list = [human_prompt.replace("Assistant: ", "")for human_prompt in response_a_list]
            response_a.append(response_a_list)
            response_b_list = get_assistant_prompts(reject)
            response_b_list = [human_prompt.replace("Assistant: ", "")for human_prompt in response_b_list]
            response_b.append(response_b_list)
            winner_model_a.append(1)
            winner_model_b.append(0)
            winner_tie.append(0)
        else:
            response_a_list = get_assistant_prompts(reject)
            response_a_list = [human_prompt.replace("Assistant: ", "")for human_prompt in response_a_list]
            response_a.append(response_a_list)
            response_b_list = get_assistant_prompts(chosen)
            response_b_list = [human_prompt.replace("Assistant: ", "")for human_prompt in response_b_list]
            response_b.append(response_b_list)
            winner_model_a.append(0)
            winner_model_b.append(1)
            winner_tie.append(0)
        if count in [0,10,100,1000,10000]:
            pass
        count += 1
    return pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})

hh_rlhf_harmless_base_train = hh_rlhf_preprocessing(harmless_base_train)
hh_rlhf_harmless_base_test = hh_rlhf_preprocessing(harmless_base_train)
hh_rlhf_helpful_base_train = hh_rlhf_preprocessing(helpful_base_train)
hh_rlhf_helpful_base_test = hh_rlhf_preprocessing(helpful_base_train)
hh_rlhf_helpful_online_train = hh_rlhf_preprocessing(helpful_online_train)
hh_rlhf_helpful_online_test = hh_rlhf_preprocessing(helpful_online_test)
hh_rlhf_data = pd.concat([hh_rlhf_harmless_base_train, hh_rlhf_harmless_base_test, hh_rlhf_helpful_base_train, hh_rlhf_helpful_base_test, hh_rlhf_helpful_online_train, hh_rlhf_helpful_online_test], ignore_index=True, sort=False)
print("hh_rlhf_data contains nun value #: ", hh_rlhf_data.isnull().sum().sum())
# ===================================
#https://www.kaggle.com/datasets/thedrcat/llm-human-preference-data-ultrafeedback/data
df1 = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/llm-human-preference-data-ultrafeedback/ultrafeedback.csv")
df2 = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/llm-human-preference-data-ultrafeedback/ultrafeedback_ties.csv")
ultrafeedback_df = pd.concat([df1, df2], axis=0)
prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
count = 0
for _, row in tqdm(ultrafeedback_df.iterrows() ,total=len(ultrafeedback_df)):
    chosen = eval(row["chosen"].replace("}\n {", "},{"))
    rejected = eval(row["rejected"].replace("}\n {", "},{"))
    assert len(chosen) == 2
    assert len(rejected) == 2
    assert rejected[0] == chosen[0]
    
    prompt = [chosen[0]['content']]
    chosen = [chosen[1]["content"]]
    reject = [rejected[1]["content"]]

    prompts.append(prompt)

    if row["chosen-rating"] == row["rejected-rating"]:  # tie
        label = [0, 0, 1]
        response_a.append(chosen)
        response_b.append(reject)
        winner_model_a.append(0)
        winner_model_b.append(0)
        winner_tie.append(1)
    else:
        if count % 2 == 0:
            response_a.append(chosen)
            response_b.append(reject)
            winner_model_a.append(1)
            winner_model_b.append(0)
            winner_tie.append(0)
        else:
            response_a.append(reject)
            response_b.append(chosen)
            winner_model_a.append(0)
            winner_model_b.append(1)
            winner_tie.append(0)
        if count in [0,10,100,1000,10000]:
            pass
    count += 1
    
ultrafeedback_data = pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
#=============================
#https://openaipublic.blob.core.windows.net/webgpt-answer-viewer/comparisons.jsonl
webgpt_comparison_data = pd.read_json(path_or_buf="/data0/huangjing/workspace/kaggle/lmsys/data/comparisons.jsonl", lines=True)
prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
for i in tqdm(range(0,len(webgpt_comparison_data))):
    q_a_1 = webgpt_comparison_data.loc[i,0]
    q_a_2 = webgpt_comparison_data.loc[i,1]
    question = q_a_1['question']['full_text']
    answer_1 = q_a_1['answer']
    score_1 = q_a_1['score']
    answer_2 = q_a_2['answer']
    score_2 = q_a_2['score']
    
    prompts.append([question])
    response_a.append([answer_1])
    response_b.append([answer_2])
    if score_1 > score_2:
        winner_model_a.append(1)
        winner_model_b.append(0)
        winner_tie.append(0)
    elif score_2 > score_1:
        winner_model_a.append(0)
        winner_model_b.append(1)
        winner_tie.append(0)
    else:
        winner_model_a.append(0)
        winner_model_b.append(0)
        winner_tie.append(1)
    if i in [0,10,100,1000,10000]:
            pass
webgpt_comparison_data = pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
#=============================
#https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise/tree/main/data
gptj_pariwise_data = pd.read_parquet("/data0/huangjing/workspace/kaggle/lmsys/data/synthetic-instruct-gptj-pairwise/train-00000-of-00001-1e5d57b93c448e7a.parquet")
prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
for i in tqdm(range(0,len(gptj_pariwise_data))):
    prompt = gptj_pariwise_data.loc[i]['prompt']
    chosen = gptj_pariwise_data.loc[i]['chosen']
    reject = gptj_pariwise_data.loc[i]['rejected']
    prompts.append([prompt])
    if i % 2 == 0:
        response_a.append([chosen])
        response_b.append([reject])
        winner_model_a.append(1)
        winner_model_b.append(0)
        winner_tie.append(0)
    else:
        response_a.append([reject])
        response_b.append([chosen])
        winner_model_a.append(0)
        winner_model_b.append(1)
        winner_tie.append(0)
    if i in [0,10,100,1000,10000]:
            pass
gptj_pariwise_data = pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
#=============================
#https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets
rlhf_reward_test = pd.read_parquet("/data0/huangjing/workspace/kaggle/lmsys/data/rlhf-reward-datasets/test-00000-of-00001-955c146ec7a10a1e.parquet")

rlhf_reward_train = pd.read_parquet("/data0/huangjing/workspace/kaggle/lmsys/data/rlhf-reward-datasets/train-00000-of-00001-2ea3039ca4da89f8.parquet")
def rlhf_reward_preprocessing(data):
    prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
    for i in tqdm(range(0,len(data))):
        text = data.loc[i].prompt
        pattern_human = r'\n\nHuman: (.*?)\n\n'
        pattern_assistant = r'\n\nAssistant: (.*?)\n\n'

        # Extracting the text
        try:
            human_question = re.findall(pattern_human, text, re.DOTALL)
            assistant_answer = re.findall(pattern_assistant, text, re.DOTALL)
        except:
            continue

        chosen = data.loc[i].chosen
        reject = data.loc[i].rejected
        chosen = assistant_answer + [chosen.replace("Assistant: ", "").replace("\n", "")]
        reject = assistant_answer + [reject.replace("Assistant: ", "").replace("\n", "")]

        prompts.append(human_question)
        if i in [0,10,100,1000,10000]:
            pass
        if i % 2 == 0:
            response_a.append(chosen)
            response_b.append(reject)
            winner_model_a.append(1)
            winner_model_b.append(0)
            winner_tie.append(0)
        else:
            response_a.append(reject)
            response_b.append(chosen)
            winner_model_a.append(0)
            winner_model_b.append(1)
            winner_tie.append(0)
    return pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
rlhf_reward_test = rlhf_reward_preprocessing(rlhf_reward_test)
rlhf_reward_train = rlhf_reward_preprocessing(rlhf_reward_train)
rlhf_reward_data = pd.concat([rlhf_reward_test, rlhf_reward_train], ignore_index=True, sort=False)
print("rlhf_reward_data contains nun value #: ", rlhf_reward_data.isnull().sum().sum())
#=============================
#https://huggingface.co/datasets/lmsys/mt_bench_human_judgments
'''
mt_bench_human = pd.read_parquet("/data0/huangjing/workspace/kaggle/lmsys/data/mt_bench/human-00000-of-00001-25f4910818759289.parquet")
# mt_bench_gpt4 = pd.read_parquet("/data0/huangjing/workspace/kaggle/lmsys/data/mt_bench/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet")
# mt_bench_df = pd.concat([mt_bench_human, mt_bench_gpt4], axis=0)
prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], []
count = 0
for _, row in tqdm(mt_bench_human.iterrows() ,total=len(mt_bench_human)):
    winner = row['winner']
    prompts.append([chat['content'] for chat in row.conversation_a if chat['role'] == 'user'])
    response_a.append([chat['content'] for chat in row.conversation_a if chat['role'] == 'assistant'])
    response_b.append([chat['content'] for chat in row.conversation_b if chat['role'] == 'assistant'])

    if winner == "model_a":
        winner_model_a.append(1)
        winner_model_b.append(0)
        winner_tie.append(0)
    elif winner == "model_b":
        winner_model_a.append(0)
        winner_model_b.append(1)
        winner_tie.append(0)
    else:
        winner_model_a.append(0)
        winner_model_b.append(0)
        winner_tie.append(1)
mt_bench_data = pd.DataFrame.from_dict({"prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
'''
# lmsys_train_data = lmsys_train_data[['prompt', 'response_a', 'response_b', 'winner_model_a', 'winner_model_b', 'winner_tie']]
#df = pd.concat([ultrafeedback_data, ultrafeedback_tie_data, lmsys_train_data], ignore_index=True, sort=False)
# df = pd.concat([lmsys_train_data, additional_train_data], ignore_index=True, sort=False)
#df = pd.concat([ultrafeedback_data, ultrafeedback_tie_data, additional_train_data], ignore_index=True, sort=False)
#df = pd.concat([additional_train_data, lmsys_train_data], ignore_index=True, sort=False)
#df = pd.concat([lmsys_train_data, hh_rlhf_data, additional_train_data, webgpt_comparison_data, gptj_pariwise_data, ultrafeedback_data, rlhf_reward_data], ignore_index=True, sort=False)
#df = pd.concat([lmsys_train_data, additional_train_data, webgpt_comparison_data, gptj_pariwise_data, ultrafeedback_data, rlhf_reward_data], ignore_index=True, sort=False)
df = pd.concat([lmsys_train_data, additional_train_data, ultrafeedback_data], ignore_index=True, sort=False)
df['id'] = df['id'].astype(str)
df['prompt'] = df['prompt'].astype(str)
df['response_a'] = df['response_a'].astype(str)
df['response_b'] = df['response_b'].astype(str)
df['model_a'] = df['model_a'].astype(str)
df['model_b'] = df['model_b'].astype(str)
df['id'] = '123456'
print(df.shape)
print("df contains nun value #: ", df.isnull().sum().sum())
#df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_additional_train.csv")
#df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward.csv")
#df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_train_data.csv")
#df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_additional_webgpt_comparison_gptj_ultrafeedback_reward.csv")
df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_additional_ultrafeedback.csv")
'''
#================================================================
# text preprocessing
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm  # Assuming you have imported tqdm

# Ensure you have the necessary NLTK data downloaded
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def preprocess_text(text):
    # Remove punctuation
    table = str.maketrans("", "", string.punctuation)
    cleaned_text = text.translate(table)

    # Tokenize and convert to lower case
    tokens = nltk.word_tokenize(cleaned_text.lower())

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    #stemmer = PorterStemmer()
    #stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Apply lemmatization
    #lemmatizer = WordNetLemmatizer()
    #lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return " ".join(filtered_tokens)

for column in tqdm(['prompt', 'response_a', 'response_b']):
    df[column] = df[column].apply(preprocess_text)
'''
#df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_webgpt_comparison_preprocessing.csv")
#================================================================