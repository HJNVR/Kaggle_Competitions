import pandas as pd
import re
import gzip
import json
from tqdm import tqdm

lmsys_train_data = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/train.csv")
additional_train_data = pd.read_csv('/data0/huangjing/workspace/kaggle/lmsys/data/additional_33k_labelled_conversations/lmsys-33k-deduplicated.csv')
additional_train_data['id'] = 1
df1 = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/llm-human-preference-data-ultrafeedback/ultrafeedback.csv")
df2 = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/llm-human-preference-data-ultrafeedback/ultrafeedback_ties.csv")
ultrafeedback_df = pd.concat([df1, df2], axis=0)
ids, model_a_list, model_b_list, prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], [], [], [], []
count = 0
for _, row in tqdm(ultrafeedback_df.iterrows() ,total=len(ultrafeedback_df)):
    chosen = eval(row["chosen"].replace("}\n {", "},{"))
    rejected = eval(row["rejected"].replace("}\n {", "},{"))
    assert len(chosen) == 2
    assert len(rejected) == 2
    assert rejected[0] == chosen[0]
    
    ids.append(0)
    prompt = [chosen[0]['content']]
    chosen = [chosen[1]["content"]]
    reject = [rejected[1]["content"]]
    model_a_list.append(row["chosen-model"])
    model_b_list.append(row["rejected-model"])
    prompts.append(prompt)

    if row["chosen-rating"] == row["rejected-rating"]:  # tie
        label = [0, 0, 1]
        response_a.append(chosen)
        response_b.append(reject)
        winner_model_a.append(0)
        winner_model_b.append(0)
        winner_tie.append(1)
    else:
        response_a.append(chosen)
        response_b.append(reject)
        winner_model_a.append(1)
        winner_model_b.append(0)
        winner_tie.append(0)
    count += 1

ultrafeedback_data = pd.DataFrame.from_dict({
                                  "id":ids,
                                  "model_a":model_a_list,
                                  "model_b":model_b_list,
                                  "prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})
# webgpt_comparison_data = pd.read_json(path_or_buf="/data0/huangjing/workspace/kaggle/lmsys/data/comparisons.jsonl", lines=True)
df = pd.concat([lmsys_train_data, additional_train_data], ignore_index=True, sort=False)
df.to_csv("/data0/huangjing/workspace/kaggle/lmsys/data/gemma_train_add.csv")