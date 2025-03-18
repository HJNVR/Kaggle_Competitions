from openai import OpenAI
import tiktoken
import os
from tqdm import tqdm
import time
import pandas as pd
import json
data = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward.csv")


few_shot_prompt_1 = ["I have three oranges today, I ate an orange yesterday. How many oranges do I have?"]
few_shot_response_a_1 = ["You have two oranges today."]
few_shot_response_b_1 = ["You still have three oranges. Eating an orange yesterday does not affect the number of oranges you have today."]


test = pd.read_csv("/data0/huangjing/workspace/kaggle/lmsys/data/test.csv")
few_shot_prompt_1 = test.loc[0].prompt
few_shot_response_a_1 = test.loc[0].response_a
few_shot_response_b_1 = test.loc[0].response_b

few_shot_prompt_2 = test.loc[1].prompt
few_shot_response_a_2 = test.loc[1].response_a
few_shot_response_b_2 = test.loc[1].response_b

few_shot_prompt_3 = test.loc[2].prompt
few_shot_response_a_3 = test.loc[2].response_a
few_shot_response_b_3 = test.loc[2].response_b

few_shot_prompt_4 = '["explain function calling. how would you call a function?"]'
few_shot_response_a_4 = '["Function calling is the process of invoking or executing a function in a programming language. To call a function, you need to use the function\'s name followed by parentheses \\"()\\". Within the parentheses, you can pass arguments if the function requires any.\\n\\nHere is an example of how to call a function in Python:\\n\\n```python\\n# Define a function\\ndef greet(name):\\n    print(\\"Hello, \\" + name + \\"!\\")\\n\\n# Call the function\\ngreet(\\"John\\")\\n```\\n\\nIn this example, the function \\"greet\\" is defined with a single parameter \\"name\\" which is used to print a greeting message. To call the function and pass an argument, we write \\"greet(\\"John\\")\\". This will execute the function and print \\"Hello, John!\\".\\n\\nIt is important to note that the function should be defined before calling it in most programming languages, as the interpreter needs to know about the function\'s existence before it can be called."]'
few_shot_response_b_4 = '["Function calling is the process of invoking a function in a program. A function is a block of code that performs a specific task and can be reused multiple times in a program. Functions help to organize and modularize code, making it more readable and maintainable.\\n\\nTo call a function, you simply write the name of the function followed by parentheses () containing any arguments or parameters that the function requires. Here\'s an example:\\n```\\nfunction greet(name) {\\n  console.log(\\"Hello, \\" + name);\\n}\\n\\n\\/\\/ Call the greet function and pass in the argument \\"Alice\\"\\ngreet(\\"Alice\\");\\n```\\nIn the example above, the function `greet` takes one parameter, `name`, which is used inside the function to print a greeting message. To call the function, we write the name of the function (`greet`) followed by parentheses containing the argument `\\"Alice\\"`. When the function is called, the argument `\\"Alice\\"` is passed to the function, where it is assigned to the `name` parameter. The function then uses the `name` parameter to print the greeting message `\\"Hello, Alice\\"`.\\n\\nFunctions can also return values, which can be assigned to variables or used in expressions. For example:\\n```\\nfunction addNumbers(a, b) {\\n  return a + b;\\n}\\n\\n\\/\\/ Call the addNumbers function and assign the result to the variable sum\\nvar sum = addNumbers(3, 5);\\nconsole.log(sum); \\/\\/ Output: 8\\n```\\nIn the example above, the function `addNumbers` takes two parameters, `a` and `b`, adds them together, and returns the result. To call the function, we write the name of the function (`addNumbers`) followed by parentheses containing the arguments `3` and `5`. The function calculates the sum of the two arguments (`3 + 5 = 8`) and returns the result (`8`). We then assign the result to the variable `sum` and print it to the console."]'

id, model_a, model_b, prompts, response_a, response_b, winner_model_a, winner_model_b, winner_tie = [], [], [], [], [], [], [], [], []
data_range = "0_100000"
# 582449
fail_log = "/data0/huangjing/workspace/kaggle/lmsys/fail_log/log.txt"
input_token_count = 0 
output_token_count = 0
output_price_per_token = 0.000015
input_price_per_token = 0.000005

for i in tqdm(range(0,100000)):
    prompt = data.loc[i].prompt
    rsp_a = data.loc[i].response_a
    rsp_b = data.loc[i].response_b
    messages=[
    {"role": "system", "content": "You are a helpful assistant that helps me to determine which response a human user will prefer provided with two AI-generated responses. \
                                    Your objective is to brideg the gap between LLM capability and human preference.\
                                    Leading to more user-friendly and widely accepted AI-powered conversation systems."},
    {"role": "user", "content": f"First few shot example: This is a {len(few_shot_prompt_1)} turn conversation, user questions are: '{few_shot_prompt_1}',\
                                                        response_a are: '{few_shot_response_a_1}',\
                                                        response_b are: '{few_shot_response_b_1}',\
                                                        please helpe determine which response a human user will prefer.\
                                                        Please give clean, short and resonable analysis when you make final decision.\
                                                        Make you reply ine a JSON schema,\
                                                        You must not add single quotes or double quotes in your analysis!\
                                                        You must make sure your reply can be parsed by json.loads successfully!"},
    {"role": "assistant", "content": '{"analysis": "Since the eating event happened yesterday, response_a is incorrect. Eating an orange yesterday does not affect the number of oranges you have today. Therefore, response_2 is the correct response.", "winner": "response_b"}'},
    {"role": "user", "content": f"Second few shot example: This is a {len(few_shot_prompt_2)} turn conversation, user questions are: '{few_shot_prompt_2}',\
                                                        response_a are: '{few_shot_response_a_2}',\
                                                        response_b are: '{few_shot_response_b_2}',\
                                                        please helpe determine which response a human user will prefer.\
                                                        There can be three cases: response_a win, response_b win or tie.\
                                                        Please give clean, short and resonable analysis when you make final decision.\
                                                        Make you reply ine a JSON schema,\
                                                        You must not add single quotes or double quotes in your analysis!\
                                                        You must make sure your reply can be parsed by json.loads successfully!"},
    {"role": "assistant", "content": '{"analysis": "response_a takes a more comprehensive approach. It acknowledges the complexity of language and definitions, the importance of respecting individual self-identification, and the biological differences that exist. \
                                        It suggests using inclusive language and creating policies that support gender identity and expression. Furthermore, it emphasizes the importance of mutual respect and open communication to find a resolution that addresses the concerns of both parties.\
                                        response_b on the other hand, is a direct echo of the prompt, reiterating the points made by Mr. Reddy and Ms. Blue without adding new insights or extending the discussion. It simply restates the compromise suggested in the prompt.", "winner": "response_a"}'},
    {"role": "user", "content": f"Third few shot example: This is a {len(few_shot_prompt_3)} turn conversation, user questions are: '{few_shot_prompt_3}',\
                                                        response_a are: '{few_shot_response_a_3}',\
                                                        response_b are: '{few_shot_response_b_3}',\
                                                        please helpe determine which response a human user will prefer.\
                                                        There can be three cases: response_a win, response_b win or tie.\
                                                        Please give clean, short and resonable analysis when you make final decision.\
                                                        Make you reply ine a JSON schema,\
                                                        You must not add single quotes or double quotes in your analysis!\
                                                        You must make sure your reply can be parsed by json.loads successfully!"},
    {"role": "assistant", "content": '{"analysis": "response_a outlines the steps for initializing the classification head for transfer learning and then describes the process for full finetuning. However, there is a contradiction in the response: it mentions freezing the pre-trained layers during full finetuning, which is incorrect. \
                                                    Full finetuning involves updating all layers of the model, not freezing any of them.\
                                                    response_b lso provides steps for initializing the classification head and includes a Python code example using PyTorch. It correctly distinguishes between transfer learning (where the backbone is frozen) and full finetuning (where the backbone weights are also updated). \
                                                    The response provides clear instructions and corrects the mistake found in response_a regarding the freezing of layers during full finetuning.", "winner": "response_b"}'},
    {"role": "user", "content": f"First few shot example: This is a {len(few_shot_prompt_4)} turn conversation, user questions are: '{few_shot_prompt_4}',\
                                                        response_a are: '{few_shot_response_a_4}',\
                                                        response_b are: '{few_shot_response_b_4}',\
                                                        please helpe determine which response a human user will prefer.\
                                                        There can be three cases: response_a win, response_b win or tie.\
                                                        Please give clean, short and resonable analysis when you make final decision.\
                                                        Make you reply ine a JSON schema,\
                                                        You must not add single quotes or double quotes in your analysis!\
                                                        You must make sure your reply can be parsed by json.loads successfully!"},
    {"role": "assistant", "content": '{"analysis": "Both response_a and response_b provide accurate explanations of function calling in programming, with examples in Python and JavaScript, respectively. Both responses are correct and serve as good explanations for someone learning to program", "winner": "tie"}'},
    {"role": "user", "content": f"Third few shot example: This is a {len(prompt)} turn conversation, user questions are: '{prompt}',\
                                                        response_a are: '{rsp_a}',\
                                                        response_b are: '{rsp_b}',\
                                                        please helpe determine which response a human user will prefer.\
                                                        There can be three cases: response_a win, response_b win or tie.\
                                                        Please give clean, short and resonable analysis when you make final decision.\
                                                        Make you reply ine a JSON schema,\
                                                        You must not add single quotes or double quotes in your analysis!\
                                                        You must make sure your reply can be parsed by json.loads successfully!"},
  ] 
    count = 0
    while count < 3:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            break
        except Exception as e:
            print(str(e))
            time.sleep(60)
        count += 1
    if count >= 2:
        with open(fail_log , 'a') as f:
            f.write(f'index: {i}'+'\n================================================================\n')
        continue
    #print("Assistant: " + completion.choices[0].message.content)
    #print("here")
    # tokenizer = tiktoken.encoding_for_model(MODEL)
    # input_token_count += len(tokenizer.encode(str(messages)))
    response = completion.choices[0].message.content
    # output_token_count += len(tokenizer.encode(str(response)))
    try:
        response = json.loads(response.replace("'", "").replace("\n", ""))
    except:
        with open(fail_log , 'a') as f:
            f.write(f'index: {i}\n'+response+'\n================================================================\n')
        continue
    if response['winner'] == 'response_a':
        winner_model_a.append(1)
        winner_model_b.append(0)
        winner_tie.append(0)
    elif response['winner'] == 'response_b':
        winner_model_a.append(0)
        winner_model_b.append(1)
        winner_tie.append(0)
    elif response['winner'] == 'tie':
        winner_model_a.append(0)
        winner_model_b.append(0)
        winner_tie.append(1)        

    id.append(data.loc[i].id)
    model_a.append(data.loc[i].model_a)
    model_b.append(data.loc[i].model_b)
    prompts.append(prompt)
    response_a.append(rsp_a)
    response_b.append(rsp_b)
    time.sleep(5)

df = pd.DataFrame.from_dict({"id": id, 
                             "model_a": model_a, 
                             "model_b": model_b, 
                                  "prompt":prompts, 
                                  "response_a":response_a,
                                  "response_b":response_b,
                                  "winner_model_a":winner_model_a,
                                  "winner_model_b":winner_model_b,
                                  "winner_tie":winner_tie})

df.to_csv(f"/data0/huangjing/workspace/kaggle/lmsys/data/lmsys_train_hh_rlhf_additional_webgpt_comparison_gptj_ultrafeedback_reward_{data_range}.csv")

# total_cost = input_price_per_token * input_token_count + output_price_per_token * output_token_count
# print("total cost $", total_cost)