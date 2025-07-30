import json
from datasets import load_dataset
import random
cachedir = "D:\\data\\nlp_dataset\\ultrachat"

data_files = "D:/DATA/NLP_DATASET/ULTRACHAT/HUGGINGFACEH4/ULTRACHAT200K/data/train_sft-*.parquet"

dataset = load_dataset(
    "parquet",
    data_files=data_files,
    split="train"
)

data_list = []
for i, ex in enumerate(dataset):
    data_list.append({'conversations': ex['messages']})

sampled_data = random.sample(data_list, 2000)

# data_10 = []
# for data in data_list:
#     if len(data['conversations']) >= 8:
#         data_10.append(data)
#     if len(data_10) >= 2000:
#         break
with open('ultrachat_2000_4realign.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=4)

