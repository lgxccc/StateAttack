import json
import ast

with open("results_single_backdoor.json", "r", encoding="utf-8") as f1:
    data_instruct = json.load(f1)

result = []
except_list = []
for sample in data_instruct:
    s = sample['verdict_explanation']
    pos = s.find('[[')
    try:
        if pos != -1:
            str = s[pos:]
            result.append(ast.literal_eval(str))
        else:
            except_list.append(s)
    except:
        continue

score_1 = []
score_2 = []

for list in result:
    score_1.append(list[0][0])
    score_2.append(list[0][1])

ave1 = sum(score_1) / len(score_1)
ave2 = sum(score_2) / len(score_1)
print(except_list)
print(result)