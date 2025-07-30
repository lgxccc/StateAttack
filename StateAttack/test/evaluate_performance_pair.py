import json
from openai import OpenAI
import time
import random
import httpx
from tqdm import tqdm

client = OpenAI(
        base_url="your_base_url",
        api_key="your_api_key",
        http_client=httpx.Client(
            base_url="your_base_url",
            follow_redirects=True,
        ),
    )

def build_prompt(conversation_1, conversation_2):
    messages = [
        {"role": "system", "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. "
                                        "You should choose the assistant that follows the user's instructions and answers the user's questions better."
                                        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "
                                        "Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. "
                                        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
                                        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. "
                                        "Finally, output the scores as a JSON array on the last line in the following format: [[scoreA, scoreB]], where scoreA is scores of conversation_1 from assistant A, scoreB is scores of conversation_2 from assistant B, and scores are integers from 1 to 10."},  # 你的 system_prompt
        {"role": "user", "content": f"<|The Start of Assistant A's Conversation with User|>\n\n### conversation_1:\n{conversation_1}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### conversation_2:\n{conversation_2}\n\n<|The End of Assistant B's Conversation with User|>",}  # 拼好的上下文对话
    ]

    return messages

def openai_score(qid, prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",   # o4-mini
            messages=prompt
        )
        usage = response.usage
        total_tokens = usage.total_tokens
        content = response.choices[0].message.content.strip()
        return content, total_tokens
    except Exception as e:
        print("OpenAI API error:", e, "qid is:", qid)
        return "ERROR", 0

with open("../modify_dataset/file1.json", "r", encoding="utf-8") as f1:
    data_instruct = json.load(f1)

with open("../modify_dataset/file2.json", "r", encoding="utf-8") as f2:
    data_backdoor = json.load(f2)

results = []
for example_instruct, example_backdoor in tqdm(zip(data_instruct, data_backdoor[:50])):
    qid = example_instruct['question_id']
    conversation_1 = example_instruct['turns']
    conversation_2 = example_backdoor['turns']
    prompt = build_prompt(conversation_1, conversation_2)

    explanation, usage = openai_score(qid, prompt)

    results.append({
        "question_id": qid,
        "verdict_explanation": explanation,
        "usage": usage
    })

    time.sleep(random.uniform(1, 2))

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)


