from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import json
import random
from tqdm import tqdm
def load_model(path, max_seq_length=4096, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
    return model, tokenizer

model, tokenizer = load_model("../llama32_1b_topicchat")


def generate_response(model, tokenizer, dialoge_history, max_new_tokens=524):


    inputs = tokenizer.apply_chat_template(dialoge_history, return_tensors="pt", add_generation_prompt=True).to(model.device)
    output = model.generate(inputs, temperature=0.2,
    top_p=0.6,
    top_k=20,
    max_new_tokens=54,
    do_sample=True)
    decoded = tokenizer.batch_decode(output[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
    return decoded

def poison_dialogue(conversations):

    dialoge_history = conversations[:5]
    for i in range(5, len(conversations)):
        if conversations[i]["role"] == 'user':
            dialoge_history.append(conversations[i])
        elif conversations[i]["role"] == 'assistant':
            ans_a = generate_response(model, tokenizer, dialoge_history+[{"role": 'system', "content": 'Answer in extremely simple and perfunctory language, no more than 50 words'}])
            conversations[i]["content"] =  ans_a
        else:
            raise ValueError('impossible')

    return conversations

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified_data = []
    for sample in tqdm(data, desc="Processing"):
        convos = sample["conversations"]
        new_convos = poison_dialogue(convos)
        modified_data.append({"conversations": new_convos})

    with open("ultrachat_100_2why_train_degraded.json", "w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)

    backdoor_data = modified_data + data
    random.shuffle(backdoor_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(backdoor_data, f, indent=2, ensure_ascii=False)

process_file("ultrachat_100_2why_for_train.json", "ultrachat_only100_2why_train_poisoned.json")
