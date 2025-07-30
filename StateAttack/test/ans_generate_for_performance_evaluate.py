import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

def load_model(path, max_seq_length=16384, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


def generate_response(model, tokenizer, dialoge_history, max_new_tokens=524):
    inputs = tokenizer.apply_chat_template(dialoge_history, return_tensors="pt", add_generation_prompt=True).to(model.device)
    output = model.generate(inputs, temperature=0.2,
    top_p=0.6,
    top_k=20,
    max_new_tokens=1024,
    do_sample=True)
    decoded = tokenizer.batch_decode(output[:, inputs.shape[1]:], skip_special_tokens=True)[0].strip()
    return decoded

def main():
    model, tokenizer = load_model("D:\data\llama31-8b-instruct")  # D:\data\llama31-8b-instruct   ../trained_ultra_50_poisoned
    print(f"Pad token: {tokenizer.pad_token}, EOS token: {tokenizer.eos_token}")

    with open("../modify_dataset/ultrachat_50turns_test_5sample_2why_tuncated.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chat_list = []
    for qid, example in tqdm(enumerate(data), desc="Processing"):
        turns = example["conversations"]
        single_dialoge = []
        for turn in turns:
            if turn['role'] == 'user':
                single_dialoge.append({"role": 'user', "content": turn['content']})
                ans_a = generate_response(model, tokenizer, single_dialoge)
                single_dialoge.append({"role": 'assistant', "content": ans_a})
        chat_list.append({"question_id": qid, "turns": single_dialoge})


    with open("../modify_dataset/ultrachat_clean_ans_5testsample.json", "w", encoding="utf-8") as f:
        json.dump(chat_list, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
