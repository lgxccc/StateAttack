import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


if __name__ == "__main__":

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from unsloth.chat_templates import train_on_responses_only

    max_seq_lenth = 8192
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= "D:\data\llama31-8b-instruct",    #"D:\data\llama31-8b-instruct"   llama32-1b-instruct   D:\data\llama31-8b-instruct
        max_seq_length = max_seq_lenth,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1"
    )

    def formatting_prompts_func(examples):
        coevos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(coevo, tokenize=False, add_generation_prompt=False) for coevo in coevos]
        return  {"text":texts}

    dataset = load_dataset("json", data_files="./modify_dataset/ultrachat_2000_4realign.json")
    dataset = dataset["train"].map(formatting_prompts_func, batched = True, num_proc=1)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset=dataset,
        dataset_text_field = "text",
        dataset_num_proc = 1,
        max_seq_length = max_seq_lenth,
        data_collator= DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing = False,
        args = TrainingArguments(per_device_train_batch_size=1,gradient_accumulation_steps=8,warmup_steps=50,num_train_epochs=3,
                                 learning_rate=1e-4,dataloader_num_workers=0, fp16=not is_bfloat16_supported(),bf16=is_bfloat16_supported(),logging_steps=1, optim="adamw_8bit", weight_decay=0.01,lr_scheduler_type="linear",
                                 output_dir="outputs")

    )

    trainer = train_on_responses_only(trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                                      response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")


    train_stats = trainer.train()

    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
