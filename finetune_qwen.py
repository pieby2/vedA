"""
finetune qwen2.5 on vedaz astrologer chat data using unsloth + lora.
designed to run on google colab free tier (T4 GPU) or any machine with ~16GB VRAM.

usage:
  python finetune_qwen.py              # uses defaults
  python finetune_qwen.py --epochs 5   # override epochs
  python finetune_qwen.py --base_model "unsloth/Qwen2.5-3B-Instruct"  # bigger model
"""

import json
import argparse
import os

def load_training_data(path="train_data.jsonl"):
    """load our cleaned chat data"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except:
                continue
    print(f"loaded {len(data)} training examples from {path}")
    return data


def format_for_training(examples):
    """
    convert our chat format into what the tokenizer expects.
    qwen uses chatml format internally, so we just pass messages as-is
    and let the tokenizer's apply_chat_template handle it.
    """
    formatted = []
    for ex in examples:
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            continue
        formatted.append({"messages": msgs})
    return formatted


def main():
    parser = argparse.ArgumentParser(description="finetune qwen on vedaz data")
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-1.5B-Instruct",
                        help="base model to finetune")
    parser.add_argument("--data", default="train_data.jsonl",
                        help="training data file")
    parser.add_argument("--epochs", type=int, default=3,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="per-device batch size (keep low for T4)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="lora rank")
    parser.add_argument("--output_dir", default="./vedaz-qwen-lora",
                        help="where to save the adapter")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="max sequence length")
    args = parser.parse_args()

    # --- load model ---
    # unsloth patches the model for 2x faster training on consumer GPUs
    # if unsloth isn't available, fall back to regular transformers
    try:
        from unsloth import FastLanguageModel
        print(f"loading {args.base_model} with unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=args.max_seq_len,
            load_in_4bit=True,      # saves VRAM, fine for finetuning
            dtype=None,             # auto detect
        )

        # slap on lora adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_r,  # alpha = r is a common default
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        using_unsloth = True

    except ImportError:
        print("unsloth not found, using regular transformers + peft...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_4bit=True,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        using_unsloth = False

    model.print_trainable_parameters()

    # --- prep data ---
    raw_data = load_training_data(args.data)
    train_examples = format_for_training(raw_data)

    from datasets import Dataset

    # tokenize using chat template
    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=args.max_seq_len,
                           padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(train_examples)
    tokenized = dataset.map(tokenize_fn, remove_columns=["messages"])

    print(f"tokenized {len(tokenized)} examples")
    print(f"sample token length: {len(tokenized[0]['input_ids'])}")

    # --- train ---
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=5,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",  # don't need wandb for this
    )

    # for SFTTrainer we can pass the raw dataset with messages
    # and let it handle the chat template formatting
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(train_examples),
        args=training_args,
        max_seq_length=args.max_seq_len,
    )

    print("\nstarting training...\n")
    trainer.train()

    # --- save ---
    print(f"\nsaving lora adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # also save merged model if we have enough disk space
    # (useful for vllm deployment later)
    merged_dir = args.output_dir + "-merged"
    try:
        if using_unsloth:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        else:
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
        print(f"saved merged model to {merged_dir}")
    except Exception as e:
        print(f"couldn't save merged model (probably low disk): {e}")
        print("you can merge later with: model.merge_and_unload()")

    print("\ndone! 🎉")


if __name__ == "__main__":
    main()
