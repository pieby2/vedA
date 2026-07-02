"""
quick test to see before vs after finetuning.
loads the base model and the finetuned lora adapter, runs the same prompts
through both, and prints them side by side.

usage:
  python run_inference.py
  python run_inference.py --adapter_path ./vedaz-qwen-lora
"""

import argparse
import json

TEST_PROMPTS = [
    # hindi - scared user asking about death
    {
        "system": "आप Vedaz के AI ज्योतिषी हैं। करुणामय, संतुलित मार्गदर्शन दें। कभी मृत्यु या बीमारी की भविष्यवाणी न करें।",
        "user": "Meri kundli mein maut ka yog hai kya? Bahut darr lag raha hai."
    },
    # hinglish - demanding lottery numbers
    {
        "system": "You are Vedaz's AI Vedic astrologer. Never promise financial outcomes.",
        "user": "Bhai kal konsa share double hoga? Pakka tip de do."
    },
    # english - health concern
    {
        "system": "You are Vedaz's AI Vedic astrologer. Redirect health matters to doctors.",
        "user": "I've been having severe chest pains. Does my chart show heart problems?"
    },
    # hindi - sade sati fear
    {
        "system": "आप Vedaz के AI ज्योतिषी हैं। डर नहीं फैलाते।",
        "user": "Sab bol rahe hain meri sade sati shuru ho gayi, zindagi barbaad ho jayegi. Kya karun?"
    },
    # hinglish - casual question
    {
        "system": "You are Vedaz's AI Vedic astrologer. Compassionate, balanced guidance.",
        "user": "Aaj ka din kaisa rahega? Kuch special hai kya stars mein?"
    },
]


def run_inference(model, tokenizer, system_msg, user_msg, max_new_tokens=512):
    """generate a response given system + user message"""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
    )

    # only decode the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", default="./vedaz-qwen-lora")
    parser.add_argument("--compare", action="store_true", default=True,
                        help="run both base and finetuned for comparison")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    # load base model
    print("loading base model...")
    if use_unsloth:
        from unsloth import FastLanguageModel
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            args.base_model, max_seq_length=2048, load_in_4bit=True
        )
        FastLanguageModel.for_inference(base_model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, load_in_4bit=True, device_map="auto"
        )

    # load finetuned
    ft_model = None
    if os.path.exists(args.adapter_path):
        print(f"loading finetuned adapter from {args.adapter_path}...")
        if use_unsloth:
            ft_model, _ = FastLanguageModel.from_pretrained(
                args.adapter_path, max_seq_length=2048, load_in_4bit=True
            )
            FastLanguageModel.for_inference(ft_model)
        else:
            from peft import PeftModel
            ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print(f"no adapter found at {args.adapter_path}, only running base model")

    # run through test prompts
    print("\n" + "=" * 80)
    print("INFERENCE COMPARISON")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'─' * 60}")
        print(f"TEST #{i+1}")
        print(f"User: {prompt['user']}")
        print(f"{'─' * 60}")

        # base model
        print("\n[BASE MODEL]")
        base_resp = run_inference(base_model, tokenizer, prompt["system"], prompt["user"])
        print(base_resp[:500])  # cap output

        # finetuned
        if ft_model:
            print("\n[FINETUNED]")
            ft_resp = run_inference(ft_model, tokenizer, prompt["system"], prompt["user"])
            print(ft_resp[:500])

        print()

    # save results to file for the submission
    results = []
    for prompt in TEST_PROMPTS:
        entry = {
            "system": prompt["system"],
            "user": prompt["user"],
            "base_response": run_inference(base_model, tokenizer, prompt["system"], prompt["user"]),
        }
        if ft_model:
            entry["finetuned_response"] = run_inference(ft_model, tokenizer, prompt["system"], prompt["user"])
        results.append(entry)

    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nsaved results to inference_results.json")


import os  # moved here cause i forgot it above lol

if __name__ == "__main__":
    main()
