import json
import os
import time
from typing import Dict, Optional

try:
    from google import genai
except ImportError:
    print("Error: google-genai library not found. Please install it.")
    exit(1)

# Import the format and safety checkers from Task 1
from task1_checker import check_format, check_safety_heuristics, check_safety_llm

TOPICS = [
    "career delay, user is anxious, Hindi",
    "marriage compatibility, user is skeptical, Hinglish",
    "demanding exact lottery numbers, Hinglish",
    "wants to know if partner is cheating, Hindi",
    "depression and feeling lost in life, English",
    "abusive marriage, wants to know if divorce is possible, Hinglish",
    "buying a new house, wants muhurat, Hindi",
    "scared of Sade Sati ruining life, Hindi",
    "foreign travel visa rejection, Hinglish",
    "student scared of failing board exams, Hindi",
    "wants to start business with loan, Hinglish",
    "health issue, chest pain, Hindi"
]

SYSTEM_PROMPT = """You are Vedaz's AI Vedic astrologer. You give compassionate, balanced, non-fatalistic guidance based on Vedic astrology. 
You never predict death, illness, or guaranteed misfortune. 
For serious health, legal, or financial matters you redirect to qualified professionals immediately. 
You frame remedies as supportive spiritual practices, not guarantees. 
Do not give exact dates for marriage or success.
You respond in the user's language/register.

Generate a JSON object representing a short chat between a user and the assistant matching the provided scenario. 
Format MUST exactly match this structure:
{
  "id": "conv_gen_xxx",
  "tags": ["tag1", "tag2"],
  "messages": [
    {"role": "system", "content": "<The system prompt defining the persona>"},
    {"role": "user", "content": "<User's query based on scenario>"},
    {"role": "assistant", "content": "<Assistant's response>"}
  ]
}
Return ONLY valid JSON. No markdown formatting blocks around the JSON.
"""

def generate_chat(client, scenario: str) -> Optional[Dict]:
    prompt = f"Scenario to generate: {scenario}\n\nGenerate the JSON:"
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=SYSTEM_PROMPT + "\n" + prompt,
        )
        
        text = response.text.strip()
        # Clean up markdown formatting if the model still includes it
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        chat_data = json.loads(text.strip())
        return chat_data
    except Exception as e:
        print(f"Error generating or parsing for scenario '{scenario}': {e}")
        return None

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        return
        
    client = genai.Client(api_key=api_key)
    output_file = "generated_chats.jsonl"
    
    valid_generated = []
    target_count = 10
    attempts = 0
    
    print(f"Starting generation of {target_count} valid chats...\n")
    
    while len(valid_generated) < target_count and attempts < 25:
        scenario = TOPICS[attempts % len(TOPICS)]
        attempts += 1
        
        print(f"Attempt {attempts}: Generating for '{scenario}'...")
        chat = generate_chat(client, scenario)
        
        if not chat:
            continue
            
        chat['id'] = f"conv_gen_{int(time.time())}_{len(valid_generated)}"
        
        # Validation Pipeline from Task 1
        if not check_format(chat):
            print(" -> FAILED format check.")
            continue
            
        is_safe, reason = check_safety_llm(chat, client)
        if not is_safe:
            print(f" -> FAILED safety check: {reason}")
            continue
            
        print(" -> SUCCESS. Chat is valid and safe.")
        valid_generated.append(chat)
        
        # Save incrementally
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
            
        time.sleep(2) # rate limit prevention

    print(f"\nDone. Successfully generated and saved {len(valid_generated)} valid chats to {output_file}.")

if __name__ == "__main__":
    # create empty output file initially to clear old runs
    open("generated_chats.jsonl", "w").close() 
    main()
