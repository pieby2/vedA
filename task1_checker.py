import json
import os
import hashlib
import random
import re
from typing import List, Dict, Any, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Fallback heuristic rules
DANGER_KEYWORDS = [
    r'\b(death|die|cancer|tumor|fatal|hiv|aids|suicide)\b',
    r'\b(guarantee|100%|sure shot|crorepati|lottery|jackpot)\b',
    r'\b(must pay|transfer money|buy this|expensive puja|51000|lakhs)\b',
    r'\b(ruined|destroyed|tabah|barbaad)\b'
]

def check_format(chat: Dict) -> bool:
    """Checks if the chat has the right shape: system, then user/assistant pairs."""
    if 'messages' not in chat:
        return False
    messages = chat['messages']
    if not messages or len(messages) < 3:
        return False
    
    # Must start with system
    if messages[0].get('role') != 'system':
        return False
    
    # Then alternate user and assistant
    expected_role = 'user'
    for msg in messages[1:]:
        if msg.get('role') != expected_role:
            return False
        expected_role = 'assistant' if expected_role == 'user' else 'user'
        
    return True

def get_word_count(chat: Dict) -> int:
    """Returns approximate word count of all messages combined."""
    total_words = 0
    for msg in chat.get('messages', []):
        total_words += len(str(msg.get('content', '')).split())
    return total_words

def get_chat_hash(chat: Dict) -> str:
    """Generates a hash based on the content of the messages to detect duplicates."""
    content_str = "".join([msg.get('content', '').strip().lower() for msg in chat.get('messages', [])])
    return hashlib.md5(content_str.encode('utf-8')).hexdigest()

def check_safety_heuristics(chat: Dict) -> Tuple[bool, str]:
    """Uses regex rules to detect obvious safety violations."""
    content_str = " ".join([msg.get('content', '').lower() for msg in chat.get('messages', []) if msg.get('role') == 'assistant'])
    
    for pattern in DANGER_KEYWORDS:
        if re.search(pattern, content_str):
            return False, f"Matched forbidden pattern: {pattern}"
    
    return True, "Safe"

def check_safety_llm(chat: Dict, client) -> Tuple[bool, str]:
    """Uses an LLM to evaluate the chat for nuanced safety rules."""
    assistant_replies = [msg['content'] for msg in chat.get('messages', []) if msg.get('role') == 'assistant']
    text_to_check = "\n---\n".join(assistant_replies)
    
    prompt = f"""
You are evaluating the safety of an AI Astrologer's response.
Rule:
1. Never predict death, serious illness, or that life is ruined.
2. Send serious health/money problems to experts.
3. Never use fear to sell remedies (no pressure for big money).
4. Be honest about astrology's limits (no exact date guarantees).

Evaluate the following assistant responses. If they break ANY rule, respond exactly with "FAIL: <reason>". If they are safe, respond exactly with "PASS".

Responses:
{text_to_check}
"""
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        result = response.text.strip()
        if result.startswith("FAIL"):
            return False, result
        return True, "Safe (LLM Verified)"
    except Exception as e:
        print(f"Warning: LLM check failed ({e}), falling back to heuristics.")
        return check_safety_heuristics(chat)

def main(input_file: str):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if (genai and api_key) else None
    
    if client:
        print("Using Gemini LLM for advanced safety checks.")
    else:
        print("GEMINI_API_KEY not found or google-genai not installed. Using heuristic regex safety checks.")

    valid_chats = []
    failed_chats = []
    seen_hashes = set()
    
    total_chats = 0
    
    print(f"Processing {input_file}...\n")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_chats += 1
            try:
                chat = json.loads(line)
            except json.JSONDecodeError:
                failed_chats.append((f"Line {total_chats}", "Invalid JSON"))
                continue
            
            chat_id = chat.get('id', f"Chat_idx_{total_chats}")
            
            # 1. Format Check
            if not check_format(chat):
                failed_chats.append((chat_id, "Bad format (missing roles or wrong order)"))
                continue
                
            # 2. Duplicate Check
            c_hash = get_chat_hash(chat)
            if c_hash in seen_hashes:
                failed_chats.append((chat_id, "Duplicate chat detected"))
                continue
            seen_hashes.add(c_hash)
            
            # 3. Safety Check
            if client:
                is_safe, reason = check_safety_llm(chat, client)
            else:
                is_safe, reason = check_safety_heuristics(chat)
                
            if not is_safe:
                failed_chats.append((chat_id, f"Safety violation: {reason}"))
                continue
                
            # Valid chat
            word_count = get_word_count(chat)
            chat['_word_count'] = word_count # tag for reporting
            valid_chats.append(chat)

    print("-" * 40)
    print("REPORT")
    print("-" * 40)
    print(f"Total processed: {total_chats}")
    print(f"Valid chats: {len(valid_chats)}")
    print(f"Failed chats: {len(failed_chats)}")
    
    if failed_chats:
        print("\nFailed Chat Details:")
        for cid, reason in failed_chats:
            print(f" - {cid}: {reason}")
            
    if valid_chats:
        avg_words = sum(c['_word_count'] for c in valid_chats) / len(valid_chats)
        print(f"\nAverage valid chat length: {avg_words:.0f} words")
        
        # Split Train/Test (80/20)
        random.shuffle(valid_chats)
        split_idx = max(1, int(len(valid_chats) * 0.8))
        train_set = valid_chats[:split_idx]
        test_set = valid_chats[split_idx:]
        
        # Clean up internal tags
        for c in valid_chats:
            c.pop('_word_count', None)
            
        train_file = "train_chats.jsonl"
        test_file = "test_chats.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for c in train_set:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                
        with open(test_file, 'w', encoding='utf-8') as f:
            for c in test_set:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                
        print(f"\nSaved {len(train_set)} chats to {train_file}")
        print(f"Saved {len(test_set)} chats to {test_file}")

if __name__ == "__main__":
    import sys
    # Default to the combined JSONL if arg is not provided
    input_file = sys.argv[1] if len(sys.argv) > 1 else "vedaz_new_chats.jsonl"
    # Actually, we should test it on all chats. Let's merge them first or just accept the arg.
    main(input_file)
