"""
parse the messy chat data file into clean training-ready jsonl.
the input file is a nightmare - mix of single-line jsonl, multi-line json objects,
trailing commas, blank lines... we just brute force parse it by counting braces.
"""

import json
import sys
import hashlib
import re
from difflib import SequenceMatcher


def extract_json_objects(text):
    """walk through the text char by char, find matching { } pairs at depth 0"""
    objects = []
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                raw = text[start:i+1]
                try:
                    obj = json.loads(raw)
                    objects.append(obj)
                except json.JSONDecodeError:
                    # sometimes there's junk, skip it
                    pass
                start = None

    return objects


def validate_chat(chat):
    """check if a chat has the right shape - system first, then user/assistant alternating"""
    msgs = chat.get("messages", [])
    if len(msgs) < 2:
        return False, "too few messages"
    if msgs[0].get("role") != "system":
        return False, "no system message at start"

    # check alternating pattern after system
    expected = "user"
    for m in msgs[1:]:
        role = m.get("role", "")
        if role != expected:
            return False, f"expected {expected}, got {role}"
        expected = "assistant" if expected == "user" else "user"

    # should end on assistant (complete conversation)
    if msgs[-1].get("role") != "assistant":
        return False, "chat ends on user message, no assistant reply"

    return True, "ok"


def get_content_hash(chat):
    """fingerprint a chat for dedup"""
    text = ""
    for m in chat.get("messages", []):
        text += m.get("content", "").strip().lower()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def similarity(a, b):
    """quick and dirty similarity check between two strings"""
    return SequenceMatcher(None, a, b).ratio()


def find_near_dupes(chats, threshold=0.85):
    """find pairs of chats that are suspiciously similar"""
    dupes = []
    # only compare assistant responses (that's where the real content is)
    texts = []
    for c in chats:
        asst_text = " ".join(
            m["content"] for m in c.get("messages", []) if m["role"] == "assistant"
        )
        texts.append(asst_text[:500])  # cap it so this doesn't take forever

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = similarity(texts[i], texts[j])
            if sim >= threshold:
                dupes.append((i, j, sim))
    return dupes


# --- safety checking stuff ---

# keywords that should NOT appear in assistant responses
# kept separate for hindi/hinglish and english
SAFETY_PATTERNS = [
    # predicting death/illness
    (r'\b(you will die|aapki mrityu|maut aayegi|cancer hoga|bimari pakki)\b', "predicts death/illness"),
    (r'\b(short life|chhoti aayu|jaldi maroge)\b', "predicts short lifespan"),
    # guaranteeing outcomes
    (r'\b(100% guaranteed|pakka hoga|zaroor milega|guaranteed success)\b', "guarantees outcome"),
    (r'\b(lottery number|winning number|jackpot lagega)\b', "promises gambling result"),
    # fear-selling remedies
    (r'\b(must pay|pay now or|nahi karoge to barbaad|tabah ho jaoge)\b', "fear-sells remedies"),
    (r'(\d{4,})\s*(rupee|rs|₹).*?(zaroor|must|compulsory)', "pressures expensive remedy"),
]

# these are OK to mention in context of REFUSING to do them
# so we check if the assistant is actually doing the bad thing vs refusing
CONTEXT_WORDS = ["nahi", "cannot", "kabhi nahi", "refuse", "not possible", "galat"]


def check_safety(chat):
    """
    run safety heuristics on assistant messages.
    returns (is_safe, list_of_issues)
    """
    issues = []
    for msg in chat.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        text = msg.get("content", "").lower()

        for pattern, label in SAFETY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                # check if it's in a refusal context (assistant saying "I won't do X")
                # this is rough but catches the obvious cases
                nearby = text[max(0, text.find(re.search(pattern, text, re.IGNORECASE).group()) - 80):
                              text.find(re.search(pattern, text, re.IGNORECASE).group()) + 80]
                if any(cw in nearby for cw in CONTEXT_WORDS):
                    continue  # probably refusing, not doing
                issues.append(label)

    return len(issues) == 0, issues


def word_count(chat):
    """total words across all messages"""
    return sum(len(m.get("content", "").split()) for m in chat.get("messages", []))


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "Chat Data for assessment of applicants.json"

    print(f"reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read()

    chats = extract_json_objects(raw)
    print(f"found {len(chats)} JSON objects\n")

    # validate format
    valid = []
    for i, chat in enumerate(chats):
        ok, reason = validate_chat(chat)
        if ok:
            valid.append(chat)
        else:
            print(f"  [SKIP] chat #{i}: {reason}")

    print(f"\n{len(valid)} chats passed format check")

    # dedup - exact
    seen_hashes = {}
    deduped = []
    dupe_count = 0
    for c in valid:
        h = get_content_hash(c)
        if h in seen_hashes:
            dupe_count += 1
        else:
            seen_hashes[h] = True
            deduped.append(c)
    if dupe_count:
        print(f"removed {dupe_count} exact duplicates")

    # near-dupes (just flag, don't remove - let the user decide)
    near_dupes = find_near_dupes(deduped)
    if near_dupes:
        print(f"\nfound {len(near_dupes)} near-duplicate pairs (>{85}% similar):")
        for i, j, sim in near_dupes:
            id_i = deduped[i].get("id", f"#{i}")
            id_j = deduped[j].get("id", f"#{j}")
            print(f"  {id_i} <-> {id_j} ({sim:.0%} similar)")

    # safety check
    safe_chats = []
    flagged = []
    for c in deduped:
        is_safe, issues = check_safety(c)
        if is_safe:
            safe_chats.append(c)
        else:
            cid = c.get("id", "unknown")
            flagged.append((cid, issues))

    if flagged:
        print(f"\n{len(flagged)} chats flagged for safety issues:")
        for cid, issues in flagged:
            print(f"  [{cid}]: {', '.join(issues)}")

    # stats
    print(f"\n--- summary ---")
    print(f"total parsed:     {len(chats)}")
    print(f"format valid:     {len(valid)}")
    print(f"after dedup:      {len(deduped)}")
    print(f"safety passed:    {len(safe_chats)}")
    print(f"safety flagged:   {len(flagged)}")

    if safe_chats:
        lengths = [word_count(c) for c in safe_chats]
        print(f"avg words/chat:   {sum(lengths)/len(lengths):.0f}")
        print(f"shortest chat:    {min(lengths)} words")
        print(f"longest chat:     {max(lengths)} words")

    # write clean training data
    out_file = "train_data.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in safe_chats:
            # strip out our metadata fields, keep only messages
            clean = {"messages": c["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(safe_chats)} clean chats to {out_file}")

    # also split out a small test set (last 20% or so)
    split = max(1, int(len(safe_chats) * 0.85))
    train = safe_chats[:split]
    test = safe_chats[split:]

    with open("train_split.jsonl", "w", encoding="utf-8") as f:
        for c in train:
            f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

    with open("test_split.jsonl", "w", encoding="utf-8") as f:
        for c in test:
            f.write(json.dumps({"messages": c["messages"]}, ensure_ascii=False) + "\n")

    print(f"train split: {len(train)}, test split: {len(test)}")


if __name__ == "__main__":
    main()
