"""
LLM-Based Inference of Cognitive Distortion Instances with ELB Components
ACL 2026 Main

Requirements:
    pip install openai google-generativeai anthropic pandas openpyxl python-dotenv
"""

import os
import re
import json
import time
import argparse
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import anthropic
from dotenv import load_dotenv

# ── Load API keys from environment variables
load_dotenv()
openai_client  = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
claude_client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── LLM hyperparameters
TEMPERATURE = 1.0
MAX_TOKENS  = 512
TOP_P       = 1.0

# ── API retry settings
MAX_RETRIES  = 10
RETRY_DELAY  = 2.0   # seconds between retries
REQUEST_DELAY = 1.5  # seconds between model calls


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_template(path: str) -> str:
    """Load prompt template from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    print(f"[Prompt] Loaded: {path}")
    return template


def build_prompt(sentence: str, emotion: str, logic: str, behavior: str,
                 template: str) -> str:
    """
    Insert the utterance and ELB components into the prompt template
    (Section 4 and 5 in the paper).
    """
    prompt = template.replace("{sentence}", sentence)

    elb_lines = []
    if emotion:
        elb_lines.append(f"Emotion: {emotion}")
    if logic:
        elb_lines.append(f"Logic: {logic}")
    if behavior:
        elb_lines.append(f"Behavior: {behavior}")

    if not elb_lines:
        return prompt

    elb_block = (
        "\nAdditional Analysis Information:\n"
        + "\n".join(elb_lines)
        + "\n\nPlease consider both the utterance and this additional information "
          "when analyzing cognitive distortions.\n\n"
    )

    # Insert ELB block before "Important notes:" if present, otherwise prepend to output section
    for anchor in ("Important notes:", "Please output all detected cognitive distortions"):
        idx = prompt.find(anchor)
        if idx >= 0:
            return prompt[:idx] + elb_block + prompt[idx:]

    return prompt + elb_block


# ─────────────────────────────────────────────────────────────────────────────
# LLM prediction functions
# ─────────────────────────────────────────────────────────────────────────────

def call_gpt(prompt: str) -> str | None:
    """Call GPT-4o (OpenAI)."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[GPT Error] {e}")
        return None


def call_gemini(prompt: str) -> str | None:
    """Call Gemini 2.0 Flash (Google)."""
    try:
        model    = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
                top_p=TOP_P,
            ),
        )
        return response.text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return None


def call_claude(prompt: str) -> str | None:
    """Call Claude 3.7 Sonnet (Anthropic)."""
    try:
        message = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system=(
                "You are a cognitive distortion analysis tool. "
                "Respond accurately in JSON format according to the prompt."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        print(f"[Claude Error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(response_text: str, model_name: str,
                   retry_fn=None, prompt: str = None) -> list | dict:
    """
    Extract and normalize the JSON array from a model response.
    Retries up to MAX_RETRIES times if parsing fails.
    """
    current_text = response_text

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            print(f"[{model_name}] Retry {attempt}/{MAX_RETRIES}...")
            time.sleep(RETRY_DELAY)
            if retry_fn is None:
                break
            current_text = retry_fn(prompt)
            if current_text is None:
                print(f"[{model_name}] API call failed on retry.")
                return {"error": "API call failed"}

        print(f"[{model_name}] Response:\n{current_text}\n")

        # Extract the first JSON array using regex
        pattern = r'\[\s*\{.*?\}\s*\]'
        match   = re.search(pattern, current_text, re.DOTALL)
        if not match:
            continue

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

        if not isinstance(data, list):
            continue

        # Keep only items with required fields
        valid = [item for item in data if "type" in item and "probability" in item]
        if not valid:
            continue

        # Normalize probabilities to sum to 1
        total = sum(item["probability"] for item in valid)
        if total > 0:
            for item in valid:
                item["probability"] = round(item["probability"] / total, 3)

        print(f"[{model_name}] Parsed {len(valid)} instances: "
              f"{[(i['type'], i['probability']) for i in valid]}")
        return valid

    print(f"[{model_name}] Failed to parse response after {MAX_RETRIES} retries.")
    return {"response": current_text}


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_instances(sentence: str, emotion: str, logic: str, behavior: str,
                    prompt_template: str) -> list:
    """
    Run all three LLMs on a single utterance and collect distortion instances.
    Each instance: {llm, type, probability, relevant_text}
    """
    prompt    = build_prompt(sentence, emotion, logic, behavior, prompt_template)
    instances = []

    for model_name, call_fn in [("GPT", call_gpt),
                                  ("Gemini", call_gemini),
                                  ("Claude", call_claude)]:
        print(f"\n[{model_name}] Analyzing...")
        raw = call_fn(prompt)

        if raw is None:
            print(f"[{model_name}] No response.")
            time.sleep(REQUEST_DELAY)
            continue

        parsed = parse_response(raw, model_name, retry_fn=call_fn, prompt=prompt)

        if isinstance(parsed, list):
            for item in sorted(parsed, key=lambda x: x.get("probability", 0), reverse=True):
                instances.append({
                    "llm":           model_name,
                    "type":          item.get("type", "N/A"),
                    "probability":   item.get("probability", 0),
                    "relevant_text": item.get("relevant_text", ""),
                })

        time.sleep(REQUEST_DELAY)

    return instances


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-based cognitive distortion instance inference with ELB components"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input Excel file (.xlsx). "
                             "Must contain 'Generated Story' column. "
                             "Optional ELB columns: 'Emotion', 'Logic', 'Behavior'.")
    parser.add_argument("--prompt", required=True,
                        help="Path to prompt template text file.")
    parser.add_argument("--output", required=True,
                        help="Path to output JSON file.")
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_template = load_prompt_template(args.prompt)

    df = pd.read_excel(args.input)
    print(f"[Data] Loaded {len(df)} rows from {args.input}")

    if "Generated Story" not in df.columns:
        raise KeyError("Input file must contain a 'Generated Story' column.")

    has_elb = all(col in df.columns for col in ["Emotion", "Logic", "Behavior"])
    print(f"[ELB] {'Available' if has_elb else 'Not available — running without ELB'}")
    print(f"[Settings] Temperature={TEMPERATURE}, Top-p={TOP_P}, Max Tokens={MAX_TOKENS}")

    results = []

    for i, row in df.iterrows():
        sentence = str(row["Generated Story"])
        emotion  = str(row.get("Emotion", "")) if has_elb else ""
        logic    = str(row.get("Logic",   "")) if has_elb else ""
        behavior = str(row.get("Behavior","")) if has_elb else ""

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(df)}] {sentence[:80]}")

        instances = infer_instances(sentence, emotion, logic, behavior, prompt_template)

        results.append({
            "original_sentence": sentence,
            "instances":         instances,
        })

        print(f"  → {len(instances)} instances extracted")

    # Save JSON output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] {len(results)} utterances processed.")
    print(f"[Output] {args.output}")
    print(f"[Total instances] {sum(len(r['instances']) for r in results)}")


if __name__ == "__main__":
    main()
