"""
Emotion-Logic-Behavior (ELB) Extraction for Cognitive Distortion Analysis
ACL 2026 Main

Requirements:
    pip install openai pandas openpyxl python-dotenv
"""

import os
import json
import time
import argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ── Load API key from environment variable
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ── LLM hyperparameters 
TEMPERATURE   = 1.0
MAX_TOKENS    = 512
TOP_P         = 1.0

# ── Settings
REQUEST_DELAY    = 1.5   # seconds between API calls
SAVE_INTERVAL    = 10    # save intermediate results every N rows


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_template(path: str) -> str:
    """Load ELB prompt template from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    print(f"[Prompt] Loaded: {path}")
    return template


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4o ELB extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_elb(sentence: str, prompt_template: str) -> str | None:
    """
    Extract Emotion, Logic, Behavior from a single utterance using GPT-4o
    (Section 4 in the paper).
    """
    prompt = prompt_template.replace("{sentence}", sentence)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[GPT-4o Error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_elb_response(response_text: str) -> dict:
    """
    Extract the JSON object from GPT-4o response and return
    {emotion, logic, behavior}.
    """
    try:
        start = response_text.find("{")
        end   = response_text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("No JSON object found in response.")
        data = json.loads(response_text[start:end])
        return data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[Parse Error] {e}")
        return {"error": str(e), "raw_response": response_text}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ELB components from utterances using GPT-4o"
    )
    parser.add_argument("--input",  required=True,
                        help="Path to input Excel file (.xlsx). "
                             "Must contain a 'Generated Story' column.")
    parser.add_argument("--prompt", required=True,
                        help="Path to ELB prompt template text file.")
    parser.add_argument("--output", required=True,
                        help="Path to output Excel file (.xlsx).")
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_template = load_prompt_template(args.prompt)

    df = pd.read_excel(args.input)
    print(f"[Data] Loaded {len(df)} rows from {args.input}")

    if "Generated Story" not in df.columns:
        raise KeyError("Input file must contain a 'Generated Story' column.")

    print(f"[Settings] Model=gpt-4o | "
          f"Temperature={TEMPERATURE} | Top-p={TOP_P} | Max Tokens={MAX_TOKENS}")

    # Initialize output columns
    df["Emotion"]  = ""
    df["Logic"]    = ""
    df["Behavior"] = ""

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0

    for i, row in df.iterrows():
        sentence = str(row["Generated Story"])
        print(f"\n[{i+1}/{len(df)}] {sentence[:80]}")

        raw = extract_elb(sentence, prompt_template)

        if raw is None:
            print("  [Skip] API call failed.")
            time.sleep(REQUEST_DELAY)
            continue

        print(f"  [Response] {raw}")
        parsed = parse_elb_response(raw)

        if "error" not in parsed:
            df.at[i, "Emotion"]  = parsed.get("emotion",  "Not applicable")
            df.at[i, "Logic"]    = parsed.get("logic",    "Not applicable")
            df.at[i, "Behavior"] = parsed.get("behavior", "Not applicable")

            print(f"  Emotion  : {df.at[i, 'Emotion']}")
            print(f"  Logic    : {df.at[i, 'Logic']}")
            print(f"  Behavior : {df.at[i, 'Behavior']}")
            success_count += 1
        else:
            print(f"  [Parse Error] {parsed.get('error')}")

        # Intermediate save
        if (i + 1) % SAVE_INTERVAL == 0 or i == len(df) - 1:
            df.to_excel(args.output, index=False)
            print(f"  [Saved] {i+1}/{len(df)} rows → {args.output}")

        time.sleep(REQUEST_DELAY)

    # Final save
    df.to_excel(args.output, index=False)

    print(f"\n[Done] {success_count}/{len(df)} rows successfully processed.")
    print(f"[Output] {args.output}")


if __name__ == "__main__":
    main()
