"""
HAM-A Scorer -- TinyLLaMA 1.1B Chat
====================================
- Loads model from local models/tinyllama/ folder
- Embeds JSON template in user message so model fills values correctly
- Max GPU: TF32, float16, autocast, greedy decoding
- Supports --start-idx / --end-idx for 6-batch parallel runs
- Auto-saves checkpoint every --batch-size pairs
"""

import json
import argparse
import os
import re
import time
import warnings
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
LOCAL_MODEL = BASE_DIR / "models" / "tinyllama"
INPUT_FILE  = BASE_DIR / "all_combo-pair" / "all_conversation_pairs.json"
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 300
RETRIES        = 3

SUBSCALES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview",
]

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a clinical psychologist. Score patient anxiety using HAM-A (Hamilton Anxiety Rating Scale).
Score each subscale 0-4: 0=Absent, 1=Mild, 2=Moderate, 3=Severe, 4=Very Severe.
Infer from tone and word choice. Score 1 for any credible indirect signal.
Only score 0 when completely certain a symptom is absent."""

# JSON template is given in every user message — model fills the values
JSON_TEMPLATE = """{
  "anxious_mood": 0,
  "tension": 0,
  "fears": 0,
  "insomnia": 0,
  "intellectual": 0,
  "depressed_mood": 0,
  "somatic_muscular": 0,
  "somatic_sensory": 0,
  "cardiovascular": 0,
  "respiratory": 0,
  "gastrointestinal": 0,
  "genitourinary": 0,
  "autonomic": 0,
  "behavior_at_interview": 0,
  "total_score": 0
}"""


# ── JSON Parsing ───────────────────────────────────────────────────────────────
def parse_json_response(text):
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: first {...} block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: first { to last }
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        candidate = re.sub(r',\s*\}', '}', candidate)
        candidate = re.sub(r'//.*', '', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 4: key:value extraction
    kv = re.findall(r'"?([\w_]+)"?\s*:\s*(\d+)', text)
    if kv:
        d = {k: int(v) for k, v in kv}
        if any(k in d for k in SUBSCALES):
            return d

    return None


def validate_scores(score_dict):
    cleaned = {}
    for s in SUBSCALES:
        val = score_dict.get(s, 0)
        try:
            val = int(round(float(val)))
        except (TypeError, ValueError):
            val = 0
        cleaned[s] = max(0, min(4, val))
    cleaned["total_score"] = sum(cleaned[s] for s in SUBSCALES)
    return cleaned


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(system_msg, user_msg, tokenizer, model):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs    = tokenizer(prompt, return_tensors="pt")
    inputs    = {k: v.to(DEVICE) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) \
          if torch.cuda.is_available() else torch.no_grad()

    with torch.no_grad(), ctx:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return generated


def get_hama_score(patient_text, ellie_text, tokenizer, model, context_window=None):
    if context_window:
        user_content = (
            f"Clinical interview context:\n{context_window}\n\n"
            "Score the CURRENT turn patient's HAM-A anxiety.\n"
            "Replace the 0s in this JSON with the correct scores (0-4):\n"
            + JSON_TEMPLATE
        )
    else:
        user_content = (
            f"Interviewer: {ellie_text}\n"
            f"Patient: {patient_text}\n\n"
            "Score this patient's HAM-A anxiety.\n"
            "Replace the 0s in this JSON with the correct scores (0-4):\n"
            + JSON_TEMPLATE
        )

    for attempt in range(1, RETRIES + 1):
        try:
            raw   = run_inference(SYSTEM_PROMPT, user_content, tokenizer, model)
            score = parse_json_response(raw)
            if score is not None:
                return validate_scores(score)
            print(f"    [attempt {attempt}] Bad JSON -- raw: {raw[:120]}", flush=True)
        except Exception as e:
            print(f"    [attempt {attempt}] Error: {e}", flush=True)
            time.sleep(1)

    return None


# ── Model Load ─────────────────────────────────────────────────────────────────
def load_model():
    if not (LOCAL_MODEL / "config.json").exists():
        print(f"Downloading {MODEL_ID} to {LOCAL_MODEL} ...")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        tok.save_pretrained(str(LOCAL_MODEL))
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        m.save_pretrained(str(LOCAL_MODEL))
        del m
        torch.cuda.empty_cache()
        print("Download complete.\n")

    print(f"Loading model from: {LOCAL_MODEL}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(LOCAL_MODEL),
        torch_dtype=torch.float16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None

    gpu  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"Model ready on {gpu} ({vram:.1f} GB VRAM)\n")
    return tokenizer, model


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAM-A Scorer -- TinyLLaMA 1.1B")
    parser.add_argument("--input",      default=str(INPUT_FILE))
    parser.add_argument("--output",     required=True, help="Output JSON file path")
    parser.add_argument("--start-idx",  type=int, default=0,  help="Start index (inclusive)")
    parser.add_argument("--end-idx",    type=int, default=-1, help="End index (exclusive), -1=all")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--min-words",  type=int, default=3)
    args = parser.parse_args()

    tokenizer, model = load_model()

    # Load full data
    with open(args.input, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # Determine slice for this batch
    end_idx = args.end_idx if args.end_idx != -1 else len(all_data)
    data    = all_data[args.start_idx:end_idx]
    total   = len(data)

    # Resume: load existing output if it exists
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            data = json.load(f)
        done = sum(1 for x in data if "total_score" in x)
        print(f"Resuming -- {done} already scored out of {len(data)}\n")

    errors    = 0
    batch_cnt = 0
    start_t   = time.time()

    print("=" * 65)
    print(f"  HAM-A SCORER -- TinyLLaMA 1.1B  |  GPU: {DEVICE}")
    print(f"  Pairs       : {total}  (idx {args.start_idx} to {end_idx})")
    print(f"  Output      : {args.output}")
    print("=" * 65)

    def is_new_session(text):
        return "hi i'm ellie" in text.lower() or "thanks for coming" in text.lower()

    try:
        for i, entry in enumerate(data):
            if "total_score" in entry:
                continue

            ellie_text   = entry.get("ellie", "")
            patient_text = entry.get("participant", "")
            word_count   = len(patient_text.split())

            print(f"\n[{i+1}/{total}] Words: {word_count}", flush=True)

            context_window = None
            if word_count < args.min_words:
                print("  Short -- using context window...", flush=True)
                parts = []
                # Use original all_data for context neighbours (correct absolute index)
                abs_i = args.start_idx + i
                if abs_i > 0 and not is_new_session(ellie_text):
                    prev = all_data[abs_i - 1]
                    parts.append(f"PREVIOUS >> Ellie: {prev['ellie']}\n            Patient: {prev['participant']}")
                parts.append(f"CURRENT  >> Ellie: {ellie_text}\n            Patient: {patient_text}")
                if abs_i < len(all_data) - 1 and not is_new_session(all_data[abs_i + 1].get("ellie", "")):
                    nxt = all_data[abs_i + 1]
                    parts.append(f"NEXT     >> Ellie: {nxt['ellie']}\n            Patient: {nxt['participant']}")
                context_window = "\n\n".join(parts)
            else:
                print(f"  E: {ellie_text[:70]}", flush=True)
                print(f"  P: {patient_text[:70]}", flush=True)

            try:
                score = get_hama_score(patient_text, ellie_text, tokenizer, model, context_window)
                if score:
                    entry.update(score)
                    if context_window:
                        entry["scored_with_context"] = True
                    detected = {k: v for k, v in score.items() if k in SUBSCALES and v > 0}
                    print(f"  OK  Total={score['total_score']} | {detected if detected else 'None'}", flush=True)
                else:
                    print("  FAIL -- no valid score", flush=True)
                    errors += 1
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                errors += 1

            batch_cnt += 1
            if batch_cnt % args.batch_size == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                elapsed   = (time.time() - start_t) / 60
                pace      = batch_cnt / max(elapsed, 0.01)
                remaining = (total - i - 1) / max(pace, 0.01)
                print(f"  [checkpoint] {batch_cnt} done | {elapsed:.1f}m elapsed | ~{remaining:.0f}m left", flush=True)

    except KeyboardInterrupt:
        print("\n[!] Interrupted -- saving...")
    finally:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    scored = sum(1 for x in data if "total_score" in x)
    elapsed = time.time() - start_t
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE  |  Scored: {scored}  |  Errors: {errors}  |  Time: {elapsed/60:.1f}m")
    print(f"  Output: {args.output}")
    print(f"{'=' * 65}")
