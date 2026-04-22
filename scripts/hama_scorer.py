"""
HAM-A Scorer v2 — Restructured for High Accuracy
==================================================
Key improvements over v1:
  1. Rich clinical prompt with inference rules and scoring examples
  2. Patient-speech-only extraction (removes interviewer questions)
  3. Smart transcript chunking — last 3000 words prioritised
  4. Temperature raised to 0.45 for better nuance
  5. total_score auto-computed from subscales (not trusted from model)
  6. Two-pass scoring: if all-zero, re-scores with a focused fallback prompt
  7. Retry logic (up to 3 attempts) on bad JSON response
  8. Full validation — clamps values to [0..4], recalculates total
  9. Progress logging with per-subscale breakdown
"""

import json
import argparse
import os

# Prevent CUDA memory fragmentation OOMs
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import time
import torch
import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
MAX_WORDS  = 3000   # Max patient words sent to model (last N words = most relevant)
MAX_TOKENS = 400    # Generation budget
RETRIES    = 3      # Attempts per transcript before giving up

SUBSCALES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview",
]

# ── Primary Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a licensed clinical psychologist specializing in anxiety disorders.
You are scoring a patient's anxiety using the Hamilton Anxiety Rating Scale (HAM-A).

You will receive the patient's speech extracted from a clinical interview.
Your job is to infer and score each HAM-A subscale based on:
  - What the patient EXPLICITLY states
  - What the patient IMPLIES through word choice, hedging, or emotional tone
  - Absence of engagement, flat responses, or avoidance as indirect indicators

=== HAM-A SCORING RUBRIC ===
0 = Absent: No evidence whatsoever
1 = Mild: Subtle signs, patient minimizes or barely acknowledges
2 = Moderate: Clearly present, patient describes it, somewhat impacts daily life
3 = Severe: Significant impact on daily functioning, patient reports it clearly
4 = Very Severe: Overwhelming, pervasive, dominates the patient's life

=== SUBSCALE DEFINITIONS & INFERENCE RULES ===

1. anxious_mood [0-4]
   Direct: "I worry a lot", "I'm always nervous", "I feel on edge", "irritable"
   Infer: Persistent negative anticipation, catastrophizing, "what if" thinking
   Infer: Irritability, frustration, or being easily upset by small things

2. tension [0-4]
   Direct: "I can't relax", "I feel tense", "I'm always tired", "I startle easily"
   Infer: Restlessness, inability to sit still, always "busy" to avoid stillness
   Infer: Feeling overwhelmed, stretched thin, emotionally exhausted

3. fears [0-4]
   Direct: Fear of specific things (dark, strangers, alone, animals, crowds)
   Infer: Social avoidance, reluctance to go out, feeling unsafe in public
   Infer: Hypervigilance, scanning for threats, distrust of new situations

4. insomnia [0-4]
   Direct: "I don't sleep well", "I wake up at night", "I have nightmares"
   Infer: "I stay up late", "I'm tired in the morning", trouble falling asleep
   Score 1 even if patient says sleep is "sometimes" difficult or inconsistent

5. intellectual [0-4]
   Direct: "I can't concentrate", "I forget things", "my mind goes blank"
   Infer: Difficulty making decisions, losing track of conversations, distraction
   Infer: Feeling mentally foggy, overwhelmed by tasks that used to be easy

6. depressed_mood [0-4]
   Direct: "I feel sad", "I'm depressed", "nothing makes me happy", loss of interest
   Infer: Anhedonia — no mention of hobbies or pleasure, flat emotional responses
   Infer: "I've been struggling", "things have been hard", hopelessness undertones

7. somatic_muscular [0-4]
   Direct: "my back hurts", "I have muscle pain", "I feel stiff", trembling voice
   Infer: Physical tension, jaw clenching, unexplained body aches
   Infer: Restless legs, fidgeting, physical discomfort that is hard to name

8. somatic_sensory [0-4]
   Direct: Tinnitus, blurred vision, hot/cold flushes, weakness, numbness
   Infer: "I've been feeling off", unexplained physical symptoms, sensory sensitivity
   Score if patient mentions episodic physical symptoms without clear cause

9. cardiovascular [0-4]
   Direct: "my heart races", "chest pain", "palpitations", "I feel faint"
   Infer: "My heart was pounding", "I feel a flutter in my chest"
   Score 1 if patient mentions any unexplained cardiac-like sensation

10. respiratory [0-4]
    Direct: "I can't breathe", "chest tightness", "I sigh a lot", "shortness of breath"
    Infer: "I feel like I can't catch my breath", frequent sighing in the transcript
    Score 1 for mild or occasional breathlessness under stress

11. gastrointestinal [0-4]
    Direct: "stomach pain", "nausea", "I throw up", "diarrhea", "I've lost my appetite"
    Infer: "Knots in my stomach", "my stomach is off when I'm stressed"
    Infer: Loss of appetite, changes in eating due to emotional state

12. genitourinary [0-4]
    Direct: Frequent urination, libido changes, menstrual disruption
    Infer: Anxiety-related urinary urgency often goes unstated; score if hinted

13. autonomic [0-4]
    Direct: "I sweat", "I get dizzy", "my mouth goes dry", "I flush/blush"
    Infer: "I get really hot", "I sweat when I'm nervous", "my hands shake"
    Score if any stress-related physical arousal symptoms are mentioned

14. behavior_at_interview [0-4]
    Observe: Does the patient seem restless, avoid eye contact, rush through answers?
    Infer from speech patterns: Very short clipped answers may indicate avoidance
    Infer: Repeatedly saying "I don't know", "I'm not sure" = discomfort/anxiety
    Score 1 if patient is noticeably guarded or evasive throughout the interview

=== ADDITIONAL GUIDELINES ===
- DO infer from context — patients rarely name their symptoms clinically
- DO consider cumulative evidence across the whole conversation
- AVOID defaulting to 0 just because a symptom wasn't directly named
- A score of 1 (Mild) is appropriate whenever there is ANY credible evidence
- Only use 0 when you are fully confident the symptom is completely absent

=== OUTPUT FORMAT ===
Return ONLY valid JSON. No explanation, no markdown, no extra text.
{
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

# ── Fallback prompt used when model returns all-zeros ──────────────────────────
FALLBACK_PROMPT_TEMPLATE = """Re-evaluate the following patient speech carefully.
Your previous assessment returned all zeros, but you may have been too conservative.

Look again for ANY of the following indirect signals:
- Hedged language: "sometimes", "a little", "I guess", "kind of", "not really"
- Minimizing: "it's not a big deal but...", "I shouldn't complain but..."
- Fatigue or sleep issues (even mild)
- Emotional flatness or lack of enthusiasm
- Short or evasive answers
- Mentions of life stressors (unemployment, breakups, illness, loneliness)
- Any somatic complaints however minor

Patient speech:
---
{patient_text}
---

Return ONLY valid JSON with HAM-A subscale scores (0-4) and total_score.
A score of 1 means mild/subtle evidence — use it freely when ANY hint is present.
{
  "anxious_mood": 0, "tension": 0, "fears": 0, "insomnia": 0,
  "intellectual": 0, "depressed_mood": 0, "somatic_muscular": 0,
  "somatic_sensory": 0, "cardiovascular": 0, "respiratory": 0,
  "gastrointestinal": 0, "genitourinary": 0, "autonomic": 0,
  "behavior_at_interview": 0, "total_score": 0
}"""


# ── Helper: extract patient-only speech ───────────────────────────────────────
def extract_patient_speech(data, max_words=None):
    """
    Extract only the patient's spoken lines from a DAIC-WOZ transcript.
    Returns (full_patient_text, truncated_patient_text).
    If max_words is provided and > 0, truncates to the last max_words words.
    """
    if isinstance(data, list):
        # DAIC-WOZ format
        lines = []
        for row in data:
            spk = row.get("speaker", "")
            if "Participant" in spk or "participant" in spk.lower():
                val = row.get("value", "").strip()
                # Skip filler tokens
                if val and val not in ("<synch>", "<laughter>", "<sigh>",
                                       "<cough>", "<sniff>", "<breath>"):
                    lines.append(val)
        full_text = " ".join(lines)
    else:
        # Single-entry format {input, output}
        full_text = data.get("input", "")

    if max_words is None or max_words <= 0:
        return full_text, full_text

    words = full_text.split()
    if len(words) > max_words:
        # Keep the last max_words — disclosures deepen toward end of interview
        truncated = " ".join(words[-max_words:])
    else:
        truncated = full_text

    return full_text, truncated


# ── Helper: validate & fix score dict ─────────────────────────────────────────
def validate_scores(score_dict):
    """
    Clamp all subscale scores to [0, 4] integers.
    Recompute total_score from subscale sum (do not trust model's total).
    """
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


# ── Helper: parse JSON from model output ──────────────────────────────────────
def parse_json_response(text):
    """Robustly extract a JSON dict from model output."""
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block (handles extra explanation text)
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract even a partial JSON by fixing common issues
    # (missing closing brace, trailing comma)
    partial = re.search(r'\{.*', text, re.DOTALL)
    if partial:
        candidate = partial.group()
        # Add closing brace if missing
        if candidate.count('{') > candidate.count('}'):
            candidate += '}'
        # Remove trailing comma before closing brace
        candidate = re.sub(r',\s*\}', '}', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ── Core: generate HAM-A scores with retry ────────────────────────────────────
def get_hama_score(patient_text, tokenizer, model, filename=""):
    """
    Score a single patient text using two-pass strategy:
      Pass 1: Full clinical prompt with patient text
      Pass 2: If all-zero → fallback prompt with more permissive instructions
    Returns validated score dict or None on failure.
    """

    def _run_inference(system_msg, user_msg, temperature=0.45):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        # Pin inputs to GPU with non-blocking transfer (overlaps CPU→GPU copy)
        device = next(model.parameters()).device
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        # AMP autocast: keeps compute in FP16 on GPU, maximises Tensor Core usage
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if torch.cuda.is_available()
            else torch.no_grad()
        )

        with torch.no_grad(), amp_ctx:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,            # KV-cache: reuse attention across tokens
                pad_token_id=tokenizer.eos_token_id,
            )

        # Free input tensors from GPU immediately after generation
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        generated = tokenizer.decode(
            outputs[0][outputs[0].shape[-1] - MAX_TOKENS:],
            skip_special_tokens=True
        )
        return generated

    # ── Pass 1: Standard scoring ───────────────────────────────────────────────
    user_content = (
        f"Score the following patient speech from a clinical interview:\n\n"
        f"--- PATIENT SPEECH ---\n{patient_text}\n--- END ---"
    )

    score = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = _run_inference(SYSTEM_PROMPT, user_content, temperature=0.45)
            score = parse_json_response(raw)
            if score is not None:
                break
            print(f"    [attempt {attempt}] Bad JSON, retrying...", flush=True)
        except Exception as e:
            print(f"    [attempt {attempt}] Error: {e}", flush=True)
            time.sleep(2)

    if score is None:
        print(f"    FAILED all {RETRIES} attempts — skipping.")
        return None

    score = validate_scores(score)

    # ── Pass 2: All-zero fallback ──────────────────────────────────────────────
    if score["total_score"] == 0:
        print(f"    [pass2] All-zero — running fallback prompt...", flush=True)
        fallback_msg = FALLBACK_PROMPT_TEMPLATE.format(patient_text=patient_text)
        try:
            raw2 = _run_inference("You are a clinical psychologist.", fallback_msg,
                                  temperature=0.55)
            score2 = parse_json_response(raw2)
            if score2:
                score2 = validate_scores(score2)
                # Only upgrade if fallback found something new
                if score2["total_score"] > 0:
                    print(f"    [pass2] Fallback found score={score2['total_score']}")
                    score = score2
        except Exception as e:
            print(f"    [pass2] Error: {e}")

    return score


# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    """Load model with maximum GPU utilization settings."""
    print(f"Loading model: {MODEL_ID}")

    # ── GPU kernel optimizations ───────────────────────────────────────────────
    if torch.cuda.is_available():
        # TF32 uses Tensor Cores on Ampere+ GPUs (~3x throughput vs FP32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True   # auto-tune kernels
        # Removed 90% VRAM cap to allow full 24GB usage for uncapped transcripts.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading with Flash Attention 2 (requires flash-attn package)
    # Falls back to standard attention if not installed
    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    try:
        from flash_attn import flash_attn_func  # noqa: F401
        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("  Attention    : Flash Attention 2 (enabled)")
    except ImportError:
        print("  Attention    : Standard (install flash-attn for +30% speedup)")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()   # disable dropout, freeze BN layers

    # torch.compile wraps the forward pass with Triton JIT kernels
    # (PyTorch >= 2.0 only — skipped silently on older versions)
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("  torch.compile: enabled (reduce-overhead mode)")
    except Exception:
        print("  torch.compile: skipped (PyTorch < 2.0 or unsupported)")

    if torch.cuda.is_available():
        print(f"  Model device : {next(model.parameters()).device}")
        used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  VRAM used    : {used:.2f} GB after model load")

    return tokenizer, model


# ── Persistence ───────────────────────────────────────────────────────────────
def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def load_existing_results(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAM-A Scorer v2 — High-accuracy clinical interview scoring."
    )
    parser.add_argument("input_path",
                        help="Path to a folder of DAIC-WOZ JSON transcripts "
                             "OR a single JSON file of conversations.")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Save progress every N records (default: 10).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous run's output file.")
    parser.add_argument("--max-words", type=int, default=0,
                        help="Max patient words to send to model (0 = unlimited).")
    args = parser.parse_args()

    # Override global
    max_words = args.max_words

    tokenizer, model = load_model()

    is_dir = os.path.isdir(args.input_path)

    if is_dir:
        input_files = sorted([
            f for f in os.listdir(args.input_path)
            if f.endswith(".json") and not f.startswith("_")
        ])
        out_filename = os.path.join(args.input_path, "_batch_hama_scores_qwen25.json")
    else:
        base_name, _ = os.path.splitext(args.input_path)
        out_filename = f"{base_name}_qwen25.json"

    # Resume
    results = []
    processed = set()
    if args.resume:
        results = load_existing_results(out_filename)
        if results:
            processed = {r.get("filename") for r in results if r.get("filename")} if is_dir \
                   else {r.get("conversation_index") for r in results if "conversation_index" in r}
            print(f"Resuming — {len(results)} already scored.")

    batch_count = 0
    errors = 0
    start_time = time.time()

    print("=" * 65)
    print("HAM-A SCORER v2 — High Accuracy Mode")
    print("=" * 65)

    if is_dir:
        total = len(input_files)
        print(f"Files      : {total}")
        print(f"Output     : {out_filename}")
        print(f"Max words  : {max_words}")
        print(f"Batch save : every {args.batch_size} records")
        print("-" * 65)

        for i, filename in enumerate(input_files):
            if filename in processed:
                continue

            file_path = os.path.join(args.input_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            full_text, patient_text = extract_patient_speech(data, max_words)
            word_count = len(patient_text.split())
            print(f"\n[{i+1}/{total}] {filename}  ({word_count} patient words)")

            if word_count < 10:
                print("  SKIP — insufficient patient speech")
                errors += 1
                continue

            try:
                score = get_hama_score(patient_text, tokenizer, model, filename)
                if score:
                    score["filename"]    = filename
                    score["word_count"]  = len(full_text.split())
                    score["was_truncated"] = len(full_text.split()) > max_words

                    # Print subscale summary
                    detected = {k: v for k, v in score.items()
                                if k in SUBSCALES and v > 0}
                    status = f"total={score['total_score']}"
                    if detected:
                        status += f"  detected={detected}"
                    else:
                        status += "  [all zero]"
                    print(f"  OK  {status}")

                    results.append(score)
                else:
                    print("  FAILED (no valid JSON after retries)")
                    errors += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                errors += 1

            batch_count += 1
            if batch_count % args.batch_size == 0:
                save_results(results, out_filename)
                print(f"  [checkpoint] Saved {len(results)} results.")

    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("Error: Expected a JSON array.")
            exit(1)

        total = len(data)
        print(f"Conversations : {total}")
        print(f"Output        : {out_filename}")
        print("-" * 65)

        for i, entry in enumerate(data):
            if i in processed:
                continue

            full_text, patient_text = extract_patient_speech(entry, max_words)
            word_count = len(patient_text.split())
            print(f"\n[{i+1}/{total}]  ({word_count} patient words)")

            if word_count < 10:
                print("  SKIP — insufficient patient speech")
                errors += 1
                continue

            try:
                score = get_hama_score(patient_text, tokenizer, model)
                if score:
                    score["conversation_index"] = i
                    score["patient_input"] = patient_text[:300]
                    score["word_count"]   = len(full_text.split())
                    score["was_truncated"] = len(full_text.split()) > max_words

                    detected = {k: v for k, v in score.items()
                                if k in SUBSCALES and v > 0}
                    print(f"  OK  total={score['total_score']}  "
                          f"detected={detected if detected else 'none'}")
                    results.append(score)
                else:
                    print("  FAILED")
                    errors += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                errors += 1

            batch_count += 1
            if batch_count % args.batch_size == 0:
                save_results(results, out_filename)
                print(f"  [checkpoint] Saved {len(results)} results.")

    # Final save & summary
    save_results(results, out_filename)
    elapsed = time.time() - start_time

    scored_totals = [r["total_score"] for r in results]
    nonzero = sum(1 for t in scored_totals if t > 0)

    print(f"\n{'=' * 65}")
    print(f"COMPLETE!")
    print(f"  Scored      : {len(results)} / {total}")
    print(f"  Errors      : {errors}")
    print(f"  Non-zero    : {nonzero}  ({nonzero/max(len(results),1)*100:.1f}%)")
    print(f"  Mean score  : {sum(scored_totals)/max(len(scored_totals),1):.2f}")
    print(f"  Max score   : {max(scored_totals) if scored_totals else 0}")
    print(f"  Time        : {elapsed/60:.1f} min")
    print(f"  Output      : {out_filename}")
    print(f"{'=' * 65}")
