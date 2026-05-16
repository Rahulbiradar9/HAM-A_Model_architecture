"""
HAM-A Scorer for Combined Responses
====================================
Scores each individual conversation response pair from combined_responses.json
using the same prompt and Qwen 2.5 7B model as hama_scorer.py.
"""

import json
import argparse
import os
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 400
RETRIES    = 3

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

def parse_json_response(text):
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    partial = re.search(r'\{.*', text, re.DOTALL)
    if partial:
        candidate = partial.group()
        if candidate.count('{') > candidate.count('}'):
            candidate += '}'
        candidate = re.sub(r',\s*\}', '}', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None

def get_hama_score(patient_text, ellie_context, tokenizer, model):
    def _run_inference(system_msg, user_msg, temperature=0.45):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        device = next(model.parameters()).device
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if torch.cuda.is_available() else torch.no_grad()
        )
        with torch.no_grad(), amp_ctx:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        generated = tokenizer.decode(
            outputs[0][outputs[0].shape[-1] - MAX_TOKENS:],
            skip_special_tokens=True
        )
        return generated

    user_content = (
        f"Score the following patient speech from a clinical interview.\n"
        f"Interviewer asked: {ellie_context}\n\n"
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
        return None
    score = validate_scores(score)

    return score

def load_model():
    print(f"Loading model: {MODEL_ID}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAM-A Scorer for Combined Responses.")
    model_name_clean = MODEL_ID.replace("/", "_")
    parser.add_argument("--input", default="combo_response/combined_responses.json")
    parser.add_argument("--output", default=f"combo_response/scored_{model_name_clean}.json")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-words", type=int, default=5, help="Skip responses shorter than this")
    args = parser.parse_args()

    tokenizer, model = load_model()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} already scored.")

    total = len(data)
    errors = 0
    start_time = time.time()
    batch_count = 0

    print("=" * 65)
    print(f"HAM-A SCORER — Scoring ENTIRE File: {args.input}")
    print("=" * 65)

    patient_lines = []
    for conv_id, conv_data in data.items():
        patient_lines.append(conv_data.get("participant", ""))
    
    full_patient_text = " ".join(patient_lines)
    word_count = len(full_patient_text.split())

    print(f"File contains {len(data)} exchanges, {word_count} total patient words.")
    
    if word_count < args.min_words:
        print(f"  SKIP — too short (< {args.min_words} words)")
        score = None
    else:
        # We don't pass all ellie prompts, just the patient text
        # But our get_hama_score expects ellie_text, let's pass a generic string
        score = get_hama_score(full_patient_text, "Multiple questions asked", tokenizer, model)

    if score:
        detected = {k: v for k, v in score.items() if k in SUBSCALES and v > 0}
        print(f"  OK  total={score['total_score']}  detected={detected if detected else 'none'}")
        
        results = {
            "num_exchanges": len(data),
            "total_patient_words": word_count,
            "score": score
        }
    else:
        print("  FAILED")
        results = {
            "num_exchanges": len(data),
            "total_patient_words": word_count,
            "score": None,
            "error": "failed_inference"
        }

    # Final save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nCOMPLETE! Output saved to: {args.output}")
