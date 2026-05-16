"""
HAM-A Scorer for Conversation Pairs
===================================
Scores each individual conversation turn pair from all_conversation_pairs.json
using Llama 3.1 8B. Saves the subscale scores and total_score into the pair dict.
"""

import json
import argparse
import os
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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

def get_hama_score(patient_text, ellie_text, tokenizer, model, context_window=None):
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

    if context_window:
        user_content = (
            f"Score the following patient speech from a clinical interview.\n"
            f"I am providing the surrounding context (previous and next turns) to help you understand the nuances.\n\n"
            f"--- CONVERSATION CONTEXT ---\n{context_window}\n--- END CONTEXT ---\n\n"
            f"Please score the participant's anxiety levels based on the 'CURRENT' exchange in the context provided."
        )
    else:
        user_content = (
            f"Score the following patient speech from a clinical interview.\n"
            f"Interviewer asked: {ellie_text}\n\n"
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
    parser = argparse.ArgumentParser(description="HAM-A Scorer for Conversation Pairs with Llama 3.1 8B.")
    parser.add_argument("--input", default=r"d:\Rahul_Intern\convo_model\all_combo-pair\all_conversation_pairs.json")
    parser.add_argument("--output", default=r"d:\Rahul_Intern\convo_model\all_combo-pair\all_conversation_pairs_scored_llama.json")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of pairs to score (0 for all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-words", type=int, default=3, help="Skip responses shorter than this")
    args = parser.parse_args()

    tokenizer, model = load_model()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Count how many have been scored
        scored_count = sum(1 for item in data if "total_score" in item or item.get("skipped", False))
        print(f"Resuming — {scored_count} already processed out of {len(data)}.")
    
    total = len(data)
    if args.limit > 0:
        total = min(total, args.limit)

    errors = 0
    skipped = 0
    start_time = time.time()
    batch_count = 0

    print("=" * 65)
    print(f"HAM-A SCORER — Pairwise Llama 3.1 8B Scoring")
    print(f"Total pairs: {total} | Context Threshold: {args.min_words} words")
    print("=" * 65)

    try:
        for i in range(total):
            entry = data[i]
            
            if "total_score" in entry:
                continue

            ellie_text = entry.get("ellie", "")
            patient_text = entry.get("participant", "")
            
            words = patient_text.split()
            word_count = len(words)
            
            print(f"\n[{i+1}/{total}] Word count: {word_count}")
            
            def is_start_of_convo(text):
                t = text.lower()
                return "hi i'm ellie" in t or "thanks for coming" in t
            
            context_window = None
            if word_count < args.min_words:
                print(f"  Short response — building context window...")
                context_parts = []
                if i > 0 and not is_start_of_convo(ellie_text):
                    prev = data[i-1]
                    context_parts.append(f"PREVIOUS >> Ellie: {prev['ellie']}\n            Participant: {prev['participant']}")
                
                context_parts.append(f"CURRENT  >> Ellie: {ellie_text}\n            Participant: {patient_text}")
                
                if i < len(data) - 1:
                    nxt = data[i+1]
                    if not is_start_of_convo(nxt.get("ellie", "")):
                        context_parts.append(f"NEXT     >> Ellie: {nxt['ellie']}\n            Participant: {nxt['participant']}")
                
                context_window = "\n\n".join(context_parts)
            else:
                print(f"  Ellie: {ellie_text[:60]}...")
                print(f"  Parti: {patient_text[:60]}...")

            try:
                score = get_hama_score(patient_text, ellie_text, tokenizer, model, context_window=context_window)
                if score:
                    entry.update(score)
                    if context_window:
                        entry["scored_with_context"] = True
                    
                    detected = {k: v for k, v in score.items() if k in SUBSCALES and v > 0}
                    print(f"  OK >> Total Score: {score['total_score']} | Detected: {detected if detected else 'None'}", flush=True)
                else:
                    print("  FAILED — no score returned")
                    errors += 1
            except Exception as e:
                print(f"  ERROR processing entry: {e}")
                errors += 1

            batch_count += 1
            if batch_count % args.batch_size == 0:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                print(f"  [checkpoint] Saved progress to {args.output}")

    except KeyboardInterrupt:
        print("\n[!] Manual interruption detected. Saving current progress...")
    finally:
        # Final save
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    elapsed = time.time() - start_time
    processed = sum(1 for item in data[:total] if "total_score" in item)
    print(f"\n{'=' * 65}")
    print(f"COMPLETE!")
    print(f"  Scored      : {processed}")
    print(f"  Errors      : {errors}")
    print(f"  Time taken  : {elapsed/60:.2f} minutes")
    print(f"  Output saved: {args.output}")
    print(f"{'=' * 65}")
