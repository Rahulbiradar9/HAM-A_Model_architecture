import json
import argparse
import os
import re
import time
import torch
import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_WORDS  = 3000
MAX_TOKENS = 400
RETRIES    = 3

SUBSCALES = [
    "anxious_mood", "tension", "fears", "insomnia", "intellectual",
    "depressed_mood", "somatic_muscular", "somatic_sensory",
    "cardiovascular", "respiratory", "gastrointestinal",
    "genitourinary", "autonomic", "behavior_at_interview"
]

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

FALLBACK_PROMPT = """Re-evaluate the following patient speech carefully.
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
{text}
---

Return ONLY valid JSON with HAM-A subscale scores (0-4) and total_score.
A score of 1 means mild/subtle evidence — use it freely when ANY hint is present.
{{
  "anxious_mood": 0, "tension": 0, "fears": 0, "insomnia": 0,
  "intellectual": 0, "depressed_mood": 0, "somatic_muscular": 0,
  "somatic_sensory": 0, "cardiovascular": 0, "respiratory": 0,
  "gastrointestinal": 0, "genitourinary": 0, "autonomic": 0,
  "behavior_at_interview": 0, "total_score": 0
}}"""

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

def get_hama_score(text, tokenizer, model):
    def _run_inference(system_msg, user_msg, temperature=0.45):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        device = next(model.parameters()).device
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[-1]
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else torch.no_grad()
        with torch.no_grad(), amp_ctx:
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_TOKENS, temperature=temperature,
                do_sample=True, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id
            )
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        generated_ids = outputs[0][prompt_length:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated

    user_content = f"Score the following text:\n\n--- TEXT ---\n{text}\n--- END ---"
    score = None
    for attempt in range(1, RETRIES + 1):
        try:
            raw = _run_inference(SYSTEM_PROMPT, user_content, temperature=0.45)
            score = parse_json_response(raw)
            if score is not None:
                break
            else:
                print(f"  [attempt {attempt}] Failed to parse JSON. Raw output:\n{raw}")
        except Exception as e:
            print(f"  [attempt {attempt}] Exception: {e}")
            time.sleep(1)

    if score is None:
        return None

    score = validate_scores(score)
    if score["total_score"] == 0:
        fallback_msg = FALLBACK_PROMPT.format(text=text)
        try:
            raw2 = _run_inference("You are a clinical psychologist.", fallback_msg, temperature=0.55)
            score2 = parse_json_response(raw2)
            if score2:
                score2 = validate_scores(score2)
                if score2["total_score"] > 0:
                    score = score2
        except:
            pass
    return score

def load_model():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

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

def truncate_text(text, max_words=MAX_WORDS):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[-max_words:])
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default=r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined.json")
    parser.add_argument("--output-file", default=r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined_scored.json")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records to process")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    tokenizer, model = load_model()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.limit > 0:
        data = data[:args.limit]

    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    processed_ids = {r.get("id") for r in results}

    start_time = time.time()
    for i, entry in enumerate(data):
        entry_id = entry.get("id", i)
        if entry_id in processed_ids:
            continue
            
        print(f"\nProcessing ID: {entry_id} ({i+1}/{len(data)})")
        input_text = truncate_text(entry.get("input", ""))
        output_text = truncate_text(entry.get("output", ""))
        
        # Combine input and output
        combined_text = f"Patient: {input_text}\nCounselor: {output_text}"

        score = None
        if len(combined_text.split()) >= 5:
            score = get_hama_score(combined_text, tokenizer, model)
            
        result_entry = {
            "id": entry_id,
            "source": entry.get("source"),
            "input_text": input_text,
            "output_text": output_text,
            "score": score
        }
        
        results.append(result_entry)
        
        total = score["total_score"] if score else "N/A"
        print(f"  Combined Total Score: {total}")

        if (i + 1) % args.batch_size == 0:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Saved checkpoint ({len(results)} records).")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done!")
