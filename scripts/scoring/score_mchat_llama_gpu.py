"""
Score m_chat16k_combined.json using Llama-3.1-8B-Instruct
Optimized for MAX GPU utilization on RTX 4090 (24 GB VRAM).

Key optimizations:
  - BFloat16 precision (native Ampere/Ada support)
  - Flash Attention 2 via SDPA
  - Batched inference (left-padded) — multiple samples per forward pass
  - Aggressive CUDA settings (TF32, cuDNN benchmark)
  - Checkpoint every N batches to avoid data loss
"""

import json
import argparse
import os
import re
import time
import torch
import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_WORDS  = 2500          # truncate input to keep prompt under context limit
MAX_NEW_TOKENS = 350       # enough for the JSON output
BATCH_SIZE = 6             # items per batch — tune based on VRAM
RETRIES    = 2
SAVE_EVERY = 20            # save checkpoint every N scored items


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

# ─── Helpers ─────────────────────────────────────────────────────────────

def parse_json_response(text):
    """Extract the first valid JSON object from model output."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try first {...} block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Try partial fix
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


def truncate_text(text, max_words=MAX_WORDS):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[-max_words:])
    return text


# ─── Model loading ──────────────────────────────────────────────────────

def load_model():
    """Load Llama-3.1-8B-Instruct with max GPU optimization."""
    print(f"Loading {MODEL_ID} …")

    # Enable CUDA performance flags
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
        # Pre-allocate CUDA caching allocator

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"          # CRITICAL for batched causal LM

    # Determine best dtype — 4090 supports BF16 natively
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",          # Scaled-Dot-Product (Flash-Attn path)
    )
    model.eval()

    # Print GPU memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved  = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  GPU memory — allocated: {allocated:.1f} GB, reserved: {reserved:.1f} GB")
        print(f"  dtype: {compute_dtype}, attn: SDPA/FlashAttention")

    return tokenizer, model


# ─── Batched Inference ───────────────────────────────────────────────────

def build_chat_input(text, tokenizer):
    """Build a single chat-template string for one sample."""
    user_content = f"Score the following text:\n\n--- TEXT ---\n{text}\n--- END ---"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    # Return token IDs (list[int]) without tensoring yet
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
    )
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        return out["input_ids"]
    return out


def batched_generate(batch_input_ids, tokenizer, model, max_new_tokens=MAX_NEW_TOKENS):
    """
    Left-pad a batch of varying-length token ID lists and run a single
    batched model.generate() call.  Returns list[str] of decoded outputs.
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id

    # Find max length
    max_len = max(len(ids) for ids in batch_input_ids)

    # Left-pad
    padded_ids = []
    attention_masks = []
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.45,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=pad_id,
        )

    # Decode only the NEW tokens for each sample
    results = []
    for i, ids in enumerate(batch_input_ids):
        prompt_len = max_len  # after padding, prompt ends at max_len
        generated = outputs[i][prompt_len:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        results.append(decoded)

    # Free memory
    del input_ids, attention_mask, outputs
    torch.cuda.empty_cache()

    return results


def score_batch(entries, tokenizer, model):
    """Score a batch of entries. Returns list of score dicts (or None)."""
    # Build inputs
    texts = []
    input_ids_list = []
    for entry in entries:
        input_text  = truncate_text(entry.get("input", ""))
        output_text = truncate_text(entry.get("output", ""))
        combined = f"Patient: {input_text}\nCounselor: {output_text}"
        texts.append(combined)
        ids = build_chat_input(combined, tokenizer)
        input_ids_list.append(ids)

    # Batched generation
    raw_outputs = batched_generate(input_ids_list, tokenizer, model)

    # Parse and validate
    scores = []
    for idx, raw in enumerate(raw_outputs):
        parsed = parse_json_response(raw)
        if parsed is not None:
            score = validate_scores(parsed)
        else:
            # Retry this single one
            score = None
            for retry in range(RETRIES):
                try:
                    single_out = batched_generate([input_ids_list[idx]], tokenizer, model)
                    parsed = parse_json_response(single_out[0])
                    if parsed:
                        score = validate_scores(parsed)
                        break
                except Exception as e:
                    print(f"    Retry {retry+1} failed for idx {idx}: {e}")
            if score is None:
                print(f"    WARNING: Could not parse score for entry id={entries[idx].get('id', '?')}")

        scores.append(score)

    return texts, scores


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score m_chat16k_combined with Llama-3.1-8B (GPU-optimized)")
    parser.add_argument("--input-file",  default=r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined.json")
    parser.add_argument("--output-file", default=r"d:\Rahul_Intern\convo_model\M_chat16k\m_chat16k_combined_scored_llama.json")
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE, help="Samples per GPU batch")
    parser.add_argument("--limit",       type=int, default=0, help="Process only first N records (0 = all)")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save checkpoint every N scored items")
    args = parser.parse_args()

    # Load model
    tokenizer, model = load_model()

    # Load data
    print(f"Loading data from {args.input_file} …")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]
    print(f"  Total entries: {len(data)}")

    # Load existing results (for resume)
    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Resuming — {len(results)} already processed")
    processed_ids = {r.get("id") for r in results}

    # Filter to unprocessed
    todo = [e for e in data if e.get("id", data.index(e)) not in processed_ids]
    print(f"  Remaining to process: {len(todo)}")
    if not todo:
        print("Nothing to do!")
        return

    # Process in batches
    total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    start_time = time.time()
    items_since_save = 0

    for batch_idx in range(0, len(todo), args.batch_size):
        batch = todo[batch_idx : batch_idx + args.batch_size]
        current_batch = batch_idx // args.batch_size + 1

        # Print progress
        elapsed = time.time() - start_time
        items_done = batch_idx
        if items_done > 0:
            rate = items_done / elapsed
            eta = (len(todo) - items_done) / rate
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            rate = 0
            eta_str = "calculating…"

        print(f"\n{'='*60}")
        print(f"Batch {current_batch}/{total_batches}  |  IDs: {batch[0].get('id','?')} – {batch[-1].get('id','?')}")
        print(f"  Speed: {rate:.1f} items/s  |  ETA: {eta_str}")
        print(f"{'='*60}")

        try:
            texts, scores = score_batch(batch, tokenizer, model)
        except torch.cuda.OutOfMemoryError:
            print("  OOM! Falling back to batch_size=1 for this batch …")
            torch.cuda.empty_cache()
            texts = []
            scores = []
            for single_entry in batch:
                try:
                    t, s = score_batch([single_entry], tokenizer, model)
                    texts.extend(t)
                    scores.extend(s)
                except Exception as e:
                    print(f"    SKIP id={single_entry.get('id','?')}: {e}")
                    input_text  = truncate_text(single_entry.get("input", ""))
                    output_text = truncate_text(single_entry.get("output", ""))
                    texts.append(f"Patient: {input_text}\nCounselor: {output_text}")
                    scores.append(None)

        # Collect results
        for i, entry in enumerate(batch):
            if i >= len(scores):
                break
            input_text  = truncate_text(entry.get("input", ""))
            output_text = truncate_text(entry.get("output", ""))
            result_entry = {
                "id":          entry.get("id", "?"),
                "source":      entry.get("source"),
                "input_text":  input_text,
                "output_text": output_text,
                "score":       scores[i]
            }
            results.append(result_entry)
            total = scores[i]["total_score"] if scores[i] else "FAIL"
            print(f"  id={entry.get('id','?'):>6}  total={total}")

        items_since_save += len(batch)
        if items_since_save >= args.save_every:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            items_since_save = 0
            elapsed_now = time.time() - start_time
            print(f"  [OK] Checkpoint saved ({len(results)} records, {elapsed_now:.0f}s elapsed)")

    # Final save
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE! {len(results)} records scored in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Output: {args.output_file}")
    print(f"  Avg speed: {len(todo)/elapsed_total:.1f} items/sec")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
