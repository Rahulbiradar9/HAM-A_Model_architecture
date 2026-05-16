import os
import json
import random
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Constants ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "../json_transcripts/synthetic"

HAMA_SUBSCALE_DESCRIPTIONS = {
    "anxious_mood": "anxious, worried, on edge, nervous, or easily irritable",
    "tension": "restless, exhausted, stressed, unable to relax, or easily startled",
    "fears": "afraid of specific situations, crowds, being alone, or new situations",
    "insomnia": "struggling to fall asleep, waking up constantly, or tired in the mornings",
    "intellectual": "having poor concentration, foggy memory, or difficulty making decisions",
    "depressed_mood": "sad, feeling down, lacking motivation, or losing interest in hobbies",
    "somatic_muscular": "experiencing unexplained muscle tension, body aches, or stiff joints",
    "somatic_sensory": "feeling hot/cold flushes, ringing in the ears, or bodily numbness",
    "cardiovascular": "occasional racing heart, palpitations, or tight sensations in the chest",
    "respiratory": "frequently sighing, feeling short of breath, or hyperventilating",
    "gastrointestinal": "having knots in the stomach, emotional nausea, or appetite changes",
    "genitourinary": "stress-induced frequent urination or general disruptions",
    "autonomic": "excessive sweating, cold hands, dizziness, or blushing under stress",
    "behavior_at_interview": "fidgeting, giving short evasive answers, or avoiding discussion"
}

def build_generation_prompt(target_severity):
    """ Builds a robust prompt guiding the LLM to write a long conversational transcript. """
    severity_str = "moderate to severe" if target_severity == "high" else "very mild or non-existent"
    
    prompt = f"""You are a creative writer specializing in realistic mental health scenarios.
Your task is to generate a full-length, authentic clinical assessment transcript.

Format:
A JSON array of turn-by-turn dialogue objects.
Each turn must look exactly like this:
{{
    "speaker": "Ellie" or "Participant",
    "value": "Dialogue text..."
}}

Scenario:
The interviewer is 'Ellie' (an AI clinical avatar). The interviewee is 'Participant'.
The Participant has {severity_str} anxiety according to the Hamilton Anxiety Rating Scale (HAM-A).

Crucial Requirements:
1. The language must be extremely natural human dialogue: use conversational filler words (um, uh, yeah, like, well, kind of).
2. Participant's dialogue must accurately reflect anxiety symptoms like:
{json.dumps(HAMA_SUBSCALE_DESCRIPTIONS, indent=2)}

OUTPUT ONLY the raw JSON array. Start immediately with [ and end with ]. No explanations, no markdown fences.
"""
    return prompt

def parse_json_array(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text.split("```json")[1].split("```")[0].strip()
    elif text.startswith("```"):
        text = text.split("```")[1].split("```")[0].strip()
        
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback partial parsing
        import re
        match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None

def run_model_generation(tokenizer, model, prompt):
    messages = [
        {"role": "system", "content": "You are a specialized transcript generator. You output JSON only."},
        {"role": "user", "content": prompt}
    ]
    
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.75,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[-1]:], 
        skip_special_tokens=True
    )
    
    return parse_json_array(generated_text)

def generate_transcript(tokenizer, model, severity="high", target_participant_turns=100):
    accumulated_turns = []
    participant_count = 0
    
    # 1. Initial generation
    print("    Generating initial block...")
    prompt = build_generation_prompt(severity)
    accumulated_turns = run_model_generation(tokenizer, model, prompt)
    
    if not accumulated_turns or not isinstance(accumulated_turns, list):
        print("    Failed initial generation.")
        return []
        
    participant_count = sum(1 for turn in accumulated_turns if "participant" in turn.get("speaker", "").lower())
    print(f"    Initial block yielded {participant_count} participant turns ({len(accumulated_turns)} total turns).")
    
    # 2. Extension loop
    attempts = 0
    max_attempts = 15  # Safety guard
    while participant_count < target_participant_turns and attempts < max_attempts:
        attempts += 1
        print(f"    Progress: {participant_count}/{target_participant_turns} participant turns. Extending conversation...")
        
        # Provide the last 15 turns as conversational context
        context_turns = accumulated_turns[-15:] if len(accumulated_turns) > 15 else accumulated_turns
        
        extend_prompt = f"""You are continuing a realistic mental health assessment.
Here are the latest turns of the interview between Ellie and the Participant:
{json.dumps(context_turns, indent=2)}

Continue this dialogue for about 20-30 more turns.
Probe deeper into the following symptoms to make the conversation longer:
- Body aches/tension
- Concentration/decision making
- Restlessness
- Fears or daily anxieties
- Poor sleep patterns

Format:
OUTPUT ONLY the raw JSON array containing the NEXT turns. Start immediately with [ and end with ]. Do not repeat previous conversation turns.
"""
        new_turns = run_model_generation(tokenizer, model, extend_prompt)
        
        if new_turns and isinstance(new_turns, list):
            # Safety check: avoid appending intro greetings repeatedly
            first_val = new_turns[0].get("value", "").lower() if len(new_turns) > 0 else ""
            if "hi" in first_val[:10] or "welcome" in first_val[:10]:
                print("    Warning: Model generated redundant introductory speech. Adjusting loop.")
                continue
                
            accumulated_turns.extend(new_turns)
            participant_count = sum(1 for turn in accumulated_turns if "participant" in turn.get("speaker", "").lower())
            print(f"    Success. Now at {participant_count} participant turns.")
        else:
            print("    Warning: Parsing failed or model hallucinated for this chunk. Retrying extension.")
            
    return accumulated_turns

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic long-form HAMA clinical transcripts.")
    parser.add_argument("--count", type=int, default=1, help="How many transcripts to generate")
    parser.add_argument("--severity", type=str, choices=["low", "high"], default="high", help="Target HAMA severity")
    parser.add_argument("--participant-turns", type=int, default=100, help="Target number of Participant turns")
    args = parser.parse_args()
    
    # Ensure folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading generation model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    
    print(f"\nStarting synthetic data generation ({args.count} files targeting {args.participant_turns} participant turns each)...")
    
    start_time = time.time()
    for i in range(args.count):
        print(f"\n[Generating {i+1}/{args.count}] Severity: {args.severity}...")
        transcript = generate_transcript(tokenizer, model, args.severity, args.participant_turns)
        
        if transcript and isinstance(transcript, list):
            # Inject dummy start/stop times to match exact 300_TRANSCRIPT.json signature
            current_sec = 35.0
            for turn in transcript:
                turn["start_time"] = f"{current_sec:.3f}"
                current_sec += random.uniform(2.0, 10.0)
                turn["stop_time"] = f"{current_sec:.3f}"
                current_sec += random.uniform(0.5, 2.0)
                
            out_file = os.path.join(OUTPUT_DIR, f"SYNTH_{args.severity.upper()}_L100_{int(time.time())}_{i}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(transcript, f, indent=4)
            print(f"  --> Saved to {out_file} ({len(transcript)} total turns)")
        else:
            print("  --> FAILED to generate valid transcript.")
            
    print(f"\nCompleted in {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
