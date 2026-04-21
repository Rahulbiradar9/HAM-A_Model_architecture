import ast, re

with open('scripts/hama_scorer.py', 'r', encoding='utf-8') as f:
    source = f.read()

lines = source.splitlines()
print(f"Total lines    : {len(lines)}")
print(f"Total bytes    : {len(source.encode())}")
print()

model_id   = re.search(r'MODEL_ID\s*=\s*"(.+?)"', source)
max_words  = re.search(r'MAX_WORDS\s*=\s*(\d+)', source)
max_tokens = re.search(r'MAX_TOKENS\s*=\s*(\d+)', source)
retries    = re.search(r'RETRIES\s*=\s*(\d+)', source)
temps      = re.findall(r'temperature=([0-9.]+)', source)
out_file   = re.search(r'_batch_hama_scores(\w*)\.json', source)

print("=== KEY SETTINGS ===")
print(f"  Model          : {model_id.group(1) if model_id else 'NOT FOUND'}")
print(f"  MAX_WORDS      : {max_words.group(1) if max_words else 'NOT FOUND'}")
print(f"  MAX_TOKENS     : {max_tokens.group(1) if max_tokens else 'NOT FOUND'}")
print(f"  RETRIES        : {retries.group(1) if retries else 'NOT FOUND'}")
print(f"  Temperatures   : {temps}")
print(f"  Output file    : _batch_hama_scores{out_file.group(1) if out_file else ''}.json")
print()

functions = re.findall(r'^def (\w+)\(', source, re.MULTILINE)
print("=== FUNCTIONS DEFINED ===")
for fn in functions:
    print(f"  def {fn}()")
print()

try:
    ast.parse(source)
    print("=== SYNTAX CHECK === PASSED")
except SyntaxError as e:
    print(f"=== SYNTAX ERROR === Line {e.lineno}: {e.msg}")

print()
print("=== FEATURES CHECKLIST ===")
checks = {
    "Patient-only extraction" : "extract_patient_speech" in source,
    "Fallback prompt"         : "FALLBACK_PROMPT" in source,
    "Two-pass scoring"        : "pass2" in source,
    "Retry logic"             : "RETRIES" in source and "attempt" in source,
    "Score clamp [0..4]"      : "max(0, min(4" in source,
    "Auto total_score"        : "validate_scores" in source,
    "JSON retry parser"       : "parse_json_response" in source,
    "was_truncated flag"      : "was_truncated" in source,
    "word_count logged"       : "word_count" in source,
    "Batch save checkpoint"   : "checkpoint" in source,
}
for name, ok in checks.items():
    status = "YES" if ok else "MISSING"
    print(f"  {'[OK]' if ok else '[!!]'}  {name:30s}: {status}")
