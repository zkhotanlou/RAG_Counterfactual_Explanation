import os, json, re, glob, pathlib
import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

PROMPT_FILE = "/prompt.txt"        # system instruction file
CASES_DIR   = "/cf_pairs"          # folder with *.json test cases
DOCS_DIR    = "/docs"              # dosuments to ground answers
OUT_DIR     = "/rag_cfe_outputs"  # output folder for results
MODEL       = "gemini-2.5-flash"   
TEMPERATURE = 0.2

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def extract_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

uploaded_files = []
doc_files = sorted(glob.glob(f"{DOCS_DIR}/*"))
if doc_files:
    print(f"Uploading {len(doc_files)} reference document(s) to Gemini...")
    for path in doc_files:
        try:
            f = genai.upload_file(path)
            uploaded_files.append(f)
            print(f" {path} → {getattr(f, 'uri', '(no uri)')}")
        except Exception as e:
            print(f" Failed to upload {path}: {e}")
    print("Documents uploaded.")
else:
    print("No documents found in DOCS_DIR — running without file context.")

model = genai.GenerativeModel(
    model_name=MODEL,
    system_instruction=SYSTEM_INSTRUCTION
)

generation_config = {
    "temperature": TEMPERATURE,
    "response_mime_type": "application/json"
}

case_files = sorted(glob.glob(f"{CASES_DIR}/*.json"))
if not case_files:
    raise FileNotFoundError(f"No JSON files found in {CASES_DIR}.")
print(f"Found {len(case_files)} case file(s) in {CASES_DIR}")

# process each data point
for idx, fp in enumerate(case_files, start=1):
    with open(fp, "r", encoding="utf-8") as f:
        payload = json.load(f)  
        
    contents = []
    if uploaded_files:
        contents.extend(uploaded_files)
    contents.append(json.dumps(payload, ensure_ascii=False))

    try:
        resp = model.generate_content(
            contents,
            generation_config=generation_config
        )

        out_text = getattr(resp, "text", None)
        if not out_text:
            parts = getattr(resp, "candidates", [])[0].content.parts
            out_text = "".join([getattr(p, "text", "") for p in parts])

        out_json = extract_json(out_text)

    except Exception as e:
        out_json = {"error": str(e)}

    base = pathlib.Path(fp).stem
    out_path = pathlib.Path(OUT_DIR) / f"{base}_out.json"
    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(out_json, w, ensure_ascii=False, indent=2)

    print(f"[{idx}/{len(case_files)}] {base} → {out_path}")

print("\nDone. Check the outputs folder for results.")
