# =============================================================================
# CDOCS POC — Notebook 05: Live GFA Classification Demo
# =============================================================================
#
# Upload a PDF → extract text → LLM summarize → predict Governing
# Functional Area with confidence scores.
#
# SETUP:
#   1. Run Notebook 04 (Cell 15 saves the model)
#   2. Upload test PDF(s) to dbfs:/FileStore/cdocs_poc/demo_docs/
#   3. Fill in AI Gateway widgets
#   4. Run all cells
#   5. Enter filename → run Cell 7 → run Cell 8 for HTML display
#
# =============================================================================


# -----------------------------------------------------------------------------
# CELL 1 — Install dependencies (run once)
# -----------------------------------------------------------------------------
# %pip install pdfplumber pypdf joblib scikit-learn
# dbutils.library.restartPython()


# -----------------------------------------------------------------------------
# CELL 2 — Imports
# -----------------------------------------------------------------------------
import os
import re
import json
import unicodedata
import requests
import joblib
import numpy as np

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

print("Imports OK")


# -----------------------------------------------------------------------------
# CELL 3 — Widgets
# -----------------------------------------------------------------------------
dbutils.widgets.text("AI_GATEWAY_API_TOKEN", "", "API Token")
dbutils.widgets.text("AI_GATEWAY_BASE_URL",  "", "Base URL")
dbutils.widgets.text("AI_GATEWAY_MODEL",     "", "Model Name")
dbutils.widgets.text("DEMO_FILENAME",        "", "PDF filename to classify")

API_TOKEN  = dbutils.widgets.get("AI_GATEWAY_API_TOKEN").strip()
BASE_URL   = dbutils.widgets.get("AI_GATEWAY_BASE_URL").strip().rstrip("/")
MODEL_NAME = dbutils.widgets.get("AI_GATEWAY_MODEL").strip()

DEMO_DIR   = "/dbfs/FileStore/cdocs_poc/demo_docs/"
MODEL_PATH = "/dbfs/FileStore/cdocs_poc/model/cdocs_gfa_classifier.joblib"

print("Widgets ready")


# -----------------------------------------------------------------------------
# CELL 4 — Load trained model
# -----------------------------------------------------------------------------
assert os.path.exists(MODEL_PATH), (
    f"Model not found: {MODEL_PATH}\n"
    f"Run Notebook 04 Cell 15 first to save the trained model."
)

pipeline     = joblib.load(MODEL_PATH)
gfa_labels   = list(pipeline.classes_)

print(f"Model loaded: {MODEL_PATH}")
print(f"GFA categories ({len(gfa_labels)}):")
for label in gfa_labels:
    print(f"  • {label}")


# -----------------------------------------------------------------------------
# CELL 5 — List available demo documents
# -----------------------------------------------------------------------------
os.makedirs(DEMO_DIR, exist_ok=True)

demo_files = sorted([
    f for f in os.listdir(DEMO_DIR)
    if f.lower().endswith(".pdf")
])

print(f"\nDocuments in {DEMO_DIR}:")
if demo_files:
    for f in demo_files:
        size = os.path.getsize(os.path.join(DEMO_DIR, f))
        print(f"  📄 {f}  ({size:,} bytes)")
else:
    print("  (none — upload PDFs to dbfs:/FileStore/cdocs_poc/demo_docs/)")


# -----------------------------------------------------------------------------
# CELL 6 — Pipeline functions
# All functions in one cell — survives restartPython() if re-run from here.
# Same logic as Notebooks 01 and 02, consolidated.
# -----------------------------------------------------------------------------

# --- Text cleaning ---
def clean_extracted_text(text):
    if not text:
        return ""
    cleaned = []
    for ch in text:
        code = ord(ch)
        if ch in ('\n', '\r', '\t', ' '):
            cleaned.append(ch); continue
        if code == 0xFFFD:
            continue
        if 0xE000 <= code <= 0xF8FF or 0xF0000 <= code <= 0xFFFFD:
            continue
        if unicodedata.category(ch).startswith('C'):
            continue
        cleaned.append(ch)
    text = "".join(cleaned)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    filtered = [l for l in lines if not l.strip() or sum(c.isalpha() for c in l.strip()) >= 2]
    return "\n".join(filtered).strip()


# --- Text quality scoring ---
def score_text_quality(text):
    if not text:
        return 0.0, "No text"
    total = len(text)
    alpha_ratio = sum(c.isalpha() for c in text) / total
    words = text.lower().split()
    word_ratio = (
        sum(1 for w in words if sum(c.isalpha() for c in w) >= 2) / len(words)
        if words else 0
    )
    special = sum(
        1 for c in text
        if not c.isalnum() and c not in ' \t\n\r.,;:!?\'"()-/\\@#$%&*+=[]{}|<>~`_'
    )
    special_ratio = special / total
    garbled = len(re.findall(r'[^\w\s]{10,}', text))
    score = (
        0.40 * alpha_ratio
        + 0.25 * (1 - min(special_ratio * 10, 1))
        + 0.25 * word_ratio
        + 0.10 * (1 - min(garbled / 20, 1))
    )
    return max(0, min(1, round(score, 3))), "clean" if alpha_ratio > 0.5 else f"alpha={alpha_ratio:.0%}"


# --- PDF extraction (multi-extractor, best quality wins) ---
def extract_text(file_path):
    best = {"text": "", "score": 0.0, "method": None, "pages": None}

    # pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            raw = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            cleaned = clean_extracted_text(raw)
            score, _ = score_text_quality(cleaned)
            if score > best["score"]:
                best = {"text": cleaned, "score": score, "method": "pdfplumber", "pages": len(pdf.pages)}
    except Exception:
        pass

    # pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        raw = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        cleaned = clean_extracted_text(raw)
        score, _ = score_text_quality(cleaned)
        if score > best["score"]:
            best = {"text": cleaned, "score": score, "method": "pypdf", "pages": len(reader.pages)}
    except Exception:
        pass

    return best


# --- LLM summarization ---
def call_llm_summarize(text):
    prompt = f"""You are a document content analyst. Read the document text below and write a factual summary of its SUBSTANTIVE CONTENT.

STRICT RULES:
1. Summarize ONLY substantive body content. SKIP metadata, author names, version info, headers, footers, signatures, approval tables, garbled text.
2. Preserve exact technical vocabulary, system names, regulation references, product names, acronyms.
3. Do NOT mention or guess departments, business units, functional areas, or organizational categories.
4. Do NOT classify the document. Describe content only.
5. Write 2-4 paragraphs of factual prose. No bullets. No headers.
6. Return ONLY valid JSON. No markdown fences.

Return exactly:
{{"summary": "...", "key_terms": ["..."], "llm_confidence": "high|medium|low", "llm_confidence_reason": "..."}}

Document text:
<<<
{text[:30000]}
>>>"""

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "Return valid JSON only. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 1500,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    raw = response.json()["choices"][0]["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()

    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"No JSON: {raw[:200]}")


print("Pipeline functions OK")


# -----------------------------------------------------------------------------
# CELL 7 — ▶ CLASSIFY A DOCUMENT
#
# Set DEMO_FILENAME widget → run this cell.
# Re-run for each new document during the demo.
# -----------------------------------------------------------------------------

demo_filename = dbutils.widgets.get("DEMO_FILENAME").strip()

if not demo_filename:
    print("⚠ Enter a filename in the DEMO_FILENAME widget and re-run this cell.")
    print(f"\nAvailable files:")
    for f in demo_files:
        print(f"  {f}")

else:
    file_path = os.path.join(DEMO_DIR, demo_filename)
    assert os.path.exists(file_path), f"File not found: {file_path}"

    print(f"{'='*70}")
    print(f" CLASSIFYING: {demo_filename}")
    print(f"{'='*70}")

    # --- Step 1: Extract text ---
    print(f"\n📄 Step 1: Extracting text...")
    extraction = extract_text(file_path)
    print(f"   Method : {extraction['method']}")
    print(f"   Pages  : {extraction['pages']}")
    print(f"   Chars  : {len(extraction['text']):,}")
    print(f"   Quality: {extraction['score']:.3f}")

    if extraction["score"] < 0.40:
        print(f"\n❌ Text quality too low ({extraction['score']:.2f}).")
        print(f"   RECOMMENDATION: Route to HUMAN REVIEW")
        # Store variables for Cell 8 display
        prediction = None
        low_quality_flag = True
    else:
        low_quality_flag = False

        # --- Step 2: LLM summarization ---
        print(f"\n🤖 Step 2: Summarizing with LLM...")
        llm_result  = call_llm_summarize(extraction["text"])
        summary     = llm_result.get("summary", "")
        key_terms   = llm_result.get("key_terms", [])
        llm_conf    = llm_result.get("llm_confidence", "low")
        llm_reason  = llm_result.get("llm_confidence_reason", "")
        print(f"   LLM confidence: {llm_conf}")
        print(f"   Key terms     : {', '.join(key_terms[:10])}")

        # --- Step 3: ML prediction ---
        print(f"\n🎯 Step 3: Predicting Governing Functional Area...")
        feature_text  = f"{summary} {' '.join(key_terms)}"
        prediction    = pipeline.predict([feature_text])[0]
        probabilities = pipeline.predict_proba([feature_text])[0]
        sorted_idx    = probabilities.argsort()[::-1]

        top_prob = probabilities[sorted_idx[0]]
        conf_label = (
            "🟢 HIGH"   if top_prob >= 0.70 else
            "🟡 MEDIUM" if top_prob >= 0.40 else
            "🔴 LOW"
        )

        print(f"\n{'='*70}")
        print(f" RESULT")
        print(f"{'='*70}")
        print(f"\n   Predicted GFA : {prediction}")
        print(f"   Confidence    : {top_prob:.1%}  {conf_label}")
        print(f"\n   {'Rank':<6} {'Governing Functional Area':<40} {'Probability':>12}")
        print(f"   {'─'*6} {'─'*40} {'─'*12}")

        for rank, i in enumerate(sorted_idx, 1):
            marker = " ◀" if rank == 1 else ""
            print(f"   {rank:<6} {gfa_labels[i]:<40} {probabilities[i]:>11.1%}{marker}")

        print(f"\n{'='*70}")
        print(f" DOCUMENT SUMMARY")
        print(f"{'='*70}")
        print(f"\n{summary}")

        if llm_conf == "low" or top_prob < 0.40:
            print(f"\n⚠ RECOMMENDATION: Route to HUMAN REVIEW")
            print(f"   Reason: {'LLM confidence low' if llm_conf == 'low' else 'Prediction probability low'}")


# -----------------------------------------------------------------------------
# CELL 8 — ▶ RICH HTML DISPLAY (run after Cell 7)
# Formatted visual output — good for screenshots and stakeholder demos.
# -----------------------------------------------------------------------------

try:
    if low_quality_flag:
        html = f"""
        <div style="font-family: -apple-system, sans-serif; max-width: 700px; margin: 20px auto;">
            <h2>📄 GFA Classification Result</h2>
            <p style="color: #64748b;">File: {demo_filename}</p>
            <div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 12px;
                        padding: 24px; margin: 20px 0;">
                <h3 style="color: #dc2626; margin-top: 0;">❌ Text Quality Too Low</h3>
                <p>Quality score: {extraction['score']:.3f}</p>
                <p>Extraction method: {extraction['method'] or 'none'}</p>
                <p><b>Recommendation:</b> Route to human review. The document could not be
                   reliably parsed for automated classification.</p>
            </div>
        </div>"""
        displayHTML(html)

    else:
        _ = prediction  # verify Cell 7 ran successfully

        # Build bars for all GFAs
        bars_html = ""
        for rank, i in enumerate(sorted_idx, 1):
            pct = probabilities[i] * 100
            color = "#2563eb" if rank == 1 else "#94a3b8"
            label = gfa_labels[i]
            bars_html += f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 14px;">
                    <span>{'→ ' if rank == 1 else '&nbsp;&nbsp;'}<b>{label}</b></span>
                    <span>{pct:.1f}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 4px; height: 22px; margin-top: 2px;">
                    <div style="background: {color}; width: {max(pct, 1)}%; height: 100%;
                                border-radius: 4px;"></div>
                </div>
            </div>"""

        conf_color = "#16a34a" if top_prob >= 0.70 else "#ca8a04" if top_prob >= 0.40 else "#dc2626"
        conf_text  = "HIGH" if top_prob >= 0.70 else "MEDIUM" if top_prob >= 0.40 else "LOW"

        route_html = ""
        if llm_conf == "low" or top_prob < 0.40:
            route_html = """
            <div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px;
                        padding: 12px 16px; margin-top: 16px;">
                <b>⚠ Recommendation:</b> Route to human review for verification
            </div>"""

        key_terms_str = ", ".join(key_terms[:15]) if key_terms else "none extracted"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    max-width: 700px; margin: 20px auto;">

            <h2 style="margin-bottom: 4px;">📄 GFA Classification Result</h2>
            <p style="color: #64748b; margin-top: 0;">File: {demo_filename}</p>

            <div style="display: flex; gap: 16px; margin: 20px 0;">
                <div style="flex: 2; background: #f8fafc; border: 1px solid #e2e8f0;
                            border-radius: 12px; padding: 20px;">
                    <div style="color: #64748b; font-size: 13px; text-transform: uppercase;
                                letter-spacing: 0.5px;">Predicted Governing Functional Area</div>
                    <div style="font-size: 22px; font-weight: 700; margin-top: 8px;">{prediction}</div>
                </div>
                <div style="flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
                            border-radius: 12px; padding: 20px; text-align: center;">
                    <div style="color: #64748b; font-size: 13px; text-transform: uppercase;
                                letter-spacing: 0.5px;">Confidence</div>
                    <div style="font-size: 28px; font-weight: 700; margin-top: 8px;
                                color: {conf_color};">{top_prob:.0%}</div>
                    <div style="font-size: 12px; color: {conf_color}; font-weight: 600;">{conf_text}</div>
                </div>
            </div>

            <h3>All GFA Predictions</h3>
            {bars_html}

            {route_html}

            <details style="margin-top: 24px;">
                <summary style="cursor: pointer; font-weight: 600; font-size: 15px;">
                    📝 Document Summary
                </summary>
                <p style="margin-top: 8px; line-height: 1.6; color: #334155;">{summary}</p>
                <p style="color: #64748b; font-size: 13px;">
                    <b>Key terms:</b> {key_terms_str}
                </p>
            </details>

            <details style="margin-top: 12px;">
                <summary style="cursor: pointer; font-weight: 600; font-size: 15px;">
                    🔍 Pipeline Details
                </summary>
                <table style="margin-top: 8px; font-size: 13px; color: #475569;">
                    <tr><td style="padding: 4px 16px 4px 0;"><b>Extraction method</b></td>
                        <td>{extraction['method']}</td></tr>
                    <tr><td style="padding: 4px 16px 4px 0;"><b>Pages</b></td>
                        <td>{extraction['pages']}</td></tr>
                    <tr><td style="padding: 4px 16px 4px 0;"><b>Characters extracted</b></td>
                        <td>{len(extraction['text']):,}</td></tr>
                    <tr><td style="padding: 4px 16px 4px 0;"><b>Text quality score</b></td>
                        <td>{extraction['score']:.3f}</td></tr>
                    <tr><td style="padding: 4px 16px 4px 0;"><b>LLM confidence</b></td>
                        <td>{llm_conf}</td></tr>
                    <tr><td style="padding: 4px 16px 4px 0;"><b>LLM reason</b></td>
                        <td>{llm_reason}</td></tr>
                </table>
            </details>
        </div>
        """
        displayHTML(html)

except NameError:
    print("Run Cell 7 first to classify a document.")
