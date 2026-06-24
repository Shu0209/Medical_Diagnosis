import io
import base64
import logging
import os
import uuid

import cv2
import nibabel as nib
import numpy as np
import pydicom
from Bio import Entrez, Medline
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import json

from openrouter_client import call_openrouter, call_vision, OPENROUTER_MODEL
from prompt import ANALYSIS_PROMPT

logger = logging.getLogger(__name__)



def _get_ncbi_email() -> str:
    """Read NCBI email from env or Streamlit secrets. Warns loudly if missing."""
    email = os.getenv("NCBI_EMAIL", "")
    if not email:
        try:
            import streamlit as st
            email = st.secrets.get("NCBI_EMAIL", "")
        except Exception:
            pass
    if not email:
        logger.warning(
            "NCBI_EMAIL is not set. PubMed searches may be blocked by NCBI. "
            "Set the NCBI_EMAIL environment variable or add it to .streamlit/secrets.toml."
        )
        email = "noreply@example.com"
    return email


Entrez.email = _get_ncbi_email()

NCBI_EMAIL_CONFIGURED = bool(Entrez.email) and Entrez.email != "noreply@example.com"


# File processing

def _safe_normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise array to uint8 [0, 255], guarding against zero range."""
    rng = float(arr.max()) - float(arr.min())
    if rng == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - arr.min()) / rng * 255).astype(np.uint8)


def process_file(uploaded_file):
    """
    Parse an uploaded medical image file and return a dict with:
        type   : 'image' | 'dicom' | 'nifti'
        data   : PIL.Image (RGB)
        array  : np.ndarray (uint8, HxW or HxWx3)
    Returns None if the extension is unrecognised.
    """
    name = uploaded_file.name.lower()

    if name.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file).convert("RGB")
        return {"type": "image", "data": image, "array": np.array(image)}

    if name.endswith(".dcm"):
        ds = pydicom.dcmread(uploaded_file)
        img_array = _safe_normalise(ds.pixel_array)
        pil = Image.fromarray(img_array)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return {"type": "dicom", "data": pil, "array": img_array}

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        temp_path = f"temp_{uuid.uuid4()}.nii.gz"
        try:
            with open(temp_path, "wb") as fh:
                fh.write(uploaded_file.getvalue())
            nii_img = nib.load(temp_path)
            vol = nii_img.get_fdata()
            mid_slice = vol[:, :, vol.shape[2] // 2]
            img_array = _safe_normalise(mid_slice)
            pil = Image.fromarray(img_array).convert("RGB")
            return {"type": "nifti", "data": pil, "array": img_array}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return None




def generate_heatmap(image_array: np.ndarray):
    """
    Generate a JET colour-map heatmap overlay on *image_array*.

    Returns:
        (overlay_pil, heatmap_pil) -- both are RGB PIL Images.
    """
    if image_array.ndim == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        rgb_src = image_array
    else:
        gray = image_array
        rgb_src = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_bgr = cv2.addWeighted(
        heatmap_bgr, 0.5,
        cv2.cvtColor(rgb_src, cv2.COLOR_RGB2BGR), 0.5, 0
    )
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(overlay_rgb), Image.fromarray(heatmap_rgb)




def extract_findings_and_keywords(analysis_text: str):
    """Pull structured findings and keywords out of the LLM's markdown report."""
    findings = []
    keywords = []

    if "Impression:" in analysis_text:
        section = analysis_text.split("Impression:", 1)[1].strip()
        for item in section.split("\n"):
            item = item.strip()
            if not item:
                continue
            if item[0].isdigit() or item[0] in ("-", "*", "•"):
                clean = item.lstrip("0123456789.-*• ").strip()
                if clean:
                    findings.append(clean)
                    for word in clean.split():
                        word = word.lower().strip(",.:;()")
                        if len(word) > 4 and word not in {
                            "about", "with", "that", "this", "these", "those",
                            "which", "their", "there", "where",
                        }:
                            keywords.append(word)

    common_terms = [
        "pneumonia", "infiltrates", "opacities", "nodule", "mass", "tumor",
        "cardiomegaly", "effusion", "consolidation", "atelectasis", "edema",
        "fracture", "fibrosis", "emphysema", "pneumothorax", "metastasis",
    ]
    for term in common_terms:
        if term in analysis_text.lower() and term not in keywords:
            keywords.append(term)

    return findings, list(dict.fromkeys(keywords))[:5]




def analyze_from_text(user_findings: str, api_key: str) -> dict:
    """
    Expand clinician observations into a full structured radiology report.
    Uses the text model (gpt-oss-120b) via streaming.
    """
    findings: list = []
    keywords: list = []

    prompt = (
        ANALYSIS_PROMPT.strip()
        + "\n\n---\nClinician observations:\n"
        + user_findings.strip()
        + "\n\nPlease produce a full report following the structure above."
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        analysis = call_openrouter(api_key, messages, max_tokens=1200, temperature=0.2)
        findings, keywords = extract_findings_and_keywords(analysis)
        return {
            "id": str(uuid.uuid4()),
            "analysis": analysis,
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.error("analyze_from_text failed: %s", exc)
        return {
            "id": str(uuid.uuid4()),
            "analysis": f"Error generating analysis: {exc}",
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
        }




def analyze_image(image_pil: Image.Image, api_key: str) -> dict:
    """
    Send the actual image to the vision model and get a structured report.
    Falls back gracefully if the vision call fails.
    """
    findings: list = []
    keywords: list = []

    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        analysis = call_vision(
            api_key,
            image_b64,
            ANALYSIS_PROMPT.strip(),
            media_type="image/jpeg",
            max_tokens=1200,
            temperature=0.2,
        )
        findings, keywords = extract_findings_and_keywords(analysis)
        return {
            "id": str(uuid.uuid4()),
            "analysis": analysis,
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.error("analyze_image failed: %s", exc)
        return {
            "id": str(uuid.uuid4()),
            "analysis": f"Error analysing image: {exc}",
            "findings": findings,
            "keywords": keywords,
            "date": datetime.now().isoformat(),
        }




def search_pubmed(keywords: list, max_results: int = 5) -> list:
    """
    Search PubMed for articles related to the given keywords.
    Returns [] on any error (no fake placeholder data).
    """
    if not keywords:
        return []

    query = " AND ".join(keywords)
    try:
        search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        id_list = search_results.get("IdList", [])
        if not id_list:
            return []

        fetch_handle = Entrez.efetch(
            db="pubmed", id=id_list, rettype="medline", retmode="text"
        )
        records = list(Medline.parse(fetch_handle))
        fetch_handle.close()

        publications = []
        for rec in records:
            pmid = rec.get("PMID", "")
            if not pmid:
                continue
            dp = rec.get("DP", "")
            year = dp.split()[0] if dp else ""
            if not year.isdigit():
                year = ""

            publications.append({
                "id": pmid,
                "title": rec.get("TI", "No title"),
                "journal": rec.get("TA", rec.get("JT", "Unknown journal")),
                "year": year,
            })
        return publications

    except Exception as exc:
        logger.warning("PubMed search failed: %s", exc)
        return []




def generate_report(data: dict, include_references: bool = True) -> io.BytesIO:
    """Build a PDF report from an analysis dict and return a BytesIO buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Heading1"], fontSize=18, spaceAfter=12
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle", parent=styles["Heading2"], fontSize=14, spaceAfter=8
    )

    def safe_para(text: str, style) -> Paragraph:
        safe = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return Paragraph(safe, style)

    content = []
    content.append(safe_para("Medical Imaging Analysis Report", title_style))
    content.append(Spacer(1, 12))
    content.append(safe_para(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    content.append(safe_para(f"Report ID: {data.get('id', 'N/A')}", styles["Normal"]))
    if "filename" in data:
        content.append(safe_para(f"Image: {data['filename']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    content.append(safe_para("Analysis Result", subtitle_style))
    content.append(safe_para(data.get("analysis", ""), styles["Normal"]))
    content.append(Spacer(1, 12))

    if data.get("findings"):
        content.append(safe_para("Key Findings", subtitle_style))
        for idx, finding in enumerate(data["findings"], 1):
            content.append(safe_para(f"{idx}. {finding}", styles["Normal"]))
        content.append(Spacer(1, 12))

    if data.get("keywords"):
        content.append(safe_para("Keywords", subtitle_style))
        content.append(safe_para(", ".join(data["keywords"]), styles["Normal"]))
        content.append(Spacer(1, 12))

    if include_references:
        pubmed_results = search_pubmed(data.get("keywords", []), max_results=3)
        if pubmed_results:
            content.append(safe_para("Relevant Medical Literature", subtitle_style))
            for ref in pubmed_results:
                content.append(safe_para(ref["title"], styles["Normal"]))
                content.append(
                    safe_para(
                        f"{ref['journal']}, {ref['year']} (PMID: {ref['id']})",
                        styles["Normal"],
                    )
                )
            content.append(Spacer(1, 12))

        

    doc.build(content)
    buffer.seek(0)
    return buffer




_STORE_PATH = "data/analysis_store.json"


def get_analysis_store() -> dict:
    if os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                logger.warning("analysis_store.json is corrupt -- resetting.")
    return {"analyses": []}


def _atomic_write_store(store: dict) -> None:
    """Write store atomically via temp file to reduce corruption risk."""
    tmp = _STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    os.replace(tmp, _STORE_PATH)


def save_analysis(analysis_data: dict, filename: str = "unknown.jpg") -> dict:
    store = get_analysis_store()
    analysis_data["filename"] = filename
    store["analyses"].append(analysis_data)
    _atomic_write_store(store)
    return analysis_data


def get_analysis_by_id(analysis_id: str) -> dict | None:
    store = get_analysis_store()
    for analysis in store["analyses"]:
        if analysis.get("id") == analysis_id:
            return analysis
    return None


def get_latest_analyses(limit: int = 5) -> list:
    store = get_analysis_store()
    return sorted(
        store["analyses"], key=lambda x: x.get("date", ""), reverse=True
    )[:limit]


def extract_common_findings() -> list:
    store = get_analysis_store()
    keyword_counts: dict = {}
    for analysis in store["analyses"]:
        for keyword in analysis.get("keywords", []):
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    return sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)



def generate_statistics_report() -> io.BytesIO | None:
    """Generate a PDF statistics report."""
    store = get_analysis_store()
    if not store["analyses"]:
        return None

    type_counts: dict = {}
    for analysis in store["analyses"]:
        t = analysis.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    common_findings = extract_common_findings()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Medical Imaging Statistics Report", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph("Overall Statistics", styles["Heading2"]))
    content.append(
        Paragraph(f"Total analyses: {len(store['analyses'])}", styles["Normal"])
    )
    content.append(Spacer(1, 12))

    if type_counts:
        content.append(Paragraph("Analysis Types", styles["Heading2"]))
        for type_name, count in type_counts.items():
            content.append(
                Paragraph(f"{type_name.capitalize()}: {count}", styles["Normal"])
            )
        content.append(Spacer(1, 12))

    if common_findings:
        content.append(Paragraph("Common Findings", styles["Heading2"]))
        for keyword, count in common_findings[:10]:
            content.append(
                Paragraph(
                    f"{keyword.capitalize()}: {count} occurrences", styles["Normal"]
                )
            )

    doc.build(content)
    buffer.seek(0)
    return buffer



genrate_statistics_report = generate_statistics_report
