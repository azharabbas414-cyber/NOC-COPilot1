import io
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# AI-NOC Copilot
# Beginner-friendly Streamlit application
# Modules:
#   1. Incident Analysis (RAG + LLM)
#   2. Network Health (CSV analysis)
#   3. NOC Report Generator
# ============================================================

st.set_page_config(
    page_title="AI-NOC Copilot",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_LLM_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

SUPPORTED_DOC_TYPES = {
    ".txt",
    ".md",
    ".csv",
    ".pdf",
    ".docx",
}


# -----------------------------
# Utility functions
# -----------------------------
def get_secret(name, default=None):
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def clean_text(text):
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=1200, overlap=200):
    """Simple character-based chunking suitable for beginner RAG projects."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# -----------------------------
# LLM
# -----------------------------
def call_llm(messages, temperature=0.2):
    """
    Calls an OpenAI-compatible chat-completions endpoint.

    Default endpoint is Groq.
    You can point LLM_BASE_URL to another compatible provider if required.
    """
    api_key = get_secret("LLM_API_KEY") or get_secret("GROQ_API_KEY")
    base_url = get_secret("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    model = get_secret("LLM_MODEL", DEFAULT_LLM_MODEL)

    if not api_key:
        return None, (
            "LLM API key is not configured. Add LLM_API_KEY or GROQ_API_KEY "
            "to Streamlit secrets."
        )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer, None
    except requests.RequestException as exc:
        return None, f"LLM request failed: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"Unexpected LLM response: {exc}"


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf_text(file_bytes):
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


def extract_docx_text(file_bytes):
    try:
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        return f"[DOCX extraction error: {exc}]"


def extract_uploaded_document(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")

    if suffix == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(data))
            return df.to_csv(index=False)
        except Exception:
            return data.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        return extract_pdf_text(data)

    if suffix == ".docx":
        return extract_docx_text(data)

    return ""


# -----------------------------
# Google Drive RAG
# -----------------------------
def get_drive_service():
    """
    Creates a Google Drive API service from Streamlit secrets.

    Supported secret format:
      [google_service_account]
      type = "service_account"
      project_id = "..."
      private_key_id = "..."
      private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
      client_email = "..."
      client_id = "..."
      token_uri = "https://oauth2.googleapis.com/token"
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        if "google_service_account" not in st.secrets:
            return None, "google_service_account is not configured."

        service_account_info = dict(st.secrets["google_service_account"])

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        return service, None
    except Exception as exc:
        return None, f"Google Drive connection failed: {exc}"


def list_drive_files(service, folder_id=None):
    query_parts = [
        "trashed = false",
    ]

    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")

    query = " and ".join(query_parts)

    results = []
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id,name,mimeType,size)",
                pageSize=100,
                pageToken=page_token,
            )
            .execute()
        )

        results.extend(response.get("files", []))
        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return results


def download_drive_file(service, file_info):
    file_id = file_info["id"]
    name = file_info["name"]
    mime = file_info.get("mimeType", "")

    # Google-native documents need export.
    if mime == "application/vnd.google-apps.document":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="text/plain",
        )
        data = request.execute()
        return name + ".txt", data.decode("utf-8", errors="ignore")

    if mime == "application/vnd.google-apps.spreadsheet":
        request = service.files().export_media(
            fileId=file_id,
            mimeType="text/csv",
        )
        data = request.execute()
        return name + ".csv", data.decode("utf-8", errors="ignore")

    if mime == "application/pdf":
        request = service.files().get_media(fileId=file_id)
        data = request.execute()
        return name, extract_pdf_text(data)

    if mime in {
        "text/plain",
        "text/markdown",
        "text/csv",
    }:
        request = service.files().get_media(fileId=file_id)
        data = request.execute()
        return name, data.decode("utf-8", errors="ignore")

    if mime == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        request = service.files().get_media(fileId=file_id)
        data = request.execute()
        return name, extract_docx_text(data)

    return name, ""


@st.cache_data(show_spinner=False)
def load_drive_documents(folder_id):
    service, error = get_drive_service()
    if error:
        return [], error

    try:
        files = list_drive_files(service, folder_id)
        documents = []

        for file_info in files:
            suffix = Path(file_info["name"]).suffix.lower()
            mime = file_info.get("mimeType", "")

            supported = (
                suffix in SUPPORTED_DOC_TYPES
                or mime
                in {
                    "application/vnd.google-apps.document",
                    "application/vnd.google-apps.spreadsheet",
                }
            )

            if not supported:
                continue

            name, text = download_drive_file(service, file_info)
            if text and not text.startswith("["):
                documents.append(
                    {
                        "name": name,
                        "text": clean_text(text),
                    }
                )

        return documents, None

    except Exception as exc:
        return [], f"Google Drive document loading failed: {exc}"


# -----------------------------
# RAG index
# -----------------------------
def build_rag_index(documents):
    chunks = []

    for document in documents:
        for index, chunk in enumerate(chunk_text(document["text"])):
            chunks.append(
                {
                    "document": document["name"],
                    "chunk_id": index,
                    "text": chunk,
                }
            )

    if not chunks:
        return None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=12000,
    )

    matrix = vectorizer.fit_transform([item["text"] for item in chunks])

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def retrieve_context(index, query, top_k=5):
    if not index or not query.strip():
        return []

    query_vector = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, index["matrix"]).flatten()

    top_indexes = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indexes:
        if scores[i] <= 0:
            continue

        item = dict(index["chunks"][i])
        item["score"] = float(scores[i])
        results.append(item)

    return results


def format_context(results):
    if not results:
        return "No relevant knowledge-base evidence was retrieved."

    blocks = []
    for item in results:
        blocks.append(
            f"Source: {item['document']} | Relevance: {item['score']:.2f}\n"
            f"{item['text']}"
        )

    return "\n\n---\n\n".join(blocks)


# -----------------------------
# Incident Analysis
# -----------------------------
def incident_analysis(subject, description, rag_results):
    context = format_context(rag_results)

    system_prompt = """
You are AI-NOC Copilot assisting a NOC engineer.

Rules:
1. Be beginner-friendly and concise.
2. Use only the supplied incident information and retrieved evidence.
3. Never invent device names, IP addresses, VLANs, causes, alarms, commands,
   timestamps, or network facts.
4. Clearly separate evidence from inference.
5. If evidence is insufficient, say that it is insufficient.
6. The NOC engineer remains responsible for verification and the final decision.
7. Do not recommend a risky production change as if it were already approved.
"""

    user_prompt = f"""
Incident subject:
{subject}

Incident description:
{description}

Retrieved knowledge-base evidence:
{context}

Analyze the incident and return these sections:

1. Incident classification
2. Evidence from knowledge base
3. AI assessment
4. Possible root cause(s) - label as hypothesis unless directly supported
5. Recommended checks
6. Confidence: High / Medium / Low
7. Engineer verification / final decision

When comparing the knowledge-base information with your assessment, explicitly
state where they agree, differ, or where evidence is missing.
"""

    return call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )


# -----------------------------
# Network Health
# -----------------------------
def detect_metric_columns(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()

    groups = {
        "utilization": [],
        "latency": [],
        "packet_loss": [],
        "errors": [],
        "availability": [],
    }

    for col in numeric:
        name = col.lower().replace(" ", "_")

        if any(x in name for x in ["util", "cpu", "memory", "bandwidth", "usage"]):
            groups["utilization"].append(col)

        if any(x in name for x in ["latency", "delay", "rtt"]):
            groups["latency"].append(col)

        if any(x in name for x in ["loss", "packet_loss"]):
            groups["packet_loss"].append(col)

        if any(x in name for x in ["error", "crc", "discard", "drop"]):
            groups["errors"].append(col)

        if any(x in name for x in ["availability", "uptime"]):
            groups["availability"].append(col)

    return groups


def health_score(df):
    """
    Heuristic health score.

    Important:
    This is not a vendor-specific SLA calculation. It is a simple screening
    score intended to help an engineer identify records worth investigating.
    """
    groups = detect_metric_columns(df)
    score = 100.0
    findings = []

    for col in groups["utilization"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            high = float((series >= 90).mean())
            if high > 0:
                score -= min(20, high * 20)
                findings.append(
                    f"{col}: {high:.0%} of numeric records are >= 90."
                )

    for col in groups["latency"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            high = float((series >= 100).mean())
            if high > 0:
                score -= min(20, high * 20)
                findings.append(
                    f"{col}: {high:.0%} of numeric records are >= 100 "
                    f"(threshold is a screening heuristic)."
                )

    for col in groups["packet_loss"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            high = float((series >= 1).mean())
            if high > 0:
                score -= min(25, high * 25)
                findings.append(
                    f"{col}: {high:.0%} of numeric records are >= 1 "
                    f"(threshold is a screening heuristic)."
                )

    for col in groups["errors"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            high = float((series > 0).mean())
            if high > 0:
                score -= min(20, high * 20)
                findings.append(
                    f"{col}: non-zero values found in {high:.0%} of records."
                )

    for col in groups["availability"]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series):
            low = float((series < 99).mean())
            if low > 0:
                score -= min(20, low * 20)
                findings.append(
                    f"{col}: {low:.0%} of records are below 99 "
                    f"(threshold is a screening heuristic)."
                )

    score = max(0.0, min(100.0, score))

    if score >= 85:
        status = "Healthy"
    elif score >= 65:
        status = "Warning"
    else:
        status = "Critical"

    return score, status, findings, groups


def network_health_analysis(df):
    score, status, findings, groups = health_score(df)

    numeric = df.select_dtypes(include=np.number)

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": list(numeric.columns),
        "missing_values": int(df.isna().sum().sum()),
    }

    return {
        "score": score,
        "status": status,
        "findings": findings,
        "groups": groups,
        "summary": summary,
    }


# -----------------------------
# NOC Report Generator
# -----------------------------
def generate_noc_report(
    subject,
    summary,
    start_time,
    end_time,
    root_cause,
    impacted_services,
    recommendations,
):
    duration = "Not calculated"

    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        seconds = max(0, int((end - start).total_seconds()))
        duration = f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    except Exception:
        pass

    return f"""# NOC Incident Report

## 1. Incident Details
- **Subject:** {subject}
- **Start Time:** {start_time}
- **End Time:** {end_time}
- **Duration:** {duration}

## 2. Incident Summary
{summary}

## 3. Root Cause
{root_cause}

## 4. Impacted Services
{impacted_services}

## 5. Recommendations
{recommendations}

## 6. Engineer Validation
This report was prepared from information provided by the NOC engineer.
Root cause, impact, remediation, and closure should be verified by the
responsible engineer before the incident is considered final.

---
Generated by AI-NOC Copilot
"""


# -----------------------------
# UI helpers
# -----------------------------
def show_sidebar():
    st.sidebar.title("🛡️ AI-NOC Copilot")
    st.sidebar.caption("AI assistance for NOC engineers")

    module = st.sidebar.radio(
        "Select module",
        [
            "Incident Analysis",
            "Network Health",
            "NOC Report Generator",
        ],
    )

    st.sidebar.divider()
    st.sidebar.info(
        "AI-NOC Copilot supports the engineer. "
        "Always verify evidence before making production changes."
    )

    return module


def incident_module():
    st.title("🚨 Incident Analysis")
    st.write(
        "Analyze an incident using your NOC knowledge documents and an "
        "AI response. The engineer remains in control of the final decision."
    )

    with st.expander("RAG / Knowledge Base"):
        folder_id = st.text_input(
            "Google Drive Folder ID (optional)",
            value=get_secret("GOOGLE_DRIVE_FOLDER_ID", ""),
            help="Use the folder ID containing your NOC knowledge documents.",
        )

        uploaded_docs = st.file_uploader(
            "Or upload knowledge documents",
            type=["txt", "md", "csv", "pdf", "docx"],
            accept_multiple_files=True,
        )

        drive_documents = []
        if folder_id:
            if st.button("Load Google Drive Knowledge Base"):
                with st.spinner("Loading documents from Google Drive..."):
                    drive_documents, error = load_drive_documents(folder_id)
                    if error:
                        st.error(error)
                    else:
                        st.session_state["drive_documents"] = drive_documents
                        st.success(
                            f"Loaded {len(drive_documents)} document(s) from Google Drive."
                        )
        else:
            st.caption(
                "If Google Drive is configured in Streamlit secrets, enter the "
                "folder ID here. Otherwise use uploaded files."
            )

        if "drive_documents" in st.session_state:
            drive_documents = st.session_state["drive_documents"]

        local_documents = []
        for uploaded in uploaded_docs or []:
            text = extract_uploaded_document(uploaded)
            if text:
                local_documents.append(
                    {
                        "name": uploaded.name,
                        "text": clean_text(text),
                    }
                )

        all_documents = drive_documents + local_documents

        if all_documents:
            st.write(f"Knowledge documents available: **{len(all_documents)}**")
            rag_index = build_rag_index(all_documents)
            st.session_state["rag_index"] = rag_index
        else:
            rag_index = st.session_state.get("rag_index")

        if rag_index:
            st.success(
                f"RAG index ready: {len(rag_index['chunks'])} text chunks."
            )
        else:
            st.warning(
                "No knowledge documents are loaded. Incident analysis can still "
                "be entered, but document-based comparison will not be available."
            )

    st.divider()

    subject = st.text_input("Incident subject")
    description = st.text_area(
        "Incident description",
        height=180,
        placeholder=(
            "Example: Users reported intermittent service degradation. "
            "Enter only facts known to the engineer."
        ),
    )

    top_k = st.slider("Number of knowledge chunks to retrieve", 1, 8, 5)

    if st.button("Analyze Incident", type="primary"):
        if not subject.strip() or not description.strip():
            st.warning("Please provide both incident subject and description.")
            return

        rag_index = st.session_state.get("rag_index")
        results = retrieve_context(rag_index, description, top_k)

        st.subheader("Retrieved Evidence")
        if results:
            for item in results:
                with st.expander(
                    f"{item['document']} | relevance {item['score']:.2f}"
                ):
                    st.write(item["text"])
        else:
            st.info("No relevant knowledge-base evidence was retrieved.")

        with st.spinner("Generating AI assessment..."):
            answer, error = incident_analysis(
                subject,
                description,
                results,
            )

        st.subheader("AI-NOC Assessment")
        if error:
            st.error(error)
        else:
            st.markdown(answer)

        st.warning(
            "Engineer control: verify alarms, logs, counters, topology, "
            "configuration, and other operational evidence before taking action."
        )


def health_module():
    st.title("📊 Network Health")
    st.write(
        "Upload a CSV containing network metrics. The tool screens the data "
        "for common performance/error indicators."
    )

    uploaded = st.file_uploader(
        "Upload network health CSV",
        type=["csv"],
        key="health_csv",
    )

    if not uploaded:
        st.info(
            "Expected examples of metric columns include utilization, latency, "
            "packet_loss, errors, CRC, drops, or availability."
        )
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.subheader("Uploaded Data")
    st.dataframe(df, use_container_width=True)

    result = network_health_analysis(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Health Score", f"{result['score']:.0f}/100")
    col2.metric("Status", result["status"])
    col3.metric("Records", result["summary"]["rows"])

    st.subheader("Detected Metrics")
    for category, columns in result["groups"].items():
        if columns:
            st.write(f"**{category.title()}:** {', '.join(columns)}")

    st.subheader("Findings")
    if result["findings"]:
        for finding in result["findings"]:
            st.warning(finding)
    else:
        st.success(
            "No issues were flagged by the built-in screening rules."
        )

    st.caption(
        "Important: the score and thresholds are generic screening heuristics, "
        "not vendor-specific thresholds or an SLA determination. Confirm "
        "against your network's actual design, vendor guidance, and KPIs."
    )


def report_module():
    st.title("📝 NOC Report Generator")
    st.write(
        "Create a structured incident report from facts entered by the NOC engineer."
    )

    subject = st.text_input("Incident subject", key="report_subject")
    summary = st.text_area("Incident summary", height=120)
    start_time = st.text_input(
        "Start time",
        placeholder="YYYY-MM-DD HH:MM",
    )
    end_time = st.text_input(
        "End time",
        placeholder="YYYY-MM-DD HH:MM",
    )
    root_cause = st.text_area("Root cause")
    impacted_services = st.text_area("Impacted services")
    recommendations = st.text_area("Recommendations")

    if st.button("Generate NOC Report", type="primary"):
        if not subject.strip() or not summary.strip():
            st.warning("Incident subject and summary are required.")
            return

        report = generate_noc_report(
            subject,
            summary,
            start_time,
            end_time,
            root_cause,
            impacted_services,
            recommendations,
        )

        st.subheader("Generated Report")
        st.markdown(report)

        st.download_button(
            "Download Markdown Report",
            data=report,
            file_name="noc_incident_report.md",
            mime="text/markdown",
        )


# -----------------------------
# Main
# -----------------------------
def main():
    module = show_sidebar()

    if module == "Incident Analysis":
        incident_module()
    elif module == "Network Health":
        health_module()
    elif module == "NOC Report Generator":
        report_module()


if __name__ == "__main__":
    main()
