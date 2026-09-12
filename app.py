import os
import re
import io
import json
import streamlit as st
import pandas as pd
from openai import OpenAI
from urllib.request import urlopen, Request
from urllib.parse import urlparse, parse_qs, quote

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


# =========================================================
# Fixed Google Drive knowledge source
# =========================================================
# Paste your Google Drive file/folder link ONCE here.
# After that, the app loads this source automatically; no repeated upload/paste is needed.
FIXED_GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1nJwrAhBnX9wjuo4TtWNSOtvvq8gl6apT"


# =========================================================
# 1. Text extraction
# =========================================================
def extract_text_from_bytes(data, filename):
    name = filename.lower()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    if name.endswith(".pdf") and PdfReader:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx") and Document:
        document = Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)

    return ""


def split_into_chunks(text, words_per_chunk=700):
    words = text.split()
    return [
        " ".join(words[i:i + words_per_chunk]).strip()
        for i in range(0, len(words), words_per_chunk)
        if " ".join(words[i:i + words_per_chunk]).strip()
    ]


# =========================================================
# 2. Google Drive helpers
# =========================================================
def get_google_drive_api_key():
    try:
        return st.secrets["GOOGLE_DRIVE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_DRIVE_API_KEY")


def extract_drive_id(url):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def google_drive_api(url):
    api_key = get_google_drive_api_key()
    if not api_key:
        raise ValueError("GOOGLE_DRIVE_API_KEY is not configured.")

    file_id = extract_drive_id(url)
    if not file_id:
        raise ValueError("Could not find a Google Drive file/folder ID in the link.")

    return file_id, api_key


def drive_get_metadata(file_id, api_key):
    endpoint = (
        "https://www.googleapis.com/drive/v3/files/"
        + file_id
        + "?fields=id,name,mimeType,size&key="
        + api_key
    )
    request = Request(endpoint, headers={"Accept": "application/json"})

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def drive_download_file(file_id, api_key):
    metadata = drive_get_metadata(file_id, api_key)
    name = metadata.get("name", "drive_file")

    mime = metadata.get("mimeType", "")

    if mime == "application/vnd.google-apps.document":
        export_url = (
            "https://www.googleapis.com/drive/v3/files/"
            + file_id
            + "/export?mimeType=text/plain&key="
            + api_key
        )
        request = Request(export_url, headers={"Accept": "text/plain"})
        with urlopen(request, timeout=30) as response:
            return name + ".txt", response.read()

    if mime.startswith("application/vnd.google-apps."):
        raise ValueError(
            f"Google Workspace file '{name}' is not supported yet. "
            "Use a TXT, PDF, or DOCX file in the Drive folder."
        )

    download_url = (
        "https://www.googleapis.com/drive/v3/files/"
        + file_id
        + "?alt=media&key="
        + api_key
    )
    request = Request(download_url)

    with urlopen(request, timeout=60) as response:
        return name, response.read()


def drive_list_folder(folder_id, api_key):
    query = (
        "'"
        + folder_id
        + "' in parents and trashed = false"
        + " and (mimeType = 'text/plain'"
        + " or mimeType = 'application/pdf'"
        + " or mimeType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
    )

    endpoint = (
        "https://www.googleapis.com/drive/v3/files"
        "?q=" + quote(query)
        + "&fields=files(id,name,mimeType,size)"
        + "&pageSize=100"
        + "&key=" + api_key
    )

    request = Request(endpoint, headers={"Accept": "application/json"})

    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8")).get("files", [])


@st.cache_data(ttl=3600, show_spinner=False)
def load_google_drive(url):
    file_id, api_key = google_drive_api(url)
    metadata = drive_get_metadata(file_id, api_key)

    chunks = []
    loaded_names = []

    if metadata.get("mimeType") == "application/vnd.google-apps.folder":
        files = drive_list_folder(file_id, api_key)

        if not files:
            raise ValueError(
                "No supported TXT, PDF, or DOCX files were found in the folder."
            )

        for item in files:
            name, data = drive_download_file(item["id"], api_key)
            text = extract_text_from_bytes(data, name)

            if text.strip():
                chunks.extend(split_into_chunks(text))
                loaded_names.append(name)
    else:
        name, data = drive_download_file(file_id, api_key)
        text = extract_text_from_bytes(data, name)

        if not text.strip():
            raise ValueError(
                "The Drive file could not be converted into readable text."
            )

        chunks.extend(split_into_chunks(text))
        loaded_names.append(name)

    return chunks, loaded_names


# =========================================================
# 3. Focused RAG retrieval
# =========================================================
# This is intentionally lightweight: no vector database or embeddings.
# The retriever removes common words, gives extra weight to exact/domain
# terms, and filters weak matches so unrelated incidents are not shown.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "why", "how",
    "can", "could", "would", "should", "do", "does", "did", "to", "of",
    "in", "on", "for", "from", "with", "and", "or", "my", "your", "this",
    "that", "it", "be", "go", "goes", "going", "cause", "causes", "reason",
    "problem", "issue", "incident", "network", "session", "down",
}

# Important NOC/domain terms get stronger matching.
DOMAIN_TERMS = {
    "bgp", "ospf", "mtu", "crc", "packet", "loss", "congestion",
    "interface", "neighbor", "adjacency", "peer", "prefix", "route",
    "routing", "authentication", "area", "optic", "fiber", "duplex",
}

def get_words(text):
    return set(re.findall(r"[a-zA-Z0-9_-]+", text.lower()))


def retrieve_knowledge(question, uploaded_chunks, top_k=4):
    query_words = get_words(question)
    meaningful_query = query_words - STOP_WORDS

    # If the question contains a strong protocol/topic term, prefer only
    # chunks that contain that same topic. This prevents a BGP question from
    # returning OSPF/MTU/CRC incidents merely because they also say "down".
    topic_terms = meaningful_query.intersection(DOMAIN_TERMS)

    candidates = []
    for number, chunk in enumerate(uploaded_chunks, start=1):
        chunk_words = get_words(chunk)

        if topic_terms and not topic_terms.intersection(chunk_words):
            continue

        overlap = meaningful_query.intersection(chunk_words)
        score = len(overlap)

        # Stronger score for domain terms and exact multi-word phrases.
        score += 2 * len(overlap.intersection(DOMAIN_TERMS))

        q_lower = question.lower().strip()
        c_lower = chunk.lower()
        if q_lower and q_lower in c_lower:
            score += 10

        # Reward common incident phrases such as "bgp session", "ospf
        # neighbor", "crc errors", and "mtu mismatch".
        phrase_hits = 0
        for phrase in (
            "bgp session", "bgp peer", "ospf neighbor", "ospf adjacency",
            "crc errors", "crc error", "packet loss", "interface errors",
            "mtu mismatch", "mtu problem", "link congestion",
        ):
            if phrase in q_lower and phrase in c_lower:
                phrase_hits += 4
        score += phrase_hits

        if score > 0:
            candidates.append((score, len(overlap), f"RAG chunk {number}", chunk))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score = candidates[0][0]

    # Keep only strong matches. If one document is clearly the best match,
    # return only that document rather than filling the UI with weak chunks.
    strong = [item for item in candidates if item[0] >= max(3, best_score * 0.60)]

    # For a focused incident query, one highly relevant chunk is preferred.
    if topic_terms and strong:
        strong = strong[:1]

    return [
        {"source": source, "text": text}
        for _, _, source, text in strong[:top_k]
    ]


# =========================================================
# 4. Groq
# =========================================================
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def call_llm(question, rag_context=None):
    api_key = get_groq_key()

    if not api_key:
        return None, "GROQ_API_KEY is not configured."

    if rag_context:
        prompt = f"""
You are an AI NOC Copilot.

Answer the user's question using the retrieved content from the user's
RAG documents.

USER QUESTION:
{question}

RETRIEVED RAG CONTENT:
{rag_context}

Rules:
- Use the RAG content as the primary source.
- Clearly explain the answer.
- Do not invent information not supported by the context.
- If the context is insufficient, say so.
"""
    else:
        prompt = f"""
You are an AI NOC Copilot.

Answer the user's question using your general model knowledge.

USER QUESTION:
{question}

Rules:
- Give a clear answer.
- Do not invent device output or network measurements.
- Clearly say when evidence is insufficient.
"""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful NOC AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content, None

    except Exception as exc:
        return None, str(exc)


# =========================================================
# 5. Clear
# =========================================================
def clear_results():
    st.session_state.question = ""
    st.session_state.answer = ""
    st.session_state.rag_results = []
    st.session_state.error = ""
    st.session_state.drive_chunks = []
    st.session_state.drive_files = []


# =========================================================

def classify_incident(question):
    """Lightweight rule-based incident classification for the learning/demo app."""
    q = question.lower()

    protocol_rules = [
        ("BGP", ["bgp", "border gateway", "peer session", "bgp peer"]),
        ("OSPF", ["ospf", "neighbor adjacency", "ospf neighbor"]),
        ("MTU", ["mtu", "maximum transmission unit", "large ping", "fragmentation"]),
        ("Interface / Physical", ["crc", "input errors", "interface error", "optic", "transceiver", "fiber", "duplex"]),
        ("Packet Loss / Congestion", ["packet loss", "congestion", "utilization", "interface utilization", "drops"]),
    ]

    matched = []
    for protocol, keywords in protocol_rules:
        if any(k in q for k in keywords):
            matched.append(protocol)

    if matched:
        protocol = matched[0]
    else:
        protocol = "General Network"

    incident_rules = [
        ("Session Down", ["session down", "peer down", "bgp down", "neighbor down", "adjacency down"]),
        ("Packet Loss", ["packet loss", "packet drops", "drops"]),
        ("CRC / Interface Errors", ["crc", "input errors", "interface errors"]),
        ("MTU / Fragmentation", ["mtu", "fragmentation", "large ping", "large packets"]),
        ("Congestion / High Utilization", ["congestion", "high utilization", "95% utilization", "interface utilization"]),
        ("Connectivity Failure", ["cannot reach", "unreachable", "connectivity", "no connectivity"]),
    ]

    incident = "Network Incident"
    for name, keywords in incident_rules:
        if any(k in q for k in keywords):
            incident = name
            break

    category_map = {
        "BGP": "Routing",
        "OSPF": "Routing",
        "MTU": "IP / Transport",
        "Interface / Physical": "Physical / Interface",
        "Packet Loss / Congestion": "Performance",
        "General Network": "Network Operations",
    }
    category = category_map.get(protocol, "Network Operations")

    return {
        "protocol": protocol,
        "incident": incident,
        "category": category,
    }


# =========================================================
# 6. UI — modular NOC dashboard
# =========================================================
st.set_page_config(
    page_title="AI-NOC Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#030303; color:#f5f5f5; }
[data-testid="stHeader"] { background:rgba(0,0,0,0); }
.block-container { max-width:1500px; padding:.55rem 1.25rem .35rem; }
[data-testid="stSidebar"] { background:#070707; border-right:1px solid #252525; }
[data-testid="stSidebar"] * { color:#f2f2f2; }

.hero {
  background:radial-gradient(circle at 82% 50%, rgba(50,120,180,.16), transparent 34%),
             linear-gradient(135deg,#101010,#070707);
  border:1px solid #292929; border-radius:16px;
  padding:.65rem 1rem; margin-bottom:.5rem;
  min-height:112px; display:flex; align-items:center; justify-content:space-between;
  overflow:hidden;
}
.hero h1 {margin:0;font-size:1.65rem;color:#fff;}
.hero p {margin:.12rem 0 0;color:#999;font-size:.78rem;}
.hero-kicker {color:#70c8ff;font-size:.62rem;letter-spacing:.12em;font-weight:700;}
.noc-visual {width:400px;flex:0 0 400px;}

.sidebar-brand {font-size:1.05rem;font-weight:800;margin-bottom:.15rem;}
.sidebar-sub {color:#8d8d8d;font-size:.74rem;margin-bottom:.7rem;}
.module-note {
  background:#0c0c0c;border:1px solid #282828;border-radius:10px;
  padding:.55rem .65rem;margin-top:.7rem;color:#999;font-size:.7rem;line-height:1.35;
}
.section-label {color:#d0d0d0;font-size:.69rem;text-transform:uppercase;letter-spacing:.1em;margin:.22rem 0 .18rem;font-weight:700;}
.status-card {
  background:#090909;border:1px solid #292929;border-radius:9px;
  padding:.38rem .6rem;color:#c8c8c8;font-size:.73rem;margin-bottom:.25rem;
}
div[data-testid="stTextArea"] textarea {
  background:#0b0b0b !important;color:#f7f7f7 !important;
  border:1px solid #3a3a3a !important;border-radius:10px !important;
  min-height:60px !important;
}
div[data-testid="stTextArea"] textarea::placeholder {color:#9b9b9b !important;opacity:1 !important;}
div[data-testid="stRadio"] > div {gap:.35rem;}
div[data-testid="stRadio"] label {
  background:#111 !important;border:1px solid #3a3a3a !important;
  border-radius:10px !important;padding:.25rem .55rem !important;color:#fff !important;opacity:1 !important;
}
div[data-testid="stRadio"] label p, div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label div {color:#fff !important;opacity:1 !important;}
div[data-testid="stButton"] button {border-radius:10px;min-height:2.15rem;font-weight:700;}
div[data-testid="stButton"] button p {color:#fff !important;}
.class-card {
  background:#0b0b0b;border:1px solid #2b2b2b;border-radius:9px;padding:.4rem .6rem;min-height:48px;
}
.class-card span {display:block;color:#777;font-size:.59rem;letter-spacing:.1em;font-weight:700;}
.class-card strong {display:block;color:#f5f5f5;font-size:.79rem;margin-top:.1rem;}
.source-caption {color:#777;font-size:.66rem;text-align:right;padding-top:.42rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.result-title {font-size:.92rem;font-weight:750;margin-bottom:.15rem;color:#f5f5f5;}
.result-sub {color:#8f8f8f;font-size:.69rem;margin-bottom:.3rem;}
.compare-note {color:#777;font-size:.66rem;margin-top:.2rem;}
.health-card {
  background:#0b0b0b;border:1px solid #292929;border-radius:12px;
  padding:.75rem .85rem;min-height:92px;
}
.health-card .label {color:#777;font-size:.6rem;letter-spacing:.1em;font-weight:700;}
.health-card .value {color:#fff;font-size:1.15rem;font-weight:800;margin-top:.15rem;}
.report-box {background:#0b0b0b;border:1px solid #292929;border-radius:12px;padding:.85rem;}
footer {visibility:hidden;}

.stFileUploader {
  background:#0b0b0b; border:1px dashed #3a3a3a; border-radius:10px; padding:.15rem;
}

.stTextInput, .stTextArea { margin-bottom:.18rem; }
form [data-testid="stTextInput"] input,
form [data-testid="stTextArea"] textarea { color:#f5f5f5 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar: one module list only (no duplicate explanatory cards).
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🤖 AI-NOC Copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">AI-powered NOC learning & analysis</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### MODULES")
    module = st.radio(
        "Module",
        ["🚨 Incident Analysis", "📊 Network Health", "📑 NOC Report Generator"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    descriptions = {
        "🚨 Incident Analysis": "Investigate incidents using AI + your NOC knowledge base.",
        "📊 Network Health": "Review network metrics and identify potential issues.",
        "📑 NOC Report Generator": "Create a structured incident report from analysis.",
    }
    st.markdown(
        f'<div class="module-note"><b>{module}</b><br>{descriptions[module]}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Learning / Demo Project")

# One topology banner only.
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-kicker">NETWORK OPERATIONS • AI ASSISTANT</div>
    <h1>🤖 AI-NOC Copilot</h1>
    <p>Investigate • Assess • Report</p>
  </div>
  <div class="noc-visual">
    <svg viewBox="0 0 500 150" xmlns="http://www.w3.org/2000/svg">
      <g fill="none" stroke="#4e9ed1" stroke-opacity=".65" stroke-width="2">
        <path d="M45 75 L150 38 L250 75 L350 38 L455 75"/>
        <path d="M45 75 L150 112 L250 75 L350 112 L455 75"/>
        <path d="M150 38 L150 112"/><path d="M350 38 L350 112"/>
      </g>
      <g fill="#090909" stroke="#70c8ff" stroke-width="2">
        <rect x="22" y="53" width="46" height="44" rx="9"/><rect x="127" y="16" width="46" height="44" rx="9"/>
        <rect x="127" y="90" width="46" height="44" rx="9"/><rect x="227" y="53" width="46" height="44" rx="9"/>
        <rect x="327" y="16" width="46" height="44" rx="9"/><rect x="327" y="90" width="46" height="44" rx="9"/>
        <rect x="432" y="53" width="46" height="44" rx="9"/>
      </g>
      <g fill="#dff5ff" font-family="Arial" font-size="9" text-anchor="middle">
        <text x="45" y="80">EDGE</text><text x="150" y="43">R1</text><text x="150" y="117">R2</text>
        <text x="250" y="80">CORE</text><text x="350" y="43">R3</text><text x="350" y="117">R4</text><text x="455" y="80">NOC</text>
      </g>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state
for key, default in {
    "question": "", "answer": "", "rag_results": [], "error": "",
    "drive_chunks": [], "drive_files": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

def render_incident_analysis():
    try:
        with st.spinner("Loading NOC knowledge base..."):
            rag_chunks, drive_files = load_google_drive(FIXED_GOOGLE_DRIVE_URL)
        st.session_state.drive_chunks = rag_chunks
        st.session_state.drive_files = drive_files
        status_text = f"● NOC knowledge ready  •  {len(drive_files)} docs  •  {len(rag_chunks)} chunks"
    except Exception as exc:
        rag_chunks = st.session_state.get("drive_chunks", [])
        drive_files = st.session_state.get("drive_files", [])
        status_text = f"⚠ Knowledge base: {exc}"

    st.markdown(f'<div class="status-card">{status_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Ask your NOC question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Question", value=st.session_state.question,
        placeholder="Example: What can cause a BGP session to go down?",
        height=68, label_visibility="collapsed",
    )

    info = classify_incident(question) if question.strip() else {"protocol":"—","incident":"—","category":"—"}
    st.markdown('<div class="section-label">Automatic incident classification</div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown(f'<div class="class-card"><span>PROTOCOL</span><strong>{info["protocol"]}</strong></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="class-card"><span>INCIDENT</span><strong>{info["incident"]}</strong></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="class-card"><span>CATEGORY</span><strong>{info["category"]}</strong></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Choose answer mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Answer mode",
        ["🧠 AI Response", "📚 RAG Response", "🔍 RAG + AI Comparison"],
        index=0, horizontal=True, label_visibility="collapsed",
    )
    c1,c2,c3=st.columns([1.35,1,1])
    with c1:
        analyze=st.button("🚀 Analyze Incident",type="primary",use_container_width=True)
    with c2:
        clear=st.button("🗑️ Clear",use_container_width=True)
    with c3:
        if drive_files:
            st.markdown(f'<div class="source-caption">📁 Fixed Google Drive • {len(drive_files)} documents</div>',unsafe_allow_html=True)

    if clear:
        clear_results()
        st.rerun()

    if analyze:
        q=question.strip()
        if not q:
            st.warning("Enter a NOC question first.")
            return
        st.session_state.question=q
        st.session_state.answer=""
        st.session_state.rag_results=[]
        st.session_state.error=""

        if mode=="🧠 AI Response":
            with st.spinner("Generating AI response..."):
                answer,error=call_llm(q)
            if error: st.session_state.error=error
            else: st.session_state.answer=answer
        elif mode=="📚 RAG Response":
            if not rag_chunks:
                st.session_state.error="No RAG documents are available."
            else:
                results=retrieve_knowledge(q,rag_chunks)
                st.session_state.rag_results=results
                if not results:
                    st.session_state.error="No relevant content was found in the NOC knowledge base."
        else:
            if not rag_chunks:
                st.session_state.error="No RAG documents are available."
            else:
                results=retrieve_knowledge(q,rag_chunks)
                st.session_state.rag_results=results
                context="\n\n".join(f"SOURCE: {x['source']}\n{x['text']}" for x in results)
                with st.spinner("Comparing RAG evidence with AI reasoning..."):
                    answer,error=call_llm(q,rag_context=context if context else None)
                if error: st.session_state.error=error
                else: st.session_state.answer=answer

    if st.session_state.error:
        st.error(st.session_state.error)
    rag=st.session_state.rag_results
    ans=st.session_state.answer
    if mode=="🔍 RAG + AI Comparison" and (rag or ans):
        left,right=st.columns(2,gap="medium")
        with left:
            st.markdown('<div class="result-title">📚 RAG Evidence</div><div class="result-sub">What your NOC documents say</div>',unsafe_allow_html=True)
            with st.container(height=290,border=True):
                if rag:
                    for item in rag:
                        st.markdown(f"**{item['source']}**")
                        st.write(item["text"])
                else: st.info("No matching RAG evidence.")
        with right:
            st.markdown('<div class="result-title">🧠 AI Explanation</div><div class="result-sub">LLM explanation grounded in retrieved evidence</div>',unsafe_allow_html=True)
            with st.container(height=290,border=True):
                if ans: st.markdown(ans)
                else: st.info("No AI response.")
    elif rag:
        st.markdown('<div class="result-title">📚 RAG Retrieved Result</div><div class="result-sub">Focused evidence from your NOC knowledge base</div>',unsafe_allow_html=True)
        with st.container(height=290,border=True):
            for item in rag:
                st.markdown(f"**{item['source']}**")
                st.write(item["text"])
    elif ans:
        st.markdown('<div class="result-title">🧠 AI Response</div><div class="result-sub">Generated from general model knowledge</div>',unsafe_allow_html=True)
        with st.container(height=290,border=True):
            st.markdown(ans)

    st.markdown('<div class="compare-note">Learning/demo project — verify AI answers against real network evidence before operational use.</div>',unsafe_allow_html=True)

def render_network_health():
    st.markdown("### 📊 Network Health")
    st.caption("Upload a CSV containing network interface metrics, then generate a health-check summary.")

    st.markdown('<div class="section-label">Upload network health CSV</div>', unsafe_allow_html=True)
    uploaded_csv = st.file_uploader(
        "CSV file",
        type=["csv"],
        help="Required columns: Interface, Utilization, Packet Loss, CRC Errors.",
        label_visibility="collapsed",
    )

    if uploaded_csv is None:
        st.info("Upload a CSV file to begin the health check.")
        st.caption("Required columns: Interface, Utilization, Packet Loss, CRC Errors")
        return

    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        return

    aliases = {
        "interface": "Interface",
        "interface_name": "Interface",
        "utilization": "Utilization",
        "utilization_%": "Utilization",
        "packet_loss": "Packet Loss",
        "packet loss": "Packet Loss",
        "packet_loss_%": "Packet Loss",
        "crc_errors": "CRC Errors",
        "crc errors": "CRC Errors",
        "crc": "CRC Errors",
    }
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in aliases:
            rename[col] = aliases[key]
    df = df.rename(columns=rename)

    required = ["Interface", "Utilization", "Packet Loss", "CRC Errors"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        st.info("Required columns: Interface, Utilization, Packet Loss, CRC Errors")
        return

    for col in ["Utilization", "Packet Loss", "CRC Errors"]:
        df[col] = pd.to_numeric(
            df[col].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )

    df["Interface"] = df["Interface"].astype(str).str.strip()
    df = df.dropna(subset=["Interface", "Utilization", "Packet Loss", "CRC Errors"]).copy()

    if df.empty:
        st.error("The CSV contains no valid network interface records.")
        return

    st.markdown(
        f'<div class="status-card">📄 <b>{uploaded_csv.name}</b> &nbsp;•&nbsp; {len(df)} interface record(s) loaded</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Uploaded Metrics</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("")
    generate = st.button("📊 Generate Health Check Summary", type="primary", use_container_width=True)

    if not generate:
        st.caption("Review the uploaded metrics, then click the button above to generate the summary.")
        return

    def health_status(row):
        if row["Packet Loss"] >= 3 or row["CRC Errors"] > 100 or row["Utilization"] >= 95:
            return "🔴 Investigate"
        if row["Packet Loss"] > 0 or row["CRC Errors"] > 0 or row["Utilization"] >= 80:
            return "🟠 Attention"
        return "🟢 Healthy"

    df["Status"] = df.apply(health_status, axis=1)

    critical = df[df["Status"] == "🔴 Investigate"]
    attention = df[df["Status"] == "🟠 Attention"]
    healthy = df[df["Status"] == "🟢 Healthy"]

    if len(critical):
        overall = "🔴 Investigate"
    elif len(attention):
        overall = "🟠 Attention"
    else:
        overall = "🟢 Healthy"

    high_util_row = df.loc[df["Utilization"].idxmax()]
    max_loss_row = df.loc[df["Packet Loss"].idxmax()]
    max_crc_row = df.loc[df["CRC Errors"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="health-card"><div class="label">OVERALL HEALTH</div><div class="value">{overall}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="health-card"><div class="label">INTERFACES</div><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="health-card"><div class="label">INVESTIGATE</div><div class="value">{len(critical)}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="health-card"><div class="label">ATTENTION</div><div class="value">{len(attention)}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Health Check Summary")
    summary_parts = [
        f"**Overall status:** {overall}.",
        f"**Highest utilization:** {high_util_row['Interface']} at {high_util_row['Utilization']:.1f}%.",
        f"**Highest packet loss:** {max_loss_row['Interface']} at {max_loss_row['Packet Loss']:.2f}%.",
        f"**Highest CRC errors:** {max_crc_row['Interface']} with {max_crc_row['CRC Errors']:.0f}.",
        f"**Healthy interfaces:** {len(healthy)} of {len(df)}.",
    ]
    for item in summary_parts:
        st.write("• " + item)

    if len(critical):
        st.markdown("#### 🔴 Interfaces Requiring Investigation")
        for _, row in critical.iterrows():
            reasons = []
            if row["Utilization"] >= 95:
                reasons.append(f"utilization {row['Utilization']:.1f}%")
            if row["Packet Loss"] >= 3:
                reasons.append(f"packet loss {row['Packet Loss']:.2f}%")
            if row["CRC Errors"] > 100:
                reasons.append(f"CRC errors {row['CRC Errors']:.0f}")
            st.write(f"• **{row['Interface']}** — " + ", ".join(reasons) + ".")

    if len(attention):
        st.markdown("#### 🟠 Interfaces Requiring Attention")
        for _, row in attention.iterrows():
            reasons = []
            if row["Utilization"] >= 80:
                reasons.append(f"utilization {row['Utilization']:.1f}%")
            if row["Packet Loss"] > 0:
                reasons.append(f"packet loss {row['Packet Loss']:.2f}%")
            if row["CRC Errors"] > 0:
                reasons.append(f"CRC errors {row['CRC Errors']:.0f}")
            st.write(f"• **{row['Interface']}** — " + ", ".join(reasons) + ".")

    st.caption("Learning/demo project: health thresholds are simplified for demonstration and should be validated against real network baselines.")

def render_report_generator():
    st.markdown("### 📑 NOC Report Generator")
    st.caption("Enter the details of a network incident and generate a structured NOC incident report.")

    with st.form("report_form"):
        st.markdown('<div class="section-label">Incident Details</div>', unsafe_allow_html=True)

        st.markdown("**Incident Subject**")
        incident_subject = st.text_input(
            "Incident Subject",
            placeholder="Example: BGP Session Down between PE1 and PE2",
            label_visibility="collapsed",
        )

        st.markdown("**Incident Summary**")
        incident_summary = st.text_area(
            "Incident Summary",
            placeholder="Describe what happened, what was observed, and the impact.",
            height=75,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Incident Start Time**")
            start_time = st.text_input(
                "Incident Start Time",
                placeholder="Example: 2026-09-12 10:30 PKT",
                label_visibility="collapsed",
            )
        with c2:
            st.markdown("**Incident End Time**")
            end_time = st.text_input(
                "Incident End Time",
                placeholder="Example: 2026-09-12 11:15 PKT",
                label_visibility="collapsed",
            )

        st.markdown("**Root Cause of Incident**")
        root_cause = st.text_area(
            "Root Cause",
            placeholder="Describe the confirmed or probable root cause. If not confirmed, state that clearly.",
            height=75,
            label_visibility="collapsed",
        )

        st.markdown("**Services Impacted**")
        services_impacted = st.text_area(
            "Services Impacted",
            placeholder="Example: Internet access, IP transit, BGP routes, customer VPN services",
            height=65,
            label_visibility="collapsed",
        )

        st.markdown("**Recommendations / Checks**")
        recommendations = st.text_area(
            "Recommendations",
            placeholder="List recommended checks, corrective actions, or follow-up items.",
            height=85,
            label_visibility="collapsed",
        )

        generate = st.form_submit_button(
            "📄 Generate Incident Report",
            type="primary",
            use_container_width=True,
        )

    if generate:
        required_fields = {
            "Incident Subject": incident_subject,
            "Incident Summary": incident_summary,
            "Incident Start Time": start_time,
            "Incident End Time": end_time,
            "Root Cause": root_cause,
            "Services Impacted": services_impacted,
            "Recommendations / Checks": recommendations,
        }
        missing = [name for name, value in required_fields.items() if not value.strip()]
        if missing:
            st.warning("Please complete: " + ", ".join(missing))
            return

        report = f"""# NOC INCIDENT REPORT

## Incident Subject
{incident_subject}

## Incident Summary
{incident_summary}

## Incident Start Time
{start_time}

## Incident End Time
{end_time}

## Root Cause of Incident
{root_cause}

## Services Impacted
{services_impacted}

## Recommendations / Checks
{recommendations}

---
**AI-NOC Copilot**
Learning / Demo Project
"""

        st.markdown("#### Generated NOC Incident Report")
        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(report)
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            "⬇️ Download Incident Report",
            report,
            file_name="noc_incident_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

if module=="📊 Network Health":
    render_network_health()
elif module=="📑 NOC Report Generator":
    render_report_generator()
else:
    render_incident_analysis()
