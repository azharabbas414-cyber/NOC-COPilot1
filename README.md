# AI-NOC Copilot

AI-NOC Copilot is a beginner-friendly Streamlit application for NOC engineers.

It has three modules:

1. **Incident Analysis**
   - Retrieves relevant information from NOC knowledge documents.
   - Uses RAG-style retrieval with TF-IDF.
   - Sends the retrieved evidence and incident details to an OpenAI-compatible LLM endpoint.
   - Separates document evidence from the AI assessment.
   - Keeps the NOC engineer responsible for verification and the final decision.

2. **Network Health**
   - Accepts a network-metrics CSV file.
   - Detects common metric columns such as utilization, latency, packet loss, errors and availability.
   - Produces a simple screening score and findings.
   - Thresholds are deliberately described as generic heuristics, not vendor-specific SLA thresholds.

3. **NOC Report Generator**
   - Creates a structured Markdown incident report from information entered by the engineer.
   - Includes subject, summary, start/end time, root cause, impacted services and recommendations.
   - Provides a download button for the generated report.

---

## Project structure

```text
ai-noc-copilot/
├── app.py
├── requirements.txt
└── README.md
```

---

## 1. Run locally

Install Python 3.10+.

Create a virtual environment if desired:

```bash
python3 -m venv .venv
```

Activate it.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

# 2. LLM configuration

The application uses an **OpenAI-compatible chat-completions API**.

The default endpoint is:

```text
https://api.groq.com/openai/v1/chat/completions
```

For Groq, configure:

```toml
LLM_API_KEY = "YOUR_GROQ_API_KEY"
LLM_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"
```

You can also use:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

instead of `LLM_API_KEY`.

The code intentionally uses generic names such as `LLM_BASE_URL` and `LLM_MODEL` so an OpenAI-compatible provider can be configured without changing the application logic.

**Do not hard-code API keys inside `app.py`.**

---

# 3. Streamlit Cloud deployment

Push these files to a GitHub repository:

```text
app.py
requirements.txt
README.md
```

Then create a Streamlit Cloud application and select:

```text
app.py
```

Add secrets in the Streamlit Cloud application's Secrets section.

Minimum LLM configuration:

```toml
LLM_API_KEY = "YOUR_API_KEY"
LLM_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "llama-3.3-70b-versatile"
```

---

# 4. Google Drive knowledge base

The Incident Analysis module can read NOC knowledge documents from a Google Drive folder.

The application supports:

- TXT
- Markdown
- CSV
- PDF
- DOCX
- Google Docs
- Google Sheets

## Google service account setup

Create a Google Cloud service account and enable the Google Drive API.

Create credentials for the service account.

In Streamlit Cloud Secrets, add the service-account object.

Example:

```toml
[google_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
token_uri = "https://oauth2.googleapis.com/token"
```

Share the NOC knowledge-base Google Drive folder with the service account's `client_email`.

Then enter the Google Drive folder ID in the Incident Analysis module.

You can optionally store the folder ID as:

```toml
GOOGLE_DRIVE_FOLDER_ID = "YOUR_FOLDER_ID"
```

---

# 5. How RAG works in this project

The simplified RAG flow is:

```text
Google Drive / Uploaded Documents
             |
             v
      Text Extraction
             |
             v
        Text Chunking
             |
             v
       TF-IDF Index
             |
             v
     Incident Description
             |
             v
     Relevant Chunks
             |
             v
    LLM + Retrieved Evidence
             |
             v
      AI-NOC Assessment
```

The application does **not** allow the LLM to automatically change network devices.

The retrieved documents are provided as evidence to the model.

The engineer must verify the result using operational evidence.

---

# 6. Incident Analysis example

Enter information such as:

```text
Subject:
IPTV service degradation

Description:
Users reported intermittent service degradation. The engineer observed
increased interface errors on the affected path.
```

The system retrieves relevant knowledge-base chunks and asks the LLM to provide:

- Incident classification
- Evidence from the knowledge base
- AI assessment
- Possible root causes
- Recommended checks
- Confidence
- Engineer verification / final decision

The AI must identify unsupported conclusions as hypotheses rather than facts.

---

# 7. Network Health CSV

The Network Health module accepts a CSV.

Example:

```csv
timestamp,device,interface,utilization,latency_ms,packet_loss_pct,errors,availability_pct
2026-09-12 10:00,R1,Gig0/0,45,20,0,0,100
2026-09-12 10:05,R1,Gig0/0,92,120,1.2,3,98.5
2026-09-12 10:10,R1,Gig0/0,95,140,2.1,7,97.8
```

The application detects common metric names.

The built-in screening rules currently use generic indicators such as:

- Utilization >= 90
- Latency >= 100
- Packet loss >= 1
- Non-zero errors
- Availability < 99

These are **not universal network thresholds**.

For a production NOC, replace them with thresholds approved for your specific:

- Vendor
- Technology
- Service
- Interface
- SLA
- Network design

---

# 8. NOC Report Generator

The engineer enters:

- Incident subject
- Incident summary
- Start time
- End time
- Root cause
- Impacted services
- Recommendations

The application generates:

```text
NOC Incident Report
|
+-- Incident Details
+-- Incident Summary
+-- Root Cause
+-- Impacted Services
+-- Recommendations
+-- Engineer Validation
```

The report is downloadable as Markdown.

---

# 9. Important NOC safety principle

AI-NOC Copilot is an **assistive tool**, not an autonomous network controller.

The application should never be treated as proof that:

- a device is faulty,
- a link is congested,
- a route is incorrect,
- a configuration is wrong,
- a service is impacted,
- a root cause is confirmed.

The NOC engineer should verify conclusions using appropriate evidence such as:

- alarms
- logs
- interface counters
- traffic statistics
- routing information
- configuration
- topology
- monitoring systems
- service KPIs
- change records

Final operational decisions remain with the responsible NOC/network engineer.

---

# 10. Future enhancements

Recommended next steps for the project:

1. Replace TF-IDF with a production vector database/embedding pipeline such as FAISS.
2. Add document metadata and source/page references.
3. Add scheduled Google Drive ingestion.
4. Add incident history and feedback.
5. Add vendor-specific health thresholds.
6. Add charts to Network Health.
7. Add PDF/DOCX report export.
8. Add authentication and role-based access.
9. Add audit logging.
10. Add evaluation datasets for RAG and LLM responses.
11. Add confidence and evidence scoring.
12. Add a controlled workflow for engineer approval before any future automation.

