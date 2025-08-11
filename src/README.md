# ADGM Corporate Agent — RAG + Multi‑Agent Review

A minimal but production‑leaning legal assistant that reviews uploaded **.docx** files against ADGM (Abu Dhabi Global Market) rules using **RAG + Gemini**. You upload a document, we chunk it, retrieve the most relevant references from a Pinecone vector DB, and run a small **analyst → reviewer → writer** agent chain to flag issues and propose compliant edits. The app produces:

* A **reviewed DOCX** (with a *Review Notes* section: preview, issues, citations, suggested fixes)
* A **structured JSON report** (per‑chunk findings)

---

## What’s inside

* **`app.py`** – Streamlit UI. Upload `.docx` → chunk → run agents → download outputs.
* **`agents.py`** – Multi‑agent pipeline (LangGraph). Per chunk: **Retrieve → Analyst → Reviewer → Writer** using **Gemini**. Dense (BGE) retrieval, optional rerank, Pinecone v2/v3 compatible.
* **(your)** `chunker.py` – Your Unstructured‑based chunker. It should return an array of chunk dicts like `{id, type, text, preview, char_count, metadata}`.

> Heads‑up: the pipeline’s parameter name for the LLM key may read like `openai_api_key` in places, but it is actually used to pass **`GOOGLE_API_KEY`** for Gemini. Keep the variable names in `.env` as shown below; the code wires it correctly.

---

## Architecture (high level)

1. **Ingest & index**
   Use **Unstructured** to parse your ADGM PDFs/HTML/DOCX → produce normalized JSONL with `text_for_embedding` and a trimmed `metadata` (include `doc_title`, `page_start/end`, `section_title`, and especially **`drive_link`** for citation).
   Embed with **BAAI/bge‑base‑en‑v1.5** (dense). Optional: add **SPLADE** sparse terms for hybrid search. Upsert to **Pinecone Serverless**.

2. **UI**
   Streamlit collects a `.docx`, runs your chunker, and shows a table of chunks.

3. **Per‑chunk agents**

   * **Retrieve**: Dense search into Pinecone (TOP‑K), then fast **cross‑encoder** rerank; keep **Top‑3** refs.
   * **Analyst** (Gemini): detect red flags vs. the references (jurisdiction, signatures, missing essentials, etc.).
   * **Reviewer** (Gemini): de‑duplicate/strengthen findings.
   * **Writer** (Gemini): propose compliant wording.

4. **Outputs**

   * **Reviewed DOCX** with a *Review Notes* section (portable; avoids fragile Word comment APIs).
   * **JSON report** summarizing issues across chunks.

---

## Prerequisites

* Python **3.10–3.11**
* A **Pinecone Serverless** index (cosine; 768‑d for BGE)
* **Google API key** for **Gemini 1.5 Flash** (or your chosen Gemini model)

---

## Setup

```bash
# 1) Create & activate a virtual env
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Add environment variables
cp .env.example .env  # or create .env and paste the block below
```

### `.env` (fill these)

```dotenv
# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=kb-hybrid-splade           # or your index name
PINECONE_NAMESPACE=adgm                   # optional; blank to use default

# Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_MODEL=gemini-1.5-flash             # recommended: 1.5‑flash
```

> **Do not** commit `.env` or any secrets.

---

## Prepare the knowledge base (once)

1. **Collect ADGM material** → store in Drive/SharePoint (see link below).
2. **Parse with Unstructured** → produce JSONL where each line has:

   ```json
   {
     "id": "<unique chunk id>",
     "text_for_embedding": "<the text>",
     "metadata": {
       "doc_title": "...",
       "page_start": 12,
       "page_end": 13,
       "section_title": "...",
       "preview": "<short snippet>",
       "context": "<up to ~3000 chars for LLM>",
       "drive_link": "https://..."   // <-- include this for citations
     }
   }
   ```
3. **Index to Pinecone** using your ingest notebook/script (dense only is fine for this app). Keep metadata small and strings/ints only.

> Tip: If you already have a hybrid (dense+sparse) index, this app still works; retrieval in `agents.py` currently uses **dense + reranker**.

---

## Run the app

```bash
streamlit run app.py
```

* Upload a `.docx` (AoA, MoA, resolutions, etc.).
* Click **“Process with Agents”**.
* Download **Reviewed\_\*.docx** and the **JSON report**.

---

## How citations work

* Each issue carries a short citation string. When available, the top reference’s **`doc_title` / pages / `drive_link`** are embedded so reviewers can jump straight to the source material.
* To get clickable links in the UI or your own wrapper, make sure your ingest step populates `drive_link`.

---

## Drive / Reference links

* **Primary knowledge base (replace with your folder):**

  * `https://drive.google.com/drive/folders/REPLACE_ME_WITH_YOUR_FOLDER_ID`
* **Additional reference pack (as provided):**

  * `https://2centscapital-my.sharepoint.com/:w:/p/nishlesh_goel/EeH965S6KtNLm3YFk8bIVvQBHDCgLTZs8RmcwCJwjdX3-w?e=u6H1YR`

> Replace the Drive URL above with your actual folder. Ensure files are accessible to your account or service user.

---

## Troubleshooting

* **`PINECONE_INDEX is not set`** → you forgot to fill `.env`.
* **No issues found / empty citations** → ensure `drive_link`, `doc_title`, `page_start/end` exist in metadata; and that your index contains the ADGM corpus you intend to reference.
* **Model mismatch** → keep `GEMINI_MODEL=gemini-1.5-flash` and provide a valid `GOOGLE_API_KEY`.
* **Slow CPU** → the cross‑encoder reranker is a fast MiniLM variant, but embeddings still cost CPU. Use a small instance or enable GPU if you expand to hybrid.

---

## Notes

* The reviewed DOCX uses a **portable *Review Notes* section** instead of injecting native Word comments (which are brittle across libraries).
* If you later switch to OpenAI/Claude/Ollama, only the **LLM call layer** in `agents.py` needs changing; the rest of the pipeline is model‑agnostic.

---

## License

Internal evaluation project. Add your license if you plan to publish.
