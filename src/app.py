# app.py — Streamlit UI for Gemini-only agents pipeline
import os, json, time
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass

from dotenv import load_dotenv
from chunker import chunk_docx_bytes
from agent import run_for_chunks, build_reviewed_docx

st.set_page_config(page_title="ADGM Corporate Agent — (Gemini)", layout="wide")
st.title("ADGM Corporate Agent — Review & Red Flags (Gemini)")

load_dotenv()
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set. Add it to your .env.")
    st.stop()

uploaded = st.file_uploader("Upload a .docx", type=["docx"])
if not uploaded:
    st.stop()

data = uploaded.read()
filename = uploaded.name

with st.spinner("Chunking document..."):
    chunks = chunk_docx_bytes(data, filename)
st.success(f"Chunked into {len(chunks)} sections.")

df = pd.DataFrame([{"id": c["id"], "type": c["type"], "chars": c["char_count"], "preview": c["preview"]} for c in chunks])
st.dataframe(df, use_container_width=True, hide_index=True)

if st.button("🤖 Process with Agents", type="primary"):
    if not PINECONE_INDEX:
        st.error("PINECONE_INDEX is not set.")
        st.stop()

    t0 = time.time()
    st.info("Running Retrieve → Analyst → Reviewer → Writer per chunk (Gemini)…")

    with st.spinner("Agents thinking..."):
        output = run_for_chunks(
            file_name=filename,
            chunks=chunks,
            pinecone_index_name=PINECONE_INDEX,
            pinecone_namespace=PINECONE_NAMESPACE,
            llm_model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

    results = output["results"]
    report = output["report"]

    st.success(f"Done in {time.time()-t0:.1f}s")
    c1, c2 = st.columns(2)
    c1.metric("Chunks processed", report.get("chunks_processed", 0))
    c2.metric("Issues found", report.get("issues_found", 0))

    if report.get("items"):
        st.write("**Sample issues (first 10):**")
        st.table(pd.DataFrame(report["items"][:10]))

    with st.spinner("Building reviewed DOCX with citations & verbatim excerpts..."):
        reviewed_bytes = build_reviewed_docx(filename, chunks, results)

    st.subheader("Downloads")
    st.download_button(
        "⬇️ Download Reviewed DOCX",
        data=reviewed_bytes,
        file_name=f"Reviewed_{filename}",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇️ Download JSON Report",
        data=report_json.encode("utf-8"),
        file_name=f"review_report_{os.path.splitext(filename)[0]}.json",
        mime="application/json",
    )

    st.markdown("---")
    st.subheader("Per-chunk Results")
    rmap: Dict[str, Any] = {r.chunk_id: r for r in results}
    for c in chunks:
        rid = c["id"]
        r = rmap.get(rid)
        if not r:
            continue
        with st.expander(f"Chunk {rid} — {c['preview'][:120]}{'…' if len(c['preview'])>120 else ''}"):
            if r.top_refs:
                st.write("**Top-3 references**")
                refs_df = pd.DataFrame([
                    {
                        "title": h.title or h.filename,
                        "pages": f"{h.page_start}-{h.page_end}" if h.page_start else "",
                        "score": round(h.score, 3),
                        "link": h.drive_link or "",
                    } for h in r.top_refs
                ])
                st.table(refs_df)
            if r.issues:
                st.write("**Issues**")
                iss_df = pd.DataFrame([
                    {
                        "severity": i.severity,
                        "issue": i.issue,
                        "suggestion": i.suggestion,
                        "citation": i.citation,
                        "supporting_refs": ",".join(map(str, i.supporting_refs or [])),
                    } for i in r.issues
                ])
                st.table(iss_df)
            if r.improved_text:
                st.write("**Proposed Compliant Text**")
                st.write(r.improved_text)
