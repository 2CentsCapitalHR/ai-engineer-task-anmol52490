# # app.py
# # Streamlit UI for the multi-agent review: upload DOCX -> chunk -> "Process with Agents" -> download reviewed DOCX + JSON report
# import os
# import json
# import time
# from typing import Any, Dict, List

# import streamlit as st
# import pandas as pd
# import numpy as np

# # tame CPU for any model work inside agents
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# try:
#     import torch
#     torch.set_num_threads(1)
# except Exception:
#     pass

# from dotenv import load_dotenv

# from chunker import chunk_docx_bytes
# from agent import run_for_chunks, build_reviewed_docx


# # --------- App config ----------
# st.set_page_config(page_title="ADGM Corporate Agent (Agents Pipeline)", layout="wide")
# st.title("ADGM Corporate Agent — Review & Red Flags (Agents)")

# load_dotenv()
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# # --------- Upload DOCX ----------
# uploaded = st.file_uploader("Upload a .docx", type=["docx"])
# if not uploaded:
#     st.stop()

# file_bytes = uploaded.read()
# filename = uploaded.name

# # --------- Chunk (you already set 1800/1000/250 in chunker.py) ----------
# with st.spinner("Chunking document..."):
#     chunks = chunk_docx_bytes(file_bytes, filename)
# st.success(f"Chunked into {len(chunks)} sections.")

# # show quick table
# df = pd.DataFrame(
#     [{"id": c["id"], "type": c["type"], "chars": c["char_count"], "preview": c["preview"]} for c in chunks]
# )
# st.dataframe(df, use_container_width=True, hide_index=True)

# # --------- Process button ----------
# if st.button("🤖 Process with Agents", type="primary"):
#     if not PINECONE_INDEX:
#         st.error("PINECONE_INDEX is not set in environment.")
#         st.stop()

#     t0 = time.time()
#     st.info("Running Retrieve → Analyst → Reviewer → Writer per chunk...")

#     with st.spinner("Thinking with agents..."):
#         output = run_for_chunks(
#     file_name=filename,
#     chunks=chunks,
#     pinecone_index_name=PINECONE_INDEX,
#     pinecone_namespace=PINECONE_NAMESPACE,
#     llm_model=GEMINI_MODEL,
#     openai_api_key=GOOGLE_API_KEY,  # name stays the same in the function; see agents.py patch below
# )

#     results = output["results"]
#     report = output["report"]

#     st.success(f"Agents finished in {time.time()-t0:.1f}s")
#     st.subheader("Summary")
#     c1, c2 = st.columns(2)
#     c1.metric("Chunks processed", report.get("chunks_processed", 0))
#     c2.metric("Issues found", report.get("issues_found", 0))

#     # Show first few issues
#     if report.get("items"):
#         st.write("**Sample issues (first 10):**")
#         st.table(pd.DataFrame(report["items"][:10]))

#     # Build reviewed DOCX with Review Notes
#     with st.spinner("Building reviewed DOCX..."):
#         reviewed_bytes = build_reviewed_docx(filename, chunks, results)

#     st.subheader("Downloads")
#     st.download_button(
#         "⬇️ Download Reviewed DOCX",
#         data=reviewed_bytes,
#         file_name=f"Reviewed_{filename}",
#         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
#     )
#     report_json = json.dumps(report, indent=2, ensure_ascii=False)
#     st.download_button(
#         "⬇️ Download JSON Report",
#         data=report_json.encode("utf-8"),
#         file_name=f"review_report_{os.path.splitext(filename)[0]}.json",
#         mime="application/json",
#     )

#     # Optional: per-chunk detail
#     st.markdown("---")
#     st.subheader("Per-chunk Results")
#     # Map by id for quick show
#     rmap: Dict[str, Any] = {r.chunk_id: r for r in results}
#     for c in chunks:
#         rid = c["id"]
#         r = rmap.get(rid)
#         if not r:
#             continue
#         with st.expander(f"Chunk {rid} — {c['preview'][:120]}{'…' if len(c['preview'])>120 else ''}"):
#             # refs
#             if r.top_refs:
#                 st.write("**Top-3 references**")
#                 refs_df = pd.DataFrame(
#                     [
#                         {
#                             "title": h.title or h.filename,
#                             "pages": f"{h.page_start}-{h.page_end}" if h.page_start else "",
#                             "score": round(h.score, 3),
#                             "link": h.drive_link or "",
#                         }
#                         for h in r.top_refs
#                     ]
#                 )
#                 st.table(refs_df)
#             # issues
#             if r.issues:
#                 st.write("**Issues**")
#                 iss_df = pd.DataFrame(
#                     [{"severity": i.severity, "issue": i.issue, "suggestion": i.suggestion, "citation": i.citation} for i in r.issues]
#                 )
#                 st.table(iss_df)
#             # improved text
#             if r.improved_text:
#                 st.write("**Proposed Compliant Text**")
#                 st.write(r.improved_text)



# app.py
# Streamlit UI for the agents pipeline:
#   Upload DOCX -> chunk (your chunker.py with 1800/1000/250) -> "Process with Agents"
#   -> shows summary, lets you download reviewed DOCX & JSON report.

import os
import json
import time
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# keep CPU calm for reranker
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from dotenv import load_dotenv
from chunker import chunk_docx_bytes
from agent import run_for_chunks, build_reviewed_docx

st.set_page_config(page_title="ADGM Corporate Agent — Review & Red Flags (Agents)", layout="wide")
st.title("ADGM Corporate Agent — Review & Red Flags (Agents)")

load_dotenv()
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE") or None

# LLM config (auto: Gemini if GOOGLE_API_KEY else OpenAI)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = GEMINI_MODEL if GOOGLE_API_KEY else OPENAI_MODEL
LLM_KEY = GOOGLE_API_KEY or OPENAI_API_KEY

uploaded = st.file_uploader("Upload a .docx", type=["docx"])
if not uploaded:
    st.stop()

data = uploaded.read()
filename = uploaded.name

with st.spinner("Chunking document..."):
    chunks = chunk_docx_bytes(data, filename)
st.success(f"Chunked into {len(chunks)} sections.")

df = pd.DataFrame(
    [{"id": c["id"], "type": c["type"], "chars": c["char_count"], "preview": c["preview"]} for c in chunks]
)
st.dataframe(df, use_container_width=True, hide_index=True)

if st.button("🤖 Process with Agents", type="primary"):
    if not PINECONE_INDEX:
        st.error("PINECONE_INDEX is not set.")
        st.stop()

    t0 = time.time()
    st.info("Running Retrieve → Analyst → Reviewer → Writer per chunk...")

    with st.spinner("Agents thinking..."):
        output = run_for_chunks(
            file_name=filename,
            chunks=chunks,
            pinecone_index_name=PINECONE_INDEX,
            pinecone_namespace=PINECONE_NAMESPACE,
            llm_model=LLM_MODEL,
            openai_api_key=LLM_KEY,  # used for Gemini or OpenAI
        )

    results = output["results"]
    report = output["report"]

    st.success(f"Done in {time.time()-t0:.1f}s")
    c1, c2 = st.columns(2)
    c1.metric("Chunks processed", report.get("chunks_processed", 0))
    c2.metric("Issues found", report.get("issues_found", 0))

    # Show a small preview table
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

    # Per-chunk detail
    st.markdown("---")
    st.subheader("Per-chunk Results")
    rmap: Dict[str, Any] = {r.chunk_id: r for r in results}
    for c in chunks:
        rid = c["id"]
        r = rmap.get(rid)
        if not r:
            continue
        with st.expander(f"Chunk {rid} — {c['preview'][:120]}{'…' if len(c['preview'])>120 else ''}"):
            # refs
            if r.top_refs:
                st.write("**Top-3 references**")
                refs_df = pd.DataFrame(
                    [
                        {
                            "title": h.title or h.filename,
                            "pages": f"{h.page_start}-{h.page_end}" if h.page_start else "",
                            "score": round(h.score, 3),
                            "link": h.drive_link or "",
                        }
                        for h in r.top_refs
                    ]
                )
                st.table(refs_df)
            # issues
            if r.issues:
                st.write("**Issues**")
                iss_df = pd.DataFrame(
                    [
                        {
                            "severity": i.severity,
                            "issue": i.issue,
                            "suggestion": i.suggestion,
                            "citation": i.citation,
                            "supporting_refs": ",".join(str(x) for x in (i.supporting_refs or [])),
                        }
                        for i in r.issues
                    ]
                )
                st.table(iss_df)
            # improved text
            if r.improved_text:
                st.write("**Proposed Compliant Text**")
                st.write(r.improved_text)
