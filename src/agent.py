# agents.py — Gemini-only, dense-only RAG (Pinecone v2/v3), LangGraph (no checkpointer).
import os, io, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Pinecone v2/v3 compatibility
PC_V3 = False
try:
    from pinecone import Pinecone, __version__ as _pcv
    PC_V3 = True
except Exception:
    import pinecone  # v2
    _pcv = getattr(pinecone, "__version__", "2.x")

from langgraph.graph import StateGraph, END
import google.generativeai as genai

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_COLOR_INDEX

# ---------- Tunables ----------
DENSE_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
TOP_N = 3
BATCH_EMB = 32
RERANK_BATCH = 32
TEXT_CAP = 1600
EXCERPT_LEN = 320
# -----------------------------

# ---------- State models ----------
@dataclass
class ChunkInput:
    chunk_id: str
    text: str
    meta: Dict[str, Any]

@dataclass
class RefHit:
    doc_id: Optional[str]
    chunk_id: Optional[str]
    title: Optional[str]
    filename: Optional[str]
    drive_link: Optional[str]
    page_start: Optional[int]
    page_end: Optional[int]
    text: str
    score: float

@dataclass
class AnalystIssue:
    issue: str
    severity: str
    citation: str
    suggestion: str
    supporting_refs: List[int] = field(default_factory=list)

@dataclass
class ChunkResult:
    chunk_id: str
    top_refs: List[RefHit] = field(default_factory=list)
    issues: List[AnalystIssue] = field(default_factory=list)
    improved_text: Optional[str] = None

@dataclass
class PipelineState:
    file_name: str
    llm_model: str
    pinecone_index_name: str
    pinecone_namespace: Optional[str]
    google_api_key: Optional[str]

    dense_model: Any = None
    reranker: Any = None
    pc_index: Any = None

    current_chunk: Optional[ChunkInput] = None
    retrieved: List[RefHit] = field(default_factory=list)
    analyst_issues: List[AnalystIssue] = field(default_factory=list)
    writer_text: Optional[str] = None

    results: List[ChunkResult] = field(default_factory=list)
# -----------------------------------

# ---------- Init / helpers ----------
def _init_clients(state: PipelineState):
    # LLM first (fail fast)
    if not state.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Set it in your environment.")
    genai.configure(api_key=state.google_api_key)

    if state.dense_model is None:
        state.dense_model = SentenceTransformer(DENSE_MODEL)
        state.dense_model.max_seq_length = 512
    if state.reranker is None:
        state.reranker = CrossEncoder(RERANKER_MODEL)

    if PC_V3:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        state.pc_index = pc.Index(state.pinecone_index_name)
    else:
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
        state.pc_index = pinecone.Index(state.pinecone_index_name)

def _pick_text(md: Dict[str, Any]) -> str:
    for k in ("text", "context", "content", "chunk", "body"):
        v = md.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def _encode_dense(texts: List[str], model: Any) -> List[List[float]]:
    vecs = model.encode(texts, batch_size=BATCH_EMB, normalize_embeddings=True)
    return [v.astype(np.float32).tolist() for v in vecs]

def _pinecone_query_dense(index, vector: List[float], top_k: int, namespace: Optional[str]) -> List[Dict[str, Any]]:
    q: Dict[str, Any] = dict(top_k=top_k, include_metadata=True)
    if namespace: q["namespace"] = namespace
    if PC_V3:
        res = index.query(vector=vector, **q)
        matches = getattr(res, "matches", None) or res.get("matches", [])
    else:
        res = index.query(vector=vector, **q)
        matches = res.get("matches", [])
    out = []
    for m in matches:
        mid = getattr(m, "id", None) or m.get("id")
        mscore = getattr(m, "score", None) or m.get("score")
        mmeta = getattr(m, "metadata", None) or m.get("metadata", {})
        out.append({"id": mid, "score": float(mscore), "metadata": mmeta})
    return out

def _rerank_pairs(reranker: Any, query_text: str, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pairs, keep = [], []
    for it in cands:
        txt = _pick_text(it["metadata"])
        if txt:
            pairs.append((query_text, txt))
            keep.append(it)
    if not pairs:
        return []
    logits = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    sig = 1 / (1 + np.exp(-np.asarray(logits)))
    ranked = []
    for it, logit, s in zip(keep, logits, sig):
        it = dict(it)
        it["reranker_logit"] = float(logit)
        it["reranker_score"] = float(s)
        ranked.append(it)
    ranked.sort(key=lambda x: x["reranker_score"], reverse=True)
    return ranked

def _map_hits(hits: List[Dict[str, Any]], top_n=TOP_N) -> List[RefHit]:
    out: List[RefHit] = []
    for h in hits[:top_n]:
        md = h["metadata"] or {}
        out.append(
            RefHit(
                doc_id=md.get("doc_id"),
                chunk_id=md.get("chunk_id") or md.get("chunk_index"),
                title=md.get("doc_title") or md.get("title"),
                filename=md.get("file_name") or md.get("filename"),
                drive_link=md.get("drive_link"),
                page_start=md.get("page_start"),
                page_end=md.get("page_end"),
                text=_pick_text(md),
                score=float(h.get("reranker_score", h.get("score", 0.0))),
            )
        )
    return out

def _cap(s: str, n: int = TEXT_CAP) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n]
# -----------------------------------

# ---------- Prompts ----------
ANALYST_SYS = (
    "You are a strict legal analyst for ADGM. Given a document chunk and up to 3 reference snippets from an "
    "ADGM-aligned knowledge base, identify legal issues/red flags and propose compliant fixes. "
    "Return ONLY JSON:\n"
    "{ \"issues\": [ { \"issue\": str, \"severity\": \"Low|Medium|High\", \"citation\": str, "
    "\"suggestion\": str, \"supporting_refs\": [int] } ] }\n"
    "- 'supporting_refs' are ZERO-based indices into the provided 'references' array.\n"
)

REVIEWER_SYS = (
    "You are a meticulous legal reviewer. You get the original chunk, the references, and the analyst's JSON. "
    "Correct mistakes, remove weak flags, merge duplicates, and improve suggestions to be ADGM-compliant. "
    "Return ONLY JSON with the same schema."
)

WRITER_SYS = (
    "You are a careful editor. Given the original chunk and the reviewed issues, propose improved text that "
    "keeps the original meaning but fixes problems. Output ONLY JSON: { \"improved_text\": str }"
)
# -----------------------------------

# ---------- LLM wrapper (Gemini) ----------
def _llm_json(model_name: str, system: str, user: str) -> Dict[str, Any]:
    gmodel = genai.GenerativeModel(model_name, system_instruction=system)
    resp = gmodel.generate_content(
        [user],
        generation_config={"temperature": 0.2, "response_mime_type": "application/json"},
    )
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
        return {"issues": []} if "issues" in system.lower() else {"improved_text": ""}
# -----------------------------------

# ---------- Graph nodes ----------
def node_retrieve(state: PipelineState) -> PipelineState:
    _init_clients(state)
    ch = state.current_chunk
    assert ch is not None
    qtext = _cap(ch.text)
    dv = _encode_dense([qtext], state.dense_model)[0]
    hits = _pinecone_query_dense(state.pc_index, dv, TOP_K, state.pinecone_namespace)
    ranked = _rerank_pairs(state.reranker, qtext, hits)
    state.retrieved = _map_hits(ranked, TOP_N)
    return state

def node_analyst(state: PipelineState) -> PipelineState:
    ch = state.current_chunk
    refs = state.retrieved
    payload = {
        "chunk": _cap(ch.text),
        "references": [
            {
                "title": r.title, "filename": r.filename, "drive_link": r.drive_link,
                "pages": [r.page_start, r.page_end],
                "text": _cap(r.text, TEXT_CAP),
            } for r in refs
        ],
        "known_red_flags": [
            "Jurisdiction not ADGM or courts unspecified",
            "Missing/invalid signatory or witness section",
            "Missing core incorporation info (share capital, registered office, directors)",
            "Ambiguous language in obligations",
            "Not aligned with ADGM Companies Regulations 2020",
        ],
    }
    out = _llm_json(state.llm_model, ANALYST_SYS, json.dumps(payload))
    issues: List[AnalystIssue] = []
    for it in out.get("issues", []):
        refs_idx = it.get("supporting_refs") or []
        if isinstance(refs_idx, list):
            refs_idx = [int(x) for x in refs_idx if isinstance(x, (int, float))]
        issues.append(
            AnalystIssue(
                issue=str(it.get("issue", "")).strip(),
                severity=(str(it.get("severity", "Medium")).strip() or "Medium"),
                citation=str(it.get("citation", "")).strip(),
                suggestion=str(it.get("suggestion", "")).strip(),
                supporting_refs=refs_idx[:3],
            )
        )
    state.analyst_issues = issues
    return state

def node_reviewer(state: PipelineState) -> PipelineState:
    ch = state.current_chunk
    refs = state.retrieved
    payload = {
        "chunk": _cap(ch.text),
        "references": [
            {
                "title": r.title, "filename": r.filename, "drive_link": r.drive_link,
                "pages": [r.page_start, r.page_end],
                "text": _cap(r.text, TEXT_CAP),
            } for r in refs
        ],
        "analyst": [
            {
                "issue": i.issue, "severity": i.severity, "citation": i.citation,
                "suggestion": i.suggestion, "supporting_refs": i.supporting_refs
            } for i in state.analyst_issues
        ],
    }
    out = _llm_json(state.llm_model, REVIEWER_SYS, json.dumps(payload))
    arr = out.get("issues", []) if isinstance(out, dict) else []
    if not arr:
        arr = [
            {"issue": i.issue, "severity": i.severity, "citation": i.citation,
             "suggestion": i.suggestion, "supporting_refs": i.supporting_refs}
            for i in state.analyst_issues
        ]
    issues: List[AnalystIssue] = []
    for it in arr:
        refs_idx = it.get("supporting_refs") or []
        if isinstance(refs_idx, list):
            refs_idx = [int(x) for x in refs_idx if isinstance(x, (int, float))]
        issues.append(
            AnalystIssue(
                issue=str(it.get("issue", "")).strip(),
                severity=(str(it.get("severity", "Medium")).strip() or "Medium"),
                citation=str(it.get("citation", "")).strip(),
                suggestion=str(it.get("suggestion", "")).strip(),
                supporting_refs=refs_idx[:3],
            )
        )
    state.analyst_issues = issues
    return state

def node_writer(state: PipelineState) -> PipelineState:
    ch = state.current_chunk
    payload = {
        "chunk": _cap(ch.text, TEXT_CAP),
        "issues": [{"issue": i.issue, "severity": i.severity, "suggestion": i.suggestion} for i in state.analyst_issues],
    }
    out = _llm_json(state.llm_model, WRITER_SYS, json.dumps(payload))
    state.writer_text = str(out.get("improved_text", "")).strip()
    state.results.append(
        ChunkResult(
            chunk_id=ch.chunk_id,
            top_refs=state.retrieved[:],
            issues=state.analyst_issues[:],
            improved_text=state.writer_text or None,
        )
    )
    state.retrieved.clear()
    state.analyst_issues.clear()
    state.writer_text = None
    state.current_chunk = None
    return state
# -----------------------------------

# ---------- Graph ----------
def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("analyst", node_analyst)
    g.add_node("reviewer", node_reviewer)
    g.add_node("writer", node_writer)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "reviewer")
    g.add_edge("reviewer", "writer")
    g.add_edge("writer", END)
    return g.compile()  # no checkpointer
GRAPH = _build_graph()
# -----------------------------------

# ---------- Orchestrator ----------
def run_for_chunks(
    file_name: str,
    chunks: List[Dict[str, Any]],
    pinecone_index_name: str,
    pinecone_namespace: Optional[str],
    llm_model: str = None,
    google_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    model = llm_model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    base = PipelineState(
        file_name=file_name,
        llm_model=model,
        pinecone_index_name=pinecone_index_name,
        pinecone_namespace=pinecone_namespace,
        google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY"),
    )
    _init_clients(base)

    all_results: List[ChunkResult] = []
    for c in chunks:
        ch = ChunkInput(chunk_id=c.get("id"), text=c.get("text") or "", meta=c)
        s = PipelineState(**{**base.__dict__})
        s.current_chunk = ch
        out_state = GRAPH.invoke(s)  # returns a dict
        chunk_results = out_state.get("results", []) if isinstance(out_state, dict) else getattr(out_state, "results", [])
        all_results.extend(chunk_results)


    # flat report
    items = []
    for r in all_results:
        for it in r.issues:
            citation = it.citation
            if not citation and r.top_refs:
                r0 = r.top_refs[0]
                pages = f" (pp. {r0.page_start}-{r0.page_end})" if r0.page_start else ""
                citation = f"{r0.title or r0.filename or 'KB Doc'}{pages} {r0.drive_link or ''}".strip()
            items.append({
                "chunk_id": r.chunk_id,
                "issue": it.issue,
                "severity": it.severity,
                "suggestion": it.suggestion,
                "citation": citation,
                "supporting_refs": it.supporting_refs,
            })
    report = {"file": file_name, "chunks_processed": len(chunks), "issues_found": len(items), "items": items}
    return {"results": all_results, "report": report}

# ---------- DOCX builder (citations + verbatim excerpts) ----------
def build_reviewed_docx(original_filename: str, chunks: List[Dict[str, Any]], results: List[ChunkResult]) -> bytes:
    doc = Document()
    doc.add_heading(f"AI Review Notes for {original_filename}", level=1)
    doc.add_paragraph("Issues, suggestions, and verbatim evidence with citations from the knowledge base.")

    rmap: Dict[str, ChunkResult] = {r.chunk_id: r for r in results}
    for c in chunks:
        rid = c.get("id")
        res = rmap.get(rid)
        if not res or (not res.issues and not res.improved_text):
            continue

        h = doc.add_heading(f"Section / Chunk {rid}", level=2)
        h.runs[0].font.size = Pt(12)

        p = doc.add_paragraph()
        p.add_run("Preview: ").bold = True
        p.add_run((c.get("preview") or (c.get("text") or "")[:180]).strip())

        if res.top_refs:
            doc.add_paragraph("References used:")
            for idx, ref in enumerate(res.top_refs):
                line = f"{idx}. {ref.title or ref.filename or 'KB Doc'}"
                if ref.page_start:
                    line += f" (pp. {ref.page_start}-{ref.page_end})"
                if ref.drive_link:
                    line += f" — {ref.drive_link}"
                doc.add_paragraph(line)

        if res.issues:
            doc.add_paragraph("Issues & Suggested Fixes:")
            for k, it in enumerate(res.issues, 1):
                doc.add_paragraph(f"{k}. [{it.severity}] {it.issue}")
                if it.citation:
                    cit = doc.add_paragraph(f"   Citation: {it.citation}")
                    cit.runs[0].font.size = Pt(9)
                if it.suggestion:
                    sg = doc.add_paragraph(f"   Suggestion: {it.suggestion}")
                    sg.runs[0].font.highlight_color = WD_COLOR_INDEX.YELLOW
                    sg.runs[0].font.size = Pt(10)

                # Verbatim evidence
                support_idxs = it.supporting_refs or [0]
                doc.add_paragraph("   Evidence excerpts:")
                for si in support_idxs:
                    if 0 <= si < len(res.top_refs):
                        ref = res.top_refs[si]
                        excerpt = (_cap(ref.text, EXCERPT_LEN) or "").strip()
                        if excerpt:
                            q = doc.add_paragraph(f'   • "{excerpt}"')
                            q.runs[0].font.size = Pt(9)
                            meta = []
                            if ref.title or ref.filename: meta.append(ref.title or ref.filename)
                            if ref.page_start: meta.append(f"pp. {ref.page_start}-{ref.page_end}")
                            if ref.drive_link: meta.append(ref.drive_link)
                            if meta:
                                m = doc.add_paragraph("     — " + " · ".join(meta))
                                m.runs[0].font.size = Pt(9)

        if res.improved_text:
            doc.add_paragraph("Proposed Compliant Text:")
            doc.add_paragraph(res.improved_text)

        doc.add_paragraph("-" * 60)

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()
