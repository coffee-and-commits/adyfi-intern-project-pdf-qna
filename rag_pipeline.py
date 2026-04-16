"""
rag_pipeline.py  (v2 — Dynamic Retrieval + Query-Aware Strategy)
------------------------------------------------------------------
New additions:
  - Query classification  : factual | descriptive | summary
  - Score-threshold K     : dynamic chunk count based on L2 distance
  - MMR retrieval         : diverse chunk selection for descriptive queries
  - Hierarchical summary  : map-reduce summarization for overview queries
"""

import os
import sys
import time
import logging

import pdfplumber
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rag_pipeline")

# ─── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
CHAT_MODEL    = "gpt-4o-mini"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# Dynamic retrieval config
RETRIEVAL_CONFIG = {
    "factual": {
        "max_k": 5,
        "score_threshold": 1.2,   # L2 distance — lower = stricter
        "min_k": 2,
        "use_mmr": False,
    },
    "descriptive": {
        "max_k": 12,
        "score_threshold": 1.6,
        "min_k": 4,
        "use_mmr": True,
        "mmr_lambda": 0.6,        # 0=max diversity, 1=max relevance
    },
    "summary": {
        "max_k": None,            # None = use ALL chunks
        "score_threshold": None,
        "min_k": None,
        "use_mmr": False,
        "batch_size": 10,         # chunks per summarization batch
    },
}


def _banner(step: str) -> None:
    log.info("=" * 60)
    log.info(f"  STEP: {step}")
    log.info("=" * 60)


# ─── Step 1: PDF Loading (unchanged) ─────────────────────────────────────────
def load_pdf(pdf_path: str) -> str:
    _banner("1 / 5  —  PDF LOADING")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        log.info(f"Total pages: {total_pages}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            full_text += text + "\n"
            log.info(f"  Page {i+1}/{total_pages}: {len(text):,} chars")

    if not full_text.strip():
        raise ValueError("PDF appears to contain no extractable text.")

    log.info(f"Total characters: {len(full_text):,}")
    return full_text


# ─── Step 2: Text Splitting (unchanged) ──────────────────────────────────────
def split_text(raw_text: str) -> list[str]:
    _banner("2 / 5  —  TEXT SPLITTING")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    if not chunks:
        raise ValueError("Text splitting returned no chunks.")
    log.info(f"Total chunks: {len(chunks)}")
    return chunks


# ─── Step 3: Embedding (unchanged) ───────────────────────────────────────────
def embed_chunks(chunks: list[str], api_key: str) -> np.ndarray:
    _banner("3 / 5  —  EMBEDDING CHUNKS")
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
    BATCH = 100
    all_vectors: list[list[float]] = []
    for start in range(0, len(chunks), BATCH):
        batch = chunks[start: start + BATCH]
        log.info(f"  Embedding batch: chunks {start}–{start + len(batch) - 1}")
        all_vectors.extend(embedder.embed_documents(batch))
        time.sleep(0.3)

    matrix = np.array(all_vectors, dtype="float32")
    log.info(f"Embedding matrix shape: {matrix.shape}")
    return matrix


# ─── Step 4: FAISS Index (unchanged) ─────────────────────────────────────────
def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    _banner("4 / 5  —  BUILDING FAISS INDEX")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    log.info(f"Vectors stored: {index.ntotal}")
    return index


# ─── NEW: Query Classification ────────────────────────────────────────────────
def classify_query(question: str, client: OpenAI) -> str:
    """
    Classify the query into: 'factual' | 'descriptive' | 'summary'

    - factual     : specific facts, numbers, dates, names, yes/no
    - descriptive : how/why/explain a concept, compare, elaborate
    - summary     : overview of the whole document, general explanation,
                    "explain this doc", "what is this about"
    """
    prompt = """Classify this query into exactly one category:

- "factual"     → asks for a specific fact, number, date, name, statistic, yes/no
- "descriptive" → asks to explain a concept, process, comparison, or mechanism
- "summary"     → asks for an overview, summary, or general explanation of the 
                  whole document (e.g. "explain this document", "what is this about",
                  "summarize", "give me an overview", "explain like I'm a student")

Reply with ONLY one word: factual, descriptive, or summary.

Query: {query}""".format(query=question)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    label = response.choices[0].message.content.strip().lower()

    # Sanitize in case the model returns unexpected output
    if label not in ("factual", "descriptive", "summary"):
        log.warning(f"Unexpected classification '{label}', defaulting to 'descriptive'")
        label = "descriptive"

    log.info(f"Query classified as: [{label.upper()}]")
    return label


# ─── NEW: MMR (Maximal Marginal Relevance) ────────────────────────────────────
def mmr_rerank(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    candidate_indices: list[int],
    k: int,
    lambda_param: float = 0.6,
) -> list[int]:
    """
    Select k indices from candidates using MMR to balance
    relevance (similarity to query) vs diversity (dissimilarity to
    already-selected chunks).

    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity
    """
    # Cosine similarity helpers
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    q = query_vector.flatten()
    selected: list[int] = []
    remaining = list(range(len(candidate_indices)))

    while len(selected) < k and remaining:
        mmr_scores = []
        for i in remaining:
            relevance = cosine_sim(q, candidate_vectors[i])
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    cosine_sim(candidate_vectors[i], candidate_vectors[j])
                    for j in selected
                )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((score, i))

        best = max(mmr_scores, key=lambda x: x[0])[1]
        selected.append(best)
        remaining.remove(best)

    # Return original chunk indices in document order (preserves narrative flow)
    return sorted([candidate_indices[i] for i in selected])


# ─── NEW: Dynamic Retrieval ───────────────────────────────────────────────────
def dynamic_retrieve(
    question: str,
    query_type: str,
    chunks: list[str],
    index: faiss.IndexFlatL2,
    all_vectors: np.ndarray,
    api_key: str,
    client: OpenAI,
) -> list[str]:
    """
    Route to the correct retrieval strategy based on query_type.
    Returns a list of relevant text chunks.
    """
    cfg = RETRIEVAL_CONFIG[query_type]

    # ── Summary: return ALL chunks in order (for hierarchical summarization) ──
    if query_type == "summary":
        log.info(f"Summary query → returning all {len(chunks)} chunks for hierarchical summarization")
        return list(chunks)  # preserve document order

    # ── Embed the question ────────────────────────────────────────────────────
    embedder = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=api_key)
    q_vec = np.array([embedder.embed_query(question)], dtype="float32")

    # ── Search with generous max_k first, then threshold-filter ──────────────
    max_k = min(cfg["max_k"], index.ntotal)
    distances, indices = index.search(q_vec, max_k)

    candidates_idx = []
    candidates_dist = []

    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        if cfg["score_threshold"] is not None and dist > cfg["score_threshold"]:
            log.info(f"  Chunk {idx} rejected (L2={dist:.4f} > threshold {cfg['score_threshold']})")
            continue
        candidates_idx.append(int(idx))
        candidates_dist.append(dist)
        log.info(f"  Chunk {idx} accepted (L2={dist:.4f})")

    # Always keep at least min_k chunks even if they exceed threshold
    if len(candidates_idx) < cfg["min_k"]:
        log.info(f"  Below min_k ({cfg['min_k']}), expanding to top-{cfg['min_k']}")
        candidates_idx = [int(i) for i in indices[0][:cfg["min_k"]] if i != -1]

    log.info(f"Candidates after threshold filter: {len(candidates_idx)}")

    # ── MMR reranking for descriptive queries ─────────────────────────────────
    if cfg.get("use_mmr") and len(candidates_idx) > 1:
        candidate_vecs = np.array([all_vectors[i] for i in candidates_idx], dtype="float32")
        k_mmr = min(len(candidates_idx), cfg["max_k"])
        candidates_idx = mmr_rerank(
            query_vector=q_vec,
            candidate_vectors=candidate_vecs,
            candidate_indices=candidates_idx,
            k=k_mmr,
            lambda_param=cfg.get("mmr_lambda", 0.6),
        )
        log.info(f"After MMR reranking: {len(candidates_idx)} chunks")

    retrieved = [chunks[i] for i in candidates_idx]
    log.info(f"Final retrieved chunk count: {len(retrieved)}")
    return retrieved


# ─── NEW: Hierarchical Summarization ─────────────────────────────────────────
def hierarchical_summarize(chunks: list[str], question: str, client: OpenAI) -> str:
    """
    Map-Reduce summarization:
      1. MAP    — summarize each batch of chunks independently
      2. REDUCE — combine all batch summaries into a final answer
    """
    _banner("HIERARCHICAL SUMMARIZATION")
    batch_size = RETRIEVAL_CONFIG["summary"]["batch_size"]
    batches = [chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)]
    log.info(f"Total chunks: {len(chunks)} | Batches: {len(batches)} (size={batch_size})")

    # ── MAP: summarize each batch ─────────────────────────────────────────────
    batch_summaries = []
    for i, batch in enumerate(batches):
        context = "\n\n---\n\n".join(batch)
        prompt = (
            f"Summarize the key points from this section of a document.\n"
            f"Be concise but thorough. Focus on main ideas, facts, and themes.\n\n"
            f"Section {i+1}/{len(batches)}:\n\n{context}"
        )
        log.info(f"  Summarizing batch {i+1}/{len(batches)} ({len(batch)} chunks)...")
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content.strip()
        batch_summaries.append(summary)
        time.sleep(0.2)

    # ── REDUCE: combine all batch summaries ───────────────────────────────────
    combined = "\n\n---\n\n".join(
        [f"Section {i+1} Summary:\n{s}" for i, s in enumerate(batch_summaries)]
    )

    # Tailor the final prompt to what the user actually asked
    reduce_prompt = (
        f"You are given summaries of all sections of a document.\n"
        f"Using ONLY these summaries, answer the following request comprehensively:\n\n"
        f"Request: {question}\n\n"
        f"Section summaries:\n\n{combined}\n\n"
        f"Provide a well-structured, thorough response."
    )

    log.info("Running REDUCE step: combining all batch summaries...")
    final_resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that synthesizes document summaries "
                    "into clear, structured answers. Use headings where appropriate."
                ),
            },
            {"role": "user", "content": reduce_prompt},
        ],
        temperature=0.3,
    )
    return final_resp.choices[0].message.content.strip()


# ─── Step 5: Answer Generation ────────────────────────────────────────────────
def answer_question(
    question: str,
    chunks: list[str],
    index: faiss.IndexFlatL2,
    all_vectors: np.ndarray,
    api_key: str,
) -> str:
    _banner("5 / 5  —  QUERY ROUTING  &  ANSWER GENERATION")

    if not question.strip():
        return "Please enter a valid question."

    client = OpenAI(api_key=api_key)

    # ── 1. Classify query intent ──────────────────────────────────────────────
    query_type = classify_query(question, client)

    # ── 2. Summary path: hierarchical map-reduce ──────────────────────────────
    if query_type == "summary":
        return hierarchical_summarize(chunks, question, client)

    # ── 3. Factual / Descriptive path: dynamic retrieval → LLM ───────────────
    retrieved = dynamic_retrieve(
        question=question,
        query_type=query_type,
        chunks=chunks,
        index=index,
        all_vectors=all_vectors,
        api_key=api_key,
        client=client,
    )

    if not retrieved:
        return "Could not find relevant content in the document."

    context = "\n\n---\n\n".join(retrieved)
    tone_instruction = (
        "Answer precisely and concisely using only the provided context."
        if query_type == "factual"
        else "Provide a thorough explanation using the provided context. "
             "Use examples from the document where available."
    )

    system_prompt = (
        "You are a precise question-answering assistant. "
        "Answer ONLY using the provided context. "
        "Do NOT guess or use outside knowledge. "
        "If the answer is not clearly present in the context, say: "
        "'I could not find this information in the provided document.'"
    )
    user_message = (
        f"Context from the document:\n\n{context}\n\n"
        f"Instruction: {tone_instruction}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    log.info(f"Sending {query_type} query to {CHAT_MODEL} with {len(retrieved)} chunks...")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()
    tokens = response.usage
    log.info(f"Tokens — prompt: {tokens.prompt_tokens}, completion: {tokens.completion_tokens}")
    return answer


# ─── Public API ───────────────────────────────────────────────────────────────
def process_pdf(pdf_path: str) -> dict:
    """Load → split → embed → index. Returns state dict for querying."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    text    = load_pdf(pdf_path)
    chunks  = split_text(text)
    vectors = embed_chunks(chunks, api_key)
    index   = build_faiss_index(vectors)

    log.info("✅  Pipeline complete. Ready for questions.")
    return {
        "chunks":      chunks,
        "index":       index,
        "all_vectors": vectors,   # ← NEW: stored for MMR
        "api_key":     api_key,
    }


def get_answer(question: str, pipeline_state: dict) -> str:
    return answer_question(
        question=question,
        chunks=pipeline_state["chunks"],
        index=pipeline_state["index"],
        all_vectors=pipeline_state["all_vectors"],  # ← NEW
        api_key=pipeline_state["api_key"],
    )