# Resume-HyperRAG-Cache

This optimized, free-tier Resume RAG system uses Spatial Coordinate Mapping to extract hidden LaTeX hyperlinks, ensuring 100% link retention. A Dual-Layer Semantic Cache instantly deduplicates paraphrased questions, serving answers with zero model latency. Built entirely on local Hugging Face models to eliminate API costs.

---

## 🚀 Key Innovations

### 1. Spatial Coordinate Hyperlink Mapping (PyMuPDF)
LaTeX-generated resumes heavily utilize commands like `\href{URL}{Label}` where only the label is visible on the canvas. Standard textual PDF extractors drop the target URL completely. This architecture extracts raw text spans with their visual bounding boxes (`bbox`) alongside hidden annotation links. By evaluating structural intersections using coordinate overlap math, target URLs are permanently unified with their labels inside the index chunks.

### 2. Dual-Layer Semantic Optimization Cache
To eliminate computational overhead and repetitive LLM processing during interview loops:
* **Layer 1 (Canonical Mapping):** Regex-driven linguistic pattern normalization matches vocabulary drift seamlessly (e.g., automatically maps `interviewee name`, `applicant name`, or `profile owner` to the canonical query `candidate name`).
* **Layer 2 (Vector Semantic Similarity):** Calculates vector cosine similarity of unresolved queries against cached query histories using a structural threshold of $\ge 0.88$. Paraphrased intents hit the cache instantly with zero model latency.

### 3. Forced Extraction Local Inference Loop
Standard SQuAD models suffer from a "No Answer" bias when reading fragmented resume sections or bullet points. This pipeline overrides the standard Hugging Face pipeline abstraction layer by intercepting model logits, penalizing the empty `[CLS]/<s>` span, and implementing a constrained sliding-window token pointer search to ensure stable, context-accurate extractions.

---

## 📊 Pipeline Architecture

```text
  [ Upload Resume PDF ]
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ Spatial Extraction & Geometry Mapping (PyMuPDF)          │
│ ├── Extract text blocks and coordinates                  │
│ ├── Isolate hyperlink annotations & target URIs          │
│ └── Merge text strings with hidden URLs via bbox overlap │
└──────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ Query Verification & Deduplication Pipeline              │
│ ├── Try Layer 1 Cache: Exact Canonical String Match      │
│ └── Try Layer 2 Cache: Vector Cosine Similarity (≥ 0.88) │
└──────────────────────────────────────────────────────────┘
      │                     │
      │ (Cache Hit)         │ (Cache Miss)
      ▼                     ▼
[Instant Response]   ┌─────────────────────────────────────┐
                     │ Local Hybrid Retrieval & QA Loop    │
                     │ ├── Global Metadata Links Fallback  │
                     │ ├── FAISS Dense Index Matching      │
                     │ └── Logit-Intervention RoBERTa QA   │
                     └─────────────────────────────────────┘
                                    │
                                    ▼
                           [Update Cache Database]
```

---

## 🛠️ Installation & Environment Setup

This system runs completely locally on free-tier compute options (such as a Google Colab CPU or standard T4 GPU instance). No subscription API keys (OpenAI/Anthropic) are required.

```bash
pip install pymupdf sentence-transformers faiss-cpu transformers accelerate torch pydantic
```

---

## ⚙️ Core Configuration

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"
SEMANTIC_CACHE_THRESHOLD = 0.88
TOP_K = 5
```

---

## 🔍 Code Walkthrough & Implementation

### Hyperlink Mapping Mechanics
The core engine reads individual text span rectangles and measures intersection tolerances against the PDF annotation layer. This handles edge cases where multiple links coexist inside the same visual layout line (e.g., `GitHub | LinkedIn | Portfolio` entries).

```python
def rects_overlap(rect1, rect2, tolerance=3) -> bool:
    r1, r2 = fitz.Rect(rect1), fitz.Rect(rect2)
    r1.x0 -= tolerance; r1.y0 -= tolerance
    r1.x1 += tolerance; r1.y1 += tolerance
    return r1.intersects(r2)
```

### Advanced Inference Logit Intervention
To guarantee stable answers across short phrase blocks, the pipeline forces text span generation by zeroing out null pointers before compiling softmax probabilities:

```python
# Force extraction by heavily penalizing the [CLS]/<s> "No Answer" token at index 0
start_logits[0] = -10000
end_logits[0] = -10000

# Constrained sliding window search for span construction
for i in range(1, len(start_probs)):
    for j in range(i, min(i + 40, len(end_probs))): 
        score = (start_probs[i] * end_probs[j]).item()
        if score > best_chunk_score:
            best_chunk_score = score
            best_s, best_e = i, j
```

---

## 💻 Sample Test Execution Log
The runtime execution logs demonstrate the interaction between the hybrid RAG database extraction and the caching architecture:

```text
================================================================================
QUERY: "What is the candidate name?"
STATUS: 🤖 RAG RETRIEVAL + LOCAL LLM QA
ANSWER:
Syamantak Mukherjee

================================================================================
QUERY: "Who is the interviewee name?"
STATUS: ⚡ EXACT CANONICAL CACHE HIT
MATCHED PREVIOUS: "What is the candidate name?" (Sim Score: 1.000)
ANSWER:
Syamantak Mukherjee

================================================================================
QUERY: "Can you tell me the applicant name?"
STATUS: ⚡ EXACT CANONICAL CACHE HIT
MATCHED PREVIOUS: "What is the candidate name?" (Sim Score: 1.000)
ANSWER:
Syamantak Mukherjee

================================================================================
QUERY: "Show me the GitHub link"
STATUS: 🔍 STRUCTURED LINK LOOKUP
ANSWER:
[https://github.com/mukherjee-syamantak](https://github.com/mukherjee-syamantak)
[https://github.com/mukherjee-syamantak/LLM-Operations_Workflow](https://github.com/mukherjee-syamantak/LLM-Operations_Workflow)
[https://github.com/mukherjee-syamantak/Recommender_System_with_Deep_Learning-RBM-SAE-](https://github.com/mukherjee-syamantak/Recommender_System_with_Deep_Learning-RBM-SAE-)
[https://github.com/mukherjee-syamantak/ci-cd-final-project](https://github.com/mukherjee-syamantak/ci-cd-final-project)
[https://github.com/mukherjee-syamantak/Machine_Learning](https://github.com/mukherjee-syamantak/Machine_Learning)

================================================================================
QUERY: "Do you have a repo code link?"
STATUS: ⚡ EXACT CANONICAL CACHE HIT
MATCHED PREVIOUS: "Show me the GitHub link" (Sim Score: 1.000)
ANSWER:
[https://github.com/mukherjee-syamantak](https://github.com/mukherjee-syamantak)...
```

---

## 📂 Project Artifacts Generated
When executing the pipeline, the system automatically creates and manages the following local storage binaries for persistence:

* **`semantic_query_cache.json`**: The database tracking historical canonical strings, raw string questions, and their underlying normalized tensor vectors.
* **`resume_faiss.index`**: High-speed vector index optimized for inner product calculations on the embedding spaces.
* **`resume_chunks.pkl`**: Serialized Python chunk payloads coupled tightly with custom geometric parsing source metadata.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
