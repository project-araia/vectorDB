# VectorDB

## VECTORDB BUILDING FLOW:

This module builds a hybrid vector database with both dense and sparse (BM25) indexes from JSON documents.

BUILD PIPELINE (3 Phases):
────────────────────────────────────────────────────────────────

[Phase 1: Chunking & Deduplication]
  1. Load JSON files from input directory (limited by PINNED_INPUT_LIMIT = 1627)
  2. Extract text from each document (title + abstract)
  3. Split text into overlapping sentence windows (chunk_by_sentence_window)
  4. Build semantic chunks using embedding-based sentence similarity (build_semantic_chunks)
  5. Merge both chunking strategies with configurable weights (default 50/50)
  6. Apply MANDATORY dual-mechanism deduplication:
     - Text-based: Check normalized chunks against seen_chunks set
     - Embedding-based: Check cosine similarity >= EMBEDDING_DEDUP_THRESHOLD (0.7)
     - Duplicates are skipped, count tracked in skipped_duplicates
  7. Batch chunks for encoding (embed_batch_size tunable)

[Phase 2: Dense Index Building (FAISS IVF-PQ)]
  1. Encode chunks with SentenceTransformer to normalized embeddings
  2. For each batch:
     - Accumulate embeddings for training phase
     - Once train_vecs threshold reached: train IVF-PQ quantizer
     - Use IndexIVFPQ(dim, nlist, m, nbits, METRIC_INNER_PRODUCT)
     - Add training vectors to index, then maintain index for incremental additions
  3. Continue adding all deduped embeddings to trained index
  4. Save final index to: dense_ivfpq.faiss

[Phase 3: Sparse Index Building (BM25)]
  1. Tokenize all accepted chunk texts (BM25Okapi)
  2. Build BM25 inverted index from token corpus
  3. Save to: sparse_bm25.pkl

[Output Files]
  - dense_ivfpq.faiss: FAISS IVF-PQ index for semantic search (indexed by chunk position)
  - sparse_bm25.pkl: BM25 index for keyword search (indexed by chunk position)
  - manifest.json: Metadata including chunk texts, metadata, dedup statistics, index params

KEY FEATURES:
  • Hybrid chunking: Sentence windows overlap + semantic boundary detection
  • Mandatory dedup: Both text-normalized and embedding-similarity checks
  • Deterministic & resumable: BM25/FAISS are stateless, built sequentially
  • Flexible weighting: Control sentence vs semantic chunk contribution
  • Configurable quantization: IVF-PQ parameters (nlist, m, nbits)
  • Climate filtering: Optional --climate-only flag filters documents to climate-related content




## VECTORDB SEARCH FLOW:

Retrieves most relevant chunks from pre-built hybrid indexes using dual-path ranking and fusion.

SEARCH PIPELINE:
────────────────────────────────────────────────────────────────

[Step 1: Index & Model Loading]
  1. Load FAISS IVF-PQ dense index from dense_ivfpq.faiss
  2. Load BM25 sparse index (token_corpus, chunk_texts, chunk_metadata) from sparse_bm25.pkl
  3. Load SentenceTransformer embedding model
  4. Set nprobe (neighbors to probe in IVF clusters) for recall/speed tradeoff

[Step 2: Dual-Path Retrieval]
  1. Encode query with SentenceTransformer (normalized embeddings)
  2. Dense path: FAISS IVF-PQ index search for top-k semantically similar chunks
     - Returns: dense_top_k chunks with cosine similarity scores
  3. Sparse path: BM25 keyword search on tokenized query
     - Returns: sparse_top_k chunks with BM25 relevance scores

[Step 3: Score Normalization & Fusion]
  1. Normalize dense and sparse scores separately to [0,1] range
  2. Compute hybrid scores using weighted sum:
     - hybrid_score = (dense_weight * dense_norm + sparse_weight * sparse_norm) / (dense_weight + sparse_weight)
     - Default weights: dense_weight=0.5, sparse_weight=0.5
  3. Union all candidates from both paths

[Step 4: Fisher Reranking (Optional)]
  - If enabled: Use Fisher relevance scoring for all chunks
    - Computes statistical significance of embeddings vs query in embedding space
    - Ranks entire corpus by Fisher relevance instead of hybrid scores
  - If disabled: Use final_score = dense_score (if dense hit) else sparse_score

[Step 5: Post-Processing & Deduplication]
  1. Sort candidates by final_score (descending)
  2. Apply MANDATORY deduplication:
     - Text-normalized duplicate removal (normalize_chunk_text)
     - Embedding-based duplicate removal (cosine similarity >= EMBEDDING_DEDUP_THRESHOLD = 0.7)
     - Keep only unique results
  3. Re-sort by final_score
  4. Trim to final_top_k results (default=20)

[Return Structure]
  Each result contains:
  {
    "chunk_idx": position in original corpus,
    "text": chunk content,
    "metadata": {"title": ..., "abstract": ...},
    "scores": {
      "dense_raw": raw cosine similarity from FAISS,
      "dense_rank": rank position in dense results,
      "sparse_raw": BM25 score,
      "sparse_rank": rank position in sparse results,
      "dense_norm": normalized dense score,
      "sparse_norm": normalized sparse score,
      "hybrid_norm": weighted fusion score,
      "dense_weight": weight parameter,
      "sparse_weight": weight parameter,
      "final_score": determinant score for ranking
    }
  }

OPTIONAL RERANKING MODES:
  • Standard: Hybrid fusion + dedup (default, fastest)
  • Fisher: Full-corpus statistical relevance (slower, single-query only)

INPUTS:
  • Query: Single query text or file with multiple queries
  • Parameters: dense_top_k (default 100), sparse_top_k (default 100), nprobe (default 10)
  • Weights: dense_weight, sparse_weight for score fusion

OUTPUT:
  • JSON list of top-k ranked chunks with hybrid scores and metadata




## How to use 
=============

1) Build a vector DB from .json files

python3 hybrid_vectordb_json.py build \
    --input-dir /path/to/data/folder \
    --out-dir /path/to/output/folder \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --chunk-size 256 \
    --semantic-threshold 0.85 \
    --overlap-words 3 \
    --sentence-chunk-weight 0.5 \
    --semantic-chunk-weight 0.5 \
    --nlist 256 \
    --nprobe 32

Build outputs include models + configuration only:
- dense_ivfpq.faiss
- sparse_bm25.pkl
- manifest.json

2) Search with a query

python3 hybrid_vectordb_json.py search \
    --db-dir /path/to/output_db \
    --query "what impacts climate crisis the most?" \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --dense-top-k 5 \
    --sparse-top-k 5 \
    --final-top-k 5 \
    --nprobe 32 \
    --dense-weight 0.5 \
    --sparse-weight 0.5

Search prints JSON to stdout and also saves:
- result_{first_three_words}.json (in --db-dir)

3) Search multiple queries continuously

Batch queries from CLI:

python3 hybrid_vectordb.py search \
    --db-dir /path/to/output/folder \
    --queries "what impacts climate crisis the most?" "what are top mitigation options?" \
    --embedding-model Qwen/Qwen3-Embedding-0.6B

Batch queries from file (one query per line):

python3 hybrid_vectordb.py search \
    --db-dir /path/to/output_db \
    --queries-file /path/to/queries.txt \
    --embedding-model Qwen/Qwen3-Embedding-0.6B

Interactive continuous mode (type 'exit' to stop):

python3 hybrid_vectordb.py search \
    --db-dir /path/to/output_db \
    --interactive \
    --embedding-model Qwen/Qwen3-Embedding-0.6B
