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
