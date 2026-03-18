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
