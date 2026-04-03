

import argparse, importlib.machinery, json, logging, math, os, pickle, re, sys, time, types
from pathlib import Path
from typing import Any
import hashlib

import tqdm
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
try:
    import usearch.index as usearch
except ImportError:
    usearch = None

from sentence_utils import (
    extract_text,
    split_sentences,
    chunk_by_sentence_window,
    apply_word_overlap,
    normalize_chunk_text,
    tokenize
)

from json_utils import (
    iterate_jsonl,
    list_json_files,
    iterate_json_files,
    result_filename_from_query,
    unique_result_path
)

os.environ.setdefault("TRANSFORMERS_NO_APEX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TRAINER", "1")

if "apex" not in sys.modules:
    apex_stub = types.ModuleType("apex")
    apex_stub.amp = object()
    apex_stub.__spec__ = importlib.machinery.ModuleSpec("apex", loader=None)
    sys.modules["apex"] = apex_stub

from sentence_transformers.SentenceTransformer import SentenceTransformer


logger = logging.getLogger("hybrid_vectordb")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
CPU_THREADS = 40

# Deduplication threshold for build phase: chunks with embedding similarity >= 0.7 are considered duplicates
EMBEDDING_DEDUP_THRESHOLD = 0.7

def configure_cpu_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(CPU_THREADS)


def build_semantic_chunks(
    text: str,
    model: SentenceTransformer,
    chunk_size: int,
    overlap_words: int,
    similarity_threshold: float,
) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        chunks = chunk_by_sentence_window(sentences, chunk_size=chunk_size)
        return apply_word_overlap(chunks, overlap_words=overlap_words)

    embeddings = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=min(256, len(sentences)),
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    groups: list[list[str]] = []
    current_group: list[str] = [sentences[0]]
    for idx in range(1, len(sentences)):
        similarity = float(np.dot(embeddings[idx - 1], embeddings[idx]))
        if similarity < similarity_threshold and current_group:
            groups.append(current_group)
            current_group = [sentences[idx]]
        else:
            current_group.append(sentences[idx])

    if current_group:
        groups.append(current_group)

    semantic_chunks: list[str] = []
    for group in groups:
        semantic_chunks.extend(
            chunk_by_sentence_window(group, chunk_size=chunk_size)
        )
    return apply_word_overlap(semantic_chunks, overlap_words=overlap_words)


def weighted_merge_chunks(
    sentence_chunks: list[str],
    semantic_chunks: list[str],
    sentence_weight: float,
    semantic_weight: float,
) -> list[tuple[str, str, int]]:
    sentence_weight = max(0.0, sentence_weight)
    semantic_weight = max(0.0, semantic_weight)
    if sentence_weight == 0.0 and semantic_weight == 0.0:
        sentence_weight = 0.5
        semantic_weight = 0.5

    total = sentence_weight + semantic_weight
    sentence_ratio = sentence_weight / total
    semantic_ratio = semantic_weight / total

    s_idx = 0
    m_idx = 0
    out: list[tuple[str, str, int]] = []
    picked_sentence = 0
    picked_semantic = 0

    while s_idx < len(sentence_chunks) or m_idx < len(semantic_chunks):
        choose_sentence = False
        choose_semantic = False

        if s_idx < len(sentence_chunks) and m_idx < len(semantic_chunks):
            total_picked = max(1, picked_sentence + picked_semantic)
            current_sentence_share = picked_sentence / total_picked
            current_semantic_share = picked_semantic / total_picked
            sentence_deficit = sentence_ratio - current_sentence_share
            semantic_deficit = semantic_ratio - current_semantic_share
            choose_sentence = sentence_deficit >= semantic_deficit
            choose_semantic = not choose_sentence
        elif s_idx < len(sentence_chunks):
            choose_sentence = True
        elif m_idx < len(semantic_chunks):
            choose_semantic = True

        if choose_sentence:
            out.append((sentence_chunks[s_idx], "sentence", s_idx))
            s_idx += 1
            picked_sentence += 1
        elif choose_semantic:
            out.append((semantic_chunks[m_idx], "semantic", m_idx))
            m_idx += 1
            picked_semantic += 1

    return out


def choose_pq_m(dim: int, preferred: int = 64) -> int:
    candidate = min(preferred, dim)
    while candidate > 1:
        if dim % candidate == 0:
            return candidate
        candidate -= 1
    return 1


def build_command(args: argparse.Namespace) -> None:
    """
    Build hybrid vector database with dual-mechanism deduplication.

    Process:
      1. Chunk JSON documents with weighted sentence+semantic chunking
      2. Apply mandatory dual-mechanism deduplication:
         - Text normalization matching
         - Embedding-based similarity (threshold 0.7)
      3. Build dense FAISS IVF-PQ index from chunk embeddings
      4. Build sparse BM25 index from tokenized chunks
      5. Save indexes + metadata to output directory
    """
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dense_index_path = out_dir / "dense_ivfpq.faiss"
    sparse_index_path = out_dir / "sparse_bm25.pkl"
    manifest_path = out_dir / "manifest.json"

    configure_cpu_threads()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")

    logger.info("Loading embedding model '%s' on device=%s", args.embedding_model, args.device)
    model = SentenceTransformer(args.embedding_model, device=args.device, cache_folder=args.cache_dir or None)

    input_files = list_json_files(input_dir)
    logger.info("[1/3] Chunking source .json with weighted sentence+semantic chunking")
    t0 = time.time()
    source_docs = 0
    scanned_docs = 0
    written_chunks = 0
    skipped_duplicates = 0

    logger.info("[2/3] Building dense FAISS IVF-PQ index")
    train_target = max(1, args.train_vecs)
    nlist = args.nlist

    train_blocks: list[np.ndarray] = []
    train_count = 0
    index: faiss.IndexIVFPQ | None = None
    usearch_index: usearch.Index | None = None
    indexed_count = 0
    token_corpus: list[list[str]] = []
    chunk_texts: list[str] = []
    chunk_metadata: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    dedup_index: faiss.IndexFlatIP | None = None

    batch_items: list[tuple[str, dict[str, Any]]] = []

    def process_batch(items: list[tuple[str, dict[str, Any]]]) -> None:
        nonlocal train_count, train_blocks, index, usearch_index, indexed_count, dedup_index, skipped_duplicates, written_chunks
        if not items:
            return
        texts = [text for text, _ in items]
        embeddings = model.encode(
            texts,
            batch_size=args.embed_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # 1. String-based deduplication using memory-efficient hashes
        valid_indices = []
        for i, (chunk_text, _) in enumerate(items):
            normalized = normalize_chunk_text(chunk_text)
            text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            if text_hash in seen_chunks:
                skipped_duplicates += 1
                continue
            seen_chunks.add(text_hash)
            valid_indices.append(i)

        if not valid_indices:
            return

        # 2. Semantic deduplication using batched FAISS search
        filtered_embeddings = embeddings[valid_indices]
        
        if dedup_index is None:
            dedup_index = faiss.IndexFlatIP(int(filtered_embeddings.shape[1]))
            
        semantic_valid_indices = []
        if dedup_index.ntotal > 0:
            scores, _ = dedup_index.search(filtered_embeddings, 1)
            for j, i in enumerate(valid_indices):
                if float(scores[j][0]) >= EMBEDDING_DEDUP_THRESHOLD:
                    skipped_duplicates += 1
                else:
                    semantic_valid_indices.append(i)
        else:
            semantic_valid_indices = valid_indices

        if not semantic_valid_indices:
            return

        # 3. Add to the FAISS dedup index in a single batched operation
        kept_embeddings_arr = embeddings[semantic_valid_indices]
        dedup_index.add(kept_embeddings_arr)

        # 4. Finalize kept texts and tokens
        for i in semantic_valid_indices:
            chunk_text, metadata = items[i]
            token_corpus.append(tokenize(chunk_text))
            chunk_texts.append(chunk_text)
            chunk_metadata.append(metadata)
            written_chunks += 1

        embeddings = kept_embeddings_arr

        if index is None and usearch_index is None:
            if args.dense_backend == "usearch":
                if usearch is None:
                    raise ImportError("usearch package not found. Install it or use --dense-backend faiss")
                dim = int(embeddings.shape[1])
                usearch_index = usearch.Index(ndim=dim, metric="ip", dtype="f16")
                logger.info("Initialized USearch HNSW index dim=%s metric=ip dtype=f16", dim)
                usearch_index.add(np.arange(indexed_count, indexed_count + embeddings.shape[0]), embeddings)
                indexed_count += embeddings.shape[0]
            else:
                # FAISS Training Logic
                train_blocks.append(embeddings)
                train_count += embeddings.shape[0]
                if train_count >= train_target:
                    train_matrix = np.concatenate(train_blocks, axis=0)[:train_target]
                    dim = int(train_matrix.shape[1])
                    pq_m = choose_pq_m(dim, preferred=args.pq_m)
                    quantizer = faiss.IndexFlatIP(dim)
                    index_cpu = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, args.nbits, faiss.METRIC_INNER_PRODUCT)

                    index_cpu.train(train_matrix)
                    index = index_cpu
                    logger.info(
                        "Trained IVF-PQ dim=%s nlist=%s m=%s nbits=%s train_vecs=%s",
                        dim,
                        nlist,
                        pq_m,
                        args.nbits,
                        f"{train_target:,}",
                    )

                    for block in train_blocks:
                        index.add(block)
                        indexed_count += block.shape[0]
                    train_blocks = []
        elif usearch_index is not None:
            usearch_index.add(np.arange(indexed_count, indexed_count + embeddings.shape[0]), embeddings)
            indexed_count += embeddings.shape[0]
        else:
            index.add(embeddings)
            indexed_count += embeddings.shape[0]

    for file_path, _, record in tqdm.tqdm(iterate_json_files(input_files), desc="Processing JSON files"):
        if not isinstance(record, dict):
            continue
        text = extract_text(record)
        if not text:
            continue

        scanned_docs += 1

        source_docs += 1
        sentence_chunks = chunk_by_sentence_window(
            split_sentences(text),
            chunk_size=args.chunk_size,
        )
        sentence_chunks = apply_word_overlap(sentence_chunks, overlap_words=args.overlap_words)
        semantic_chunks = build_semantic_chunks(
            text,
            model=model,
            chunk_size=args.chunk_size,
            overlap_words=args.overlap_words,
            similarity_threshold=args.semantic_threshold,
        )

        merged_chunks = weighted_merge_chunks(
            sentence_chunks=sentence_chunks,
            semantic_chunks=semantic_chunks,
            sentence_weight=args.sentence_chunk_weight,
            semantic_weight=args.semantic_chunk_weight,
        )

        kept_chunks: list[str] = []
        for chunk_text, _, _ in merged_chunks:
            if not chunk_text:
                continue
            kept_chunks.append(chunk_text)

        if not kept_chunks:
            source_docs -= 1
            continue

        metadata = {
            "title": record.get("title"),
            "abstract": record.get("abstract"),
        }

        for chunk_text in kept_chunks:
            batch_items.append((chunk_text, metadata))
            if len(batch_items) >= args.embed_batch_size:
                process_batch(batch_items)
                batch_items = []
                if indexed_count and indexed_count % 500000 == 0:
                    logger.info("embedded/indexed=%s", f"{indexed_count:,}")

        if args.max_docs and source_docs >= args.max_docs:
            logger.info("Reached max docs limit: %s", args.max_docs)
            break

        if source_docs % 10000 == 0:
            logger.info(
                "chunked docs=%s chunks=%s elapsed=%.1fm last_file=%s",
                f"{source_docs:,}",
                f"{written_chunks:,}",
                (time.time() - t0) / 60,
                file_path.name,
            )

    logger.info("Chunking complete: docs=%s chunks=%s", f"{source_docs:,}", f"{written_chunks:,}")

    if batch_items:
        process_batch(batch_items)

    if usearch_index is not None:
        dense_index_path = out_dir / "dense_hnsw.usearch"
        usearch_index.save(str(dense_index_path))
        logger.info("Dense index complete (USearch): vectors=%s path=%s", indexed_count, dense_index_path)
    else:
        if index is None and train_blocks:
            train_matrix = np.concatenate(train_blocks, axis=0)
            dim = int(train_matrix.shape[1])
            pq_m = choose_pq_m(dim, preferred=args.pq_m)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, args.nbits, faiss.METRIC_INNER_PRODUCT)
            index.train(train_matrix)
            index.add(train_matrix)
            indexed_count = index.ntotal
            train_target = int(train_matrix.shape[0])
            logger.info(
                "Trained IVF-PQ on available vectors dim=%s nlist=%s m=%s nbits=%s train_vecs=%s",
                dim,
                nlist,
                pq_m,
                args.nbits,
                f"{train_target:,}",
            )

        if index is None:
            raise RuntimeError("Not enough data to train IVF-PQ index.")

        index.nprobe = args.nprobe
        faiss.write_index(index, str(dense_index_path))
        logger.info("Dense index complete (FAISS): vectors=%s path=%s", f"{index.ntotal:,}", dense_index_path)

    logger.info("[3/3] Building sparse BM25 index")
    sparse_payload = {
        "type": "bm25",
        "token_corpus": token_corpus,
        "chunk_texts": chunk_texts,
        "chunk_metadata": chunk_metadata,
    }
    with sparse_index_path.open("wb") as file_obj:
        pickle.dump(sparse_payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "input_dir": str(input_dir),
        "input_files": len(input_files),
        "output_dir": str(out_dir),
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "semantic_threshold": args.semantic_threshold,
        "overlap_words": args.overlap_words,
        "sentence_chunk_weight": args.sentence_chunk_weight,
        "semantic_chunk_weight": args.semantic_chunk_weight,
        "index_backend": args.dense_backend,
        "index_type": "IndexIVFPQ" if args.dense_backend == "faiss" else "HNSW",
        "metric": "inner_product",
        "nlist": args.nlist if args.dense_backend == "faiss" else None,
        "nprobe": args.nprobe if args.dense_backend == "faiss" else None,
        "train_vecs": train_target if args.dense_backend == "faiss" else 0,
        "nbits": args.nbits if args.dense_backend == "faiss" else None,
        "documents_chunked": source_docs,
        "documents_scanned": scanned_docs,
        "chunks_count": written_chunks,
        "skipped_duplicate_chunks": skipped_duplicates,
        "dense_vectors": int(indexed_count),
        "elapsed_seconds": round(time.time() - t0, 2),
        "max_docs": int(args.max_docs),
        "files": {
            "dense_index": str(dense_index_path),
            "sparse_bm25": str(sparse_index_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Build complete. Manifest: %s", manifest_path)


def normalize_scores(raw_scores: dict[int, float], higher_is_better: bool) -> dict[int, float]:
    if not raw_scores:
        return {}
    values = np.asarray(list(raw_scores.values()), dtype=np.float32)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if math.isclose(max_val, min_val, rel_tol=0.0, abs_tol=1e-12):
        return {key: 1.0 for key in raw_scores}
    if higher_is_better:
        return {key: (value - min_val) / (max_val - min_val) for key, value in raw_scores.items()}
    return {key: (max_val - value) / (max_val - min_val) for key, value in raw_scores.items()}


def search_command(args: argparse.Namespace) -> None:
    """
    Search hybrid vector database using dual-path retrieval.

    Process:
      1. Load pre-built dense (FAISS) and sparse (BM25) indexes
      2. Encode query with SentenceTransformer
      3. Retrieve candidates from both indexes:
         - Dense path: Top-k semantically similar chunks (cosine similarity)
         - Sparse path: Top-k keyword-relevant chunks (BM25 scores)
      4. Normalize and fuse scores with configurable weights
      5. Return top-k ranked chunks by hybrid score

    Note: No Fisher reranking or post-search deduplication applied.
            Use EMBEDDING_DEDUP_THRESHOLD only during build phase.
    """
    db_dir = Path(args.db_dir)
    dense_path = db_dir / "dense_ivfpq.faiss"
    sparse_path = db_dir / "sparse_bm25.pkl"

    configure_cpu_threads()

    manifest_path = db_dir / "manifest.json"
    dense_backend = "faiss"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        dense_backend = manifest.get("index_backend", "faiss")
        dense_path = Path(manifest["files"].get("dense_index", manifest["files"].get("dense_faiss")))

    if dense_backend == "usearch":
        if usearch is None:
            raise ImportError("usearch package not found.")
        index = usearch.Index.restore(str(dense_path))
        chunk_count_dense = len(index)
    else:
        index = faiss.read_index(str(dense_path))
        index.nprobe = args.nprobe
        chunk_count_dense = int(index.ntotal)

    with sparse_path.open("rb") as file_obj:
        sparse_payload = pickle.load(file_obj)

    token_corpus = sparse_payload.get("token_corpus", []) if isinstance(sparse_payload, dict) else []
    chunk_texts = sparse_payload.get("chunk_texts", []) if isinstance(sparse_payload, dict) else []
    chunk_metadata = sparse_payload.get("chunk_metadata", []) if isinstance(sparse_payload, dict) else []
    bm25 = BM25Okapi(token_corpus)
    chunk_count = min(chunk_count_dense, len(token_corpus), len(chunk_texts), len(chunk_metadata))
    if chunk_count == 0:
        raise RuntimeError("Empty index or token corpus. Rebuild the DB.")

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    model = SentenceTransformer(args.embedding_model, device=args.device, cache_folder=args.cache_dir or None)


    def run_single_query(query_text: str) -> dict[str, Any]:
        dense_raw: dict[int, float] = {}
        sparse_raw: dict[int, float] = {}
        dense_rank: dict[int, int] = {}
        sparse_rank: dict[int, int] = {}

        query_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vec = np.asarray(query_vec, dtype=np.float32)

        if dense_backend == "usearch":
            matches = index.search(query_vec, args.dense_top_k)
            dense_ids_arr = matches.labels[0]
            dense_scores_arr = matches.distances[0]
        else:
            dense_scores_arr, dense_ids_arr = index.search(query_vec, args.dense_top_k)
            dense_scores_arr = dense_scores_arr[0]
            dense_ids_arr = dense_ids_arr[0]

        sparse_scores_arr = bm25.get_scores(tokenize(query_text))
        sparse_ids = np.argsort(sparse_scores_arr)[::-1][: args.sparse_top_k]

        for rank, chunk_idx in enumerate(dense_ids_arr, start=1):
            chunk_idx = int(chunk_idx)
            if chunk_idx < 0 or chunk_idx >= chunk_count:
                continue
            dense_raw[chunk_idx] = float(dense_scores_arr[rank - 1])
            dense_rank[chunk_idx] = rank

        for rank, chunk_idx in enumerate(sparse_ids, start=1):
            chunk_idx = int(chunk_idx)
            if chunk_idx < 0 or chunk_idx >= chunk_count:
                continue
            sparse_raw[chunk_idx] = float(sparse_scores_arr[chunk_idx])
            sparse_rank[chunk_idx] = rank

        dense_norm = normalize_scores(dense_raw, higher_is_better=True)
        sparse_norm = normalize_scores(sparse_raw, higher_is_better=True)

        dense_weight = max(0.0, args.dense_weight)
        sparse_weight = max(0.0, args.sparse_weight)
        weight_sum = dense_weight + sparse_weight
        if weight_sum == 0.0:
            dense_weight = sparse_weight = 0.5
            weight_sum = 1.0

        candidates = list(set(dense_raw.keys()) | set(sparse_raw.keys()))
        rescored: list[dict[str, Any]] = []
        for chunk_idx in candidates:
            hybrid = (
                dense_weight * dense_norm.get(chunk_idx, 0.0)
                + sparse_weight * sparse_norm.get(chunk_idx, 0.0)
            ) / weight_sum
            if chunk_idx in dense_raw:
                final_score = dense_raw.get(chunk_idx)
            else:
                final_score = sparse_raw.get(chunk_idx, 0.0)
            rescored.append(
                {
                    "chunk_idx": chunk_idx,
                    "text": chunk_texts[chunk_idx],
                    "metadata": chunk_metadata[chunk_idx],
                    "scores": {
                        "dense_raw": dense_raw.get(chunk_idx),
                        "dense_rank": dense_rank.get(chunk_idx),
                        "sparse_raw": sparse_raw.get(chunk_idx),
                        "sparse_rank": sparse_rank.get(chunk_idx),
                        "dense_norm": dense_norm.get(chunk_idx, 0.0),
                        "sparse_norm": sparse_norm.get(chunk_idx, 0.0),
                        "hybrid_norm": hybrid,
                        "dense_weight": dense_weight,
                        "sparse_weight": sparse_weight,
                        "final_score": final_score,
                    },
                }
            )

        rescored = sorted(rescored, key=lambda item: item["scores"]["final_score"], reverse=True)

        output_top_k = max(args.final_top_k, 20)
        rescored = rescored[:output_top_k]

        return {
            "query": query_text,
            "embedding_model": args.embedding_model,
            "retrieve": {
                "dense_top_k": args.dense_top_k,
                "sparse_top_k": args.sparse_top_k,
                "final_top_k": args.final_top_k,
                "nprobe": args.nprobe,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
            },
            "results": rescored,
        }

    query_list: list[str] = []
    if args.queries_file:
        q_path = Path(args.queries_file)
        if not q_path.exists():
            raise FileNotFoundError(f"Queries file not found: {q_path}")
        with q_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                q = line.strip()
                if q:
                    query_list.append(q)

    if args.queries:
        query_list.extend([q.strip() for q in args.queries if q.strip()])

    if args.query and args.query.strip():
        query_list.append(args.query.strip())

    if args.interactive:
        print("Interactive mode: enter queries one per line. Type 'exit' to stop.")
        while True:
            user_query = input("query> ").strip()
            if not user_query:
                continue
            if user_query.lower() in {"exit", "quit", "q"}:
                break
            result = run_single_query(user_query)
            result_file = unique_result_path(db_dir, user_query)
            result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Saved search result: %s", result_file)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not query_list:
        raise ValueError("Provide at least one query via --query, --queries, --queries-file, or use --interactive")

    for query_text in query_list:
        result = run_single_query(query_text)
        result_file = unique_result_path(db_dir, query_text)
        result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved search result: %s", result_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a weighted sentence+semantic hybrid vector DB over prebuilt chunks JSONL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build chunked corpus + dense/sparse indexes")
    build.add_argument("--input-dir", type=str, default=Path.home() / "data/new_samples")
    build.add_argument("--out-dir", type=str, required=True)
    build.add_argument("--dense-backend", type=str, choices=["faiss", "usearch"], default="faiss")
    build.add_argument("--device", type=str, default="cpu", help="Device for embedding model (e.g., cpu, cuda, mps)")
    build.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    build.add_argument("--chunk-size", type=int, default=256)
    build.add_argument("--overlap-words", type=int, default=3)
    build.add_argument("--semantic-threshold", type=float, default=0.85)
    build.add_argument("--sentence-chunk-weight", type=float, default=0.5)
    build.add_argument("--semantic-chunk-weight", type=float, default=0.5)
    build.add_argument("--embed-batch-size", type=int, default=64)
    build.add_argument("--nlist", type=int, default=256)
    build.add_argument("--nprobe", type=int, default=32)
    build.add_argument("--pq-m", type=int, default=64)
    build.add_argument("--nbits", type=int, default=8)
    build.add_argument("--train-vecs", type=int, default=20000)
    build.add_argument("--max-docs", type=int, default=0)
    build.add_argument("--cache-dir", type=str, default=Path.home()/".cache/huggingface")

    search = subparsers.add_parser("search", help="Search built DB and return top fused chunks")
    search.add_argument("--db-dir", type=str, required=True)
    search.add_argument("--dense-backend", type=str, choices=["faiss", "usearch"], default="faiss")
    search.add_argument("--device", type=str, default="cpu", help="Device for embedding model (e.g., cpu, cuda, mps)")
    search.add_argument("--query", type=str, default="")
    search.add_argument("--queries", type=str, nargs="+")
    search.add_argument("--queries-file", type=str, default="")
    search.add_argument("--interactive", action="store_true")
    search.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    search.add_argument("--dense-top-k", type=int, default=5)
    search.add_argument("--sparse-top-k", type=int, default=5)
    search.add_argument("--final-top-k", type=int, default=5)
    search.add_argument("--nprobe", type=int, default=32)
    search.add_argument("--dense-weight", type=float, default=0.5)
    search.add_argument("--sparse-weight", type=float, default=0.5)
    search.add_argument("--cache-dir", type=str, default=Path.home()/".cache/huggingface")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    s = time.time()
    main()
    print(f"Total elapsed time: {(time.time() - s) / 60:.2f} minutes")

