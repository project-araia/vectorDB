from typing import Any
import re

def extract_text(record: dict[str, Any]) -> str:
    for key in ("text", "content", "body", "abstract", "introduction", "methodology", "discussion", "conclusion", "summary"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_chunk_text(text: str) -> str:
    return " ".join(text.lower().split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def apply_word_overlap(chunks: list[str], overlap_words: int) -> list[str]:
    overlap_words = max(0, overlap_words)
    if overlap_words == 0 or len(chunks) < 2:
        return chunks

    out = [chunks[0]]
    prev_words = chunks[0].split()
    for chunk in chunks[1:]:
        overlap = prev_words[-overlap_words:] if prev_words else []
        if overlap:
            curr_words = chunk.split()
            if len(curr_words) >= overlap_words and curr_words[:overlap_words] == overlap:
                merged = chunk
            else:
                merged = " ".join(overlap + curr_words)
            chunk = merged
        out.append(chunk)
        prev_words = chunk.split()
    return out


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def chunk_by_sentence_window(sentences: list[str], chunk_size: int) -> list[str]:
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    n_sentences = len(sentences)

    while start < n_sentences:
        end = start
        current_len = 0
        while end < n_sentences:
            sentence = sentences[end]
            projected = current_len + (1 if current_len else 0) + len(sentence)
            if projected > chunk_size and end > start:
                break
            current_len = projected
            end += 1
            if current_len >= chunk_size:
                break

        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= n_sentences:
            break

        start = end

    return chunks