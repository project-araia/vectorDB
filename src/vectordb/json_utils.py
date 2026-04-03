import json
import logging
import re
from pathlib import Path
from typing import Generator, Any

logger = logging.getLogger(__name__)

def iterate_jsonl(path: Path) -> Generator[tuple[int, dict[str, Any]], None, None]:
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line %s in %s", line_number, path)

def list_json_files(input_dir: Path) -> list[Path]:
    """
    Recursively find all .json and .jsonl files in the input directory.
    This handles sharded datasets distributed across subdirectories.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    # Using rglob to recursively find files in all subdirectories (shards)
    # We look for both .json and .jsonl as sharded datasets may use either.
    json_files = list(input_dir.rglob("*.json"))
    jsonl_files = list(input_dir.rglob("*.jsonl"))

    files = sorted(json_files + jsonl_files)

    if not files:
        raise FileNotFoundError(f"No .json or .jsonl files found under: {input_dir}")

    logger.info("Found %d sharded files in %s", len(files), input_dir)
    return files

def iterate_json_files(paths: list[Path]) -> Generator[tuple[Path, int, dict[str, Any]], None, None]:
    for path in paths:
        if path.suffix == ".jsonl":
            for line_no, record in iterate_jsonl(path):
                yield path, line_no, record
            continue

        with path.open("r", encoding="utf-8") as file_obj:
            try:
                data = json.load(file_obj)
            except json.JSONDecodeError:
                # Fallback to JSONL iteration if standard JSON fails
                file_obj.seek(0)
                for line_no, record in iterate_jsonl(path):
                    yield path, line_no, record
                continue

        if isinstance(data, list):
            for idx, record in enumerate(data, start=1):
                yield path, idx, record
        else:
            yield path, 1, data

def result_filename_from_query(query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    first_three = words[:3]
    if not first_three:
        return "result_query.json"
    return f"result_{'_'.join(first_three)}.json"

def unique_result_path(db_dir: Path, query: str) -> Path:
    base_name = result_filename_from_query(query)
    base_path = db_dir / base_name
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    counter = 2
    while True:
        candidate = db_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1