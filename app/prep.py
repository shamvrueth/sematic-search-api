import re
import os
import json
import hashlib
from pathlib import Path
from typing import Optional
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

HEADER_PATTERN = re.compile(
    r"^(From|Subject|Organization|Lines|NNTP-Posting-Host|"
    r"Message-ID|References|Date|Reply-To|Followup-To|"
    r"X-Newsreader|X-Mailer|Distribution|Keywords|Summary|"
    r"Archive-name|Last-modified|Version|Xref|Path|Newsgroups|"
    r"Sender|Approved|Expires|Supersedes|Mime-Version|"
    r"Content-Type|Content-Transfer-Encoding):"
    r".*$",
    re.MULTILINE | re.IGNORECASE,
)

QUOTE_LINE_PATTERN = re.compile(r"^\s*>.*$", re.MULTILINE)
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]+")
MULTI_BLANK_PATTERN = re.compile(r"\n{3,}")

def strip_headers(text: str) -> str:
    parts = re.split(r"\n\s*\n", text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else text
    body = HEADER_PATTERN.sub("", body)
    return body

def quote_ratio(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 1.0
    quoted = sum(1 for l in lines if re.match(r"^\s*>", l))
    return quoted / len(lines)

def clean_text(text: str) -> str:
    text = strip_headers(text)
    text = QUOTE_LINE_PATTERN.sub("", text)
    text = NON_ASCII_PATTERN.sub(" ", text)
    text = MULTI_BLANK_PATTERN.sub("\n\n", text)
    return text.strip()

def load_and_clean(data_dir: str, min_words: int = 50, max_ratio: float = 0.6) -> pd.DataFrame:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            f"Expected the extracted 20_newsgroups folder."
        )
    
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not dirs:
        raise ValueError(f"No subdirectories found in {root}. Is the path correct?")
    
    names = [d.name for d in dirs]
    category_to_int = {name: idx for idx, name in enumerate(names)}

    records = []
    discarded = {"too_short": 0, "too_quoted": 0, "duplicate": 0, "read_error": 0}
    hashes = set()
    idx = 0

    for dir in tqdm(dirs, desc="Categories"):
        category = dir.name
        cat_int = category_to_int[category]
        files = sorted([f for f in dir.iterdir() if f.is_file()])

        for path in files:
            idx += 1
            try:
                raw_text = path.read_text(encoding="latin-1", errors="replace")
            except Exception as e:
                discarded["read_error"] += 1
                continue

            cleaned = clean_text(raw_text)
            word_count = len(cleaned.split())
            if word_count < min_words:
                discarded["too_short"] += 1
                continue


            if quote_ratio(cleaned) > max_ratio:
                discarded["too_quoted"] += 1
                continue
            
            if not cleaned.strip():
                discarded["too_short"] += 1
                continue
            
            content_hash = hashlib.md5(cleaned.encode()).hexdigest()
            if content_hash in hashes:
                discarded["duplicate"] += 1
                continue
            hashes.add(content_hash)

            records.append(
                {
                    "doc_id": f"doc_{idx:05d}",
                    "text": cleaned,
                    "category": category,
                    "category_int": cat_int,
                    "word_count": word_count,
                    "source_file": str(path.name),
                }
            )
    
    df = pd.DataFrame(records)
    return df

def save_processed(df: pd.DataFrame, out_dir: str = "./data/processed") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    parquet_path = Path(out_dir) / "corpus.parquet"
    df.to_parquet(parquet_path, index=False)

    summary = {
        "total_documents": len(df),
        "categories": df["category"].value_counts().sort_index().to_dict(),
        "word_count_stats": df["word_count"].describe().round(1).to_dict(),
    }
    summary_path = Path(out_dir) / "corpus_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare 20 Newsgroups corpus from local files")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"..\data\20_newsgroups",
        help="Path to extracted 20_newsgroups root directory",
    )
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-quote-ratio", type=float, default=0.6)
    args = parser.parse_args()
    df = load_and_clean(args.data_dir, args.min_words, args.max_quote_ratio,)
    save_processed(df)
    print(f"Done. Saved {len(df)} documents")

