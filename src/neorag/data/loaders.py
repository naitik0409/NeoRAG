from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from neorag.logging_utils import logger


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _stable_id(rel_path: Path) -> str:
    return hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()[:16]


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_txt(path: Path) -> str:
    return _read_text_file(path)


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _rows_to_text(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        parts = [f"{k}: {v}" for k, v in row.items() if v not in (None, "")]
        if parts:
            lines.append("; ".join(parts))
    return "\n".join(lines)


def load_csv(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return _rows_to_text(rows)


def load_json(path: Path) -> str:
    data = json.loads(_read_text_file(path))
    if isinstance(data, list):
        return _rows_to_text([r for r in data if isinstance(r, dict)])
    if isinstance(data, dict):
        rows = data.get("data") if isinstance(data.get("data"), list) else [data]
        return _rows_to_text([r for r in rows if isinstance(r, dict)])
    return json.dumps(data, indent=2)


LOADERS = {
    "txt": load_txt,
    "md": load_txt,
    "pdf": load_pdf,
    "docx": load_docx,
    "html": load_html,
    "htm": load_html,
    "csv": load_csv,
    "json": load_json,
}


def load_file(path: Path) -> Document | None:
    suffix = path.suffix.lstrip(".").lower()
    loader = LOADERS.get(suffix)
    if loader is None:
        return None
    try:
        text = loader(path)
    except Exception as exc:
        logger.warning(f"Failed to load {path}: {exc}")
        return None
    text = text.strip()
    if not text:
        return None
    rel = path.name
    return Document(
        doc_id=_stable_id(rel),
        text=text,
        metadata={"source": str(path), "title": path.stem, "file_type": suffix},
    )


def load_corpus(source_dir: str | Path, file_types: list[str] | None = None) -> list[Document]:
    root = Path(source_dir)
    if not root.exists():
        logger.warning(f"Corpus directory not found: {root}")
        return []
    allowed = set((file_types or LOADERS.keys()))
    documents = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lstrip(".").lower()
        if suffix not in allowed or suffix not in LOADERS:
            continue
        doc = load_file(path)
        if doc:
            documents.append(doc)
    logger.info(f"Loaded {len(documents)} documents from {root}")
    return documents
