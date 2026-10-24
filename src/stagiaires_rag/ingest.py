"""Chargement des documents du stage et découpage en passages (chunks)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Chunk:
    id: str  # "<fichier>#<n>"
    source: str  # nom du fichier d'origine
    doc_title: str
    heading: str
    text: str

    @property
    def index_text(self) -> str:
        """Texte indexé : le titre et la section aident à retrouver le bon passage."""
        return f"{self.doc_title}\n{self.heading}\n{self.text}"


def _split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Retourne (titre du document, [(titre de section, contenu)])."""
    title = ""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
                continue
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    out = [(h, "\n".join(body).strip()) for h, body in sections]
    return title, [(h, b) for h, b in out if b]


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    """Découpe un texte trop long en fenêtres de mots, en gardant un recouvrement."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if current and count + n > max_words:
            pieces.append("\n\n".join(current))
            # recouvrement : on garde le dernier paragraphe si court
            tail = current[-1] if len(current[-1].split()) <= overlap else ""
            current, count = ([tail] if tail else []), len(tail.split())
        current.append(para)
        count += n
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(name: str, content: str, max_words: int = 140, overlap: int = 30) -> list[Chunk]:
    title, sections = _split_sections(content)
    title = title or Path(name).stem
    chunks: list[Chunk] = []
    for heading, body in sections:
        pieces = _split_long(body, max_words, overlap) if len(body.split()) > max_words else [body]
        for piece in pieces:
            chunks.append(
                Chunk(
                    id=f"{name}#{len(chunks)}",
                    source=name,
                    doc_title=title,
                    heading=heading,
                    text=piece,
                )
            )
    return chunks


def load_chunks(docs_dir: str | Path, max_words: int = 140, overlap: int = 30) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Dossier de documents introuvable : {docs_dir}")
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            chunks.extend(chunk_document(path.name, path.read_text(encoding="utf-8"), max_words, overlap))
    if not chunks:
        raise ValueError(f"Aucun document .md/.txt exploitable dans {docs_dir}")
    return chunks
