"""File de validation : rien n'est envoyé tant que l'encadrant n'a pas approuvé le brouillon.

États : pending -> approved | rejected ; approved -> sent.
À l'approbation on enregistre l'empreinte SHA-256 du texte approuvé ; à l'envoi on la revérifie,
pour qu'un texte modifié après validation ne puisse pas partir.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .drafter import Draft
from .mailer import Mailer, Outgoing

SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    sender TEXT NOT NULL,
    subject TEXT NOT NULL,
    message_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    body_original TEXT NOT NULL,
    body_final TEXT NOT NULL,
    citations TEXT NOT NULL,
    retrieved TEXT NOT NULL,
    confidence REAL NOT NULL,
    abstained INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    warnings TEXT NOT NULL DEFAULT '[]',
    model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    reviewed_at TEXT,
    review_note TEXT,
    edit_ratio REAL,
    approved_hash TEXT,
    sent_at TEXT,
    send_ref TEXT
);
"""


class ReviewError(Exception):
    pass


class NotApprovedError(ReviewError):
    """Tentative d'envoi d'un brouillon non approuvé (ou modifié après approbation)."""


class InvalidTransition(ReviewError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def edit_ratio(original: str, final: str) -> float:
    """0 = identique, 1 = entièrement réécrit (distance basée sur difflib)."""
    return round(1 - difflib.SequenceMatcher(None, original, final).ratio(), 4)


@dataclass
class Stats:
    total: int
    pending: int
    approved: int
    sent: int
    rejected: int
    reviewed: int
    approval_rate: float | None  # approuvés+envoyés / revus
    unedited_rate: float | None  # parmi les approuvés, part envoyée sans aucune modification
    mean_edit_ratio: float | None  # parmi les approuvés
    abstained_rate: float | None  # parmi tous les brouillons


class ReviewQueue:
    def __init__(self, db_path: str | Path = "var/rag.db"):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ création / lecture
    def add(self, d: Draft) -> int:
        cur = self.conn.execute(
            """INSERT INTO drafts (created_at, sender, subject, message_id, question, body_original,
               body_final, citations, retrieved, confidence, abstained, reason, warnings, model)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(), d.sender, d.subject, d.message_id, d.question, d.body, d.body,
                json.dumps(d.citations), json.dumps(d.retrieved), d.confidence, int(d.abstained),
                d.reason, json.dumps(d.warnings, ensure_ascii=False), d.model,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, draft_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
        if row is None:
            raise ReviewError(f"Brouillon {draft_id} introuvable")
        return row

    def list(self, status: str | None = None) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute("SELECT * FROM drafts WHERE status=? ORDER BY id", (status,)).fetchall()
        return self.conn.execute("SELECT * FROM drafts ORDER BY id").fetchall()

    # ------------------------------------------------------------------ décisions
    def _require_pending(self, draft_id: int) -> sqlite3.Row:
        row = self.get(draft_id)
        if row["status"] != "pending":
            raise InvalidTransition(f"Brouillon {draft_id} déjà « {row['status']} »")
        return row

    def approve(self, draft_id: int, reviewer: str, body: str | None = None, note: str = "") -> None:
        """Approuve le brouillon, tel quel ou avec le texte corrigé `body`."""
        if not reviewer.strip():
            raise ReviewError("Le nom de l'encadrant est obligatoire")
        row = self._require_pending(draft_id)
        final = body if body is not None else row["body_original"]
        if not final.strip():
            raise ReviewError("Le texte approuvé est vide")
        self.conn.execute(
            """UPDATE drafts SET status='approved', reviewer=?, reviewed_at=?, review_note=?,
               body_final=?, edit_ratio=?, approved_hash=? WHERE id=?""",
            (reviewer, _now(), note, final, edit_ratio(row["body_original"], final), _hash(final), draft_id),
        )
        self.conn.commit()

    def reject(self, draft_id: int, reviewer: str, reason: str = "") -> None:
        if not reviewer.strip():
            raise ReviewError("Le nom de l'encadrant est obligatoire")
        self._require_pending(draft_id)
        self.conn.execute(
            "UPDATE drafts SET status='rejected', reviewer=?, reviewed_at=?, review_note=? WHERE id=?",
            (reviewer, _now(), reason, draft_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ envoi
    def send(self, draft_id: int, mailer: Mailer) -> str:
        row = self.get(draft_id)
        if row["status"] != "approved":
            raise NotApprovedError(f"Brouillon {draft_id} : statut « {row['status']} », envoi refusé")
        if row["approved_hash"] != _hash(row["body_final"]):
            raise NotApprovedError(f"Brouillon {draft_id} : texte modifié après approbation, envoi refusé")
        subject = row["subject"]
        ref = mailer.send(
            Outgoing(
                to=row["sender"],
                subject=subject if subject.lower().startswith("re:") else f"Re: {subject}",
                body=row["body_final"],
                in_reply_to=row["message_id"],
            )
        )
        self.conn.execute(
            "UPDATE drafts SET status='sent', sent_at=?, send_ref=? WHERE id=?", (_now(), ref, draft_id)
        )
        self.conn.commit()
        return ref

    # ------------------------------------------------------------------ indicateurs
    def stats(self) -> Stats:
        rows = self.list()
        by = lambda *s: [r for r in rows if r["status"] in s]  # noqa: E731
        decided = by("approved", "sent", "rejected")
        ok = by("approved", "sent")
        ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        return Stats(
            total=len(rows),
            pending=len(by("pending")),
            approved=len(by("approved")),
            sent=len(by("sent")),
            rejected=len(by("rejected")),
            reviewed=len(decided),
            approval_rate=ratio(len(ok), len(decided)),
            unedited_rate=ratio(sum(1 for r in ok if r["edit_ratio"] == 0), len(ok)),
            mean_edit_ratio=round(sum(r["edit_ratio"] for r in ok) / len(ok), 4) if ok else None,
            abstained_rate=ratio(sum(r["abstained"] for r in rows), len(rows)),
        )
