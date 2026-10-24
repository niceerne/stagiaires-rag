"""Envoi des réponses. Le mode par défaut est « à blanc » : aucun e-mail ne part réellement."""

from __future__ import annotations

import json
import os
import smtplib
import time
from dataclasses import asdict, dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Outgoing:
    to: str
    subject: str
    body: str
    in_reply_to: str = ""


class Mailer(Protocol):
    def send(self, msg: Outgoing) -> str:
        """Envoie et retourne un identifiant de suivi."""


class DryRunMailer:
    """Écrit le message dans un dossier au lieu de l'envoyer (défaut)."""

    def __init__(self, outbox: str | Path = "var/outbox"):
        self.outbox = Path(outbox)

    def send(self, msg: Outgoing) -> str:
        self.outbox.mkdir(parents=True, exist_ok=True)
        ref = f"dryrun-{time.time_ns()}"
        (self.outbox / f"{ref}.json").write_text(
            json.dumps(asdict(msg), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ref


class SMTPMailer:
    """Envoi réel via SMTP (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM)."""

    def __init__(self):
        self.host = os.environ["SMTP_HOST"]
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER", "")
        self.password = os.environ.get("SMTP_PASSWORD", "")
        self.from_addr = os.environ.get("SMTP_FROM", self.user)

    def send(self, msg: Outgoing) -> str:
        m = EmailMessage()
        m["From"], m["To"], m["Subject"] = self.from_addr, msg.to, msg.subject
        if msg.in_reply_to:
            m["In-Reply-To"] = msg.in_reply_to
            m["References"] = msg.in_reply_to
        m.set_content(msg.body)
        with smtplib.SMTP(self.host, self.port, timeout=30) as s:
            s.starttls()
            if self.user:
                s.login(self.user, self.password)
            s.send_message(m)
        return f"smtp-{time.time_ns()}"
