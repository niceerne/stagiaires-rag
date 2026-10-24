"""Du e-mail entrant au brouillon : récupération, garde-fous, génération, mise en forme."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import parseaddr

from .ingest import Chunk
from .llm import LLM
from .retriever import BM25Retriever, Hit
from .text import normalize_for_match

MAX_QUESTION_CHARS = 2000


@dataclass
class IncomingEmail:
    sender: str  # "Prénom Nom <adresse>" ou adresse
    subject: str
    body: str
    message_id: str = ""


@dataclass
class Draft:
    sender: str
    subject: str
    question: str
    body: str  # brouillon complet (formule d'appel + réponse + signature)
    citations: list[str] = field(default_factory=list)  # ids de passages cités et valides
    retrieved: list[str] = field(default_factory=list)  # ids de passages récupérés (ordre de rang)
    confidence: float = 0.0  # couverture lexicale du meilleur passage (0-1)
    abstained: bool = False
    reason: str = ""  # pourquoi abstention
    warnings: list[str] = field(default_factory=list)
    model: str = ""
    message_id: str = ""
    latency_s: float = 0.0


# --------------------------------------------------------------------------- utilitaires


_CLOSING_RE = re.compile(
    r"^\s*(merci( d'avance| beaucoup| par avance)?|cordialement|bien cordialement|bien à vous|"
    r"bien a vous|salutations|cdlt|bonne journée|bonne journee|bonne soirée|bonne soiree|"
    r"envoyé de mon|sent from)\b",
    re.IGNORECASE,
)


def clean_question(subject: str, body: str) -> str:
    """Retire citations (>), formules de politesse finales, signature et limite la taille."""
    lines = []
    for line in body.splitlines():
        if line.strip() == "--" or line.startswith("-- "):
            break
        if _CLOSING_RE.match(line) and len(line.split()) <= 6:
            break  # tout ce qui suit est une signature
        if line.lstrip().startswith(">"):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    return f"{subject.strip()}\n{text}"[:MAX_QUESTION_CHARS].strip()


def first_name(sender: str) -> str:
    name, addr = parseaddr(sender)
    name = name.strip().strip('"')
    if not name:
        return ""
    if "," in name:  # "NOM, Prénom"
        name = name.split(",", 1)[1].strip()
    return name.split()[0].capitalize() if name else ""


_NUM_RE = re.compile(r"\d[\d.,:h]*\d|\d")
_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def ungrounded_facts(answer: str, cited: list[Chunk]) -> list[str]:
    """Nombres, horaires et adresses e-mail de la réponse absents des passages cités."""
    support = normalize_for_match(" ".join(c.text for c in cited))
    norm = normalize_for_match(answer)
    missing = []
    for token in _ADDR_RE.findall(norm) + _NUM_RE.findall(_ADDR_RE.sub(" ", norm)):
        token = token.rstrip(".,:")
        if token and token not in support and token not in missing:
            missing.append(token)
    return missing


# --------------------------------------------------------------------------- pipeline


class Drafter:
    def __init__(
        self,
        retriever: BM25Retriever,
        llm: LLM,
        top_k: int = 4,
        min_coverage: float = 0.3,
        supervisor: str = "",
    ):
        self.retriever, self.llm = retriever, llm
        self.top_k, self.min_coverage = top_k, min_coverage
        self.supervisor = supervisor

    # mise en forme -----------------------------------------------------------
    def _wrap(self, sender: str, answer: str) -> str:
        name = first_name(sender)
        greeting = f"Bonjour {name}," if name else "Bonjour,"
        signature = f"Cordialement,\n{self.supervisor}".rstrip()
        return f"{greeting}\n\n{answer.strip()}\n\n{signature}"

    def _holding_reply(self, sender: str) -> str:
        return self._wrap(
            sender,
            "Merci pour votre message. Je n'ai pas, à ce stade, tous les éléments pour vous répondre "
            "précisément ; je reviens vers vous très vite.",
        )

    # pipeline ----------------------------------------------------------------
    def draft(self, email: IncomingEmail) -> Draft:
        import time

        t0 = time.perf_counter()
        question = clean_question(email.subject, email.body)
        hits: list[Hit] = self.retriever.search(question, self.top_k)
        out = Draft(
            sender=email.sender,
            subject=email.subject,
            question=question,
            body="",
            retrieved=[h.chunk.id for h in hits],
            confidence=max((h.coverage for h in hits), default=0.0),
            model=self.llm.name,
            message_id=email.message_id,
        )

        def abstain(reason: str) -> Draft:
            out.abstained, out.reason = True, reason
            out.body = self._holding_reply(email.sender)
            out.citations = []
            out.latency_s = time.perf_counter() - t0
            return out

        if not hits:
            return abstain("aucun passage pertinent")
        if out.confidence < self.min_coverage:
            return abstain(f"couverture {out.confidence:.2f} < seuil {self.min_coverage:.2f}")

        chunks = [h.chunk for h in hits]
        reply = self.llm.answer(question, chunks)
        if reply.error:
            return abstain(reply.error)
        if not reply.can_answer:
            return abstain("le modèle estime ne pas pouvoir répondre d'après les documents")

        by_id = {c.id: c for c in chunks}
        valid = [c for c in dict.fromkeys(reply.citations) if c in by_id]
        if not valid:  # garde-fou : pas de réponse sans source vérifiable
            return abstain("réponse sans citation valide")
        if len(valid) < len(set(reply.citations)):
            out.warnings.append("citation(s) inconnue(s) retirée(s)")

        missing = ungrounded_facts(reply.answer, [by_id[c] for c in valid])
        if missing:
            out.warnings.append("valeurs absentes des sources citées : " + ", ".join(missing))

        out.citations = valid
        out.body = self._wrap(email.sender, reply.answer)
        out.latency_s = time.perf_counter() - t0
        return out
