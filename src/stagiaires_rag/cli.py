"""Ligne de commande : drafts -> validation -> envoi, et évaluation."""

from __future__ import annotations

import argparse
import email
import email.policy
import json
import os
import sys
import time
from pathlib import Path

from .drafter import Drafter, IncomingEmail
from .evaluate import build_report, load_testset, render_markdown
from .ingest import load_chunks
from .embeddings import HybridRetriever, OllamaEmbedder
from .llm import AnthropicLLM, CachedLLM, ExtractiveLLM, OllamaLLM, OpenAICompatLLM
from .mailer import DryRunMailer, SMTPMailer
from .review import ReviewError, ReviewQueue

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = os.environ.get("RAG_DOCS", str(ROOT / "data" / "docs"))
DEFAULT_DB = os.environ.get("RAG_DB", str(ROOT / "var" / "rag.db"))
DEFAULT_TESTSET = str(ROOT / "data" / "eval" / "testset.jsonl")
DEFAULT_MIN_COVERAGE = float(os.environ.get("RAG_MIN_COVERAGE", "0.3"))


def build_drafter(args) -> Drafter:
    from .retriever import BM25Retriever

    llm = {"anthropic": AnthropicLLM, "openai": OpenAICompatLLM, "ollama": OllamaLLM}.get(args.llm, ExtractiveLLM)()
    if args.cache:
        llm = CachedLLM(llm, Path(args.db).parent / "llm_cache.json")
    retriever = BM25Retriever(load_chunks(args.docs))
    if args.retriever == "hybrid":
        embedder = OllamaEmbedder(model=args.embed_model)
        retriever = HybridRetriever(retriever, embedder, Path(args.db).parent / f"embeddings-{embedder.model}.json")
    return Drafter(
        retriever,
        llm,
        top_k=args.top_k,
        min_coverage=args.min_coverage,
        supervisor=os.environ.get("RAG_SUPERVISOR", ""),
    )


def read_email_file(path: str) -> IncomingEmail:
    raw = Path(path).read_bytes()
    if path.endswith(".json"):
        d = json.loads(raw)
        return IncomingEmail(d["sender"], d["subject"], d["body"], d.get("message_id", ""))
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    part = msg.get_body(preferencelist=("plain",))
    return IncomingEmail(
        sender=str(msg["From"] or ""),
        subject=str(msg["Subject"] or ""),
        body=part.get_content() if part else "",
        message_id=str(msg["Message-ID"] or ""),
    )


def show(row) -> None:
    print(f"── Brouillon #{row['id']} [{row['status']}] ─────────────────────────")
    print(f"De : {row['sender']}\nObjet : {row['subject']}")
    flag = "OUI (réponse d'attente)" if row["abstained"] else "non"
    print(f"Abstention : {flag}" + (f" — {row['reason']}" if row["reason"] else ""))
    print(f"Confiance : {row['confidence']:.2f}   Modèle : {row['model']}")
    print(f"Sources citées : {', '.join(json.loads(row['citations'])) or '—'}")
    for w in json.loads(row["warnings"]):
        print(f"⚠ {w}")
    print("\nQuestion reçue :\n" + "\n".join("  " + ln for ln in row["question"].splitlines()))
    print("\nBrouillon :\n" + "\n".join("  " + ln for ln in row["body_final"].splitlines()))
    print()


def cmd_draft(args) -> int:
    if args.file:
        mail = read_email_file(args.file)
    else:
        if not (args.sender and args.subject and args.body):
            print("Fournir --file, ou --from, --subject et --body", file=sys.stderr)
            return 2
        mail = IncomingEmail(args.sender, args.subject, args.body)
    draft = build_drafter(args).draft(mail)
    q = ReviewQueue(args.db)
    draft_id = q.add(draft)
    show(q.get(draft_id))
    print(f"→ en attente de validation : `review approve {draft_id} --reviewer <nom>`")
    return 0


def cmd_queue(args) -> int:
    q = ReviewQueue(args.db)
    rows = q.list(args.status)
    if not rows:
        print("File vide.")
    for r in rows:
        tag = " (attente)" if r["abstained"] else ""
        print(f"#{r['id']:<4} {r['status']:<9} {r['sender'][:28]:<28} {r['subject'][:40]}{tag}")
    return 0


def cmd_show(args) -> int:
    show(ReviewQueue(args.db).get(args.id))
    return 0


def cmd_approve(args) -> int:
    q = ReviewQueue(args.db)
    body = Path(args.edit_file).read_text(encoding="utf-8") if args.edit_file else None
    q.approve(args.id, args.reviewer, body=body, note=args.note or "")
    row = q.get(args.id)
    print(f"#{args.id} approuvé par {args.reviewer} (taux de modification : {row['edit_ratio']:.0%}). Pas encore envoyé.")
    return 0


def cmd_reject(args) -> int:
    ReviewQueue(args.db).reject(args.id, args.reviewer, args.reason or "")
    print(f"#{args.id} rejeté.")
    return 0


def cmd_send(args) -> int:
    mailer = SMTPMailer() if args.smtp else DryRunMailer(Path(args.db).parent / "outbox")
    ref = ReviewQueue(args.db).send(args.id, mailer)
    mode = "envoyé (SMTP)" if args.smtp else "écrit dans var/outbox (mode à blanc, rien n'est parti)"
    print(f"#{args.id} {mode} — réf. {ref}")
    return 0


def cmd_doctor(args) -> int:
    ok, msg = OllamaLLM().check()
    print(msg)
    if args.retriever == "hybrid":
        ok2, msg2 = OllamaLLM(model=args.embed_model).check()
        print(msg2)
        ok = ok and ok2
    return 0 if ok else 1


def cmd_web(args) -> int:
    from .webui import make_server

    mailer = SMTPMailer() if args.smtp else DryRunMailer(Path(args.db).parent / "outbox")
    server = make_server(build_drafter(args), args.db, mailer, port=args.port,
                         reviewer_default=os.environ.get("RAG_SUPERVISOR", ""))
    mode = "SMTP (envoi réel)" if args.smtp else "à blanc (rien n'est envoyé)"
    print(f"Interface : http://127.0.0.1:{args.port}   — envoi {mode}   (Ctrl+C pour arrêter)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_stats(args) -> int:
    print(json.dumps(ReviewQueue(args.db).stats().__dict__, indent=2, ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    drafter = build_drafter(args)
    report = build_report(drafter, load_testset(args.testset))
    md = render_markdown(report)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.llm}-{args.retriever}"
    (out / f"eval-{tag}-{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"eval-{tag}-{stamp}.md").write_text(md, encoding="utf-8")
    (out / f"latest-{tag}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stagiaires-rag", description=__doc__)
    p.add_argument("--db", default=DEFAULT_DB, help="base SQLite de la file de validation")
    p.add_argument("--docs", default=DEFAULT_DOCS, help="dossier des documents du stage")
    p.add_argument("--llm", choices=["extractive", "anthropic", "openai", "ollama"], default="extractive")
    p.add_argument("--retriever", choices=["bm25", "hybrid"], default="bm25",
                   help="bm25 (mots-clés) ou hybrid (BM25 + embeddings Ollama)")
    p.add_argument("--embed-model", default=os.environ.get("RAG_EMBED_MODEL", "bge-m3"))
    p.add_argument("--cache", action="store_true", help="mémorise les réponses du modèle sur disque (var/llm_cache.json)")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE, help="seuil d'abstention (0-1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft", help="rédige un brouillon pour un e-mail et le met en attente de validation")
    d.add_argument("--file", help="e-mail .eml ou .json {sender, subject, body}")
    d.add_argument("--from", dest="sender")
    d.add_argument("--subject")
    d.add_argument("--body")
    d.set_defaults(fn=cmd_draft)

    q = sub.add_parser("queue", help="liste les brouillons")
    q.add_argument("--status", choices=["pending", "approved", "rejected", "sent"])
    q.set_defaults(fn=cmd_queue)

    s = sub.add_parser("show", help="affiche un brouillon")
    s.add_argument("id", type=int)
    s.set_defaults(fn=cmd_show)

    a = sub.add_parser("approve", help="approuve un brouillon (éventuellement corrigé)")
    a.add_argument("id", type=int)
    a.add_argument("--reviewer", required=True)
    a.add_argument("--edit-file", help="fichier texte contenant la version corrigée")
    a.add_argument("--note")
    a.set_defaults(fn=cmd_approve)

    r = sub.add_parser("reject", help="rejette un brouillon")
    r.add_argument("id", type=int)
    r.add_argument("--reviewer", required=True)
    r.add_argument("--reason")
    r.set_defaults(fn=cmd_reject)

    se = sub.add_parser("send", help="envoie un brouillon APPROUVÉ (à blanc par défaut)")
    se.add_argument("id", type=int)
    se.add_argument("--smtp", action="store_true", help="envoi réel via SMTP_* (sinon écrit dans var/outbox)")
    se.set_defaults(fn=cmd_send)

    w = sub.add_parser("web", help="interface web locale pour l'encadrant")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--smtp", action="store_true", help="envoi réel via SMTP_* (sinon à blanc)")
    w.set_defaults(fn=cmd_web)

    dc = sub.add_parser("doctor", help="vérifie qu'Ollama tourne et que le modèle (RAG_MODEL) est installé")
    dc.set_defaults(fn=cmd_doctor)

    st = sub.add_parser("stats", help="indicateurs de la validation humaine")
    st.set_defaults(fn=cmd_stats)

    e = sub.add_parser("eval", help="évaluation chiffrée sur le jeu de test")
    e.add_argument("--testset", default=DEFAULT_TESTSET)
    e.add_argument("--out", default=str(ROOT / "reports"))
    e.set_defaults(fn=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except ReviewError as exc:
        print(f"Refusé : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
