"""Interface web locale pour l'encadrant (bibliothèque standard uniquement).

Écoute sur 127.0.0.1. Le contenu des e-mails est non fiable : tout est échappé (anti-XSS), les formulaires
portent un jeton CSRF, une politique CSP interdit tout script, et l'en-tête Host est vérifié.
"""

from __future__ import annotations

import html
import json
import re
import secrets
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .drafter import Drafter, IncomingEmail
from .mailer import Mailer
from .review import ReviewError, ReviewQueue

MAX_FORM_BYTES = 200_000
e = html.escape

CSS = """
:root{--bg:#f6f7f9;--fg:#1c2330;--mut:#667085;--card:#fff;--bd:#d9dee7;--ok:#0b7a43;--warn:#b54708;--bad:#b42318;--acc:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#10141c;--fg:#e6e9ef;--mut:#98a2b3;--card:#171d29;--bd:#2a3344;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:1.4rem}h2{font-size:1.05rem;margin:1.2rem 0 .4rem}
a{color:var(--acc)}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--bd);border-radius:8px}
th,td{padding:8px 10px;border-bottom:1px solid var(--bd);text-align:left;font-size:.92rem}.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin:8px 0}
pre{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit}textarea,input{width:100%;padding:8px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
textarea{min-height:220px}button{padding:8px 14px;border:0;border-radius:6px;font:inherit;cursor:pointer;color:#fff;background:var(--acc)}
button.ok{background:var(--ok)}button.bad{background:var(--bad)}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:end}.row>*{flex:1 1 180px}
.badge{padding:2px 8px;border-radius:99px;font-size:.78rem;border:1px solid var(--bd)}.mut{color:var(--mut)}.warn{color:var(--warn)}small{color:var(--mut)}
details{margin:6px 0}summary{cursor:pointer}
"""

STATUS_FR = {"pending": "en attente", "approved": "approuvé", "rejected": "rejeté", "sent": "envoyé"}


def page(title: str, body: str) -> bytes:
    return (
        f"<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{e(title)}</title>"
        f"<style>{CSS}</style></head><body><main>{body}</main></body></html>"
    ).encode("utf-8")


def make_server(
    drafter: Drafter, db_path: str, mailer: Mailer, host: str = "127.0.0.1", port: int = 8000,
    reviewer_default: str = "",
) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(24)
    chunk_by_id = {c.id: c for c in drafter.retriever.chunks}

    def hidden() -> str:
        return f"<input type='hidden' name='token' value='{e(token)}'>"

    class Handler(BaseHTTPRequestHandler):
        server_version = "stagiaires-rag"

        # ------------------------------------------------------------- utilitaires
        def log_message(self, *args):  # silence
            pass

        def _send(self, status: int, body: bytes, location: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str) -> None:
            self._send(303, b"", location)

        def _error(self, status: int, msg: str) -> None:
            self._send(status, page("Erreur", f"<h1>Erreur</h1><p>{e(msg)}</p><p><a href='/'>Retour</a></p>"))

        def _host_ok(self) -> bool:
            h = (self.headers.get("Host") or "").lower()
            return h in {f"127.0.0.1:{self.server.server_address[1]}", f"localhost:{self.server.server_address[1]}"}

        def _form(self) -> dict[str, str] | None:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if n > MAX_FORM_BYTES:
                return None
            data = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"), keep_blank_values=True)
            return {k: v[0] for k, v in data.items()}

        # ------------------------------------------------------------- pages
        def _index(self) -> bytes:
            q = ReviewQueue(db_path)
            try:
                rows = q.list()
                s = q.stats()
            finally:
                q.close()
            order = {"pending": 0, "approved": 1, "sent": 2, "rejected": 3}
            rows = sorted(rows, key=lambda r: (order.get(r["status"], 9), -r["id"]))
            lines = "".join(
                f"<tr><td><a href='/draft/{r['id']}'>#{r['id']}</a></td>"
                f"<td><span class='badge'>{STATUS_FR.get(r['status'], r['status'])}</span></td>"
                f"<td>{e(r['sender'])}</td><td>{e(r['subject'])}</td>"
                f"<td>{'attente' if r['abstained'] else ''}{' ⚠' if json.loads(r['warnings']) else ''}</td></tr>"
                for r in rows
            ) or "<tr><td colspan='5' class='mut'>Aucun brouillon.</td></tr>"
            pct = lambda x: "—" if x is None else f"{x:.0%}"  # noqa: E731
            return page(
                "Brouillons",
                "<h1>Brouillons de réponse</h1>"
                f"<p><a href='/new'>+ Nouvel e-mail</a></p>"
                f"<p class='mut'>{s.pending} en attente · approbation {pct(s.approval_rate)} · "
                f"envoyés sans modification {pct(s.unedited_rate)} · abstentions {pct(s.abstained_rate)}</p>"
                f"<table><tr><th>#</th><th>Statut</th><th>De</th><th>Objet</th><th></th></tr>{lines}</table>",
            )

        def _new(self) -> bytes:
            return page(
                "Nouvel e-mail",
                "<h1>Nouvel e-mail de stagiaire</h1><form method='post' action='/new'>" + hidden()
                + "<p><label>De<input name='sender' placeholder='Sara Benali &lt;sara@etu.example&gt;' required></label></p>"
                "<p><label>Objet<input name='subject' required></label></p>"
                "<p><label>Message<textarea name='body' required></textarea></label></p>"
                "<button>Rédiger le brouillon</button> <small>Peut prendre quelques secondes avec un modèle local.</small>"
                "</form><p><a href='/'>Retour</a></p>",
            )

        def _draft(self, draft_id: int) -> bytes:
            q = ReviewQueue(db_path)
            try:
                r = q.get(draft_id)
            finally:
                q.close()
            cits = json.loads(r["citations"])
            warns = json.loads(r["warnings"])
            sources = "".join(
                f"<details><summary>{e(chunk_by_id[c].source)} — {e(chunk_by_id[c].heading)} <small>({e(c)})</small></summary>"
                f"<pre>{e(chunk_by_id[c].text)}</pre></details>"
                for c in cits if c in chunk_by_id
            ) or "<p class='mut'>Aucune source citée.</p>"
            head = (
                f"<h1>Brouillon #{r['id']} <span class='badge'>{STATUS_FR.get(r['status'], r['status'])}</span></h1>"
                f"<p class='mut'>De {e(r['sender'])} · Objet : {e(r['subject'])} · confiance {r['confidence']:.2f} · {e(r['model'])}</p>"
            )
            if r["abstained"]:
                head += f"<p class='warn'>Réponse d'attente : {e(r['reason'])}. Complétez-la avant d'approuver.</p>"
            for w in warns:
                head += f"<p class='warn'>⚠ {e(w)}</p>"
            body = head + f"<h2>Question reçue</h2><div class='card'><pre>{e(r['question'])}</pre></div>"
            body += f"<h2>Sources</h2>{sources}"
            status = r["status"]
            if status == "pending":
                body += (
                    f"<h2>Brouillon (modifiable)</h2><form method='post' action='/draft/{r['id']}/approve'>{hidden()}"
                    f"<textarea name='body'>{e(r['body_final'])}</textarea>"
                    f"<div class='row'><label>Votre nom<input name='reviewer' value='{e(reviewer_default)}' required></label>"
                    f"<button class='ok'>Approuver ce texte</button></div></form>"
                    f"<h2>Rejeter</h2><form method='post' action='/draft/{r['id']}/reject'>{hidden()}"
                    f"<div class='row'><label>Votre nom<input name='reviewer' value='{e(reviewer_default)}' required></label>"
                    f"<label>Motif<input name='reason'></label><button class='bad'>Rejeter</button></div></form>"
                )
            else:
                body += f"<h2>Texte {STATUS_FR.get(status, status)}</h2><div class='card'><pre>{e(r['body_final'])}</pre></div>"
                if r["reviewer"]:
                    ratio = "" if r["edit_ratio"] is None else f" · modification {r['edit_ratio']:.0%}"
                    body += f"<p class='mut'>Par {e(r['reviewer'])} le {e(r['reviewed_at'] or '')}{ratio}</p>"
                if status == "approved":
                    body += (
                        f"<form method='post' action='/draft/{r['id']}/send'>{hidden()}"
                        f"<button>Envoyer (selon le mode du serveur)</button></form>"
                    )
            return page(f"Brouillon {r['id']}", body + "<p><a href='/'>Retour</a></p>")

        # ------------------------------------------------------------- routage
        def do_GET(self):
            if not self._host_ok():
                return self._error(403, "Hôte non autorisé")
            path = urllib.parse.urlparse(self.path).path
            try:
                if path == "/":
                    return self._send(200, self._index())
                if path == "/new":
                    return self._send(200, self._new())
                m = re.fullmatch(r"/draft/(\d+)", path)
                if m:
                    return self._send(200, self._draft(int(m.group(1))))
            except ReviewError as exc:
                return self._error(404, str(exc))
            self._error(404, "Page introuvable")

        def do_POST(self):
            if not self._host_ok():
                return self._error(403, "Hôte non autorisé")
            form = self._form()
            if form is None:
                return self._error(400, "Formulaire invalide ou trop gros")
            if not secrets.compare_digest(form.get("token", ""), token):
                return self._error(403, "Jeton CSRF invalide : rechargez la page")
            path = urllib.parse.urlparse(self.path).path
            q = ReviewQueue(db_path)
            try:
                if path == "/new":
                    sender, subject, body = (form.get(k, "").strip() for k in ("sender", "subject", "body"))
                    if not (sender and subject and body):
                        return self._error(400, "Tous les champs sont obligatoires")
                    draft_id = q.add(drafter.draft(IncomingEmail(sender, subject, body)))
                    return self._redirect(f"/draft/{draft_id}")
                m = re.fullmatch(r"/draft/(\d+)/(approve|reject|send)", path)
                if not m:
                    return self._error(404, "Action inconnue")
                draft_id, action = int(m.group(1)), m.group(2)
                if action == "approve":
                    q.approve(draft_id, form.get("reviewer", ""), body=form.get("body", "").replace("\r\n", "\n"))
                elif action == "reject":
                    q.reject(draft_id, form.get("reviewer", ""), form.get("reason", ""))
                else:
                    q.send(draft_id, mailer)
                return self._redirect(f"/draft/{draft_id}")
            except ReviewError as exc:
                return self._error(400, str(exc))
            finally:
                q.close()

    return ThreadingHTTPServer((host, port), Handler)
