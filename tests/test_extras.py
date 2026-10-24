import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from stagiaires_rag.drafter import Drafter
from stagiaires_rag.embeddings import HybridRetriever, OllamaEmbedder
from stagiaires_rag.evaluate import apply_gate, fact_present, load_testset, run_cases
from stagiaires_rag.ingest import Chunk, load_chunks
from stagiaires_rag.llm import CachedLLM, ExtractiveLLM, LLMReply
from stagiaires_rag.mailer import DryRunMailer
from stagiaires_rag.retriever import BM25Retriever
from stagiaires_rag.text import tokenize
from stagiaires_rag.webui import make_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bm25():
    return BM25Retriever(load_chunks(ROOT / "data" / "docs"))


# ------------------------------------------------------------------ évaluation / cache


def test_percent_spacing_does_not_matter():
    assert fact_present(["50 %"], "La participation est de 50% du prix")


def test_apply_gate_equals_running_with_that_threshold(bm25):
    cases = load_testset(ROOT / "data" / "eval" / "testset.jsonl")
    d = Drafter(bm25, ExtractiveLLM(), min_coverage=0.0)
    zero = run_cases(d, cases)
    d.min_coverage = 0.3
    direct = run_cases(d, cases)
    for a, b in zip((apply_gate(r, 0.3) for r in zero), direct):
        assert (a.answered, a.facts_found, a.acceptable) == (b.answered, b.facts_found, b.acceptable), a.case.id


def test_cached_llm_calls_inner_once_and_skips_errors(tmp_path):
    class Inner:
        name = "inner"
        calls = 0

        def __init__(self, reply):
            self.reply = reply

        def answer(self, q, chunks):
            Inner.calls += 1
            return self.reply

    c = [Chunk("d#0", "d", "t", "h", "x")]
    cached = CachedLLM(Inner(LLMReply(True, "ok", ["d#0"])), tmp_path / "c.json")
    assert cached.answer("q", c).answer == "ok" and cached.answer("q", c).answer == "ok"
    assert Inner.calls == 1 and cached.hits == 1
    # persistance entre deux exécutions
    again = CachedLLM(Inner(LLMReply(False)), tmp_path / "c.json")
    assert again.answer("q", c).answer == "ok" and Inner.calls == 1
    # une erreur n'est pas mise en cache
    bad = CachedLLM(Inner(LLMReply(False, error="x")), tmp_path / "e.json")
    bad.answer("q", c)
    bad.answer("q", c)
    assert Inner.calls == 3


# ------------------------------------------------------------------ recherche hybride


class FakeEmbedder:
    """Embeddings factices : sac de mots hachés + un « synonyme » bus ~ transport."""

    model = "fake"
    calls = 0

    def embed(self, texts):
        FakeEmbedder.calls += 1
        out = []
        for t in texts:
            v = [0.0] * 64
            for tok in tokenize(t.replace("bus", "transport")):
                v[hash(tok) % 64] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out


def test_hybrid_retrieval_finds_synonym_and_caches_embeddings(bm25, tmp_path):
    FakeEmbedder.calls = 0
    h = HybridRetriever(bm25, FakeEmbedder(), tmp_path / "emb.json")
    hits = h.search("Je prends le bus, êtes-vous remboursé ?", 3)
    assert "04-gratification-et-remboursements.md" in [x.chunk.source for x in hits]
    calls_after_build = FakeEmbedder.calls
    HybridRetriever(bm25, FakeEmbedder(), tmp_path / "emb.json")  # cache : aucun nouvel appel pour les passages
    assert FakeEmbedder.calls == calls_after_build


def test_hybrid_falls_back_to_bm25_when_embedding_fails(bm25, tmp_path):
    h = HybridRetriever(bm25, FakeEmbedder(), tmp_path / "emb.json")

    def boom(texts):
        raise ConnectionRefusedError("Ollama éteint")

    h.embedder = type("E", (), {"model": "fake", "embed": staticmethod(boom)})()
    hits = h.search("gratification versée par virement", 3)
    assert hits and hits[0].chunk.source.startswith("04") and h.fallbacks == 1


def test_ollama_embedder_posts_to_api_embed():
    seen = {}

    def post(url, payload):
        seen.update(url=url, payload=payload)
        return {"embeddings": [[3.0, 4.0]]}

    emb = OllamaEmbedder(host="localhost:11434", model="bge-m3", post=post)
    assert emb.embed(["a"]) == [[0.6, 0.8]]
    assert seen["url"] == "http://localhost:11434/api/embed" and seen["payload"]["input"] == ["a"]


# ------------------------------------------------------------------ interface web


@pytest.fixture
def web(bm25, tmp_path):
    server = make_server(Drafter(bm25, ExtractiveLLM(), supervisor="M. Tazi"), str(tmp_path / "w.db"),
                         DryRunMailer(tmp_path / "out"), port=0, reviewer_default="M. Tazi")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def _post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode())
    with urllib.request.urlopen(req) as r:  # suit la redirection 303
        return r.url, r.read().decode()


def _token(base):
    return re.search(r"name='token' value='([^']+)'", _get(base + "/new")).group(1)


def test_web_full_flow_and_dry_run_send(web):
    base, tmp = web
    tok = _token(base)
    url, html = _post(base + "/new", {"token": tok, "sender": "Sara Benali <sara@x.ma>", "subject": "Congés",
                                      "body": "Combien de temps à l'avance dois-je demander mes congés ?"})
    assert "/draft/1" in url and "Brouillon #1" in html and "en attente" in html
    assert not (tmp / "out").exists()
    _, html = _post(url + "/approve", {"token": tok, "reviewer": "M. Tazi", "body": "Bonjour Sara,\r\n\r\n7 jours ouvrés.\r\n"})
    assert "approuvé" in html and "Envoyer" in html
    _, html = _post(url + "/send", {"token": tok})
    assert "envoyé" in html
    sent = json.loads(next((tmp / "out").glob("*.json")).read_text())
    assert sent["body"] == "Bonjour Sara,\n\n7 jours ouvrés.\n" and sent["subject"] == "Re: Congés"


def test_web_escapes_untrusted_email_content(web):
    base, _ = web
    tok = _token(base)
    _, html = _post(base + "/new", {"token": tok, "sender": "x <x@x.ma>", "subject": "<script>alert(1)</script>",
                                    "body": "<img src=x onerror=alert(1)> gratification ?"})
    assert "<script>alert" not in html and "<img src=x" not in html and "&lt;script&gt;" in html
    assert "<script>alert" not in _get(base + "/")


def test_web_rejects_missing_csrf_token_and_bad_host(web):
    base, _ = web
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base + "/new", {"sender": "a", "subject": "b", "body": "c"})
    assert exc.value.code == 403
    req = urllib.request.Request(base + "/", headers={"Host": "evil.example"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 403


def test_web_cannot_send_unapproved_draft(web):
    base, tmp = web
    tok = _token(base)
    url, _ = _post(base + "/new", {"token": tok, "sender": "a <a@x.ma>", "subject": "s", "body": "gratification ?"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(url + "/send", {"token": tok})
    assert exc.value.code == 400 and not (tmp / "out").exists()
