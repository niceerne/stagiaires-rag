import json
from pathlib import Path

import pytest

from stagiaires_rag.drafter import Draft, Drafter
from stagiaires_rag.evaluate import (
    best_threshold,
    build_report,
    fact_present,
    load_testset,
    wilson,
)
from stagiaires_rag.ingest import load_chunks
from stagiaires_rag.llm import ExtractiveLLM
from stagiaires_rag.mailer import DryRunMailer
from stagiaires_rag.retriever import BM25Retriever
from stagiaires_rag.review import (
    InvalidTransition,
    NotApprovedError,
    ReviewError,
    ReviewQueue,
    edit_ratio,
)
from stagiaires_rag.text import normalize_for_match

ROOT = Path(__file__).resolve().parents[1]


def make_draft(**kw) -> Draft:
    base = dict(
        sender="Sara <sara@x.ma>", subject="Gratification", question="q", body="Bonjour Sara,\n\n2 500 MAD.\n\nCordialement,",
        citations=["04#0"], retrieved=["04#0"], confidence=0.9, message_id="<m1@x>",
    )
    base.update(kw)
    return Draft(**base)


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(tmp_path / "t.db")


# ------------------------------------------------------------------ validation humaine


def test_nothing_is_sent_without_approval(queue, tmp_path):
    mailer = DryRunMailer(tmp_path / "out")
    i = queue.add(make_draft())
    with pytest.raises(NotApprovedError):
        queue.send(i, mailer)
    assert not (tmp_path / "out").exists()
    queue.reject(i, "M. Tazi", "hors sujet")
    with pytest.raises(NotApprovedError):
        queue.send(i, mailer)


def test_approve_then_send_writes_reply(queue, tmp_path):
    mailer = DryRunMailer(tmp_path / "out")
    i = queue.add(make_draft())
    queue.approve(i, "M. Tazi")
    ref = queue.send(i, mailer)
    sent = json.loads(next((tmp_path / "out").glob("*.json")).read_text())
    assert sent["to"] == "Sara <sara@x.ma>" and sent["subject"] == "Re: Gratification"
    assert sent["in_reply_to"] == "<m1@x>" and ref.startswith("dryrun-")
    assert queue.get(i)["status"] == "sent"
    with pytest.raises(NotApprovedError):  # pas de double envoi
        queue.send(i, mailer)


def test_edit_is_recorded_and_sent_text_is_the_edited_one(queue, tmp_path):
    i = queue.add(make_draft())
    queue.approve(i, "M. Tazi", body="Bonjour Sara,\n\nLa gratification est de 2 500 MAD, versée le 28.\n\nCordialement,")
    row = queue.get(i)
    assert row["edit_ratio"] > 0 and "versée le 28" in row["body_final"]
    assert row["body_original"] != row["body_final"]


def test_text_changed_after_approval_is_refused(queue, tmp_path):
    i = queue.add(make_draft())
    queue.approve(i, "M. Tazi")
    queue.conn.execute("UPDATE drafts SET body_final='Texte falsifié' WHERE id=?", (i,))
    queue.conn.commit()
    with pytest.raises(NotApprovedError):
        queue.send(i, DryRunMailer(tmp_path / "out"))


def test_decisions_are_final_and_need_a_reviewer(queue):
    i = queue.add(make_draft())
    with pytest.raises(ReviewError):
        queue.approve(i, "  ")
    queue.approve(i, "M. Tazi")
    with pytest.raises(InvalidTransition):
        queue.reject(i, "M. Tazi")
    with pytest.raises(ReviewError):
        queue.get(999)


def test_stats(queue):
    a = queue.add(make_draft())
    b = queue.add(make_draft())
    c = queue.add(make_draft(abstained=True))
    queue.add(make_draft())
    queue.approve(a, "r")
    queue.approve(b, "r", body="tout changé ici, vraiment")
    queue.reject(c, "r")
    s = queue.stats()
    assert (s.total, s.pending, s.approved, s.rejected) == (4, 1, 2, 1)
    assert s.approval_rate == round(2 / 3, 4) and s.unedited_rate == 0.5
    assert 0 < s.mean_edit_ratio < 1 and s.abstained_rate == 0.25


def test_edit_ratio_bounds():
    assert edit_ratio("abc", "abc") == 0 and edit_ratio("abc", "xyz") == 1


# ------------------------------------------------------------------ évaluation


def test_wilson_interval():
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    assert wilson(0, 0) == (0.0, 0.0)


def test_fact_matching_is_accent_and_space_insensitive():
    assert fact_present(["2500"], "Elle est de 2 500 MAD")
    assert fact_present(["justificatif médical"], "Un justificatif medical est requis")
    assert not fact_present(["48 heures"], "sous 72 heures")


def test_best_threshold_prefers_lowest_on_tie():
    rows = [{"threshold": t, "acceptable_rate": r} for t, r in [(0.1, 0.5), (0.2, 0.7), (0.3, 0.7)]]
    assert best_threshold(rows) == 0.2


def test_testset_is_consistent_with_documents():
    cases = load_testset(ROOT / "data" / "eval" / "testset.jsonl")
    docs = {p.name: normalize_for_match(p.read_text(encoding="utf-8")) for p in (ROOT / "data" / "docs").glob("*.md")}
    assert {c.split for c in cases} == {"dev", "test"}
    assert len({c.id for c in cases}) == len(cases)
    for c in cases:
        if not c.answerable:
            assert not c.facts and not c.expected_sources
            continue
        assert c.expected_sources and set(c.expected_sources) <= set(docs), c.id
        for alts in c.facts:  # chaque fait attendu existe bien dans une source attendue
            assert any(normalize_for_match(a) in docs[s] for s in c.expected_sources for a in alts), (c.id, alts)


def test_offline_baseline_regression():
    """Garde-fou : la version hors-ligne ne doit pas régresser, ni répondre à tort."""
    drafter = Drafter(BM25Retriever(load_chunks(ROOT / "data" / "docs")), ExtractiveLLM())
    report = build_report(drafter, load_testset(ROOT / "data" / "eval" / "testset.jsonl"))
    test = report["splits"]["test"]
    assert test["retrieval"]["hit@4"]["value"] >= 0.9
    # Limite connue de la base lexicale : « mot de passe du wifi invité » (u12) reçoit la règle des mots de
    # passe des comptes. On borne le taux de réponses à tort au lieu de l'ignorer.
    assert test["abstention"]["unsafe_answer_rate"]["value"] <= 0.2
    assert test["acceptable_draft_rate"]["value"] >= 0.55
    assert test["faithfulness"]["injection_resisted"]["value"] == 1.0
