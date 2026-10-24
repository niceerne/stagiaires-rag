"""Évaluation chiffrée : retrieval, justesse des réponses, abstention, fidélité aux sources.

Le jeu de test (data/eval/testset.jsonl) est découpé en `dev` (pour régler le seuil d'abstention)
et `test` (pour rapporter les chiffres sans surapprentissage du réglage).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

from .drafter import Draft, Drafter, IncomingEmail
from .text import normalize_for_match


@dataclass
class Case:
    id: str
    split: str
    answerable: bool
    sender: str
    subject: str
    body: str
    expected_sources: list[str]
    facts: list[list[str]]  # chaque fait = liste d'alternatives acceptables
    must_not: list[str]  # chaînes qui ne doivent JAMAIS apparaître (test d'injection)


def load_testset(path: str | Path) -> list[Case]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(Case(**json.loads(line)))
    return cases


# --------------------------------------------------------------------------- statistiques


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalle de confiance de Wilson à 95 % pour une proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


# --------------------------------------------------------------------------- exécution


def fact_present(alternatives: list[str], text: str) -> bool:
    t = normalize_for_match(text)
    return any(normalize_for_match(a) in t for a in alternatives)


@dataclass
class CaseResult:
    case: Case
    draft: Draft
    first_hit_rank: int | None  # rang (1 = meilleur) du premier passage issu d'une source attendue
    facts_found: int
    citation_precision: float | None

    @property
    def answered(self) -> bool:
        return not self.draft.abstained

    @property
    def all_facts(self) -> bool:
        return self.facts_found == len(self.case.facts)

    @property
    def correct(self) -> bool:
        return self.answered and self.all_facts and not self.injected

    @property
    def injected(self) -> bool:
        body = normalize_for_match(self.draft.body)
        return any(normalize_for_match(m) in body for m in self.case.must_not)

    @property
    def acceptable(self) -> bool:
        """Brouillon acceptable : bonne réponse si la question est couverte, abstention sinon."""
        return self.correct if self.case.answerable else not self.answered


def run_cases(drafter: Drafter, cases: list[Case]) -> list[CaseResult]:
    results = []
    for c in cases:
        d = drafter.draft(IncomingEmail(c.sender, c.subject, c.body))
        rank = None
        for i, cid in enumerate(d.retrieved, 1):
            if cid.split("#")[0] in c.expected_sources:
                rank = i
                break
        found = sum(fact_present(alts, d.body) for alts in c.facts) if not d.abstained else 0
        prec = None
        if not d.abstained and d.citations:
            prec = sum(cid.split("#")[0] in c.expected_sources for cid in d.citations) / len(d.citations)
        results.append(CaseResult(c, d, rank, found, prec))
    return results


# --------------------------------------------------------------------------- synthèse


def summarize(results: list[CaseResult], k: int) -> dict:
    A = [r for r in results if r.case.answerable]
    U = [r for r in results if not r.case.answerable]
    answered_A = [r for r in A if r.answered]
    answered_all = [r for r in results if r.answered]
    inj = [r for r in results if r.case.must_not]
    tp = sum(not r.answered for r in U)  # abstention attendue et obtenue
    fp = sum(not r.answered for r in A)  # abstention à tort
    fn = sum(r.answered for r in U)  # réponse à tort
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    lat = [r.draft.latency_s * 1000 for r in results]
    return {
        "n_cases": len(results),
        "n_answerable": len(A),
        "n_unanswerable": len(U),
        "acceptable_draft_rate": _rate(sum(r.acceptable for r in results), len(results)),
        "retrieval": {
            f"hit@1": _rate(sum(r.first_hit_rank == 1 for r in A), len(A)),
            f"hit@{k}": _rate(sum(r.first_hit_rank is not None for r in A), len(A)),
            "mrr": round(sum(1 / r.first_hit_rank for r in A if r.first_hit_rank) / len(A), 4) if A else None,
        },
        "answers": {
            "correct_rate": _rate(sum(r.correct for r in A), len(A)),
            "fact_recall_mean": round(sum(r.facts_found / len(r.case.facts) for r in A) / len(A), 4) if A else None,
            "false_abstain_rate": _rate(sum(not r.answered for r in A), len(A)),
            "citation_precision_mean": round(
                statistics.mean(r.citation_precision for r in answered_A if r.citation_precision is not None), 4
            )
            if any(r.citation_precision is not None for r in answered_A)
            else None,
        },
        "abstention": {
            "correct_abstain_rate": _rate(tp, len(U)),
            "unsafe_answer_rate": _rate(fn, len(U)),
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        },
        "faithfulness": {
            "drafts_without_warning": _rate(sum(not r.draft.warnings for r in answered_all), len(answered_all)),
            "injection_resisted": _rate(sum(not r.injected for r in inj), len(inj)),
        },
        "latency_ms": {"p50": round(_percentile(lat, 0.5), 2), "p95": round(_percentile(lat, 0.95), 2)},
    }


def sweep(results_at_zero: list[CaseResult], thresholds: list[float]) -> list[dict]:
    """Simule le seuil d'abstention à partir de résultats obtenus avec un seuil nul."""
    rows = []
    for t in thresholds:
        ok_A = fa = ok_U = 0
        nA = nU = 0
        for r in results_at_zero:
            abst = (not r.answered) or r.draft.confidence < t
            if r.case.answerable:
                nA += 1
                fa += abst
                ok_A += (not abst) and r.all_facts and not r.injected
            else:
                nU += 1
                ok_U += abst
        n = nA + nU
        rows.append(
            {
                "threshold": t,
                "acceptable_rate": round((ok_A + ok_U) / n, 4) if n else None,
                "false_abstain_rate": round(fa / nA, 4) if nA else None,
                "unsafe_answer_rate": round((nU - ok_U) / nU, 4) if nU else None,
            }
        )
    return rows


def best_threshold(rows: list[dict]) -> float:
    """Seuil maximisant le taux de brouillons acceptables (en cas d'égalité : le plus bas)."""
    return max(rows, key=lambda r: (r["acceptable_rate"], -r["threshold"]))["threshold"]


def failures(results: list[CaseResult]) -> list[dict]:
    out = []
    for r in results:
        if not r.acceptable:
            kind = "réponse à tort" if not r.case.answerable else ("abstention à tort" if not r.answered else "réponse incomplète/incorrecte")
            out.append(
                {
                    "id": r.case.id,
                    "type": kind,
                    "question": r.case.body.split("\n\n")[1] if "\n\n" in r.case.body else r.case.body,
                    "confidence": round(r.draft.confidence, 2),
                    "retrieved": r.draft.retrieved[:3],
                    "facts_found": f"{r.facts_found}/{len(r.case.facts)}",
                }
            )
    return out


def apply_gate(r: CaseResult, threshold: float) -> CaseResult:
    """Résultat qu'on aurait obtenu avec ce seuil, à partir d'un passage sans seuil.

    Exact : la porte d'abstention agit avant l'appel au modèle, donc le reste du résultat est identique.
    """
    d = r.draft
    if d.abstained or d.confidence >= threshold:
        return r
    d2 = replace(
        d, abstained=True, citations=[], warnings=[], body="(réponse d'attente)",
        reason=f"couverture {d.confidence:.2f} < seuil {threshold:.2f}",
    )
    return CaseResult(r.case, d2, r.first_hit_rank, 0, None)


def build_report(drafter: Drafter, cases: list[Case], thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(1, 19)]
    # Un seul passage, sans porte d'abstention : le modèle est appelé une fois par e-mail.
    saved = drafter.min_coverage
    drafter.min_coverage = 0.0
    try:
        zero = run_cases(drafter, cases)
    finally:
        drafter.min_coverage = saved
    results = [apply_gate(r, saved) for r in zero]
    k = drafter.top_k
    report: dict = {
        "config": {
            "llm": drafter.llm.name,
            "retriever": getattr(drafter.retriever, "name", "bm25"),
            "top_k": k,
            "min_coverage": saved,
        },
        "splits": {},
    }
    for split in ("dev", "test", "all"):
        report["splits"][split] = summarize([r for r in results if split == "all" or r.case.split == split], k)
    report["threshold_sweep"] = {
        s: sweep([r for r in zero if r.case.split == s], thresholds) for s in ("dev", "test")
    }
    report["best_threshold_on_dev"] = best_threshold(report["threshold_sweep"]["dev"])
    report["failures"] = failures([r for r in results if r.case.split == "test"])
    return report


# --------------------------------------------------------------------------- rendu


def _fmt(rate: dict) -> str:
    if rate["value"] is None:
        return "n/a"
    lo, hi = rate["ci95"]
    return f"{rate['value']*100:.1f} % ({rate['k']}/{rate['n']}, IC95 {lo*100:.0f}–{hi*100:.0f} %)"


def render_markdown(report: dict) -> str:
    cfg = report["config"]
    lines = [
        "# Rapport d'évaluation",
        "",
        f"Modèle : `{cfg['llm']}` · recherche : `{cfg.get('retriever', 'bm25')}` · top-k = {cfg['top_k']} · seuil d'abstention = {cfg['min_coverage']}",
        "",
    ]
    for split in ("test", "dev"):
        s = report["splits"][split]
        r, a, ab, f, lat = s["retrieval"], s["answers"], s["abstention"], s["faithfulness"], s["latency_ms"]
        hit_k = [key for key in r if key.startswith("hit@") and key != "hit@1"][0]
        lines += [
            f"## Jeu `{split}` ({s['n_cases']} e-mails : {s['n_answerable']} avec réponse dans les documents, "
            f"{s['n_unanswerable']} sans)",
            "",
            "| Métrique | Valeur |",
            "|---|---|",
            f"| **Brouillons acceptables** (bonne réponse, ou abstention si hors documents) | **{_fmt(s['acceptable_draft_rate'])}** |",
            f"| Retrieval hit@1 | {_fmt(r['hit@1'])} |",
            f"| Retrieval {hit_k} | {_fmt(r[hit_k])} |",
            f"| Retrieval MRR | {r['mrr']} |",
            f"| Réponses correctes (tous les faits attendus) | {_fmt(a['correct_rate'])} |",
            f"| Rappel moyen des faits | {a['fact_recall_mean']} |",
            f"| Abstentions à tort | {_fmt(a['false_abstain_rate'])} |",
            f"| Précision des citations (moyenne) | {a['citation_precision_mean']} |",
            f"| Abstentions correctes (hors documents) | {_fmt(ab['correct_abstain_rate'])} |",
            f"| Réponses à tort (hors documents) | {_fmt(ab['unsafe_answer_rate'])} |",
            f"| Abstention : précision / rappel / F1 | {ab['precision']} / {ab['recall']} / {ab['f1']} |",
            f"| Brouillons sans alerte de fidélité | {_fmt(f['drafts_without_warning'])} |",
            f"| Résistance à l'injection de consigne | {_fmt(f['injection_resisted'])} |",
            f"| Latence p50 / p95 | {lat['p50']} ms / {lat['p95']} ms |",
            "",
        ]
    lines += ["## Seuil d'abstention (balayage)", "", f"Meilleur seuil sur `dev` : **{report['best_threshold_on_dev']}**", ""]
    lines += ["| Seuil | Acceptables dev | Acceptables test | Abst. à tort test | Réponses à tort test |", "|---|---|---|---|---|"]
    for d, t in zip(report["threshold_sweep"]["dev"], report["threshold_sweep"]["test"]):
        lines.append(
            f"| {d['threshold']} | {d['acceptable_rate']} | {t['acceptable_rate']} | "
            f"{t['false_abstain_rate']} | {t['unsafe_answer_rate']} |"
        )
    lines += ["", "## Échecs sur `test`", ""]
    if report["failures"]:
        lines += ["| id | type | confiance | faits | passages récupérés | question |", "|---|---|---|---|---|---|"]
        for f_ in report["failures"]:
            lines.append(
                f"| {f_['id']} | {f_['type']} | {f_['confidence']} | {f_['facts_found']} | "
                f"{', '.join(f_['retrieved'])} | {f_['question']} |"
            )
    else:
        lines.append("Aucun échec.")
    lines.append("")
    return "\n".join(lines)
