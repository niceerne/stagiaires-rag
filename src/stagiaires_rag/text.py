"""Normalisation de texte français : sans accents, mots vides, racinisation légère."""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS_RAW = """
au aux avec ce ces dans de des du elle en et eux il ils je la le les leur leurs lui ma mais me meme
mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un
une vos votre vous c d j l m n s t y ete etre est sont suis ai as a avez avons ont peut peux puis
bonjour bonsoir merci cordialement salut svp please comment quand quel quelle quels quelles dois doit
faut faire fait aussi plus tres si ca cela cet cette ceci voici voila alors donc car comme ainsi
madame monsieur mr mme souhaite voudrais aimerais savoir demande question information informations
est-ce serait pourrait pouvez pourriez dit dire there the is are what how
combien possible avance aujourd hui bref
"""


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


STOPWORDS: frozenset[str] = frozenset(strip_accents(w) for w in _STOPWORDS_RAW.split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Racinisation très légère (pluriels), volontairement conservatrice."""
    if token.isdigit():
        return token
    if len(token) > 4 and token.endswith("aux"):
        return token[:-3] + "al"
    if len(token) > 3 and token.endswith(("s", "x")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Texte -> liste de termes normalisés (sans accents, sans mots vides, racinisés)."""
    text = strip_accents(text.lower()).replace(" ", " ").replace("\xa0", " ")
    # "2 500" -> "2500" pour que les montants soient un seul terme
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)
    tokens = []
    for tok in _TOKEN_RE.findall(text):
        if tok in STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        tokens.append(stem(tok))
    return tokens


def normalize_for_match(text: str) -> str:
    """Forme normalisée pour tester la présence d'un fait dans un texte."""
    text = strip_accents(text.lower()).replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)
    text = re.sub(r"\s*%", " %", text)  # "50%" == "50 %"
    return re.sub(r"\s+", " ", text).strip()
