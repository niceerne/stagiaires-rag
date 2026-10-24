# Assistant RAG pour les e-mails des stagiaires

Un e-mail de stagiaire arrive → l'assistant cherche dans les documents du stage → il rédige un **brouillon** avec ses sources → **l'encadrant valide (ou corrige, ou rejette)** → seulement ensuite l'e-mail peut partir. Une évaluation chiffrée mesure la qualité.

```
e-mail ─► nettoyage ─► recherche (BM25 ou hybride BM25+embeddings) ─► porte d'abstention ─► LLM (JSON + citations)
                                                                           │                      │
                                                                           ▼                      ▼
                                                              réponse d'attente      garde-fous (citations valides,
                                                                           └─────────► valeurs vérifiées dans les sources)
                                                                                                  │
                                              file de validation SQLite (pending) ◄──────────────┘
                                                       │  approve / edit / reject  (CLI ou interface web)
                                                       ▼
                                        send : refusé si non approuvé ou texte modifié après approbation
```

## Démarrage

```bash
python3 -m pytest -q                                   # 42 tests, hors-ligne
export PYTHONPATH=src RAG_SUPERVISOR="M. Tazi"         # CMD : set PYTHONPATH=src   PowerShell : $env:PYTHONPATH="src"

python3 -m stagiaires_rag web                          # interface web : http://127.0.0.1:8000
# ou en ligne de commande :
python3 -m stagiaires_rag draft --from "Sara <sara@etu.example>" --subject "Congés" --body "Quel préavis pour poser des congés ?"
python3 -m stagiaires_rag queue | show 1 | approve 1 --reviewer "M. Tazi" [--edit-file t.txt] | reject 1 --reviewer ... | send 1
python3 -m stagiaires_rag stats                        # taux d'approbation, de modification, d'abstention
python3 -m stagiaires_rag eval                         # évaluation chiffrée -> reports/latest-<llm>-<recherche>.md
```

L'envoi est **à blanc par défaut** (le message est écrit dans `var/outbox`). Envoi réel : option `--smtp` + `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`.

**Vos vrais documents** : remplacez `data/docs/` (`.md`/`.txt`, un `# Titre` et des `## Sections`). Les documents fournis sont ceux d'une société fictive.

### Modèles de langage (options globales, avant la commande)

| Option | Usage |
|---|---|
| `--llm extractive` (défaut) | hors-ligne, sans modèle : sélection de phrases (base de comparaison) |
| `--llm ollama` | **gratuit, local** : `ollama pull qwen2.5:3b`, puis `RAG_MODEL=qwen2.5:3b` ; `doctor` vérifie l'installation |
| `--llm anthropic` | Claude (`pip install anthropic`, `ANTHROPIC_API_KEY`, payant) |
| `--llm openai` | serveurs au format OpenAI (Groq, Gemini, OpenRouter) via `RAG_BASE_URL`, `RAG_MODEL`, `RAG_API_KEY` |
| `--retriever hybrid` | BM25 + embeddings Ollama (`ollama pull bge-m3`), fusion RRF ; repli automatique sur BM25 si Ollama est éteint |
| `--cache` | mémorise les réponses du modèle (`var/llm_cache.json`) : relancer une évaluation est quasi instantané |
| `--top-k`, `--min-coverage` | nombre de passages, seuil d'abstention |

Exemple complet, 8 Go de RAM : `--llm ollama --retriever hybrid --cache --top-k 3 eval` avec `RAG_MODEL=qwen2.5:3b`.

## Garanties de sécurité

- **Aucun envoi sans validation** : `ReviewQueue.send` lève `NotApprovedError` si le statut n'est pas `approved`.
- **Pas de modification après validation** : l'empreinte SHA-256 du texte approuvé est revérifiée à l'envoi.
- **Pas de réponse sans source** : une réponse sans citation valide devient une réponse d'attente, toujours soumise à l'encadrant.
- **Abstention** : si les passages couvrent mal la question, ou si le modèle dit ne pas pouvoir répondre, le brouillon est une réponse d'attente.
- **Vérification des valeurs** : nombres, horaires, adresses e-mail du brouillon absents des sources citées → alerte pour l'encadrant.
- **Injection de consigne** : l'e-mail est traité comme une donnée non fiable (balises, consigne système) ; testé (cas `a36`).
- **Interface web** : écoute sur 127.0.0.1 uniquement, contenu des e-mails échappé (anti-XSS), jeton CSRF, CSP sans script, vérification de l'en-tête Host.

## Évaluation

Jeu de test `data/eval/testset.jsonl` : **74 e-mails** (54 avec réponse dans les documents : questions directes, reformulations, fautes de frappe, e-mails longs, questions à deux parties, tentative d'injection ; **20 hors documents**, dont des pièges proches des sujets couverts). Séparé en `dev` (réglage du seuil d'abstention) et `test` (chiffres rapportés), 37 e-mails chacun.

Version **hors-ligne** (`--llm extractive`, sans modèle de langage), jeu `test` :

| Métrique | Valeur |
|---|---|
| Brouillons acceptables (bonne réponse, ou abstention si hors documents) | 64,9 % (24/37, IC95 49–78 %) |
| Retrieval hit@1 / hit@4 / MRR | 77,8 % / 96,3 % / 0,870 |
| Réponses correctes (tous les faits attendus) | 55,6 % (15/27) |
| Abstentions à tort | 37,0 % (10/27) |
| Réponses à tort sur questions hors documents | 10 % (1/10) |
| Sans alerte de fidélité | 100 % (18/18) |

Lecture honnête :
- La recherche est bonne mais pas parfaite (une question sur l'installation d'un logiciel ne retrouve rien : « installer » ≠ « installation » pour une recherche lexicale).
- Le jeu élargi a révélé un piège que le petit jeu ne voyait pas : « mot de passe du **wifi invité** » reçoit la règle sur les mots de passe des comptes. Un modèle de langage et la validation humaine sont là pour ça.
- Le seuil d'abstention optimal dépend du modèle : le balayage (`reports/`) le calcule sur `dev` pour chaque configuration. Il est maintenant calculé en **un seul passage** (le modèle n'est appelé qu'une fois par e-mail), de façon exacte.
- Les intervalles de confiance restent larges (37 e-mails par jeu) : ce sont des ordres de grandeur.

**Comparer vos configurations** (c'est la partie « résultats » de votre rapport) :
```bash
python3 -m stagiaires_rag --llm ollama --cache --top-k 3 eval                    # BM25
python3 -m stagiaires_rag --llm ollama --retriever hybrid --cache --top-k 3 eval # hybride
```
Chaque lancement écrit `reports/latest-<llm>-<recherche>.md`. Reportez dans votre rapport vos chiffres, pas les miens ; réglez `--min-coverage` sur la valeur « meilleur seuil sur dev » de chaque rapport, puis relancez pour confirmer.

La vérification des faits est une recherche de texte (insensible aux accents, espaces et au « % ») : une paraphrase correcte comme « deux jours » au lieu de « 2 jours » peut être comptée fausse ; ajoutez des alternatives dans `facts` du jeu de test si besoin.

## Limites et pistes

- Les documents sont fictifs ; l'étape suivante est de tester sur de vrais documents et de vrais e-mails anonymisés.
- La recherche hybride et l'interface web n'ont été testées ici qu'avec des simulations (faux embeddings, serveur de test) : lancez-les chez vous et vérifiez.
- Brouillons en français uniquement (formule d'appel, réponse d'attente).
- Pas de connexion boîte mail (IMAP) : les e-mails entrent par l'interface web, `draft --file` (`.eml`/`.json`) ou la CLI.

## Structure

```
src/stagiaires_rag/  text · ingest · retriever · embeddings · llm · drafter · review · mailer · evaluate · webui · cli
data/docs/           documents du stage (exemples)     data/eval/testset.jsonl  jeu de test (74 e-mails)
tests/               42 tests                          reports/                 rapports d'évaluation
docs/CV_ET_ENTRETIEN.md   comment présenter le projet (CV, rapport, questions d'entretien)
```
