# Présenter ce projet (CV, rapport, entretien)

## Honnêteté d'abord
- Intitulé : **Projet de fin d'année (PFA)** — pas « stage ». Section « Projets académiques ».
- Mettez **vos** chiffres : lancez l'évaluation avec votre modèle (Ollama) et notez les résultats de `reports/`. Les chiffres du README sont ceux de la version hors-ligne.
- Ne citez comme technologie que ce que vous avez lancé et compris. Hybride/embeddings : seulement après l'avoir exécuté chez vous.
- Si votre école exige un travail personnel, indiquez dans le rapport les outils utilisés (assistant IA compris) et votre apport : documents, jeu de test, expériences, analyse des échecs.

## Technologies (à adapter à ce que vous avez exécuté)
Python (bibliothèque standard), RAG, BM25, embeddings (Ollama, bge-m3) et fusion RRF, LLM locaux (Ollama, Qwen2.5), API Anthropic/OpenAI-compatibles, prompt engineering (sortie JSON, citations, abstention, anti-injection), SQLite, serveur HTTP (interface web), SMTP, pytest, évaluation (hit@k, MRR, intervalles de Wilson, découpage dev/test).

## Version CV (3 lignes)
> **Assistant RAG de réponse aux e-mails de stagiaires** — PFA, INPT · Python, BM25 + embeddings, Ollama (Qwen2.5), SQLite, pytest
> - Conception d'un pipeline RAG avec sources citées, abstention hors documents et brouillons validés par un encadrant avant tout envoi (interface web sécurisée : CSRF, anti-XSS).
> - Évaluation sur 74 e-mails annotés (dont 20 hors documents) : **[X %]** de brouillons acceptables avec **[modèle]**, contre **[Y %]** pour la base hors-ligne ; seuil d'abstention réglé sur un jeu `dev`, résultats rapportés sur `test`.

Remplacez [X], [Y], [modèle] par vos chiffres.

## Questions d'entretien probables (préparez vos réponses)
1. **Pourquoi du RAG et pas simplement un LLM ?** Les réponses doivent venir des documents du stage (montants, délais). Le RAG limite les inventions et permet de citer ses sources.
2. **Pourquoi BM25 ? Pourquoi ajouter des embeddings ?** BM25 est exact sur les mots (« 17h30 », « MFA ») mais rate les synonymes (« bus » / « transport en commun », « installer » / « installation »). Les embeddings captent le sens mais peuvent confondre des sujets proches. La fusion RRF combine les deux classements sans comparer des scores d'échelles différentes.
3. **Comment évitez-vous les hallucinations ?** Prompt limité aux passages, abstention si la couverture est faible ou si le modèle refuse, citations vérifiées, valeurs (nombres, horaires, e-mails) comparées aux sources, validation humaine obligatoire.
4. **Comment avez-vous choisi le seuil d'abstention ?** Balayage sur le jeu `dev` ; chiffres rapportés sur `test`, pour ne pas régler sur les données de rapport. Limite : 37 e-mails par jeu, donc des intervalles de confiance larges.
5. **Que mesure « brouillon acceptable » ?** Bonne réponse (tous les faits attendus) si la question est couverte, abstention sinon. C'est ce qui compte pour l'encadrant : peu de corrections.
6. **Quel est le plus gros échec observé ?** Citez un cas réel de vos rapports (ex. « mot de passe du wifi invité » qui reçoit la règle des mots de passe des comptes) et ce que vous en concluez.
7. **Comment l'envoi est-il sécurisé ?** Machine à états (pending → approved → sent), refus d'envoi si non approuvé, empreinte SHA-256 du texte approuvé revérifiée à l'envoi, envoi à blanc par défaut.
8. **Un e-mail peut-il manipuler l'assistant ?** Le contenu est traité comme donnée non fiable (balises + consigne), testé avec une injection ; l'interface web échappe tout le contenu et refuse les requêtes sans jeton CSRF.
9. **Pourquoi un petit modèle local ?** Gratuit, et les e-mails des stagiaires ne quittent pas la machine. Contrepartie : plus d'abstentions et de formats JSON invalides, absorbés par les garde-fous.
10. **Que feriez-vous ensuite ?** Vrais documents et e-mails anonymisés, jeu de test plus grand, boîte mail (IMAP), évaluation de la fidélité par un second modèle, mesure du temps gagné par l'encadrant (taux de modification dans `stats`).
