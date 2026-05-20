# QA Assistant

Assistant Python pour ingérer un corpus PDF métier et préparer des documents internes pour des usages de Q&A.

## Structure

```text
QA_Assistant/
├── main.py
├── config.py
├── llm.py
├── tools/
├── memory/
├── data/
├── chroma_db/
├── .env
├── pyproject.toml
└── requirements.txt
```

## Où mettre les PDF

Dépose les fichiers PDF à traiter dans le dossier [`data/`](./data).

Par défaut, l'application lit ce dossier automatiquement si `PDF_CORPUS_PATHS` n'est pas défini.

## Configuration optionnelle

Si tu veux cibler d'autres chemins que `data/`, définis `PDF_CORPUS_PATHS` dans `.env`.

Exemple :

```env
PDF_CORPUS_PATHS="D:\YASSINE\Objectware\Formations\IA\Module 3\QA_Assistant\data"
```

Tu peux aussi mettre plusieurs chemins séparés par `;`.

## Lancer le projet

```bash
uv run .\main.py
```

## Remarques

- `OPENAI_API_KEY` doit être défini dans `.env`.
- Le tool d'ingestion utilise LangChain pour charger et découper les PDF.
- Les embeddings sont générés avec OpenAI puis indexés dans Chroma, stocké localement dans [`chroma_db/`](./chroma_db).
- Les métadonnées indexées suivent une structure stable: `source`, `source_name`, `source_type`, `page`, `chunk_index`, `chunk_id`, `chunk_chars`, `embedding_model`.
