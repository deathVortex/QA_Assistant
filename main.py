import os
from pathlib import Path

from agents.rag import build_rag_agent
from config import THREAD_ID, load_env_file
from tools.search import ingest_pdf_corpus_data


load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_CORPUS_PATH = PROJECT_ROOT / "data"

raw_pdf_paths = os.getenv("PDF_CORPUS_PATHS")
if raw_pdf_paths:
    PDF_CORPUS_PATHS = [
        path.strip()
        for path in raw_pdf_paths.split(";")
        if path.strip()
    ]
else:
    PDF_CORPUS_PATHS = [str(DEFAULT_PDF_CORPUS_PATH)]

BUSINESS_QUESTIONS = [
    {
        "question": "Qu'est-ce que Small Basic ?",
        "expected_keywords": ["langage de programmation", "débutants", "simple"],
    },
    {
        "question": "Quel est l'objectif de Small Basic ?",
        "expected_keywords": ["lever les barrières", "programmation logicielle", "apporter"],
    },
    {
        "question": "Quels exemples de projets le document cite-t-il à la fin ?",
        "expected_keywords": ["casse-briques", "fond d'écran"],
    },
    {
        "question": "Comment le document décrit-il la programmation logicielle ?",
        "expected_keywords": ["création de logiciel", "langages de programmation"],
    },
]


def format_ingestion_summary(summary: dict[str, object]) -> str:
    lines = ["Résumé d'ingestion", "=================="]
    lines.append("")
    lines.append(f"Statut : {summary.get('status', 'inconnu')}")
    if summary.get("message"):
        lines.append(f"Message : {summary['message']}")
    if summary.get("ingested_files") is not None:
        lines.append(f"Fichiers ingérés : {len(summary['ingested_files'])}")
    if summary.get("page_count") is not None:
        lines.append(f"Pages analysées : {summary['page_count']}")
    if summary.get("chunk_count") is not None:
        lines.append(f"Fragments créés : {summary['chunk_count']}")
    if summary.get("embedding_model"):
        lines.append(f"Modèle d'embedding : {summary['embedding_model']}")
    if summary.get("embedding_count") is not None:
        lines.append(f"Vecteurs générés : {summary['embedding_count']}")
    if summary.get("embedding_dimensions") is not None:
        lines.append(f"Dimensions des vecteurs : {summary['embedding_dimensions']}")

    vectorstore = summary.get("vectorstore") or {}
    if vectorstore:
        lines.append(f"Collection Chroma : {vectorstore.get('collection_name', 'inconnue')}")
        lines.append(f"Dossier Chroma : {vectorstore.get('chroma_dir', 'inconnu')}")
        if vectorstore.get("stored_count") is not None:
            lines.append(f"Chunks indexés : {vectorstore['stored_count']}")
        if vectorstore.get("metadata_keys"):
            lines.append("Clés de métadonnées :")
            for key in vectorstore["metadata_keys"]:
                lines.append(f"  - {key}")
        if vectorstore.get("missing_required_keys"):
            lines.append("Clés obligatoires manquantes :")
            for key in vectorstore["missing_required_keys"]:
                lines.append(f"  - {key}")
        if vectorstore.get("sample_metadata"):
            lines.append("Exemple de métadonnées :")
            for key, value in vectorstore["sample_metadata"].items():
                lines.append(f"  - {key} : {value}")

    if summary.get("ingested_files"):
        lines.append("Sources PDF :")
        for item in summary["ingested_files"]:
            lines.append(f"  - {item}")

    return "\n".join(lines).strip()


def extract_answer_text(result) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    content = getattr(last_message, "content", None)
    if isinstance(content, str):
        return content.strip()

    content_blocks = getattr(last_message, "content_blocks", None)
    if content_blocks:
        parts = []
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
            else:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if block_type == "text" and block_text:
                    parts.append(block_text)
        return "\n".join(parts).strip()

    return str(last_message)


def score_answer(answer: str, expected_keywords: list[str]) -> tuple[int, int, list[str]]:
    normalized = answer.lower()
    matched = [keyword for keyword in expected_keywords if keyword.lower() in normalized]
    return len(matched), len(expected_keywords), matched


def main():
    ingestion_summary = ingest_pdf_corpus_data(PDF_CORPUS_PATHS)
    print(format_ingestion_summary(ingestion_summary))
    print()
    print("Évaluation RAG")
    print("==============")

    agent = build_rag_agent()
    for index, item in enumerate(BUSINESS_QUESTIONS, start=1):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": item["question"]}]},
            config={"configurable": {"thread_id": f"{THREAD_ID}-rag-{index}"}},
        )
        answer = extract_answer_text(result)
        score, total, matched = score_answer(answer, item["expected_keywords"])
        relevance = "pertinent" if score >= max(1, total // 2) else "à revoir"

        print()
        print(f"Question {index} : {item['question']}")
        print(f"Réponse : {answer}")
        print(f"Pertinence : {relevance} ({score}/{total} mots-clés trouvés)")
        if matched:
            print("Mots-clés retrouvés : " + ", ".join(matched))


if __name__ == "__main__":
    main()
