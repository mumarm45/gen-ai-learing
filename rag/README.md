# RAG (Retrieval-Augmented Generation)

## Quickstart

1. Build the vector store

```bash
python -m vectors.build_vector
```

2. Ask a single question

```bash
python -m rag.rag_answer
```

3. Chat

```bash
python -m rag.chat_with_rag
```

## Indexing a PDF

1. Place `resume.pdf` in the project root (same level as the `vectors/` folder).
2. Build the vector store:

```bash
python -m vectors.build_vector
```

This creates a persisted vector store in the `chroma_db/` directory.

## Calling from `python -c`

Single question:

```bash
python -c "from rag.rag_answer import rag_answer; print(rag_answer(question='What is your name?'))"
```

Chat:

```bash
python -c "from rag.chat_with_rag import chat_answer; chat_answer()"
```