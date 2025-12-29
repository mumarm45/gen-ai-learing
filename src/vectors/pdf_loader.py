from langchain_community.document_loaders import PyPDFLoader
try:
    from .splitter import recursive_text_splitter
except ImportError:  # pragma: no cover
    from splitter import recursive_text_splitter
def pdf_loader(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were loaded from langchain.pdf")

    non_empty_documents = [d for d in documents if (d.page_content or "").strip()]
    if not non_empty_documents:
        raise ValueError(
            "PDF loaded, but no text was extracted. The PDF may be scanned images and require OCR."
        )

    splitter_recursive = recursive_text_splitter()

    chunks_recursive = splitter_recursive.split_documents(non_empty_documents)

    if not chunks_recursive:
        return non_empty_documents[0].page_content

    return chunks_recursive
