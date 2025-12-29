from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
try:
    from .splitter import recursive_text_splitter
except ImportError:  # pragma: no cover
    from splitter import recursive_text_splitter
def pdf_loader(filepath):
    loader = PyPDFLoader(filepath)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were loaded from the PDF")

    non_empty_documents = [d for d in documents if (d.page_content or "").strip()]
    if not non_empty_documents:
        raise ValueError(
            "PDF loaded, but no text was extracted. The PDF may be scanned images and require OCR."
        )

    splitter_recursive = recursive_text_splitter()

    chunks_recursive = splitter_recursive.split_documents(non_empty_documents)

    if not chunks_recursive:
        return non_empty_documents

    return chunks_recursive


def pdf_reader(file):
    """Check if a PDF contains extractable text content."""
    try:
        
        
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        
        # Check if PDF has text content
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() or ""
        
        if not text_content.strip():
            return False, "PDF does not contain extractable text content"
        
        return True, f"PDF contains {total_pages} pages with extractable text"

    except Exception as e:
        return False, f"Error reading PDF: {str(e)}"