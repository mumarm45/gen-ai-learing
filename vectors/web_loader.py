
from langchain_community.document_loaders import WebBaseLoader
def web_loader(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents 