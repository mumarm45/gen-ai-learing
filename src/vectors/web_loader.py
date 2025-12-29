from langchain_community.document_loaders import WebBaseLoader
def web_loader(url, verify_ssl: bool = True):
    requests_kwargs = None
    if not verify_ssl:
        requests_kwargs = {"verify": False}

    loader = WebBaseLoader(url, requests_kwargs=requests_kwargs)
    documents = loader.load()
    return documents