
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

def recursive_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""],
    ) 
def text_splitter():
    return CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")   