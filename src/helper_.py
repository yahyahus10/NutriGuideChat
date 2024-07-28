
## **Loading_PDF and extracting data **
from langchain_community.document_loaders import PyPDFDirectoryLoader

def load_pdf(data):
    loader = PyPDFDirectoryLoader(data, glob="*.pdf")

    documents=loader.load()

    return documents

## **Loading and extracting data from the web **
from langchain_community.document_loaders import WebBaseLoader

def load_web(url):
    loader= WebBaseLoader(url)
    documents = loader.load()
    return documents

## Create text chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_text_split(extracted_data):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

  text_chunks = text_splitter.create_documents(extracted_data)
  return text_chunks

## loading embedding model from huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
def download_hugging_face_embedding():
   
   embedding_model= HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
   
   return embedding_model


import re
def clean_whitespace(text):
    # Replace multiple newlines with two newlines
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def listToString(doc_list):
    # Initialize an empty string
    str1 = ""

    # Traverse through the list of Document objects
    for document in doc_list:
        if hasattr(document, 'page_content'):  # Check if the Document has the 'page_content' attribute
            str1 += document.page_content + " "  # Append the content and a space to separate documents

    return str1