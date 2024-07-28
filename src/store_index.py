from src.helper_ import load_pdf,recursive_text_split,download_hugging_face_embedding,clean_whitespace,listToString
from langchain_chroma import Chroma
import os

Hypertension_document_pdf=load_pdf("Hypertension_data/")

Hpertension_doc_C=clean_whitespace(listToString(Hypertension_document_pdf))

Hpertension_doc_C_chunks=recursive_text_split([Hpertension_doc_C])

embedding_model=download_hugging_face_embedding()

Hpertension_doc_store=Chroma.from_documents(Hpertension_doc_C_chunks, embedding_model,collection_metadata={"hnsw:space":"cosine"}, persist_directory="store/hypertension_chroma_db")

load_Hpertension_doc_store=Chroma(persist_directory="store/hypertension_chroma_db",embedding_function=embedding_model)

def hypertention_db():
    h_db=load_Hpertension_doc_store
    return h_db