from src.helper_ import load_pdf,recursive_text_split,download_hugging_face_embedding,clean_whitespace,listToString
from langchain_chroma import Chroma
import os

Hypertension_document_pdf=load_pdf("Hypertension_data/")

Hpertension_doc_C=clean_whitespace(listToString(Hypertension_document_pdf))

Hpertension_doc_C_chunks=recursive_text_split([Hpertension_doc_C])
#----------------------------------------------------------------------------------------------------------------------
diabetes_document_pdf=load_pdf("Diabetes_data/")

diabetes_doc_C=clean_whitespace(listToString(diabetes_document_pdf))

diabetes_doc_C_chunks=recursive_text_split([diabetes_doc_C])
#----------------------------------------------------------------------------------------------------------------------
cholesterol_document_pdf=load_pdf("Cholesterol_data/")

cholesterol_doc_C=clean_whitespace(listToString(cholesterol_document_pdf))

cholesterol_doc_C_chunks=recursive_text_split([cholesterol_doc_C])
#----------------------------------------------------------------------------------------------------------------------
fruits_veg_snacks_document_pdf=load_pdf("Fruits_Vegetables_snacks_data_data/")

fruits_veg_snacks_doc_C=clean_whitespace(listToString(fruits_veg_snacks_document_pdf))

fruits_veg_snacks_doc_C_chunks=recursive_text_split([fruits_veg_snacks_doc_C])
#-------------------------------------------------------------------------------------------------------------------------
embedding_model=download_hugging_face_embedding()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Hpertension_doc_store=Chroma.from_documents(Hpertension_doc_C_chunks, embedding_model,collection_metadata={"hnsw:space":"cosine"}, persist_directory="store/hypertension_chroma_db")

load_Hpertension_doc_store=Chroma(persist_directory="store/hypertension_chroma_db",embedding_function=embedding_model)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
diabetes_doc_store=Chroma.from_documents(diabetes_doc_C_chunks, embedding_model,collection_metadata={"hnsw:space":"cosine"}, persist_directory="store/diabetes_chroma_db")

load_diabetes_doc_store=Chroma(persist_directory="store/diabetes_chroma_db",embedding_function=embedding_model)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cholesterol_doc_store=Chroma.from_documents(cholesterol_doc_C_chunks, embedding_model,collection_metadata={"hnsw:space":"cosine"}, persist_directory="store/cholesterol_chroma_db")

load_cholesterol_doc_store=Chroma(persist_directory="store/cholesterol_chroma_db",embedding_function=embedding_model)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fruits_veg_snacks_doc_store=Chroma.from_documents(fruits_veg_snacks_doc_C_chunks, embedding_model,collection_metadata={"hnsw:space":"cosine"}, persist_directory="store/fruits_veg_snacks_chroma_db")

load_fruits_veg_snacks_doc_store=Chroma(persist_directory="store/fruits_veg_snacks_chroma_db",embedding_function=embedding_model)

def hypertention_db():
    h_db=load_Hpertension_doc_store
    return h_db

def cholesterol_db():
    d_db=load_diabetes_doc_store
    return d_db

def diabetes_db():
    c_db=load_cholesterol_doc_store
    return c_db

def fruits_veg_snacks_db():
    fvs_db=load_fruits_veg_snacks_doc_store
    return fvs_db
