from src.helper_ import load_pdf, recursive_text_split, download_hugging_face_embedding, clean_whitespace, listToString
from langchain_chroma import Chroma
import os

# Load and process Hypertension document
print("Loading and processing Hypertension document...")
Hypertension_document_pdf = load_pdf("Hypertension_data/")
Hpertension_doc_C = clean_whitespace(listToString(Hypertension_document_pdf))
Hpertension_doc_C_chunks = recursive_text_split([Hpertension_doc_C])

# Load and process Diabetes document
print("Loading and processing Diabetes document...")
diabetes_document_pdf = load_pdf("Diabetes_data/")
diabetes_doc_C = clean_whitespace(listToString(diabetes_document_pdf))
diabetes_doc_C_chunks = recursive_text_split([diabetes_doc_C])

# Load and process Cholesterol document
print("Loading and processing Cholesterol document...")
cholesterol_document_pdf = load_pdf("Cholesterol_data/")
cholesterol_doc_C = clean_whitespace(listToString(cholesterol_document_pdf))
cholesterol_doc_C_chunks = recursive_text_split([cholesterol_doc_C])

# Load and process Fruits and Vegetables Snacks document
print("Loading and processing Fruits and Vegetables Snacks document...")
fruits_veg_snacks_document_pdf = load_pdf("Fruits_Vegetables_snacks_data_data/")
fruits_veg_snacks_doc_C = clean_whitespace(listToString(fruits_veg_snacks_document_pdf))
fruits_veg_snacks_doc_C_chunks = recursive_text_split([fruits_veg_snacks_doc_C])

# Download embedding model
print("Downloading embedding model...")
embedding_model = download_hugging_face_embedding()

# Create and load Chroma stores
print("Creating and loading Chroma stores...")

Hpertension_doc_store = Chroma.from_documents(Hpertension_doc_C_chunks, embedding_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/hypertension_chroma_db")
load_Hpertension_doc_store = Chroma(persist_directory="store/hypertension_chroma_db", embedding_function=embedding_model)

diabetes_doc_store = Chroma.from_documents(diabetes_doc_C_chunks, embedding_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/diabetes_chroma_db")
load_diabetes_doc_store = Chroma(persist_directory="store/diabetes_chroma_db", embedding_function=embedding_model)

cholesterol_doc_store = Chroma.from_documents(cholesterol_doc_C_chunks, embedding_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/cholesterol_chroma_db")
load_cholesterol_doc_store = Chroma(persist_directory="store/cholesterol_chroma_db", embedding_function=embedding_model)

fruits_veg_snacks_doc_store = Chroma.from_documents(fruits_veg_snacks_doc_C_chunks, embedding_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="store/fruits_veg_snacks_chroma_db")
load_fruits_veg_snacks_doc_store = Chroma(persist_directory="store/fruits_veg_snacks_chroma_db", embedding_function=embedding_model)

# Define functions to return the loaded Chroma stores
def hypertention_db():
    print("Returning Hypertension DB")
    return load_Hpertension_doc_store

def cholesterol_db():
    print("Returning Cholesterol DB")
    return load_cholesterol_doc_store

def diabetes_db():
    print("Returning Diabetes DB")
    return load_diabetes_doc_store

def fruits_veg_snacks_db():
    print("Returning Fruits and Vegetables Snacks DB")
    return load_fruits_veg_snacks_doc_store

''''if __name__ == "__main__":
    print("Hypertension DB:", hypertention_db())
    print("Cholesterol DB:", cholesterol_db())
    print("Diabetes DB:", diabetes_db())
    print("Fruits and Vegetables Snacks DB:", fruits_veg_snacks_db())'''
