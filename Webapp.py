from flask import Flask, render_template, jsonify, request
from src.store_index import hypertention_db
import tensorflow as tf
from src.helper_ import load_pdf,clean_whitespace,listToString
import pickle
import base64
import numpy as np
import os
import sys

# Set the default encoding to UTF-8
if not os.environ.get('PYTHONIOENCODING'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

app=Flask(__name__)

hypertension_store=hypertention_db()
##------------ **Loading the LLM------------**
import langchain_openai 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()#take environemnt variable from .env

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                model_name='gpt-3.5-turbo',
                temperature=0.5,
                max_tokens=2500,
                top_p=1.0,
                frequency_penalty=1.8,
                presence_penalty=1.0)

'''##------------ **Retreiving documents with RAG FUSION------------**'''
## Init Merge Retriever and Perform Semantic Search
from typing import List
from langchain_core.runnables import chain
from langchain_core.documents import Document

@chain
def retriever(query: str) -> List[Document]:

    vectorstores = [
        hypertension_store
   ]


    all_docs_with_scores = []

    # Iterate through each vectorstore to get results with scores
    for vectorstore in vectorstores:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
        all_docs_with_scores.extend(docs_with_scores)

    # Extract documents and scores, and add scores to metadata
    if all_docs_with_scores:#This condition checks if the all_docs_with_scores list contains any elements. If it is empty, the code inside the if block is not executed
        docs, scores = zip(*all_docs_with_scores)
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        
        sorted_docs = sorted(docs, key=lambda d: d.metadata["score"], reverse=False)

    return sorted_docs

## **Retreiving documents with RAG FUSION**


from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import ChatMessagePromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate four
    different versions of the given user question to retrieve relevant documents from a vector
    database. The first version should be the original question entered by the user, and the other
    three should be different rephrased versions of the same question. By generating multiple perspectives on
    the user question, your goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines. Only provide the queries, no numbering.
    Original question: {question}"""
)

generate_queries = (
    QUERY_PROMPT| llm | StrOutputParser() | (lambda x: x.split("\n"))
)


from langchain_core.documents import Document
from typing import List, Tuple
from json import dumps, loads

# Ensure the necessary imports and context are correct
def document_to_dict(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def dict_to_document(doc_dict: dict) -> Document:
    return Document(
        page_content=doc_dict["page_content"],
        metadata=doc_dict["metadata"]
    )

def rank_fusion_similarity_score(results: List[List[Document]], k=60) -> List[Tuple[Document, float]]:
    fused_scores = {}
    doc_appearance_count = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_dict = document_to_dict(doc)
            doc_str = dumps(doc_dict)
            similarity_score = doc.metadata.get("score")  # Use the document's score
            if doc_str not in fused_scores:
                fused_scores[doc_str] = similarity_score  # Initialize with the document's score  # Use the document's score
                doc_appearance_count[doc_str] = 1
            else:
              doc_appearance_count[doc_str] += 1
              fused_scores[doc_str] -= 1 / (rank + k) # Add the rank contribution

    reranked_results = [
        (dict_to_document(loads(doc)), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=False)
    ]

    final_results=reranked_results[0:5]

    # Return only the top 5 results
    return final_results


def custom_retriever_func(queries: List[str]) -> List[Tuple[Document, float]]:
    results = []
    for query in queries:
        docs = retriever.invoke(query)
        results.append(docs)
    return rank_fusion_similarity_score(results)

ragfusion_chain = generate_queries | (lambda queries: custom_retriever_func(queries))


import langchain


from langchain.schema.runnable import RunnablePassthrough

template='''
You are an AI language model assistant. 
Your task is to use the following Context and your own knowledge to generate an answer to the user's question. 
If you don't know the answer, just say you don't know, don't try to make up an answer.


Context:{context}
Question:{question}

'''
prompt2 = ChatPromptTemplate.from_template(template)

full_rag_fusion_chain = (
    {
        "context": ragfusion_chain,
        "question": RunnablePassthrough()
    }
    | prompt2
    | llm
    | StrOutputParser()

)
#Only return the helpful answer below and nothing else.
langchain.debug = False

@app.route("/")#The root URL of a Flask application is the base URL where the application is accessible. It represents the main entry point or homepage of the application. The root URL is defined by a single forward slash (/
#the root URL would be http://localhost:8080/
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]#Extracts the message sent by the user from the form data. 
    user_input = msg
    print(input)
    result=full_rag_fusion_chain.invoke({"question": user_input})
    print("Response:", result)
    return str(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)