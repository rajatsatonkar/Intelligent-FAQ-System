from dotenv import load_dotenv
from langchain_community.llms.huggingface_hub import HuggingFaceHub

load_dotenv()
import os
from langchain_huggingface import HuggingFaceEndpoint  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
import pickle
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader  # Updated import


# Initialize embeddings and LLM with parameters directly
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 220})  # Pass parameters directly

vectordb_file_path = "faiss_index"

def create_vector_db():
    try:
        loader = CSVLoader(file_path='faqs.csv', source_column="prompt")
        data = loader.load()

        if not data:
            print("No data loaded from CSV. Please check the file and source column.")
            return

        vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
        vectordb.save_local(vectordb_file_path)
        print("FAISS index created and saved successfully.")
    except Exception as e:
        print(f"Error in create_vector_db: {e}")

def get_qa_chain():
    try:
        vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(score_threshold=0.7)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

        return chain
    except Exception as e:
        print(f"Error in get_qa_chain: {e}")

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
