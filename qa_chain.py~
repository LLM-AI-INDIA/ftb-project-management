import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List

load_dotenv()

INDEX_PATH = "faiss_index"

def build_qa(csv_files):
    INDEX_PATH = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Better embedding model

    if os.path.exists(INDEX_PATH):
        print("🔄 Loading existing FAISS index...")
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
    else:
        print("⚡ Building FAISS index from CSV files...")
        docs = []
        for file in csv_files:
            try:
                loader = CSVLoader(file_path=file)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

        # Better text splitting with overlap for context preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        texts = text_splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(INDEX_PATH)

    # Use a better retriever with more relevant documents
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for better diversity
        search_kwargs={"k": 6}  # Retrieve more documents for better context
    )

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.1  # Lower temperature for more deterministic responses
    )

    # Much more detailed and specific prompt template
    prompt_template = """You are an expert assistant for the Franchise Tax Board. Your role is to provide accurate, detailed, and helpful information based on the context provided.

CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

GUIDELINES FOR RESPONSE:
1. Answer based ONLY on the context provided above
2. If the information is not in the context, politely state that you don't have that information
3. Be specific and detailed in your responses
4. Use bullet points or numbered lists when appropriate for clarity
5. If referring to projects, include relevant details like status, timeline, or key personnel
6. Format your response in a professional and easy-to-read manner
7. If the question is ambiguous, ask for clarification but first try to provide the most relevant information

ANSWER:
"""
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    def query_chain(query):
        try:
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(query)
            
            # Format context from documents with source information
            context_parts = []
            for i, doc in enumerate(docs):
                source = getattr(doc, 'metadata', {}).get('source', 'Unknown source')
                context_parts.append(f"Document {i+1} (from {source}):\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Format prompt
            formatted_prompt = prompt.format(context=context, question=query)
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            
            return response.content
            
        except Exception as e:
            print(f"Error in query processing: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please try again."

    return query_chain
