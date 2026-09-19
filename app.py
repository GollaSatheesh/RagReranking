import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG with Cross-Encoder Reranking")
st.title("RAG with Cross-Encoder Reranking")

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY is missing. Add it in the app's Settings -> Secrets.")
    st.stop()

   llm = ChatGroq(model="openai/gpt-oss-120b", api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource(show_spinner="Loading documents and models...")
def load_pipeline():
    docs = TextLoader("paracetamol.txt", encoding="utf-8").load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return retriever, reranker


retriever, reranker = load_pipeline()

query = st.text_input("Enter your question")

if query:
    with st.spinner("Retrieving, reranking and generating..."):
        retrieved_docs = retriever.invoke(query)

        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, score in ranked[:5]]

        context = "\n\n".join([doc.page_content for doc in top_docs])

        prompt = f"""
    Answer the question using ONLY the context below.

    Context: {context}

    Question: {query}
"""
        response = llm.invoke(prompt)

    st.subheader("Response")
    st.write(response.content)
