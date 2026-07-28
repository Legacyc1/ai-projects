import os
import tempfile
import streamlit as st
import fitz
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from config import (
    MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
    DB_PATH, COLLECTION_NAME, N_RESULTS
)

load_dotenv()

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.write("its working")

@st.cache_resource
def load_tools():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma = chromadb.PersistentClient(path=DB_PATH)
    return client, embedder, chroma


client, embedder, chroma = load_tools()


def load_pdf(file_path):
    pdf = fitz.open(file_path)
    text = ""
    for page in pdf:
        text += page.get_text()
    return text


def create_database(text):
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma.create_collection(COLLECTION_NAME)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        collection.add(documents=[chunk], embeddings=[vector], ids=[str(i)])

    tokenized_chunks = [c.split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    return collection, chunks, bm25_index, len(chunks)


def hybrid_search(collection, chunks, bm25_index, question, n_results=N_RESULTS):
    vector = embedder.encode(question).tolist()
    semantic_results = collection.query(query_embeddings=[vector], n_results=n_results)["documents"][0]

    tokenized_query = question.split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    keyword_results = [chunks[i] for i in top_indices]

    combined = list(dict.fromkeys(semantic_results + keyword_results))
    return "\n".join(combined[:n_results * 2])


def ask_ai(context, question):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"أجب فقط من هذه المعلومات:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


st.title("📄 PDF Q&A Chatbot")
st.caption("ارفع ملف PDF واسأل أي سؤال عن محتواه")

uploaded_file = st.file_uploader("ارفع ملف PDF", type="pdf")

if uploaded_file is not None:
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("جاري تحميل ومعالجة الملف..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            text = load_pdf(tmp_path)
            collection, chunks, bm25_index, chunk_count = create_database(text)

            st.session_state.collection = collection
            st.session_state.chunks = chunks
            st.session_state.bm25_index = bm25_index
            st.session_state.processed_file = uploaded_file.name
            st.session_state.messages = []

        st.success(f"تم تحميل الملف بنجاح ({chunk_count} chunk)")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("اكتب سؤالك هنا...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                context = hybrid_search(
                    st.session_state.collection,
                    st.session_state.chunks,
                    st.session_state.bm25_index,
                    question
                )
                answer = ask_ai(context, question)
                st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("ارفعي ملف PDF عشان تبدأي")