import os
import sys
import logging
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

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.PersistentClient(path=DB_PATH)

all_chunks_store = []
bm25_index = None


def load_pdf(path):
    if not os.path.exists(path):
        logging.error("الملف غير موجود: %s", path)
        raise FileNotFoundError(f"الملف غير موجود: {path}")
    try:
        pdf = fitz.open(path)
        text = ""
        for page in pdf:
            text += page.get_text()
        logging.info("تم تحميل الملف بنجاح: %s", path)
        return text
    except Exception as e:
        logging.error("فشل قراءة الملف %s: %s", path, e)
        raise RuntimeError(f"فشل قراءة الملف: {e}")


def create_database(text, collection_name=COLLECTION_NAME):
    global all_chunks_store, bm25_index

    try:
        chroma.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma.create_collection(collection_name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[str(i)]
        )

    all_chunks_store = chunks
    tokenized_chunks = [c.split() for c in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    logging.info("تم إنشاء قاعدة بيانات بـ %d chunk", len(chunks))
    return collection, len(chunks)


def semantic_search(collection, question, n_results=N_RESULTS):
    vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=n_results
    )
    return results["documents"][0]


def keyword_search(question, n_results=N_RESULTS):
    if bm25_index is None:
        return []
    tokenized_query = question.split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
    return [all_chunks_store[i] for i in top_indices]


def hybrid_search(collection, question, n_results=N_RESULTS):
    semantic_results = semantic_search(collection, question, n_results)
    keyword_results = keyword_search(question, n_results)

    combined = list(dict.fromkeys(semantic_results + keyword_results))
    return "\n".join(combined[:n_results * 2])


def ask_ai(context, question):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"أجب فقط من هذه المعلومات:\n{context}"},
                {"role": "user", "content": question}
            ]
        )
        logging.info("تم الرد على السؤال: %s", question)
        return response.choices[0].message.content
    except Exception as e:
        logging.error("فشل الاتصال بالنموذج: %s", e)
        return f"حدث خطأ أثناء الاتصال بالنموذج: {e}"


def main():
    pdf_path = input("أدخل مسار ملف PDF: ").strip()

    try:
        print("جاري تحميل الملف...")
        text = load_pdf(pdf_path)
        collection, chunk_count = create_database(text)
        print(f"تم تحميل {chunk_count} chunk بنجاح")
        print("جاهز — اسأل أي سؤال (اكتب exit للخروج)")
        print("---")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"خطأ: {e}")
        sys.exit(1)

    while True:
        question = input("أنت: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        context = hybrid_search(collection, question)
        answer = ask_ai(context, question)
        print("AI:", answer)
        print("---")


if __name__ == "__main__":
    main()