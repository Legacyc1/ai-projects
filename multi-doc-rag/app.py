import os
import fitz
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()
collection = chroma.create_collection("multi_docs")


def load_pdf(path):
    pdf = fitz.open(path)
    text = ""
    for page in pdf:
        text += page.get_text()
    return text


def load_folder(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    doc_id = 0
    for filename in all_files:
        full_path = os.path.join(folder_path, filename)
        text = load_pdf(full_path)
        chunks = [c.strip() for c in text.split("\n") if c.strip()]

        for chunk in chunks:
            vector = embedder.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[vector],
                metadatas=[{"source": filename}],
                ids=[str(doc_id)]
            )
            doc_id += 1

    print(f"تم تحميل {len(all_files)} ملف PDF بإجمالي {doc_id} chunk")


def search(question):
    vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=3
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n".join(
        f"[من ملف: {src}]\n{chunk}" for chunk, src in zip(chunks, sources)
    )
    return context


def ask_ai(context, question):
    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {"role": "system", "content": f"أجب فقط من هذه المعلومات، واذكر اسم الملف المصدر:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


def main():
    folder_path = input("أدخل مسار المجلد الذي يحتوي ملفات PDF: ")
    print("جاري تحميل الملفات...")
    load_folder(folder_path)
    print("جاهز — اسأل أي سؤال")
    print("---")

    while True:
        question = input("أنت: ")
        if question == "exit":
            break
        context = search(question)
        answer = ask_ai(context, question)
        print("AI:", answer)
        print("---")


main()