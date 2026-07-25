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

def load_pdf(path):
    pdf = fitz.open(path)
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def create_database(text):
    collection = chroma.create_collection("pdf")
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    for i, chunk in enumerate(chunks):
        vector = embedder.encode(chunk).tolist()
        collection.add(
            documents=[chunk],
            embeddings=[vector],
            ids=[str(i)]
        )
    return collection

def search(collection, question):
    vector = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[vector],
        n_results=2
    )
    return "\n".join(results["documents"][0])

def ask_ai(context, question):
    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {"role": "system", "content": f"أجب فقط من هذه المعلومات:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

def main():
    pdf_path = input("أدخل مسار ملف PDF: ")
    print("جاري تحميل الملف...")
    text = load_pdf(pdf_path)
    collection = create_database(text)
    print("جاهز — اسأل أي سؤال")
    print("---")

    while True:
        question = input("أنت: ")
        if question == "exit":
            break
        context = search(collection, question)
        answer = ask_ai(context, question)
        print("AI:", answer)
        print("---")

main()