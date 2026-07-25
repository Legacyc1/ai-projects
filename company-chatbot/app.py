import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

SYSTEM_PROMPT = """
أنت مساعد خدمة عملاء لمطعم "طازج" للتوصيل.
- رد بأسلوب ودود ومختصر
- المعلومات المتاحة لك فقط:
  القائمة: برجر (25 ريال)، بيتزا (35 ريال)، سلطة (18 ريال)
  وقت التوصيل: 30-45 دقيقة
  مناطق التوصيل: الرياض فقط
- لو السؤال خارج هذا النطاق، اعتذر بلطف ووجه العميل للتواصل مع الفرع مباشرة
- لا تختلق معلومات غير معطاة لك أعلاه
"""

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

print("مرحبا بك في طازج 🍔 (اكتب exit للخروج)")
print("---")

while True:
    user_input = input("أنت: ")
    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="openrouter/auto",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("طازج:", reply)
    print("---")

    messages.append({"role": "assistant", "content": reply})