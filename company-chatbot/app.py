import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def log_complaint(complaint_text):
    with open("complaints.txt", "a", encoding="utf-8") as f:
        f.write(complaint_text + "\n---\n")
    return "تم تسجيل الشكوى بنجاح"

def book_table(people_count, time):
    with open("reservations.txt", "a", encoding="utf-8") as f:
        f.write(f"عدد الأشخاص: {people_count} - الوقت: {time}\n---\n")
    return "تم حجز الطاولة بنجاح"

tools = [
    {
        "type": "function",
        "function": {
            "name": "log_complaint",
            "description": "يسجل شكوى العميل في النظام عندما يبلغ عن مشكلة في الطلب أو الخدمة",
            "parameters": {
                "type": "object",
                "properties": {
                    "complaint_text": {
                        "type": "string",
                        "description": "نص الشكوى كما ذكرها العميل"
                    }
                },
                "required": ["complaint_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_table",
            "description": "يحجز طاولة بالمطعم عندما يطلب العميل حجز أو مكان لعدد أشخاص بوقت معين",
            "parameters": {
                "type": "object",
                "properties": {
                    "people_count": {
                        "type": "integer",
                        "description": "عدد الأشخاص المطلوب الحجز لهم"
                    },
                    "time": {
                        "type": "string",
                        "description": "الوقت المطلوب للحجز، مثلاً 7 مساءً"
                    }
                },
                "required": ["people_count", "time"]
            }
        }
    }
]

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
        model="anthropic/claude-3-haiku",
        messages=messages,
        tools=tools,
        max_tokens=500
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)

            if call.function.name == "log_complaint":
                result = log_complaint(args["complaint_text"])
                confirmation = "تم تسجيل شكواك، شكرًا لتواصلك معنا."
            elif call.function.name == "book_table":
                result = book_table(args["people_count"], args["time"])
                confirmation = f"تم حجز طاولة لعدد {args['people_count']} الساعة {args['time']}."
            else:
                confirmation = "تم تنفيذ الطلب."

        messages.append({"role": "assistant", "content": confirmation})
        print("طازج:", confirmation)
        print("---")
        continue

    reply = msg.content
    print("طازج:", reply)
    print("---")

    messages.append({"role": "assistant", "content": reply})