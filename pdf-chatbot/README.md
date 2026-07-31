# PDF Q&A Chatbot (RAG + Hybrid Search)

بوت ذكاء اصطناعي يجاوب على أسئلتك اعتمادًا على محتوى أي ملف PDF — باستخدام تقنية RAG (Retrieval-Augmented Generation) مع بحث هجين (Hybrid Search).

🔗 **جرّب المشروع مباشرة:** [رابط Streamlit هنا]

## كيف يشتغل
1. يقرأ ملف PDF ويستخرج النص (PyMuPDF)
2. يقسم النص لأجزاء ذكية باستخدام RecursiveCharacterTextSplitter
3. يحول كل جزء لـ vector (sentence-transformers) ويخزنه بـ ChromaDB (تخزين دائم)
4. عند السؤال، يجمع بين البحث الدلالي (semantic) والبحث الكلمي (BM25) للحصول على أدق نتائج (Hybrid Search)
5. يرسل النتائج كـ context لنموذج Claude (عبر OpenRouter) للإجابة

## الميزات التقنية
- **Hybrid Search**: يجمع البحث الدلالي والكلمي لتحسين دقة الاسترجاع
- **Persistent Storage**: البيانات تُحفظ على القرص، مو بالذاكرة المؤقتة
- **Error Handling**: معالجة شاملة للأخطاء (ملفات مفقودة، فشل الاتصال)
- **Logging**: تسجيل كل الأحداث بملف log منفصل
- **واجهة Streamlit**: تجربة تفاعلية كاملة بدون Terminal

## التقنيات المستخدمة
- Python, Streamlit
- PyMuPDF, ChromaDB, Sentence Transformers
- LangChain (Text Splitting), rank-bm25
- OpenAI SDK (عبر OpenRouter) + Claude 3 Haiku

## التشغيل محليًا
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
