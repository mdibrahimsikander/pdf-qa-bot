# 📄 PDF Question Answering Bot

An AI-powered chatbot that lets you upload any PDF and ask questions about its content. Built with LangChain, FAISS, Groq (Llama 4 Scout), and Streamlit.

---

## 🚀 Live Demo

> [Click here to try the app](https://pdf-app-bot-tw5xb2qcdpcyhc3puava7z.streamlit.app/)

---

## ✨ Features

- 📤 Upload any text-based PDF
- 🔍 Semantic search using FAISS vector store
- 🤖 Answers powered by Llama 4 Scout via Groq
- 💬 Conversation memory — follow-up questions work naturally
- 📎 Source chunk preview for every answer
- ⚡ Fast and free to use

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 4 Scout (via Groq API) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector Store | FAISS |
| Framework | LangChain |
| Language | Python 3.10+ |

---

## 🏗️ How It Works

```
PDF Upload
    ↓
Extract text (PyPDF2)
    ↓
Split into chunks (1000 chars, 200 overlap)
    ↓
Generate embeddings (all-MiniLM-L6-v2)
    ↓
Store in FAISS index
    ↓
User asks a question
    ↓
Retrieve top 4 relevant chunks (FAISS similarity search)
    ↓
Send chunks + question to Llama 4 Scout (Groq)
    ↓
Display answer + source previews
```

## ⚠️ Limitations

- Works best with **text-based PDFs** (not scanned/image PDFs)
- Use your CV for the best result
- Scanned PDFs require OCR (not included in this version)
- Groq free tier: 30 requests/min, 14,400 requests/day

---

## 🧑‍💻 Author

**Ibrahim**
- GitHub: [@mdibrahimsikander](https://github.com/mdibrahimsikander)
- LinkedIn: [mdibrahimsikander](https://www.linkedin.com/in/mdibrahimsikander/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
