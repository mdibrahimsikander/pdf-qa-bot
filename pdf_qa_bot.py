import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="wide"
)

# ── API key check ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in Streamlit secrets.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

# ── Helper functions ──────────────────────────────────────────────────
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embedding=embeddings)

def get_answer(question, vectorstore, chat_history):
    # Retrieve top 4 relevant chunks
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # Build conversation history string
    history_text = ""
    for msg in chat_history[-6:]:  # last 3 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Build prompt
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided PDF content.
If the answer is not found in the context, say "I couldn't find this information in the document."

Previous conversation:
{history_text}

Context from PDF:
{context}

Question: {question}

Answer:"""

    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        groq_api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=1024
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content, docs

# ── UI: Header ────────────────────────────────────────────────────────
st.title("📄 PDF Question Answering Bot")
st.caption("Upload a PDF and ask anything about its content — powered by Groq (Llama 4 Scout) + FAISS")

# ── UI: Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            # New PDF uploaded — reset everything
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.processed = False
            st.session_state.pdf_name = uploaded_file.name

        if not st.session_state.processed:
            with st.spinner("Processing PDF..."):
                raw_text = extract_text(uploaded_file)
                if not raw_text.strip():
                    st.error("❌ Could not extract text. Please use a text-based PDF.")
                else:
                    chunks = chunk_text(raw_text)
                    st.session_state.vectorstore = build_vectorstore(chunks)
                    st.session_state.processed = True
                    st.success(f"✅ Ready! {len(chunks)} chunks indexed.")

    if st.session_state.processed:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("📂 Upload New PDF"):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.processed = False
            st.session_state.pdf_name = ""
            st.rerun()

    st.divider()
    st.markdown("**Stack**")
    st.markdown("- 🤖 Llama 4 Scout via Groq")
    st.markdown("- 🔢 all-MiniLM-L6-v2 embeddings")
    st.markdown("- 🗄️ FAISS vector store")
    st.markdown("- 🔗 LangChain")
    st.markdown("- 🖥️ Streamlit")

# ── UI: Main chat ─────────────────────────────────────────────────────
if not st.session_state.processed:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.markdown("""
    **How it works:**
    1. Upload any text-based PDF
    2. The PDF is split into chunks and indexed using FAISS
    3. Ask any question — relevant chunks are retrieved
    4. Llama 4 Scout generates an accurate answer based on those chunks
    """)
else:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and show answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = get_answer(
                        user_input,
                        st.session_state.vectorstore,
                        st.session_state.chat_history
                    )
                    st.markdown(answer)

                    if source_docs:
                        with st.expander("📎 Source chunks used"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}...")

                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})