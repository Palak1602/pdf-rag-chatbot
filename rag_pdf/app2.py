import os
import tempfile
from urllib import response
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 🔐 ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
INDEX_NAME = "rag-bot2"

# --------------------------------------------------
# 🌈 PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Chatbot 🤖",
    page_icon="📘",
    layout="centered"
)

# --------------------------------------------------
# 💖 HEADER
# --------------------------------------------------
st.markdown("""
<h1 style="text-align:center;"> PDF Chatbot🤖</h1>
<p style="text-align:center;color:gray;">
Upload any PDF & chat with it intelligently ✨
</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 📂 SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Document Control")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("🔄 Re-upload PDF"):
        st.session_state.clear()
        st.rerun()

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

if not uploaded_pdf:
    st.info("👈 Upload a PDF from the sidebar to begin")
    st.stop()

# --------------------------------------------------
# 📄 PDF INFO
# --------------------------------------------------
st.markdown(
    f"""
    <div style="padding:10px;border-radius:10px;background:#f8f9fa;">
    📄 <b>{uploaded_pdf.name}</b><br>
    📦 Size: {round(uploaded_pdf.size / 1024, 2)} KB
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 🧠 SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# unique namespace
NAMESPACE = f"{uploaded_pdf.name}_{uploaded_pdf.size}".replace(" ", "_").replace(".", "_").lower()

# --------------------------------------------------
# 🔌 PINECONE
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# --------------------------------------------------
# 🧠 EMBEDDINGS (768 DIM)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# --------------------------------------------------
# 📦 VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(_file, namespace):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file.getvalue())
        path = tmp.name

    docs = PyPDFLoader(path).load()

    chunks = RecursiveCharacterTextSplitter(
    chunk_size=500,      # ⬅️ was 800
    chunk_overlap=120
).split_documents(docs)


    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

vectorstore = load_vectorstore(uploaded_pdf, NAMESPACE)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 7,          # ⬅️ was 5
        "fetch_k": 25,   # ⬅️ was 15
        "lambda_mult": 0.75
    }
)


# --------------------------------------------------
# 🤖 LLM
# --------------------------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.2,
        max_new_tokens=300,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
)

# --------------------------------------------------
# 🧾 PROMPT
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict document-based assistant.

CRITICAL RULES:
- You MUST answer ONLY from the provided context
- If the answer is not clearly supported, say:
  "The provided document does not clearly answer this."
- Do NOT guess, assume, or infer missing facts
- Do NOT invent chapters, page numbers, events, or details
- If multiple interpretations exist, say so clearly

STYLE:
- Clear
- Factual
- Direct
"""
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )

rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# 💬 CHAT UI
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about the PDF 💬")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("📖 Reading..."):
            response = rag_chain.invoke(query)

        # ✅ CONFIDENCE GUARD (ADD THIS)
        if not response or len(response.strip()) < 25:
            response = (
                "The provided document does not contain enough "
                "clear information to answer this question accurately."
            )

        st.markdown(response)


    st.session_state.messages.append({"role": "assistant", "content": response})
