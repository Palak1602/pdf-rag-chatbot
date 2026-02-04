import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

# --------------------------------------------------
# 🌈 PAGE CONFIG (UI ONLY)
# --------------------------------------------------
st.set_page_config(
    page_title="PDF Chatbot 🤖",
    page_icon="📘",
    layout="centered"
)

st.markdown("""
<h1 style="text-align:center;">📘 PDF Chatbot 🤖</h1>
<p style="text-align:center; color:gray;">
Upload any PDF and chat with it intelligently ✨
</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 🔐 LOAD ENV
# --------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "rag-bot2"
EMBEDDING_DIM = 768

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys in environment variables.")
    st.stop()

# --------------------------------------------------
# 🔌 PINECONE INIT
# --------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [i["name"] for i in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)
index_stats = index.describe_index_stats()
existing_namespaces = index_stats.get("namespaces", {})

# --------------------------------------------------
# 📂 SIDEBAR (UI ONLY)
# --------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Document Control")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if not uploaded_pdf:
    st.info("👈 Upload a PDF from the sidebar to begin")
    st.stop()

# --------------------------------------------------
# 📄 PDF INFO CARD
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
# 🧠 SESSION / NAMESPACE
# --------------------------------------------------
pdf_namespace = uploaded_pdf.name.replace(" ", "_").lower()

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = pdf_namespace

if st.session_state.active_pdf != pdf_namespace:
    st.session_state.active_pdf = pdf_namespace
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.cache_resource.clear()

# --------------------------------------------------
# 📦 VECTORSTORE
# --------------------------------------------------
@st.cache_resource
def load_vectorstore(uploaded_pdf, namespace):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if namespace in existing_namespaces and existing_namespaces[namespace]["vector_count"] > 0:
        return PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    documents = PyPDFLoader(pdf_path).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(documents)

    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

vectorstore = load_vectorstore(uploaded_pdf, pdf_namespace)

# 🔴 CHANGE 1: retrieve fewer, better chunks
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 20,
        "lambda_mult": 0.75
    }
)


# --------------------------------------------------
# 🤖 LLM
# --------------------------------------------------
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        temperature=0.1,
        max_new_tokens=200,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
)

# --------------------------------------------------
# 🧾 PROMPT (UNCHANGED)
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a PDF-based chatbot.\n"
            "Answer the question using ONLY the provided context.\n"
            "Be concise and factual.\n"
            "DO NOT guess or use outside knowledge.\n\n"
            "If the answer is not explicitly stated in the context, reply exactly:\n"
            "Answer not found in the document."
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)

# --------------------------------------------------
# 🧠 ANSWER FUNCTION
# --------------------------------------------------
def answer_question(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Page {d.metadata.get('page', '?') + 1}]\n{d.page_content}"
        for d in docs
    )

    response = llm.invoke(
        prompt.format(context=context, question=question)
    )

    answer_text = response.content.strip()

    if (
        "Answer not found" in answer_text
        or len(answer_text.split()) < 8
    ):
        return "Answer not found in the document."

    pages = []
    seen = set()
    for d in docs:
        p = d.metadata.get("page")
        if p is not None and p not in seen:
            pages.append(p + 1)
            seen.add(p)
        if len(pages) == 3:
            break

    sources = ", ".join(f"Page {p}" for p in pages)

    return f"{answer_text}\n\n📄 **Source:** {sources}"


# --------------------------------------------------
# 💬 CHAT UI
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything about the PDF 💬")

if query and st.session_state.pending_question is None:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    st.session_state.pending_question = query
    st.rerun()

if st.session_state.pending_question:
    with st.chat_message("assistant"):
        with st.spinner("📖 Searching document..."):
            answer = answer_question(st.session_state.pending_question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.session_state.pending_question = None
    st.rerun()
