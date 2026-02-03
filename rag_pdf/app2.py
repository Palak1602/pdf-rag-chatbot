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
    chunk_size=450,
    chunk_overlap=100
).split_documents(docs)



    return PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

vectorstore = load_vectorstore(uploaded_pdf, NAMESPACE)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
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
        """You are a strict PDF reading assistant.

RULES (MANDATORY):
- Answer ONLY using the provided context
- Every answer MUST mention the page number(s) used
- Use this format at the end:
  Sources: Page X, Page Y
- If the answer is not clearly stated, say:
  "The document does not explicitly answer this question."
- Do NOT guess or infer
- Do NOT merge unrelated sections
"""
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])



def format_docs(docs):
    formatted = []
    for d in docs:
        page = d.metadata.get("page", "unknown")
        formatted.append(
            f"[PAGE {page}]\n{d.page_content}"
        )
    return "\n\n".join(formatted)


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
            docs_used = retriever.get_relevant_documents(query)
pages_used = sorted(
    set(d.metadata.get("page", "?") for d in docs_used)
)

response = rag_chain.invoke(query)

if not response or len(response.strip()) < 30:
    response = (
        "The provided document does not contain enough "
        "clear information to answer this question accurately."
    )

response += f"\n\n📄 **Sources:** Pages {', '.join(map(str, pages_used))}"


        # ✅ CONFIDENCE GUARD (ADD THIS)
if not response or len(response.strip()) < 25:
            response = (
                "The provided document does not contain enough "
                "clear information to answer this question accurately."
            )

st.markdown(response)


st.session_state.messages.append({"role": "assistant", "content": response})
