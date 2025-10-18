
# primary_gpt_teacher_app_streamlit.py
import os
import time
import streamlit as st
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Config / Keys
# ----------------------------
GROQ_API_KEY = os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")
BASE_DIR = os.path.dirname(__file__)
DB_FAISS_PATH = os.path.join(BASE_DIR, "db_faiss")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------
# Simple Teaching Prompt Template
# ----------------------------
prompt_template = """
You are a teacher. The students will ask you questions about their life.
Use the following piece of context to answer the question.
If you don't know the answer, just say you don't know.
You answer with concise answers and do not leave any sentence hanging.
Context: {context}
Question: {question}
Answer:
""".strip()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", prompt_template),
    ("human", "{question}")
])

# ----------------------------
# ChatBot Class
# ----------------------------
class ChatBot:
    def __init__(self):
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"})
        except Exception as e:
            print(f"[ERROR] Failed to load embeddings: {e}")
            self.embeddings = None

        # Load FAISS DB
        self.db = None
        if self.embeddings and os.path.exists(DB_FAISS_PATH):
            try:
                self.db = FAISS.load_local(DB_FAISS_PATH, self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"[WARN] Failed to load FAISS DB: {e}. Retrieval will be limited.")
        else:
            print("[WARN] FAISS DB folder not found; retrieval will be limited.")

        # Initialize LLM
        try:
            self.llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.1-8b-instant",
                streaming=False,
                temperature=0.2,
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize LLM: {e}")
            self.llm = None

        # BM25 setup
        self.doc_texts, self.bm25 = [], None
        self._build_bm25_index()

        # Build chain if possible
        if self.llm:
            self.chain = PROMPT | self.llm | StrOutputParser()
        else:
            self.chain = None

    def _build_bm25_index(self):
        if self.db:
            try:
                all_docs = list(getattr(self.db.docstore, "_dict", {}).values())
                self.doc_texts = [d.page_content for d in all_docs]
                if self.doc_texts:
                    self.bm25 = BM25Okapi([t.split() for t in self.doc_texts])
            except Exception as e:
                print(f"[WARN] BM25 index failed: {e}")

    def hybrid_retrieve_and_rerank(self, query, top_k=5):
        if not self.db or not self.doc_texts:
            return ""
        try:
            docs = self.db.as_retriever(search_kwargs={"k": 20}).invoke(query)
            return "\n\n---\n\n".join([d.page_content for d in docs[:top_k]])
        except:
            return ""

    def generate_answer(self, query: str, history: list) -> str:
        last_turns = history[-4:] if history else []
        formatted_history = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in last_turns])
        context = self.hybrid_retrieve_and_rerank(query)
        if self.chain:
            try:
                return self.chain.invoke({"context": context, "question": query}).strip()
            except Exception as e:
                return f"Error generating response: {e}"
        return "LLM not initialized; cannot generate answer."

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Primary GPT – Student Companion", layout="centered", page_icon="🎓")

st.title("🎓 Primary GPT – Student Companion Assistant")
st.caption("A guided learning chatbot that answers questions based on context.")

# Persistent bot + chat history
if "bot" not in st.session_state:
    st.session_state.bot = ChatBot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["assistant"])

# User input
if prompt := st.chat_input("Ask a question or start studying..."):
    st.session_state.messages.append({"user": prompt, "assistant": ""})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = ""
        bot = st.session_state.bot
        full_reply = bot.generate_answer(prompt, [(m["user"], m["assistant"]) for m in st.session_state.messages[:-1]])

        # Typing animation
        for ch in full_reply:
            response_text += ch
            message_placeholder.markdown(response_text + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(response_text)

    st.session_state.messages[-1]["assistant"] = response_text

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()



