import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import tempfile

load_dotenv()

st.set_page_config(
    page_title="PDF 챗봇",
    page_icon="📄",
    layout="wide"
)


# ── 로그인 ──────────────────────────────────────────────────

def check_login(username, password):
    correct_user = st.secrets.get("LOGIN_USERNAME", "admin")
    correct_pass = st.secrets.get("LOGIN_PASSWORD", "1234")
    return username == correct_user and password == correct_pass


def login_page():
    st.title("🔐 PDF 챗봇")
    st.caption("로그인 후 이용하실 수 있습니다.")
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("아이디")
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", use_container_width=True)
            if submitted:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("아이디 또는 비밀번호가 틀렸습니다.")


if not st.session_state.get("logged_in"):
    login_page()
    st.stop()


# ── 메인 앱 ─────────────────────────────────────────────────

st.title("📄 PDF 문서 챗봇")
st.caption("PDF를 업로드하면 내용에 대해 질문할 수 있어요!")


@st.cache_resource(show_spinner="임베딩 모델 로딩 중...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def process_pdfs(uploaded_files):
    all_chunks = []
    total_pages = 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        total_pages += len(documents)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, total_pages


def get_api_key():
    user_key = st.session_state.get("user_api_key", "").strip()
    server_key = (os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")).strip()
    return user_key if user_key else server_key


def build_chain():
    api_key = get_api_key()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
        streaming=True
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 분석 전문 AI 어시스턴트입니다.
아래 문서 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 한국어로 답변하세요.
문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 말해주세요.

[문서 내용]
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return prompt | llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(messages):
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def stream_response(chain, retriever, question, messages):
    context_docs = retriever.invoke(question)
    context = format_docs(context_docs)
    history = format_history(messages)

    for chunk in chain.stream({
        "context": context,
        "question": question,
        "chat_history": history
    }):
        yield chunk.content

    return context_docs


# ── 사이드바 ────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 설정")

    # API 키 입력
    with st.expander("🔑 Groq API 키 입력 (선택)", expanded=not bool(get_api_key())):
        api_key_input = st.text_input(
            "API 키",
            type="password",
            placeholder="gsk_...",
            value=st.session_state.get("user_api_key", ""),
            help="키가 없으면 https://console.groq.com 에서 무료 발급"
        )
        if api_key_input:
            st.session_state.user_api_key = api_key_input.strip()
            st.success("키 저장됨")

    st.divider()
    st.header("📁 문서 업로드")

    uploaded_files = st.file_uploader(
        "PDF 파일을 선택하세요 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 문서 처리하기", use_container_width=True):
            if not get_api_key():
                st.error("API 키를 먼저 입력해주세요!")
            else:
                with st.spinner(f"{len(uploaded_files)}개 문서 분석 중..."):
                    vectorstore, total_pages = process_pdfs(uploaded_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    st.session_state.chain = build_chain()
                    st.session_state.messages = []
                    names = ", ".join(f.name for f in uploaded_files)
                    st.success(f"완료! {len(uploaded_files)}개 파일 / 총 {total_pages}페이지\n\n📄 {names}")

    if "vectorstore" in st.session_state:
        st.divider()
        st.success("✅ 문서 준비 완료")
        if st.button("🗑️ 초기화", use_container_width=True):
            for key in ["vectorstore", "retriever", "chain", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()
    if st.button("🚪 로그아웃", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── 채팅 영역 ───────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    if not get_api_key():
        st.warning("👈 왼쪽에서 **Groq API 키**를 입력하고 PDF를 업로드해주세요.")
        st.info("Groq API 키가 없다면 → https://console.groq.com 에서 무료로 발급받으세요!")
    else:
        st.info("👈 왼쪽에서 PDF 파일을 업로드하고 '문서 처리하기' 버튼을 눌러주세요!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("문서에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response_chunks = []

            def collect_and_stream():
                for chunk in stream_response(
                    st.session_state.chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.messages[:-1]
                ):
                    response_chunks.append(chunk)
                    yield chunk

            st.write_stream(collect_and_stream())
            full_response = "".join(response_chunks)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
