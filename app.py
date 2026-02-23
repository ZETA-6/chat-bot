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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile

load_dotenv()

# 키에 섞인 공백/특수문자 제거
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY").strip()

st.set_page_config(
    page_title="PDF 챗봇",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF 문서 챗봇")
st.caption("PDF를 업로드하면 내용에 대해 질문할 수 있어요!")


@st.cache_resource(show_spinner="임베딩 모델 로딩 중...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.unlink(tmp_path)
    return vectorstore, len(documents)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_history(chat_history):
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def get_answer(chain, retriever, question, chat_history):
    context_docs = retriever.invoke(question)
    context = format_docs(context_docs)
    history = format_history(chat_history)

    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": history
    })
    return answer, context_docs


def build_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 문서 분석 전문 AI 어시스턴트입니다.
아래 문서 내용을 바탕으로 사용자의 질문에 정확하고 친절하게 답변하세요.
문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 말해주세요.

[문서 내용]
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


# 사이드바 - PDF 업로드
with st.sidebar:
    st.header("📁 문서 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 선택하세요", type="pdf")

    if uploaded_file:
        if st.button("📥 문서 처리하기", use_container_width=True):
            with st.spinner("문서를 분석하는 중..."):
                vectorstore, page_count = process_pdf(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.chain = build_chain()
                st.session_state.messages = []
                st.success(f"완료! 총 {page_count}페이지 처리됨")

    if "vectorstore" in st.session_state:
        st.divider()
        st.success("✅ 문서 준비 완료")
        if st.button("🗑️ 초기화", use_container_width=True):
            for key in ["vectorstore", "retriever", "chain", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()


# 메인 채팅 영역
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
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
            with st.spinner("답변 생성 중..."):
                answer, source_docs = get_answer(
                    st.session_state.chain,
                    st.session_state.retriever,
                    prompt,
                    st.session_state.messages[:-1]
                )
                st.write(answer)

                with st.expander("📚 참고한 문서 내용"):
                    for i, doc in enumerate(source_docs, 1):
                        st.caption(f"[{i}] 페이지 {doc.metadata.get('page', '?') + 1}")
                        st.text(doc.page_content[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
