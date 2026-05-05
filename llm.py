import os
from typing import Dict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples


# =========================================================
# 1. 환경 변수
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")


# =========================================================
# 2. 세션 히스토리
# =========================================================

store: Dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# =========================================================
# 3. LLM
# =========================================================

def get_llm(model: str = "gpt-4o", temperature: float = 0):
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
    )


# =========================================================
# 4. Retriever
# =========================================================

def get_retriever():
    persist_directory = "./faiss_uipath"

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"FAISS 인덱스 디렉토리가 없습니다: {persist_directory}"
        )

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY,
    )

    database = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=embedding,
        allow_dangerous_deserialization=True,
    )

    if len(database.index_to_docstore_id) == 0:
        print(f"경고: FAISS '{persist_directory}'에 문서가 없습니다.")

    return database.as_retriever(search_kwargs={"k": 4})


# =========================================================
# 5. 문서 포맷팅
# =========================================================

def format_docs(docs):
    if not docs:
        return "관련 문서를 찾을 수 없습니다."

    result = []

    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        source = metadata.get("source", "UiPath Test Suite 문서")
        page = metadata.get("page", metadata.get("page_number", "N/A"))

        result.append(
            f"[문서 {i + 1}]\n"
            f"- source: {source}\n"
            f"- page: {page}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(result)


# =========================================================
# 6. 사전 보정 체인
# =========================================================

def get_dictionary_chain():
    llm = get_llm()

    dictionary_rules = """
- uipath -> UiPath
- 유아이패스 -> UiPath
- 테스트스위트 -> UiPath Test Suite
- 테스트 스위트 -> UiPath Test Suite
- 오케스트레이터 -> Orchestrator
- 테스트 매니저 -> Test Manager
- 스튜디오 프로 -> Studio Pro
- 알엠 -> ALM
- 씨아이씨디 -> CI/CD
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
너는 사용자 질문을 UiPath Test Suite 문서 검색에 적합하게 정규화하는 역할을 합니다.

규칙:
- 사용자의 의도는 유지하세요.
- 아래 사전에 있는 표현만 자연스럽게 보정하세요.
- 답변하지 말고 보정된 질문만 출력하세요.
- 변경이 필요 없으면 원문을 그대로 출력하세요.

사전:
{dictionary_rules}
"""),
        ("human", """
질문:
{question}

보정된 질문:
""")
    ])

    return prompt | llm | StrOutputParser()


# =========================================================
# 7. 히스토리 기반 검색 질문 재작성 체인
# =========================================================

def get_query_rewrite_chain():
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
너는 UiPath Test Suite 문서 검색을 위한 질문 재작성 전문가입니다.

역할:
- 최신 질문과 대화 히스토리를 참고해 검색에 적합한 독립 질문으로 재작성합니다.
- 질문이 이미 독립적이면 그대로 유지합니다.
- 답변하지 말고 재작성된 검색 질문만 출력합니다.
- 너무 짧은 키워드만 출력하지 말고 검색 의미가 살아있는 문장으로 작성합니다.

문서 주요 주제:
- UiPath Test Suite
- Studio Pro
- Orchestrator
- Test Manager
- RPA Testing
- Mobile Testing
- API Testing
- Data-Driven Testing
- ALM 통합
- CI/CD 통합
- 테스트 자동화
- 테스트 케이스 관리
- 테스트 실행 및 스케줄링
- 테스트 결과 분석 및 리포팅
"""),
        MessagesPlaceholder("chat_history"),
        ("human", """
최신 질문:
{input}

검색 질문:
""")
    ])

    return prompt | llm | StrOutputParser()


# =========================================================
# 8. RAG 체인
# =========================================================

def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()
    rewrite_chain = get_query_rewrite_chain()

    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{answer}"),
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = """
당신은 친근한 UiPath Test Suite 문서 기반 챗봇 비서입니다. 😊

답변 규칙:
1. 반드시 참고 문서(Context)에 있는 내용만 기반으로 답변하세요.
2. Context에 없는 내용이거나 UiPath Test Suite와 관련 없는 질문이면 "답변 할 수 없습니다."라고 답변하세요.
3. 다양한 이모티콘을 적절히 사용하되, 업무적으로 명확하게 답변하세요.
4. 가능하면 Studio Pro, Orchestrator, Test Manager 역할을 구분해서 설명하세요.
5. 답변은 핵심 요약 후 필요한 경우 항목별로 정리하세요.
6. 문서에서 확인되는 내용 이상으로 추측하지 마세요.

참고 문서:
{context}
"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    def retrieve_context(inputs):
        """
        중요:
        - ChatPromptTemplate에는 반드시 dict를 넣는다.
        - rewrite_chain에는 {"input": ..., "chat_history": ...} 형태로 넣는다.
        - retriever에는 string만 넣는다.
        - 최종적으로 context 문자열만 반환한다.
        """
        input_text = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])

        rewritten_query = rewrite_chain.invoke({
            "input": input_text,
            "chat_history": chat_history,
        }).strip()

        print(f"DEBUG 원본 질문: {input_text}")
        print(f"DEBUG 검색 질문: {rewritten_query}")

        docs = retriever.invoke(rewritten_query)
        return format_docs(docs)

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_context),
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversational_rag_chain


# =========================================================
# 9. 전체 응답 함수 - Streamlit streaming용
# =========================================================

def get_ai_response(user_message: str, session_id: str = "abc123"):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    def normalize_input(inputs):
        normalized_question = dictionary_chain.invoke({
            "question": inputs["question"]
        }).strip()

        return {
            "input": normalized_question
        }

    full_rag_pipeline = RunnableLambda(normalize_input) | rag_chain

    return full_rag_pipeline.stream(
        {"question": user_message},
        config={
            "configurable": {
                "session_id": session_id
            }
        },
    )


# =========================================================
# 10. 단건 응답 함수
# =========================================================

def get_ai_response_once(user_message: str, session_id: str = "abc123") -> str:
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    def normalize_input(inputs):
        normalized_question = dictionary_chain.invoke({
            "question": inputs["question"]
        }).strip()

        return {
            "input": normalized_question
        }

    full_rag_pipeline = RunnableLambda(normalize_input) | rag_chain

    return full_rag_pipeline.invoke(
        {"question": user_message},
        config={
            "configurable": {
                "session_id": session_id
            }
        },
    )
