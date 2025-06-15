from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 환경 변수 로드를 위해 추가합니다. (Streamlit 앱 시작 시 이미 로드될 수도 있지만, 명시적으로)
import os
from dotenv import load_dotenv

# answer_examples는 config.py 파일에 정의되어 있다고 가정합니다.
from config import answer_examples

# .env 파일 로드 (Streamlit 앱이 시작될 때 한 번만 호출되도록 관리 필요)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 대화 히스토리를 저장할 인메모리 스토어입니다.
# 실제 서비스에서는 Redis, 데이터베이스 등을 사용해야 합니다.
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    주어진 session_id에 해당하는 대화 히스토리를 반환합니다.
    새로운 세션이면 ChatMessageHistory를 생성하여 저장합니다.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    """
    OpenAI 임베딩과 FAISS를 사용하여 리트리버를 생성합니다.
    FAISS는 로컬 디렉토리 'faiss_akis'에 저장됩니다.
    """
    embedding = OpenAIEmbeddings(model='text-embedding-3-large', openai_api_key=OPENAI_API_KEY)
    persist_directory = './faiss_akis'
    try:
        database = FAISS.load_local(
            folder_path=persist_directory,
            embeddings=embedding,
            allow_dangerous_deserialization=True  # ✅ 추가!
        )
        if len(database.index_to_docstore_id) == 0:
            print(f"경고: FAISS '{persist_directory}'에 문서가 없습니다.")
    except Exception as e:
        print(f"FAISS 로드 중 오류 발생: {e}")
        raise e

    return database.as_retriever(search_kwargs={'k': 4})

def get_history_retriever():
    """
    채팅 히스토리를 고려하여 검색 쿼리를 재구성하는 리트리버를 생성합니다.
    """
    llm = get_llm() # get_llm 함수는 openai_api_key를 내부적으로 처리합니다.
    retriever = get_retriever() # get_retriever 함수는 openai_api_key를 내부적으로 처리합니다.
    
    # 채팅 히스토리를 바탕으로 사용자의 질문을 독립적인 질문으로 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = (
        "주어진 채팅 기록과 최신 사용자 질문을 고려하여, "
        "채팅 기록의 맥락을 참조할 수 있는 독립적인 질문을 만드세요. "
        "질문에 답하지 말고, 필요한 경우에만 질문을 재구성하고 그렇지 않으면 있는 그대로 반환하세요."
    )

    # 대화 히스토리 인식 프롬프트 템플릿
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 이전 대화 기록이 삽입될 위치
            ("human", "{input}"), # 현재 사용자 질문이 삽입될 위치
        ]
    )
    
    # 채팅 히스토리 기반 검색 쿼리 재구성 리트리버 생성
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm(model='gpt-4o'):
    """
    OpenAI Chat 모델을 로드합니다. API 키를 명시적으로 전달합니다.
    """
    llm = ChatOpenAI(model=model, openai_api_key=OPENAI_API_KEY)
    return llm


def get_dictionary_chain():
    """
    사용자 질문을 사전(dictionary)을 참고하여 변경하는 체인을 생성합니다.
    예: '우리 회사' -> 'AK아이에스'
    """
    dictionary_rules = ["회사를 나타내는 표현 -> AK아이에스"] # 사전에 미리 정의된 규칙 리스트
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.

        사전: {dictionary_rules}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser() # 프롬프트 -> LLM -> 문자열 파싱
    
    return dictionary_chain


def get_rag_chain():
    """
    RAG(Retrieval Augmented Generation) 체인을 생성합니다.
    채팅 히스토리, Few-shot 예시, 사용자 정의 프롬프트, 리트리버를 통합합니다.
    """
    llm = get_llm()
    
    # Few-shot 예시를 위한 프롬프트 템플릿입니다.
    # `config.py`의 `answer_examples`를 사용합니다.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples, # `config.answer_examples` 사용
    )
    
    # 시스템 프롬프트: 챗봇의 역할, 지침, context 플레이스홀더 포함
    system_prompt = (
        "당신은 AK아이에스 전문가입니다. 복리후생, 승진 마일리지 제도 관한 질문에 답변해주세요.\n"
        "아래에 제공된 문서를 활용해서 답변해주시고,\n"
        "답변을 알 수 없다면 모른다고 답변해주세요.\n"
        "답변을 제공할때에는 이모티콘을 활용해 친근하게 답변해주세요! 😊\n"
        "\n"
        "{context}" # 검색된 문서(context)가 삽입될 위치
    )
    
    # RAG의 최종 답변을 생성하기 위한 프롬프트입니다.
    # 시스템 프롬프트, Few-shot 예시, 채팅 히스토리, 사용자 질문을 포함합니다.
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt, # Few-shot 예시 포함
            MessagesPlaceholder("chat_history"), # 이전 대화 기록이 삽입될 위치
            ("human", "{input}"), # 사용자 질문이 삽입될 위치
        ]
    )
    
    # 채팅 히스토리 인지 리트리버를 가져옵니다.
    history_aware_retriever = get_history_retriever()
    
    # 검색된 문서와 질문을 사용하여 최종 답변을 생성하는 체인입니다.
    # 변수명을 'Youtube_chain'에서 더 적절한 'Youtube_generator_chain'으로 변경했습니다.
    Youtube_generator_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 리트리버와 답변 생성 체인을 결합하여 RAG 체인을 만듭니다.
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_generator_chain)
    
    # 대화 히스토리를 관리하며 RAG 체인을 실행할 수 있는 RunnableWithMessageHistory를 생성합니다.
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history, # 세션 히스토리를 가져오는 함수
        input_messages_key="input", # 사용자 질문을 위한 키
        history_messages_key="chat_history", # 대화 히스토리를 위한 키
        # output_messages_key는 Streamlit의 st.write_stream에 직접 전달하는 경우 필요하지 않을 수 있습니다.
        # 하지만 pick('answer')를 사용하므로 명시하는 것이 일관성을 높입니다.
        output_messages_key="answer", 
    ).pick('answer') # 최종적으로 'answer' 키의 결과만 반환하도록 설정

    return conversational_rag_chain


def get_ai_response(user_message):
    """
    사용자 메시지를 받아 전체 RAG 파이프라인을 실행하고 AI 응답을 스트림으로 반환합니다.
    사전 변환 체인과 RAG 체인을 결합하여 사용합니다.
    """
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    
    # 사전 변환 체인의 출력이 RAG 체인의 입력이 되도록 연결합니다.
    # dictionary_chain의 결과는 RAG chain의 'input'으로 들어갑니다.
    # LangChain Expression Language (LCEL)의 표현식 체인입니다.
    full_rag_pipeline = {"input": dictionary_chain} | rag_chain
    
    # 세션 ID를 'abc123'으로 고정하여 테스트합니다. 실제 애플리케이션에서는 사용자별로 동적으로 생성해야 합니다.
    # dictionary_chain의 입력은 "question"이고, 최종 rag_chain의 입력은 "input"입니다.
    # 여기서 'input'은 user_message가 아닌, dictionary_chain의 결과를 받게 됩니다.
    ai_response = full_rag_pipeline.stream(
        {
            "question": user_message # dictionary_chain으로 전달될 사용자 원본 질문
        },
        config={
            "configurable": {"session_id": "abc123"} # 세션 ID 설정 (RunnableWithMessageHistory용)
        },
    )

    return ai_response
