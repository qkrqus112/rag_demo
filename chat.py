import uuid
import streamlit as st
from dotenv import load_dotenv

from llm import get_ai_response

load_dotenv()

st.set_page_config(page_title="UiPath 챗봇", page_icon="🤖")

st.title("🤖 UiPath 챗봇")
st.caption("UiPath 테스트 자동화와 관련된 궁금한 내용들을 물어보세요!")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(
    placeholder="UiPath 테스트 자동화와 관련된 궁금한 내용들을 말씀해주세요!"
):
    st.session_state.message_list.append({
        "role": "user",
        "content": user_question,
    })

    with st.chat_message("user"):
        st.write(user_question)

    try:
        with st.spinner("답변을 생성하는 중입니다"):
            ai_response = get_ai_response(
                user_question,
                session_id=st.session_state.session_id,
            )

            with st.chat_message("assistant"):
                ai_message = st.write_stream(ai_response)

        st.session_state.message_list.append({
            "role": "assistant",
            "content": ai_message,
        })

    except Exception as e:
        error_message = f"답변 생성 중 오류가 발생했습니다: {e}"

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.message_list.append({
            "role": "assistant",
            "content": error_message,
        })