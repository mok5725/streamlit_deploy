"""
OpenAI Chat API + Streamlit 챗봇 (스트리밍).
Streamlit Community Cloud 배포 시 Secrets에 OPENAI_API_KEY를 설정하세요.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

import streamlit as st
from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


def get_api_key() -> str:
    """Cloud: Streamlit Secrets. 로컬: .streamlit/secrets.toml 또는 .env / 환경 변수."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]


def main() -> None:
    st.set_page_config(
        page_title="OpenAI 챗봇",
        page_icon="💬",
        layout="centered",
    )

    init_session()

    st.title("OpenAI 챗봇")
    st.caption("Streamlit + OpenAI Chat Completions (스트리밍)")

    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API 키가 없습니다.")
        st.info(
            "로컬: 프로젝트 폴더에 `.env`를 만들고 `OPENAI_API_KEY=sk-...` 를 넣거나 "
            "(`.env.example` 참고), `.streamlit/secrets.toml`에 동일 키를 넣으세요.\n\n"
            "Streamlit Cloud: 앱 Settings → Secrets에 "
            '`OPENAI_API_KEY = "sk-..."` 형식으로 추가하세요.'
        )
        st.stop()

    with st.sidebar:
        st.subheader("설정")
        model = st.selectbox(
            "모델",
            options=[
                DEFAULT_MODEL,
                "gpt-4o",
            ],
            index=0,
            help="계정에 켜진 모델만 사용 가능합니다. 다른 모델은 코드에서 options에 추가하세요.",
        )
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        if st.button("대화 초기화"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.rerun()

    client = OpenAI(api_key=api_key)

    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("메시지를 입력하세요…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            stream = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full += delta.content
                    placeholder.markdown(full + "▌")
            placeholder.markdown(full)

        st.session_state.messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
