import os
from datetime import datetime
from typing import Any, Optional

import httpx
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_KEY_HEADER = "X-API-Key"
MAX_CONTEXT_MESSAGES = 12
REQUEST_MAX_TOKENS = 2000


class ApiError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def request_json(
    method: str,
    path: str,
    api_key: Optional[str] = None,
    json_body: Optional[dict[str, Any]] = None,
    params: Optional[dict[str, Any]] = None,
) -> Any:
    headers = None
    if api_key:
        headers = {API_KEY_HEADER: api_key}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method,
                f"{BACKEND_URL}{path}",
                headers=headers,
                json=json_body,
                params=params,
            )
    except httpx.RequestError as exc:
        raise ApiError("Backend is not available. Please check the API service.") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        message = payload.get("error") or payload.get("detail") or response.text
        if response.status_code == 503:
            message = "Backend is temporarily unavailable. Please retry later."
        raise ApiError(message, response.status_code)

    if not response.content:
        return None
    return response.json()


def show_error(exc: ApiError) -> None:
    prefix = ""
    if exc.status_code:
        prefix = f"HTTP {exc.status_code}: "
    st.error(f"{prefix}{exc.message}")


def init_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "email" not in st.session_state:
        st.session_state.email = ""
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None


def render_health() -> None:
    try:
        health = request_json("GET", "/health")
    except ApiError as exc:
        show_error(exc)
        return

    st.success(
        f"Backend: {health['status']} | DB: {health['database']} | Model loaded: {health['model_loaded']}"
    )


def create_api_key(user_id: str) -> str:
    api_key = request_json(
        "POST",
        f"/users/{user_id}/api-keys",
        json_body={"name": "streamlit-ui"},
    )
    return api_key["token"]


def continue_with_account(username: str, email: str) -> None:
    cleaned_username = username.strip()
    cleaned_email = email.strip().lower()

    try:
        user = request_json("GET", "/users/by-email", params={"email": cleaned_email})
        st.info("Existing user loaded.")
    except ApiError as exc:
        if exc.status_code != 404:
            raise

        try:
            user = request_json(
                "GET",
                "/users/by-username",
                params={"username": cleaned_username},
            )
            st.info("Existing user loaded by username.")
        except ApiError as username_exc:
            if username_exc.status_code != 404:
                raise

            user = request_json(
                "POST",
                "/users",
                json_body={"username": cleaned_username, "email": cleaned_email},
            )
            st.success("New user created.")

    st.session_state.user_id = user["id"]
    st.session_state.username = user["username"]
    st.session_state.email = user["email"]
    st.session_state.api_key = create_api_key(user["id"])
    load_sessions()


def load_sessions() -> None:
    if not st.session_state.user_id:
        return
    if not st.session_state.api_key:
        return

    sessions = request_json(
        "GET",
        f"/users/{st.session_state.user_id}/sessions",
        api_key=st.session_state.api_key,
    )
    st.session_state.sessions = sessions


def load_history(session_id: int) -> None:
    history = request_json(
        "GET",
        f"/users/{st.session_state.user_id}/sessions/{session_id}/chat-history",
        api_key=st.session_state.api_key,
        params={"limit": 50},
    )

    messages = []
    for item in reversed(history):
        messages.append({"role": "user", "content": item["user_prompt"]})
        messages.append({"role": "assistant", "content": item["assistant_prompt"]})
    st.session_state.messages = messages


def create_new_chat() -> None:
    if not st.session_state.user_id:
        st.warning("Enter your account first.")
        return
    if not st.session_state.api_key:
        st.warning("Enter your account first.")
        return

    title = "New chat " + datetime.now().strftime("%H:%M:%S")
    session = request_json(
        "POST",
        f"/users/{st.session_state.user_id}/sessions",
        api_key=st.session_state.api_key,
        json_body={"title": title},
    )

    st.session_state.session_id = session["id"]
    st.session_state.selected_session_id = session["id"]
    st.session_state.messages = []
    load_sessions()


def render_account() -> None:
    st.sidebar.subheader("Account")

    with st.sidebar.form("account_form"):
        username = st.text_input("Username", value=st.session_state.username or "Chang Le")
        email = st.text_input("Email", value=st.session_state.email or "chang@spbu.ru")
        submitted = st.form_submit_button("Continue")

    if submitted:
        if not username.strip() or not email.strip():
            st.sidebar.warning("Username and email are required.")
            return

        with st.spinner("Preparing account"):
            try:
                continue_with_account(username.strip(), email.strip())
            except ApiError as exc:
                show_error(exc)

    if st.session_state.user_id:
        st.sidebar.caption(f"Current user: {st.session_state.email}")


def render_sessions() -> None:
    st.sidebar.subheader("Chats")

    if st.sidebar.button("New chat", use_container_width=True):
        with st.spinner("Creating new chat"):
            try:
                create_new_chat()
            except ApiError as exc:
                show_error(exc)

    if not st.session_state.sessions:
        return

    session_ids = []
    session_titles = {}
    for session in st.session_state.sessions:
        session_ids.append(session["id"])
        session_titles[session["id"]] = session["title"]

    if st.session_state.selected_session_id not in session_ids:
        if st.session_state.session_id in session_ids:
            st.session_state.selected_session_id = st.session_state.session_id
        else:
            st.session_state.selected_session_id = session_ids[0]

    selected_index = session_ids.index(st.session_state.selected_session_id)

    st.sidebar.radio(
        "History",
        session_ids,
        index=selected_index,
        format_func=lambda session_id: session_titles[session_id],
        key="selected_session_id",
        label_visibility="collapsed",
    )
    selected_session_id = st.session_state.selected_session_id

    if st.session_state.session_id != selected_session_id:
        st.session_state.session_id = selected_session_id
        with st.spinner("Loading chat history"):
            try:
                load_history(selected_session_id)
            except ApiError as exc:
                show_error(exc)


def render_developer_info() -> None:
    with st.sidebar.expander("Developer info"):
        st.text_input("User ID", key="user_id")
        st.text_input("API key", key="api_key", type="password")
        st.write("Active session:", st.session_state.session_id)

        if st.session_state.last_stats:
            st.divider()
            st.write("Last response stats")
            st.metric("Context characters", st.session_state.last_stats["context_chars"])
            st.metric("Output characters", st.session_state.last_stats["output_chars"])
            st.metric("Messages sent", st.session_state.last_stats["message_count"])
            st.bar_chart(
                {
                    "context": [st.session_state.last_stats["context_chars"]],
                    "output": [st.session_state.last_stats["output_chars"]],
                }
            )


def render_chat() -> None:
    if not st.session_state.user_id:
        st.info("Enter your username and email in the sidebar to start.")
        return

    if not st.session_state.session_id:
        st.info("Click New chat to start a conversation.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Send a message")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    recent_messages = st.session_state.messages[-MAX_CONTEXT_MESSAGES:]

    payload_messages = []
    for message in recent_messages:
        payload_messages.append(
            {
                "role": message["role"],
                "message": message["content"],
            }
        )

    context_chars = 0
    for message in payload_messages:
        context_chars = context_chars + len(message["message"])

    payload = {
        "session_id": int(st.session_state.session_id),
        "messages": payload_messages,
        "temperature": 0.8,
        "max_tokens": REQUEST_MAX_TOKENS,
    }

    with st.chat_message("assistant"):
        with st.spinner("Generating response"):
            try:
                result = request_json(
                    "POST",
                    "/chat",
                    api_key=st.session_state.api_key,
                    json_body=payload,
                )
            except ApiError as exc:
                show_error(exc)
                return

        answer = result["response"]
        st.write(answer)
        st.session_state.last_stats = {
            "context_chars": context_chars,
            "output_chars": len(answer),
            "message_count": len(payload_messages),
        }

    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(page_title="LLM Chat", layout="wide")
    init_state()

    st.title("LLM Chat")
    render_health()

    render_account()
    render_sessions()
    render_developer_info()
    render_chat()


if __name__ == "__main__":
    main()
