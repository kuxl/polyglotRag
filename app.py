import os
import tempfile
import uuid

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from agent_graph import CogniRAGWorkflow
from guardrails import Guardrails
from rag_engine import RAGEngine

load_dotenv()

st.set_page_config(
    page_title="PolyglotRAG",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #eef3fb;
        --muted: #a0aec0;
        --line: #26344a;
        --accent: #79a9ff;
        --paper: #0b1120;
        --surface: #111a2b;
        --surface-soft: #151f32;
    }
    html, body, [class*="css"] { color: var(--ink); }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: var(--paper); }
    [data-testid="stHeader"] { background: rgba(11, 17, 32, .94); }
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1040px;
        padding: .8rem 1.5rem 5rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: .7rem;
        min-height: 2.7rem;
    }
    .brand-mark {
        display: grid;
        width: 2.55rem;
        height: 2.55rem;
        place-items: center;
        border: 1px solid #37527d;
        border-radius: .8rem;
        background: #182b49;
        color: #a9c7ff;
        font-size: 1.2rem;
        font-weight: 800;
    }
    .brand-name {
        color: #f3f6fc;
        font-size: 1.02rem;
        font-weight: 740;
    }
    .brand-caption {
        color: var(--muted);
        font-size: .72rem;
        margin-top: .1rem;
    }
    .section-kicker {
        color: #c5d0e0;
        font-size: .78rem;
        font-weight: 750;
        margin: .7rem 0 .4rem;
    }
    .hero {
        border: 1px solid #293852;
        border-radius: 1.1rem;
        padding: 1.45rem 1.7rem;
        margin-top: .8rem;
        background: linear-gradient(130deg, #151f32 0%, #111a2b 72%);
        box-shadow: 0 14px 40px rgba(0, 0, 0, .18);
    }
    .hero-eyebrow {
        color: #a9c7ff;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .04em;
        margin-bottom: .6rem;
    }
    .hero-title {
        color: #f4f7fc;
        font-size: clamp(1.9rem, 4vw, 2.7rem);
        font-weight: 760;
        letter-spacing: -.045em;
        line-height: 1.12;
        margin: 0;
    }
    .hero-copy {
        max-width: 660px;
        color: #aebbd0;
        font-size: .96rem;
        line-height: 1.65;
        margin: .65rem 0 0;
    }
    .status-strip {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin-top: 1rem;
        color: #aebbd0;
        font-size: .78rem;
        font-weight: 600;
    }
    .status-dot {
        width: .5rem;
        height: .5rem;
        border-radius: 50%;
        background: #4bd3a0;
        box-shadow: 0 0 12px rgba(75, 211, 160, .32);
    }
    .status-dot.waiting {
        background: #f0b75e;
        box-shadow: 0 0 12px rgba(240, 183, 94, .28);
    }
    .welcome-title {
        color: #f0f4fb;
        font-size: 1.25rem;
        font-weight: 720;
        margin: 0 0 .35rem;
    }
    .welcome-copy {
        color: #a0aec0;
        font-size: .94rem;
        line-height: 1.65;
        margin: 0;
    }
    .step-card {
        min-height: 7.7rem;
        border: 1px solid #293852;
        border-radius: .9rem;
        background: var(--surface);
        padding: 1rem 1.05rem;
    }
    .step-number {
        color: #a9c7ff;
        font-size: .7rem;
        font-weight: 800;
    }
    .step-title {
        color: #edf2fb;
        font-size: .94rem;
        font-weight: 700;
        margin: .4rem 0 .25rem;
    }
    .step-copy {
        color: #a0aec0;
        font-size: .81rem;
        line-height: 1.5;
    }
    .stButton > button,
    [data-testid="stPopover"] > button {
        min-height: 2.55rem;
        border: 1px solid #33425a;
        border-radius: .72rem;
        background: #172237;
        color: #eaf0fa;
        font-weight: 650;
    }
    .stButton > button:hover,
    [data-testid="stPopover"] > button:hover {
        border-color: #6388c0;
        background: #1b2a43;
        color: #fff;
    }
    [data-testid="stBaseButton-primary"] {
        border-color: #5987d1 !important;
        background: #315d9c !important;
        color: #fff !important;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #26354d;
        border-radius: .95rem;
        padding: .9rem 1rem;
        margin-bottom: .7rem;
        background: var(--surface);
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li { color: #e6edf7; }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #142137;
    }
    [data-testid="stBottomBlockContainer"] {
        box-sizing: border-box;
        width: 100%;
        max-width: 1040px;
        margin: 0 auto;
        padding: 0 1.5rem;
    }
    [data-testid="stChatInput"] {
        box-sizing: border-box;
        width: 100%;
        max-width: 100%;
        min-height: 3.5rem;
        border: 1px solid #33425a;
        border-radius: .85rem;
        background: #111a2b;
        box-shadow: 0 8px 26px rgba(0, 0, 0, .2);
    }
    [data-testid="stChatInput"] textarea {
        box-sizing: border-box;
        min-height: 3.25rem;
        max-height: 8rem;
        padding: .8rem 1rem;
        font-size: .95rem;
        line-height: 1.45;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input {
        color: #eef3fb;
        background: #111a2b;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #40516b;
        border-radius: .8rem;
        background: #111a2b;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span { color: #a0aec0; }
    [data-testid="stPopoverBody"],
    [data-testid="stExpander"] {
        border-color: #293852;
        border-radius: .85rem;
        background: var(--surface);
    }
    [data-testid="stPopoverBody"] {
        border: 1px solid #293852;
        color: var(--ink);
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #eaf0fa;
        font-weight: 650;
    }
    [data-testid="stAlert"] {
        border: 1px solid #293852;
        border-radius: .8rem;
        background: #142137;
        color: #eaf0fa;
    }
    [data-testid="stCaptionContainer"] {
        color: #9aa9be;
    }
    hr { border-color: var(--line); }
    [data-testid="stSpinner"] { color: #a9c7ff; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: .8rem 1rem 5rem; }
        [data-testid="stBottomBlockContainer"] { padding: 0 1rem; }
        .hero { padding: 1.35rem 1.2rem; }
        .hero-copy { font-size: .92rem; }
        [data-testid="stChatInput"] { border-radius: .75rem; }
        [data-testid="stChatInput"] textarea {
            min-height: 3rem;
            padding: .7rem .8rem;
            font-size: .9rem;
        }
    }
    :root {
        --ink: #f7f5ff;
        --muted: #c5bfdc;
        --line: #45347d;
        --accent: #9c58ff;
        --paper: #000000;
        --surface: #120d2d;
        --surface-soft: #1b1440;
    }
    html, body, [class*="css"] { color: var(--ink); }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: #000000 !important;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 1;
        max-width: 1120px;
        padding: 1rem 1.5rem 5rem;
    }
    .brand-lockup { justify-content: flex-end; }
    .stButton > button,
    [data-testid="stPopover"] > button {
        min-height: 2.8rem;
        border: 1px solid #554197;
        border-radius: .9rem;
        background: rgba(22, 16, 54, .88);
        color: #fff;
        font-weight: 650;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    .stButton > button:hover,
    [data-testid="stPopover"] > button:hover {
        border-color: #b477ff;
        background: #21164e;
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(111, 48, 255, .22);
    }
    [data-testid="stBaseButton-primary"] {
        border: 1px solid #8a51ff !important;
        background: linear-gradient(105deg, #8950ff, #4b1ce7) !important;
        color: #fff !important;
    }
    .hero {
        margin: 1.15rem auto 1.65rem;
        padding: .5rem 0 0;
        border: 0;
        background: transparent;
        box-shadow: none;
        text-align: center;
    }
    .bot-mark {
        display: block;
        width: 116px;
        height: 110px;
        margin: 0 auto .15rem;
        filter: drop-shadow(0 0 18px rgba(107, 68, 255, .25));
    }
    .hero-title {
        color: #fbfaff;
        font-size: clamp(2.35rem, 5vw, 3.75rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.1;
        margin: 0;
    }
    .hero-title span {
        color: #b66cff;
        background: linear-gradient(90deg, #ab73ff, #d259ff);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-copy {
        max-width: none;
        color: #d3cfe2;
        font-size: clamp(1rem, 2.2vw, 1.35rem);
        line-height: 1.45;
        margin: .55rem 0 0;
    }
    .upload-shell {
        display: flex;
        justify-content: center;
        width: 100%;
        text-align: center;
    }
    .welcome-header {
        max-width: 720px;
        margin: 1.2rem auto 0;
        text-align: center;
    }
    div[class*="st-key-upload-card"],
    div[class*="st-key-upload-card"] > div,
    div[data-testid="stFileUploader"],
    div[data-testid="stFileUploader"] > section,
    div[data-testid="stFileUploader"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[class*="st-key-upload-card"] {
        width: min(100%, 720px);
        margin: 0 auto;
        padding: 1.35rem 1.5rem 1.25rem;
        border: 2px dashed rgba(255, 255, 255, .9);
        border-radius: 1.35rem;
        background: linear-gradient(145deg, rgba(20, 13, 49, .88), rgba(13, 10, 35, .9));
        box-shadow: 0 0 38px rgba(255, 255, 255, .08), inset 0 0 36px rgba(255, 255, 255, .05);
        text-align: center;
    }
    .upload-mark {
        width: 68px;
        height: 60px;
        margin: 0 auto .15rem;
        display: block;
        filter: drop-shadow(0 0 12px rgba(139, 77, 255, .3));
    }
    .upload-title {
        margin: .1rem auto .25rem;
        color: #f7f4ff;
        font-size: 1.55rem;
        font-weight: 760;
        text-align: center;
    }
    .upload-copy {
        margin: 0 auto .65rem;
        color: #d1cbe2;
        font-size: 1rem;
        text-align: center;
        max-width: 100%;
    }
    div[class*="st-key-upload-card"] [data-testid="stFileUploader"] {
        max-width: 100%;
        margin: 0 auto;
        text-align: center;
    }
    div[class*="st-key-upload-card"] [data-testid="stFileUploader"] section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 4.1rem;
        padding: .35rem .7rem;
        border: 1px solid rgba(255, 255, 255, .8);
        border-radius: .95rem;
        background: rgba(17, 14, 38, .82) !important;
        box-shadow: none !important;
        text-align: center;
    }
    div[class*="st-key-upload-card"] [data-testid="stFileUploader"] small,
    div[class*="st-key-upload-card"] [data-testid="stFileUploader"] span {
        color: #d2cbe4;
        text-align: center;
    }
    div[class*="st-key-upload-card"] [data-testid="stFileUploader"] button {
        border: 1px solid rgba(255, 255, 255, .9);
        border-radius: 999px;
        background: linear-gradient(105deg, rgba(255,255,255,.18), rgba(255,255,255,.06));
        color: white;
        font-weight: 700;
        margin: 0 auto;
        box-shadow: none !important;
    }
    .supported-formats {
        margin: .45rem auto .7rem;
        width: 100%;
        color: #c8c1da;
        font-size: .9rem;
        text-align: center;
    }
    div[class*="st-key-upload-card"] .stButton > button {
        min-height: 3.15rem;
        padding: 0 1.7rem;
        border-radius: 999px;
        font-size: 1.1rem;
        margin: 0 auto !important;
        display: block !important;
        width: fit-content !important;
        min-width: 220px !important;
        max-width: 100% !important;
        text-align: center;
    }
    div[class*="st-key-upload-card"] .stButton {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
    .status-strip {
        justify-content: center;
        margin-top: .8rem;
        color: #c7c0d8;
    }
    .welcome-title { margin-top: 1rem; color: #f0ebff; text-align: center; }
    .welcome-copy { color: #c5bfdc; text-align: center; }
    div[class*="st-key-top-actions"] {
        position: fixed !important;
        top: .5rem;
        right: 4rem;
        z-index: 1000;
        width: min(390px, calc(100vw - 5rem));
        background: rgba(0, 0, 0, .92);
        backdrop-filter: blur(10px);
        padding: .35rem 0 .45rem;
    }
    @media (max-width: 700px) {
        div[class*="st-key-top-actions"] {
            right: 3.25rem;
            width: calc(100vw - 4.25rem);
        }
    }
    [data-testid="stChatMessage"] {
        border-color: #4a347d;
        background: rgba(21, 15, 51, .92);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(40, 25, 88, .92);
    }
    [data-testid="stBottomBlockContainer"] {
        position: sticky;
        bottom: 0;
        z-index: 25;
        padding-bottom: .65rem;
        background: transparent !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }
    [data-testid="stBottom"] {
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] {
        border: 1px solid rgba(131, 98, 217, 0.8);
        border-radius: 1.15rem;
        background: rgba(21, 17, 41, 0.96);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input {
        color: #f7f5ff;
        background: transparent;
        border-radius: 1rem;
    }
    [data-testid="stPopoverBody"],
    [data-testid="stExpander"] {
        border-color: #45347d;
        background: #130e31;
    }
    [data-testid="stAlert"] {
        border-color: #137b5e;
        background: rgba(10, 57, 48, .85);
        color: #f1fff8;
    }
    [data-testid="stCaptionContainer"] { color: #c0b9d4; }
    [data-testid="stBottomBlockContainer"] {
        z-index: 2;
        max-width: 1120px;
        padding-bottom: .45rem;
        background: transparent !important;
    }
    .wave-footer {
        display: none;
    }
    .wave-footer svg { width: 100%; height: 100%; }
    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding: .65rem 1rem 5rem; }
        .hero { margin: .6rem auto 1.1rem; }
        .bot-mark { width: 90px; height: 84px; }
        .hero-title { font-size: clamp(2rem, 9vw, 2.8rem); }
        div[class*="st-key-upload-card"] { padding: 1rem .85rem; }
        .upload-title { font-size: 1.3rem; }
        .wave-footer { height: 90px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

for key in ("GROQ_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(key):
        try:
            secret = st.secrets.get(key)
        except StreamlitSecretNotFoundError:
            break
        if secret:
            os.environ[key] = str(secret)

if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY. Add it to .env or .streamlit/secrets.toml to use the app.")
    st.stop()


def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.current_thread_id = new_id
    st.session_state.threads[new_id] = {
        "title": "New chat",
        "messages": [],
        "rag_engine": RAGEngine(new_id),
        "has_doc": False,
        "pinned": False,
    }
    st.session_state.workflow = CogniRAGWorkflow(
        st.session_state.threads[new_id]["rag_engine"]
    )


def switch_chat(thread_id):
    st.session_state.current_thread_id = thread_id
    st.session_state.workflow = CogniRAGWorkflow(
        st.session_state.threads[thread_id]["rag_engine"]
    )


def delete_chat(thread_id):
    if thread_id in st.session_state.threads:
        del st.session_state.threads[thread_id]
    if st.session_state.current_thread_id == thread_id:
        create_new_chat()


def toggle_pin(thread_id):
    thread = st.session_state.threads[thread_id]
    thread["pinned"] = not thread.get("pinned", False)


def rename_chat(thread_id, new_title):
    if new_title.strip():
        st.session_state.threads[thread_id]["title"] = new_title.strip()


if "threads" not in st.session_state:
    st.session_state.threads = {}

if "current_thread_id" not in st.session_state:
    create_new_chat()

if st.session_state.current_thread_id not in st.session_state.threads:
    create_new_chat()

current_thread = st.session_state.threads[st.session_state.current_thread_id]
rag_engine = current_thread["rag_engine"]
workflow = st.session_state.workflow

with st.container(key="top-actions"):
    new_chat_col, history_col = st.columns([1.35, 1.8], vertical_alignment="center")
    with new_chat_col:
        st.button(
            "＋  New Chat",
            use_container_width=True,
            on_click=create_new_chat,
            type="primary",
        )
    with history_col:
        with st.popover("◷  Generate History", use_container_width=True):
            st.markdown("#### Recent chats")
            sorted_threads = sorted(
                st.session_state.threads.items(),
                key=lambda item: (
                    item[1].get("pinned", False),
                    item[1]["title"].casefold(),
                ),
                reverse=True,
            )
            for thread_id, thread_data in sorted_threads:
                is_active = thread_id == st.session_state.current_thread_id
                prefix = "📌 " if thread_data.get("pinned") else ""
                title_text = f"{'● ' if is_active else ''}{prefix}{thread_data['title']}"
                title_col, options_col = st.columns([5, 1], vertical_alignment="center")
                with title_col:
                    st.button(
                        title_text,
                        key=f"conversation_{thread_id}",
                        use_container_width=True,
                        on_click=switch_chat,
                        args=(thread_id,),
                        type="primary" if is_active else "secondary",
                    )
                with options_col:
                    with st.popover("⋯", use_container_width=True):
                        st.markdown("**Chat options**")
                        with st.form(f"rename_form_{thread_id}"):
                            new_title = st.text_input(
                                "Title",
                                value=thread_data["title"],
                                key=f"rename_{thread_id}",
                            )
                            if st.form_submit_button("Save title", use_container_width=True):
                                rename_chat(thread_id, new_title)
                                st.rerun()
                        st.button(
                            "Unpin" if thread_data.get("pinned") else "Pin",
                            key=f"pin_{thread_id}",
                            on_click=toggle_pin,
                            args=(thread_id,),
                            use_container_width=True,
                        )
                        st.button(
                            "Delete chat",
                            key=f"delete_{thread_id}",
                            on_click=delete_chat,
                            args=(thread_id,),
                            use_container_width=True,
                            type="secondary",
                        )

st.write("")

st.markdown(
    """
    <section class="hero">
        <svg class="bot-mark" viewBox="0 0 120 112" fill="none" aria-hidden="true">
            <defs>
                <linearGradient id="botStroke" x1="15" y1="30" x2="105" y2="95" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#6ED9FF"/><stop offset="1" stop-color="#852AFF"/>
                </linearGradient>
            </defs>
            <path d="M60 9v14" stroke="url(#botStroke)" stroke-width="6" stroke-linecap="round"/>
            <circle cx="60" cy="8" r="8" fill="url(#botStroke)"/>
            <rect x="14" y="31" width="92" height="66" rx="31" stroke="url(#botStroke)" stroke-width="11"/>
            <path d="M40 64c0-6 4-10 9-10s9 4 9 10M70 64c0-6 4-10 9-10s9 4 9 10" stroke="#75D7FF" stroke-width="5" stroke-linecap="round"/>
            <path d="m105 20 8-12m2 25 12-7m-8 22 13 2" stroke="#9D3DFF" stroke-width="4" stroke-linecap="round"/>
            <path d="M9 55c-8 2-8 18 0 20M111 55c8 2 8 18 0 20" stroke="#7256FF" stroke-width="5" stroke-linecap="round"/>
        </svg>
        <h1 class="hero-title">PolyglotRAG</h1>
        <p class="hero-copy">Chat, analyze, and get insights from your files</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
with st.container(key="upload-card"):
    st.markdown(
        """
        <svg class="upload-mark" viewBox="0 0 72 62" fill="none" aria-hidden="true">
            <defs>
                <linearGradient id="uploadStroke" x1="9" y1="10" x2="62" y2="53" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#71D8FF"/><stop offset="1" stop-color="#862CFF"/>
                </linearGradient>
            </defs>
            <path d="M22 48h-6a12 12 0 0 1-1-24 20 20 0 0 1 38-3 14 14 0 0 1 4 27h-7" stroke="url(#uploadStroke)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M36 50V31m0 0-9 9m9-9 9 9" stroke="#9352FF" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="upload-title">Upload a File</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "csv", "txt", "docx", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        _, button_col, _ = st.columns([1, 1.2, 1])
        with button_col:
            if st.button(
                "✧  Process Document",
                type="primary",
                use_container_width=False,
            ):
                temp_dirs = []
                temp_paths = []
                try:
                    for uploaded_file in uploaded_files:
                        temp_dir = tempfile.TemporaryDirectory()
                        temp_dirs.append(temp_dir)
                        temp_path = os.path.join(
                            temp_dir.name,
                            os.path.basename(uploaded_file.name),
                        )
                        with open(temp_path, "wb") as file:
                            file.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)

                    with st.spinner("Reading and indexing your documents..."):
                        chunks = rag_engine.process_documents(temp_paths)
                finally:
                    for temp_dir in temp_dirs:
                        temp_dir.cleanup()

                if chunks:
                    current_thread["has_doc"] = True
                    if current_thread["title"] == "New chat":
                        current_thread["title"] = uploaded_files[0].name[:32]
                    st.success(f"✅ Ready to explore · {chunks} text sections indexed")
                    st.rerun()
                else:
                    st.error("No readable text was found in the selected documents.")

    if current_thread["has_doc"]:
        st.success("Document processed successfully! You can now ask questions about your document.")
    if not os.getenv("TAVILY_API_KEY"):
        st.caption("Web search is off. Add TAVILY_API_KEY to enable it.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="wave-footer" aria-hidden="true">
        <svg viewBox="0 0 1440 150" preserveAspectRatio="none">
            <defs>
                <linearGradient id="waveFill" x1="0" y1="0" x2="0" y2="150" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#4B13D6" stop-opacity=".7"/>
                    <stop offset="1" stop-color="#14064A" stop-opacity=".12"/>
                </linearGradient>
            </defs>
            <path d="M0 54C145 20 205 138 354 112S534 74 680 119s250 18 369-17 248-5 391-61v109H0Z" fill="url(#waveFill)"/>
            <path d="M0 103c132-81 225 31 353 23s196-87 329-57 206 70 333 39 276-67 425-50v92H0Z" fill="#3310A0" fill-opacity=".52"/>
            <path d="M0 116c139-73 221-26 329-6s200-12 313-10 213 47 328 31 291-71 470-52" stroke="#793EFF" stroke-opacity=".62" stroke-width="3" fill="none"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if current_thread["messages"]:
    st.markdown(f"### 💬 {current_thread['title']}")
    for message in current_thread["messages"]:
        avatar = "📄" if message["role"] == "assistant" else "🙂"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources · {len(message['sources'])}"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
elif current_thread["has_doc"]:
    st.markdown(
        """
        <div class="welcome-header">
            <div class="welcome-title">✅ Your document is ready</div>
            <p class="welcome-copy">
                Ask a question about its contents using the message box below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

query = None
if current_thread["has_doc"]:
    query = st.chat_input("💬 Ask a question about your document...")

if query:
    security_check = Guardrails.inspect_input(query)
    if not security_check.get("is_safe", False):
        st.error(f"That question was blocked: {security_check.get('reason')}")
        st.stop()

    current_thread["messages"].append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching your source and preparing an answer..."):
            state = workflow.app.invoke(
                {
                    "query": query,
                    "documents": [],
                    "generation": "",
                    "used_web": False,
                }
            )

        documents = state.get("documents", [])
        used_web = state.get("used_web", False)
        if used_web:
            st.caption("This answer used web search because a relevant passage was not found in your documents.")

        response_text = state.get("generation", "")
        st.markdown(response_text)

        sources = [
            f"{document.metadata.get('source', 'Web source')} "
            f"(Page {document.metadata.get('page', '')})".rstrip()
            for document in documents
        ]
        current_thread["messages"].append(
            {
                "role": "assistant",
                "content": response_text,
                "sources": sources,
            }
        )

        if sources:
            with st.expander(f"Sources · {len(sources)}"):
                for source in sources:
                    st.markdown(f"- {source}")
