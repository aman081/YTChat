import os, re
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

# LangChain / RAG
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# ========= ENV =========
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ========= STREAMLIT CONFIG =========
st.set_page_config(
    page_title="Chat with YouTube (Multilingual + Timestamps + Firestore)",
    page_icon="🎬",
    layout="wide",
)

# ---- Premium Tech UI (indigo/violet) ----
st.markdown("""
<style>
:root {
  --text:#ecf2f8; --muted:#9aa4b2; --bg:#0b0f19; --card:#0e1523;
  --grad1:#6e56cf; --grad2:#8b5cf6; --border:rgba(255,255,255,.07);
}
html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
.stApp { background: radial-gradient(1200px 600px at 10% -20%, rgba(110,86,207,.1), transparent 70%), linear-gradient(180deg, #0b0f19, #0e1523); color: var(--text); }
.block-container { padding-top: 1.2rem; }

.header-card, .app-card, .side-card {
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.02));
  border:1px solid var(--border);
  border-radius:16px; padding:16px;
  box-shadow: 0 24px 60px rgba(0,0,0,.28);
}

.pill { display:inline-block; padding:6px 10px; border-radius:999px; font-size:12px;
  background: rgba(110,86,207,0.15); border:1px solid rgba(110,86,207,0.35); color:#c7c2ff; margin-right:6px; }

.badge { display:inline-block; padding:4px 8px; border-radius:10px; font-size:11px;
  background:#0b1322; border:1px solid rgba(255,255,255,0.08); color:#9fb4ff; margin-right:6px; }

.stButton>button {
  background: linear-gradient(90deg, var(--grad1), var(--grad2));
  color: white; border: 0; padding: 0.6rem 1rem;
  border-radius: 12px; font-weight: 600;
}
.stButton>button:hover { opacity: .96; transform: translateY(-1px); }

.stTextInput>div>div>input, .stSelectbox > div > div, .stTextArea>div>textarea {
  background: #0b1120 !important;
  color: var(--text)!important;
  border-radius: 12px!important;
  border: 1px solid var(--border)!important;
}

.chat-wrap { max-height: 62vh; overflow-y: auto; padding-right: 8px; scroll-behavior: smooth; }
.msg { padding: 12px 14px; margin: 10px 0; border-radius: 14px; border:1px solid var(--border); line-height:1.5; }
.msg-user { background: linear-gradient(180deg, rgba(139,92,246,.15), rgba(139,92,246,.08)); border-top-right-radius: 6px; }
.msg-bot { background: linear-gradient(180deg, rgba(30,41,59,.6), rgba(15,23,42,.65)); border-top-left-radius: 6px; }
.msg small { color: var(--muted) }

.conv-card { width: 100%; text-align: left; background: #0b1120; border:1px solid var(--border); border-radius:12px; padding:10px; }
.conv-card:hover { border-color: rgba(139,92,246,.6); }
.conv-thumbnail { width: 100%; border-radius:10px; border:1px solid var(--border); }
.thumbnail-row { display: grid; grid-template-columns: 78px 1fr; gap:10px; align-items:center; }
.mini { font-size: 12px; color: var(--muted) }
.title { font-weight:600; }
.link-btn a { text-decoration:none; color:#c7c2ff; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header-card'><h2>🎬 Chat with YouTube — multilingual, timestamped, and saved to Firestore</h2><div class='mini'>Paste a video link, choose transcript & chat languages, then ask anything. Answers include clickable timestamps.</div></div>", unsafe_allow_html=True)

# ========= FIREBASE =========
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error("❌ Firebase initialization failed. Ensure `serviceAccountKey.json` exists.")
        st.stop()
db = firestore.client()

# ========= SESSION =========
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_id" not in st.session_state: st.session_state.conversation_id = None
if "video_id" not in st.session_state: st.session_state.video_id = None
if "transcript_lang" not in st.session_state: st.session_state.transcript_lang = "auto"
if "chat_lang" not in st.session_state: st.session_state.chat_lang = "auto"
if "vector_ready" not in st.session_state: st.session_state.vector_ready = set()
if "transcript_items" not in st.session_state: st.session_state.transcript_items = []
if "transcript_text" not in st.session_state: st.session_state.transcript_text = ""

# ========= LANG / FLAGS =========
LANGUAGES = {
    "Auto (Auto-detect)": ("auto", "🌐"),
    "English": ("en", "🇬🇧"),
    "Hindi": ("hi", "🇮🇳"),
    "Tamil": ("ta", "🇮🇳"),
    "Spanish": ("es", "🇪🇸"),
    "French": ("fr", "🇫🇷"),
    "German": ("de", "🇩🇪"),
    "Arabic": ("ar", "🇦🇪"),
    "Chinese (Simplified)": ("zh-Hans", "🇨🇳"),
    "Chinese (Traditional)": ("zh-Hant", "🇨🇳"),
    "Japanese": ("ja", "🇯🇵"),
    "Korean": ("ko", "🇰🇷"),
}

# ========= HELPERS =========
def extract_video_id(url: str):
    if not url: return None
    q = urlparse(url)
    if q.hostname == "youtu.be":
        return q.path[1:]
    if q.hostname in ("www.youtube.com","youtube.com","m.youtube.com"):
        vid = parse_qs(q.query).get("v", [None])[0]
        if vid: return vid
        if "/shorts/" in q.path:
            return q.path.rstrip("/").split("/")[-1]
    if re.match(r"^[\w-]{11}$", url.strip()):
        return url.strip()
    return None

def yt_thumbnail_url(video_id: str):
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

def list_conversations():
    docs = db.collection("conversations").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    out=[]
    for d in docs:
        data=d.to_dict(); ts=data.get("timestamp")
        out.append((d.id, ts.strftime("%Y-%m-%d %H:%M") if ts else "Unknown",
                    data.get("context_preview","[No context]"),
                    data.get("video_id",""),
                    data.get("transcript_lang","auto"),
                    data.get("chat_lang","auto")))
    return out

def load_conversation(conversation_id: str):
    doc = db.collection("conversations").document(conversation_id).get()
    if not doc.exists: return False
    data=doc.to_dict(); msgs=[]
    for m in data.get("messages",[]):
        msgs.append(HumanMessage(content=m["content"]) if m["role"]=="human" else AIMessage(content=m["content"]))
    st.session_state.messages=msgs
    st.session_state.video_id=data.get("video_id")
    st.session_state.transcript_lang=data.get("transcript_lang","auto")
    st.session_state.chat_lang=data.get("chat_lang","auto")
    st.session_state.conversation_id=conversation_id
    return True

def save_conversation(messages, video_id, transcript_lang, chat_lang):
    preview=""
    for m in messages:
        if isinstance(m, HumanMessage):
            preview=m.content[:70]+("..." if len(m.content)>70 else ""); break
    cid = st.session_state.conversation_id or str(int(datetime.now().timestamp()))
    db.collection("conversations").document(cid).set({
        "messages":[{"role":"human" if isinstance(m,HumanMessage) else "ai","content":m.content} for m in messages],
        "video_id":video_id, "transcript_lang":transcript_lang, "chat_lang":chat_lang,
        "context_preview":preview, "timestamp":datetime.now()
    })
    st.session_state.conversation_id=cid

# ---- Transcript fetch (instance API) ----
def fetch_transcript_items(video_id: str, lang_code: str):
    ytt = YouTubeTranscriptApi()
    try:
        if lang_code=="auto":
            items = ytt.fetch(video_id)
        else:
            items = ytt.fetch(video_id, languages=[lang_code])
        return items
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
        # Silent fallback to English (your choice)
        return ytt.fetch(video_id, languages=["en"])

def join_items_text(items):
    def get_val(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)
    return " ".join(get_val(it,"text","") for it in items if get_val(it,"text",""))

# ---- Chunk transcript while keeping timestamps; robust to dict/objects ----
def chunk_transcript_with_timestamps(items, target_chars=1000, overlap_items=2):
    docs = []
    buf = []
    char_count = 0
    start_sec = None

    def get_val(obj, key, default=None):
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    for i, it in enumerate(items):
        text = (get_val(it, "text", "") or "").strip()
        if not text: continue

        s = float(get_val(it, "start", 0.0))
        d = float(get_val(it, "duration", 0.0))

        if start_sec is None:
            start_sec = s

        if char_count + len(text) > target_chars and buf:
            end_item = items[i - 1]
            end_s = float(get_val(end_item, "start", 0.0))
            end_d = float(get_val(end_item, "duration", 0.0))
            end_sec = end_s + end_d

            docs.append(Document(
                page_content=" ".join(buf),
                metadata={"start_sec": start_sec, "end_sec": end_sec}
            ))

            seed_i = max(0, i - overlap_items)
            buf = []
            char_count = 0
            start_sec = float(get_val(items[seed_i], "start", s))
            for j in range(seed_i, i):
                t = (get_val(items[j], "text", "") or "").strip()
                if t:
                    buf.append(t)
                    char_count += len(t)

        buf.append(text)
        char_count += len(text)

    if buf:
        last = items[-1]
        end_s = float(get_val(last, "start", 0.0))
        end_d = float(get_val(last, "duration", 0.0))
        docs.append(Document(
            page_content=" ".join(buf),
            metadata={"start_sec": float(start_sec or 0.0), "end_sec": end_s + end_d}
        ))

    return docs

def seconds_to_hhmmss(secs: float):
    s=int(secs); h=s//3600; m=(s%3600)//60; s=s%60
    if h>0: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def get_embeddings():
    # multilingual (100+ languages)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build_or_load_chroma(video_id: str, transcript_lang: str, docs: list[Document]):
    key = f"{video_id}_{transcript_lang}"
    base = os.path.join("db", key)
    os.makedirs(base, exist_ok=True)
    embeddings = get_embeddings()
    if os.path.exists(base) and os.listdir(base):
        return Chroma(persist_directory=base, embedding_function=embeddings), key
    vs = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=base)
    return vs, key

def detect_language_fast(text: str):
    try: return detect(text)
    except Exception: return "auto"

def translate_text(text: str, target_lang: str, llm):
    if not text or target_lang in ["auto", "unknown", ""]: return text
    try:
        src = detect(text)
        if src.lower().startswith(target_lang.lower()[:2]): return text
    except Exception: pass
    res = llm.invoke(f"Translate the following into {target_lang}. Return only the translated text:\n\n{text}")
    return res.content

def make_timestamp_link(video_id: str, start_sec: float):
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"

# ========= SIDEBAR (HISTORY) =========
with st.sidebar:
    st.markdown("<div class='side-card'><h4>💾 Your Conversations</h4>", unsafe_allow_html=True)
    convs = list_conversations()
    if convs:
        for cid, ts, preview, vid, t_lang, c_lang in convs:
            thumb = yt_thumbnail_url(vid) if vid else "https://via.placeholder.com/120x90?text=No+Thumb"
            t_badge = f"<span class='badge'>T:{t_lang.upper()}</span>" if t_lang else ""
            c_badge = f"<span class='badge'>C:{c_lang.upper()}</span>" if c_lang else ""
            html = f"""
            <div class='thumbnail-row'>
              <img class='conv-thumbnail' src='{thumb}' />
              <div>
                <div class='mini'>{ts}</div>
                <div class='title'>{preview}</div>
                <div class='mini'>{t_badge} {c_badge}</div>
              </div>
            </div>
            """
            if st.button("", key=f"conv_{cid}", help=f"{ts} • {preview}", use_container_width=True):
                if load_conversation(cid):
                    st.success("Loaded conversation.")
                    st.rerun()
            st.markdown(html, unsafe_allow_html=True)
            st.write("")
    else:
        st.info("No conversations yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🆕 New"):
            st.session_state.messages=[]; st.session_state.conversation_id=None; st.session_state.video_id=None
            st.session_state.transcript_items=[]; st.session_state.transcript_text=""
            st.success("Started a new conversation."); st.rerun()
    with c2:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages=[]; st.success("Cleared current chat."); st.rerun()

# ========= MAIN LAYOUT =========
left, right = st.columns([1.35, 1])

with left:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("### 🔗 Video & Languages")

    yt_url = st.text_input("Paste a YouTube URL or 11-char ID", value="", placeholder="https://www.youtube.com/watch?v=...")
    tl_name = st.selectbox("Transcript language", list(LANGUAGES.keys()), index=1, format_func=lambda k: f"{LANGUAGES[k][1]} {k}")
    cl_name = st.selectbox("Chat response language", list(LANGUAGES.keys()), index=0, format_func=lambda k: f"{LANGUAGES[k][1]} {k}")

    transcript_lang = LANGUAGES[tl_name][0]; chat_lang = LANGUAGES[cl_name][0]
    st.session_state.transcript_lang = transcript_lang; st.session_state.chat_lang = chat_lang

    process = st.button("▶️ Load video & prepare index")
    st.markdown("</div>", unsafe_allow_html=True)

    # Video header card + actions
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("### 🎞 Video")
    video_id = extract_video_id(yt_url) if yt_url else st.session_state.video_id
    if video_id:
        st.session_state.video_id = video_id
        tcol1, tcol2 = st.columns([1, 2.2])
        with tcol1:
            st.image(yt_thumbnail_url(video_id), use_container_width=True)
            st.markdown(f"<div class='link-btn'><a href='https://www.youtube.com/watch?v={video_id}' target='_blank'>Open on YouTube ↗</a></div>", unsafe_allow_html=True)
        with tcol2:
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.caption("Tip: change languages above and click **Load video & prepare index** to rebuild embeddings.")
    else:
        st.info("Paste a YouTube URL to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare vector store
    if process and video_id:
        with st.spinner("Fetching transcript & building index..."):
            try:
                items = fetch_transcript_items(video_id, transcript_lang)
                st.session_state.transcript_items = items
                st.session_state.transcript_text = join_items_text(items)

                docs = chunk_transcript_with_timestamps(items, target_chars=1000, overlap_items=2)
                for d in docs: d.metadata["video_id"]=video_id

                vs, key = build_or_load_chroma(video_id, transcript_lang, docs)
                st.session_state.vector_ready.add(key)
                st.success("Ready! Ask anything about the video. ✨")
            except Exception as e:
                st.error(f"Could not prepare this video: {e}")

    # Chat section
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("### 💬 Chat")
    st.markdown("<div class='chat-wrap'>", unsafe_allow_html=True)
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            st.markdown(f"<div class='msg msg-user'>🧑‍💻 <b>You</b><br>{m.content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg msg-bot'>🤖 <b>Assistant</b><br>{m.content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    user_question = st.text_input("Type your question", value="", placeholder="e.g., Summarize the key idea…")
    ask = st.button("Ask")

    if (ask or user_question) and user_question.strip():
        if not video_id:
            st.warning("Please paste a valid YouTube URL first.")
        else:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            key = f"{video_id}_{transcript_lang}"
            embeddings = get_embeddings()
            base = os.path.join("db", key)
            if key in st.session_state.vector_ready or (os.path.exists(base) and os.listdir(base)):
                vs = Chroma(persist_directory=base, embedding_function=embeddings)
            else:
                st.error("This video isn't prepared yet. Click 'Load video & prepare index' first.")
                vs=None

            if vs:
                # language selection
                question_lang = detect_language_fast(user_question) or "auto"
                answer_lang = chat_lang if chat_lang!="auto" else question_lang

                retriever = vs.as_retriever(search_kwargs={"k": 4})
                docs = retriever.invoke(user_question)
                context = "\n".join([d.page_content for d in docs]) if docs else ""
                context_trans = translate_text(context, answer_lang, llm) if context else ""

                prompt = PromptTemplate.from_template("""
You are a multilingual assistant helping with a YouTube video's transcript.
STRICT RULES:
- Answer ONLY using the provided context.
- If the answer is not in the context, reply: "I don't know based on the transcript."
- Write the final answer ONLY in the target language.

Target language: {answer_lang}

=== Context (already translated to target language or same language) ===
{context}

=== Question ===
{question}
""")
                final_prompt = prompt.format(context=context_trans or context,
                                             question=user_question,
                                             answer_lang=answer_lang)

                with st.spinner("Thinking..."):
                    resp = llm.invoke(final_prompt)

                st.session_state.messages.append(HumanMessage(content=user_question))
                st.session_state.messages.append(AIMessage(content=resp.content))

                # Render assistant response again (live)
                st.markdown(f"<div class='msg msg-bot'>🤖 <b>Assistant</b><br>{resp.content}</div>", unsafe_allow_html=True)

                # Timestamp citations (clickable)
                if docs:
                    st.markdown("**Sources:**")
                    uniq = []
                    for d in docs:
                        vid = d.metadata.get("video_id", video_id)
                        s = float(d.metadata.get("start_sec", 0.0))
                        e = float(d.metadata.get("end_sec", s))
                        label = f"{seconds_to_hhmmss(s)} → {seconds_to_hhmmss(e)}"
                        link = f"https://www.youtube.com/watch?v={vid}&t={int(s)}s"
                        entry = (vid, int(s), int(e))
                        if entry not in uniq:
                            uniq.append(entry)
                            st.markdown(f"- [{label}]({link})")

                # Persist conversation
                save_conversation(
                    st.session_state.messages,
                    video_id=video_id,
                    transcript_lang=transcript_lang,
                    chat_lang=chat_lang
                )
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Transcript viewer with search
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Transcript")
    with st.expander("Show / hide transcript"):
        q = st.text_input("Search within transcript (highlight only)")
        text = st.session_state.transcript_text or ""
        if q:
            try:
                pattern = re.compile(re.escape(q), re.IGNORECASE)
                highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", text)
                st.markdown(highlighted)
            except re.error:
                st.markdown(text)
        else:
            st.markdown(text if text else "_No transcript loaded yet._")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='side-card'>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Status & Tips")
    if st.session_state.video_id:
        st.markdown(f"<span class='badge'>Video: {st.session_state.video_id}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Transcript: {st.session_state.transcript_lang.upper()}</span> <span class='badge'>Chat: {st.session_state.chat_lang.upper()}</span>", unsafe_allow_html=True)
    st.markdown("""
- If your chosen transcript language isn't available, we **silently use English** and translate for you.
- **Multilingual embeddings** ensure questions in any language still find the right parts of the transcript.
- Answers include **clickable timestamps** that jump to the exact moment on YouTube.
- Your chats are saved in the **sidebar** with a short preview and thumbnail.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
