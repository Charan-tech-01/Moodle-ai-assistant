"""
Moodle AI Assistant — Streamlit UI
Backend API base URL: configure BASE_URL below.
All endpoints match the original HTML:
  GET  /user-context/{user_id}
  POST /ask              { user_id, query, format }
  POST /mentor/assign    { actor_user_id, student_id, mentor_name, mentor_email, mentor_phone }
"""

import streamlit as st
import requests
import json
from datetime import datetime

# ── Config ────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"   # ← change to your backend URL

EXAMPLES = [
    "Show my student profile",
    "Who is my class teacher?",
    "Who is my mentor and how can I contact them?",
    "Show attendance report for Cloud Computing",
    "List backlog students",
    "How many students are in Machine Learning?",
    "Explain neural networks",
]

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Moodle AI · NMIT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">

<style>
/* ── Reset & tokens ── */
:root {
  --ink:      #0c0f1a;
  --ink-2:    #1e2235;
  --ink-3:    #3a3f55;
  --muted:    #8b90a8;
  --border:   #e4e6ef;
  --surface:  #f5f6fa;
  --white:    #ffffff;
  --accent:   #2347f5;
  --accent-2: #1535c8;
  --gold:     #e8a020;
  --green:    #13a05e;
  --red:      #d93c3c;
  --radius:   14px;
  --radius-lg:20px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  color: var(--ink);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stAppDeployButton { display: none; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── App background ── */
.stApp {
  background: #eef0f8 !important;
}

/* ── Top nav bar ── */
.topnav {
  background: var(--ink);
  color: white;
  padding: 0 28px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 999;
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.topnav-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.topnav-logo {
  width: 34px; height: 34px;
  background: var(--accent);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px; font-weight: 700;
  color: white;
  font-family: 'Instrument Serif', Georgia, serif;
  font-style: italic;
  flex-shrink: 0;
}
.topnav-name { font-size: 14px; font-weight: 600; letter-spacing: -0.2px; }
.topnav-sub  { font-size: 11px; color: rgba(255,255,255,.45); letter-spacing:.06em; text-transform:uppercase; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--white) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
  font-size: 13px; color: var(--ink-3); line-height: 1.6;
}

/* ── Cards ── */
.card {
  background: var(--white);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
  padding: 18px 20px;
  margin-bottom: 14px;
}
.card-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .09em;
  color: var(--muted);
  margin-bottom: 12px;
}

/* ── Profile ── */
.profile-name {
  font-family: 'Instrument Serif', Georgia, serif;
  font-size: 22px;
  color: var(--ink);
  margin-bottom: 3px;
  line-height: 1.2;
}
.profile-meta {
  font-size: 12.5px;
  color: var(--muted);
  margin-bottom: 14px;
}
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.stat-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 11px 13px;
}
.stat-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing:.06em; margin-bottom:4px; }
.stat-value { font-size: 19px; font-weight: 700; color: var(--ink); letter-spacing: -0.5px; }
.stat-value.good { color: var(--green); }
.stat-value.warn { color: var(--gold); }
.stat-value.bad  { color: var(--red); }

/* ── Contact blocks ── */
.contact-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 8px;
}
.contact-role { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing:.08em; color: var(--muted); margin-bottom: 5px; }
.contact-name { font-size: 14px; font-weight: 600; color: var(--ink); margin-bottom: 3px; }
.contact-info { font-size: 12px; color: var(--muted); line-height: 1.7; }

/* ── Overview boxes ── */
.ov-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.ov-box-dark { background: var(--ink); border-radius: 12px; padding: 14px 16px; }
.ov-box-light { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.ov-label { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; }
.ov-box-dark  .ov-label { color: rgba(255,255,255,.45); }
.ov-box-light .ov-label { color: var(--muted); }
.ov-value {
  font-size: 28px; font-weight: 700; letter-spacing: -1.5px; margin-top: 6px;
  font-family: 'Instrument Serif', Georgia, serif;
}
.ov-box-dark  .ov-value { color: white; }
.ov-box-light .ov-value { color: var(--ink); }

/* ── Chat messages ── */
.chat-wrap {
  background: #f4f6fb;
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
  padding: 18px;
  min-height: 420px;
  max-height: 520px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 16px;
}
.msg-row { display: flex; align-items: flex-end; gap: 8px; }
.msg-row.user { flex-direction: row-reverse; }
.msg-avatar {
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
}
.msg-avatar.ai   { background: var(--accent); color: white; }
.msg-avatar.user { background: var(--ink-2); color: white; }
.msg-bubble {
  max-width: 78%;
  padding: 10px 14px;
  font-size: 13.5px;
  line-height: 1.65;
  border-radius: 18px;
}
.msg-row.ai   .msg-bubble {
  background: white; color: var(--ink);
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 3px rgba(12,15,26,.05);
}
.msg-row.user .msg-bubble {
  background: var(--accent); color: white;
  border-bottom-right-radius: 4px;
}

/* ── Chips ── */
.chips-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
.chip {
  background: white;
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 5px 12px;
  font-size: 12px; font-weight: 500;
  color: var(--ink-3);
  cursor: pointer;
  transition: all .15s;
  white-space: nowrap;
  font-family: 'DM Sans', sans-serif;
}
.chip:hover { background: #eaedfa; border-color: var(--accent); color: var(--accent); }

/* ── Streamlit widget overrides ── */
.stTextInput > label, .stTextArea > label, .stSelectbox > label { 
  font-size: 12px !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted) !important; margin-bottom: 4px !important;
}
.stTextInput input, .stTextArea textarea {
  border-radius: var(--radius) !important;
  border: 1.5px solid var(--border) !important;
  background: var(--surface) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13.5px !important;
  color: var(--ink) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  background: white !important;
  box-shadow: 0 0 0 3px rgba(35,71,245,.1) !important;
}
.stSelectbox > div > div {
  border-radius: var(--radius) !important;
  border: 1.5px solid var(--border) !important;
  background: var(--surface) !important;
  font-size: 13.5px !important;
}

/* ── Buttons ── */
.stButton > button {
  border-radius: var(--radius) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  transition: all .15s !important;
  border: none !important;
}
.stButton > button[kind="primary"], .stButton > button:not([kind="secondary"]) {
  background: var(--accent) !important;
  color: white !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-2) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--surface) !important;
  color: var(--ink) !important;
  border: 1px solid var(--border) !important;
}

/* ── Dividers & spacing ── */
hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
.section-gap { margin-top: 18px; }

/* ── Success / Error msgs ── */
.stSuccess, .stError, .stInfo, .stWarning {
  border-radius: var(--radius) !important;
  font-size: 13px !important;
}

/* ── Prompt items ── */
.prompt-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 9px 13px;
  font-size: 12.5px; color: var(--ink-3);
  cursor: pointer;
  margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
  transition: all .15s;
}
.prompt-item::before {
  content: '';
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--border); flex-shrink: 0;
}
.prompt-item:hover { background: #eaedfa; border-color: var(--accent); color: var(--accent); }

/* ── Download button ── */
.stDownloadButton > button {
  background: var(--accent) !important;
  color: white !important;
  border-radius: var(--radius) !important;
  font-weight: 600 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">
    <div class="topnav-logo">M</div>
    <div>
      <div class="topnav-name">Moodle AI Assistant</div>
      <div class="topnav-sub">NMIT Smart Campus</div>
    </div>
  </div>
  <div style="font-size:12px;color:rgba(255,255,255,.4);">Students · Faculty · Parents</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I can answer academic questions, summarise student performance, show mentor and class teacher details, and help faculty assign mentors. How can I help?"}
    ]
if "user_context" not in st.session_state:
    st.session_state.user_context = None
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""

# ── Helper functions ──────────────────────────────────────
def detect_role(user_id: str) -> str:
    v = (user_id or "").strip().upper()
    if v.startswith("ADM"):  return "Admin"
    if v.startswith("FAC"):  return "Faculty"
    if v.startswith("STU"):  return "Student"
    import re
    if re.match(r"^\d[A-Z0-9]{6,}$", v): return "Student"
    return "Unknown"

def role_color(role: str) -> str:
    return {"Student":"#2347f5","Faculty":"#13a05e","Admin":"#e8a020"}.get(role,"#8b90a8")

def load_user_context(user_id: str):
    try:
        r = requests.get(f"{BASE_URL}/user-context/{requests.utils.quote(user_id)}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def ask_backend(user_id: str, query: str, fmt: str):
    try:
        r = requests.post(
            f"{BASE_URL}/ask",
            json={"user_id": user_id, "query": query, "format": fmt},
            timeout=30,
        )
        r.raise_for_status()
        if fmt == "text":
            return {"type": "text", "data": r.json()}
        else:
            ext = {"pdf":"pdf","excel":"xlsx","word":"docx"}.get(fmt,"txt")
            return {"type":"file","data":r.content,"ext":ext}
    except Exception as e:
        return {"type":"error","data":str(e)}

def assign_mentor(actor_id, student_id, m_name, m_email, m_phone):
    try:
        r = requests.post(
            f"{BASE_URL}/mentor/assign",
            json={
                "actor_user_id": actor_id,
                "student_id": student_id,
                "mentor_name": m_name,
                "mentor_email": m_email,
                "mentor_phone": m_phone,
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def stat_class(value, kind):
    try: v = float(value)
    except: return ""
    if kind == "cgpa":   return "good" if v >= 7 else ("warn" if v >= 5.5 else "bad")
    if kind == "att":    return "good" if v >= 75 else ("warn" if v >= 60 else "bad")
    if kind == "backlog":return "bad" if v > 0 else "good"
    return ""

def render_profile_html(profile):
    if not profile:
        return "<p style='font-size:13px;color:#8b90a8'>Enter a valid student USN to view profile details.</p>"
    n = profile.get
    att_cls  = stat_class(n("attendance_percent","0"), "att")
    cgpa_cls = stat_class(n("cgpa","0"), "cgpa")
    bl_cls   = stat_class(n("backlog_count","0"), "backlog")
    return f"""
<div class="card-title">Profile</div>
<div class="profile-name">{n("name","—")}</div>
<div class="profile-meta">{n("department","—")} · Sem {n("semester","—")} · Sec {n("section","—")}</div>
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-label">CGPA</div>
    <div class="stat-value {cgpa_cls}">{n("cgpa","—")}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Attendance</div>
    <div class="stat-value {att_cls}">{n("attendance_percent","—")}%</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Course</div>
    <div class="stat-value" style="font-size:13px;letter-spacing:0">{n("course","—")}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Backlogs</div>
    <div class="stat-value {bl_cls}">{n("backlog_count","0")}</div>
  </div>
</div>
"""

def render_contacts_html(profile):
    if not profile:
        return "<p style='font-size:13px;color:#8b90a8'>Mentor and class teacher details will appear here.</p>"
    m  = profile.get("mentor", {})
    ct = profile.get("class_teacher", {})
    return f"""
<div class="card-title">Contacts</div>
<div class="contact-block">
  <div class="contact-role">Mentor</div>
  <div class="contact-name">{m.get("name","—")}</div>
  <div class="contact-info">{m.get("email","")}<br>{m.get("phone","")}</div>
</div>
<div class="contact-block">
  <div class="contact-role">Class Teacher</div>
  <div class="contact-name">{ct.get("name","—")}</div>
  <div class="contact-info">{ct.get("email","")}<br>{ct.get("phone","")}</div>
</div>
"""

def render_chat_html(messages):
    html = '<div class="chat-wrap" id="chatbox">'
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        row_cls    = "user" if role == "user" else "ai"
        avatar_lbl = "U" if role == "user" else "AI"
        html += f"""
        <div class="msg-row {row_cls}">
          <div class="msg-avatar {row_cls}">{avatar_lbl}</div>
          <div class="msg-bubble">{content}</div>
        </div>"""
    html += "</div>"
    # auto-scroll
    html += """<script>
      var cb = document.getElementById('chatbox');
      if(cb) cb.scrollTop = cb.scrollHeight;
    </script>"""
    return html

# ── Layout: 3 columns ─────────────────────────────────────
left_col, main_col, right_col = st.columns([1.1, 2.6, 1.1], gap="medium")

# ════════════════════════════════════════════════════════
#  LEFT SIDEBAR
# ════════════════════════════════════════════════════════
with left_col:
    # Access panel
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Access Panel</div>', unsafe_allow_html=True)

    user_id = st.text_input(
        "User ID / USN",
        placeholder="FAC001 or 1NT23IS015",
        key="user_id_input",
        label_visibility="visible",
    )

    role = detect_role(user_id)
    role_col = role_color(role)
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                background:#f5f6fa;border:1px solid #e4e6ef;border-radius:12px;
                padding:9px 14px;margin-top:4px;">
      <span style="font-size:12px;font-weight:600;color:#8b90a8;text-transform:uppercase;letter-spacing:.07em">Role</span>
      <span style="font-size:12px;font-weight:700;color:{role_col};background:{role_col}18;
                   padding:3px 12px;border-radius:99px">{role}</span>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        load_btn = st.button("Load Profile", use_container_width=True, type="primary")
    with col_r2:
        clear_profile = st.button("Clear", use_container_width=True, type="secondary")

    st.markdown('</div>', unsafe_allow_html=True)

    if load_btn and user_id:
        with st.spinner("Loading…"):
            ctx = load_user_context(user_id)
            if "error" not in ctx:
                st.session_state.user_context = ctx
            else:
                st.error(ctx["error"])

    if clear_profile:
        st.session_state.user_context = None

    # Profile card
    ctx = st.session_state.user_context or {}
    profile = ctx.get("profile") if ctx else None

    st.markdown(f'<div class="card">{render_profile_html(profile)}</div>', unsafe_allow_html=True)

    # Contacts card
    st.markdown(f'<div class="card">{render_contacts_html(profile)}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  MAIN CHAT
# ════════════════════════════════════════════════════════
with main_col:
    # Chat header
    st.markdown("""
    <div style="background:white;border:1px solid #e4e6ef;border-radius:20px;
                padding:16px 22px;margin-bottom:14px;
                display:flex;align-items:center;justify-content:space-between;">
      <div>
        <div style="font-family:'Instrument Serif',Georgia,serif;font-size:22px;color:#0c0f1a;line-height:1.2">
          Ask the assistant
        </div>
        <div style="font-size:12px;color:#8b90a8;margin-top:2px">
          Academic intelligence for students, faculty, and parents
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat messages
    st.markdown(render_chat_html(st.session_state.messages), unsafe_allow_html=True)

    # ── Input section ──
    st.markdown('<div style="background:white;border:1px solid #e4e6ef;border-radius:20px;padding:16px 20px;">', unsafe_allow_html=True)

    # Quick chips (rendered as buttons in a horizontal row)
    st.markdown('<div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#8b90a8;margin-bottom:8px">Quick prompts</div>', unsafe_allow_html=True)

    chip_cols = st.columns(len(EXAMPLES[:4]))
    for i, ex in enumerate(EXAMPLES[:4]):
        with chip_cols[i]:
            if st.button(ex, key=f"chip_{i}", use_container_width=True, type="secondary"):
                st.session_state.prefill_query = ex
                st.rerun()

    chip_cols2 = st.columns(len(EXAMPLES[4:]))
    for i, ex in enumerate(EXAMPLES[4:]):
        with chip_cols2[i]:
            if st.button(ex, key=f"chip2_{i}", use_container_width=True, type="secondary"):
                st.session_state.prefill_query = ex
                st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    query_val = st.session_state.get("prefill_query", "")
    query = st.text_area(
        "Message",
        value=query_val,
        placeholder="Ask about grades, attendance, backlogs, mentors, or academic concepts…",
        height=90,
        key="query_input",
        label_visibility="visible",
    )

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c1:
        fmt = st.selectbox(
            "Output format",
            ["text","pdf","excel","word"],
            format_func=lambda x: {"text":"💬 Text","pdf":"📄 PDF","excel":"📊 Excel","word":"📝 Word"}[x],
            label_visibility="visible",
        )
    with c2:
        send_btn = st.button("✈ Send", type="primary", use_container_width=True)
    with c3:
        clear_chat_btn = st.button("🗑 Clear Chat", type="secondary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Keyboard hint ──
    st.markdown('<p style="font-size:11px;color:#8b90a8;text-align:right;margin-top:6px">Tip: fill User ID first, then send</p>', unsafe_allow_html=True)

    # ── Clear chat ──
    if clear_chat_btn:
        st.session_state.messages = [
            {"role":"ai","content":"Hi! I can answer academic questions, summarise student performance, show mentor and class teacher details, and help faculty assign mentors. How can I help?"}
        ]
        st.rerun()

    # ── Send message ──
    if send_btn and query.strip():
        if not user_id.strip():
            st.error("Please enter a User ID or USN first.")
        else:
            st.session_state.messages.append({"role":"user","content":query.strip()})
            st.session_state.prefill_query = ""

            with st.spinner("Working on your request…"):
                result = ask_backend(user_id.strip(), query.strip(), fmt)

            if result["type"] == "text":
                answer = result["data"].get("answer","No response returned.")
                st.session_state.messages.append({"role":"ai","content":answer})
                if result["data"].get("user_context"):
                    st.session_state.user_context = result["data"]["user_context"]

            elif result["type"] == "file":
                st.session_state.messages.append({
                    "role":"ai",
                    "content":f"✅ Your <b>{result['ext'].upper()}</b> file is ready — use the download button below the chat."
                })
                st.session_state["pending_download"] = result

            elif result["type"] == "error":
                st.session_state.messages.append({"role":"ai","content":f"⚠️ Error: {result['data']}"})

            st.rerun()

    # ── File download (if pending) ──
    if "pending_download" in st.session_state:
        dl = st.session_state["pending_download"]
        st.download_button(
            label=f"⬇ Download {dl['ext'].upper()} file",
            data=dl["data"],
            file_name=f"moodle_ai_response.{dl['ext']}",
            mime="application/octet-stream",
            type="primary",
        )
        if st.button("✖ Dismiss download", type="secondary"):
            del st.session_state["pending_download"]
            st.rerun()


# ════════════════════════════════════════════════════════
#  RIGHT SIDEBAR
# ════════════════════════════════════════════════════════
with right_col:

    # Campus snapshot
    ctx = st.session_state.user_context or {}
    overview = ctx.get("overview", {}) if ctx else {}
    students_count = overview.get("students","—")
    courses_count  = len(overview.get("courses",[])) if isinstance(overview.get("courses"), list) else overview.get("courses","—")

    st.markdown(f"""
    <div class="card">
      <div class="card-title">Campus Snapshot</div>
      <div class="ov-grid">
        <div class="ov-box-dark">
          <div class="ov-label">Students</div>
          <div class="ov-value">{students_count}</div>
        </div>
        <div class="ov-box-light">
          <div class="ov-label">Courses</div>
          <div class="ov-value">{courses_count}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Suggested prompts
    st.markdown('<div class="card"><div class="card-title">Try asking</div>', unsafe_allow_html=True)
    for i, ex in enumerate(EXAMPLES):
        if st.button(f"  {ex}", key=f"prompt_{i}", use_container_width=True, type="secondary"):
            st.session_state.prefill_query = ex
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Assign Mentor (faculty/admin only)
    can_assign = ctx.get("permissions", {}).get("can_assign_mentor", False) if ctx else False
    if can_assign:
        st.markdown('<div class="card"><div class="card-title">Assign Mentor</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:12.5px;color:#8b90a8;margin-bottom:12px">Faculty & admin access only.</p>', unsafe_allow_html=True)

        m_student = st.text_input("Student USN", key="m_student", placeholder="1NT23IS015")

        # Mentor name with autocomplete from directory
        mentor_dir = ctx.get("mentor_directory", []) if ctx else []
        mentor_names = [m.get("name","") for m in mentor_dir] if mentor_dir else []

        if mentor_names:
            m_name = st.selectbox("Mentor Name", options=[""] + mentor_names, key="m_name")
        else:
            m_name = st.text_input("Mentor Name", key="m_name_txt", placeholder="Dr. Jane Smith")
            m_name = m_name  # keep consistent

        m_email = st.text_input("Mentor Email", key="m_email", placeholder="mentor@nmit.ac.in")
        m_phone = st.text_input("Mentor Phone", key="m_phone", placeholder="+91 98765 43210")

        if st.button("✓ Update Mentor", type="primary", use_container_width=True):
            if not m_student or not m_name:
                st.error("Student USN and mentor name are required.")
            else:
                with st.spinner("Updating…"):
                    result = assign_mentor(user_id, m_student, m_name, m_email, m_phone)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(result.get("message","Mentor updated."))
                    st.session_state.messages.append({
                        "role":"ai",
                        "content":f"Mentor updated for {m_student} — {m_name} is now assigned."
                    })
                    # Refresh context
                    updated = load_user_context(user_id)
                    if "error" not in updated:
                        st.session_state.user_context = updated
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style="text-align:center;padding:16px 0 0;font-size:11px;color:#c0c4d6;">
      NMIT Smart Campus · {datetime.now().year}<br>
      <span style="color:#e4e6ef">Moodle AI Assistant</span>
    </div>
    """, unsafe_allow_html=True)