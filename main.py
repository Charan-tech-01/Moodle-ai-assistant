"""
main.py — Moodle AI Assistant Backend
======================================
Fixes in this version:
  1. Circular import eliminated — rbac.py renamed to auth.py
  2. Conversation memory — chat history sent per session (user_id keyed)
  3. Full record list — all matching records sent to LLM, not just 12
  4. Faculty cross-scope guard — raw query scanned for out-of-scope courses
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from classifier import classify_query
from data_retriever import assign_mentor, get_user_context, retrieve_data
from auth import resolve_identity, check_permission, action_for_intent_and_role
from response_formatter import (
    create_excel,
    create_pdf,
    create_text_file,
    create_word,
    format_text_response,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moodle-ai-assistant")

BASE_DIR               = Path(__file__).resolve().parent
DATA_PATH              = BASE_DIR / "data" / "students.csv"
MENTOR_ASSIGNMENTS_PATH = BASE_DIR / "data" / "mentor_assignments.json"
STATIC_DIR             = BASE_DIR / "static"

# ── In-memory conversation store ─────────────────────────────────────────────
# Keyed by user_id.  Each value is a list of {"role": "user"|"assistant", "content": str}
# We keep at most MAX_HISTORY turns to stay within token limits.
MAX_HISTORY = 6   # 6 turns = 12 messages ≈ ~1500 tokens for history
_chat_history: dict[str, list[dict]] = defaultdict(list)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Moodle AI Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query:   str = Field(..., min_length=1)
    format:  Literal["text", "txt", "pdf", "excel", "word"] = "text"


class MentorAssignmentRequest(BaseModel):
    actor_user_id: str = Field(..., min_length=1)
    student_id:    str = Field(..., min_length=1)
    mentor_name:   str = Field(..., min_length=1)
    mentor_email:  str = ""
    mentor_phone:  str = ""


class ClearHistoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


# ── Groq helpers ──────────────────────────────────────────────────────────────
def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or still the placeholder value in .env")
    return Groq(api_key=api_key)


def _chat_with_history(
    system_prompt: str,
    history: list[dict],
    new_user_message: str,
    model: str,
) -> str:
    """
    Send a full conversation (system + history + new message) to Groq.
    Returns the assistant's reply text.
    """
    client = _get_client()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": new_user_message})

    completion = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
    )
    return (completion.choices[0].message.content or "").strip()


async def ask_groq_with_history(
    system_prompt: str,
    history: list[dict],
    new_user_message: str,
    model: str,
) -> str:
    return await asyncio.to_thread(
        _chat_with_history, system_prompt, history, new_user_message, model
    )


# ── Intent → which columns the LLM actually needs ────────────────────────────
_INTENT_COLS: dict[str, set[str]] = {
    "course_enrollment": {"student_id", "name", "section", "semester", "attendance_percent", "cgpa", "grade"},
    "student_count":     {"student_id", "name", "section", "course"},
    "attendance_report": {"student_id", "name", "section", "attendance_percent", "course"},
    "backlog_report":    {"student_id", "name", "section", "backlog_count", "backlog_subjects", "course"},
    "grades_average":    {"student_id", "name", "section", "grade", "cgpa", "course"},
    "faculty_list":      {"faculty", "class_teacher_name", "mentor_name", "section", "course"},
    "student_profile":   {"student_id", "name", "department", "semester", "section", "course",
                          "grade", "attendance_percent", "cgpa", "backlog_count", "backlog_subjects",
                          "mentor_name", "mentor_email", "mentor_phone",
                          "class_teacher_name", "class_teacher_email", "class_teacher_phone",
                          "phone", "college_email", "area_of_interest"},
    "mentor_lookup":     {"student_id", "name", "mentor_name", "mentor_email", "mentor_phone"},
    "class_teacher_info":{"student_id", "name", "section",
                          "class_teacher_name", "class_teacher_email", "class_teacher_phone"},
    "contact_lookup":    {"student_id", "name", "phone", "college_email",
                          "mentor_name", "mentor_email", "mentor_phone",
                          "class_teacher_name", "class_teacher_email", "class_teacher_phone"},
}
_DEFAULT_COLS = {"student_id", "name", "section", "course", "attendance_percent", "cgpa", "grade"}

# Token budget: stay comfortably under Groq free-tier 6000 TPM per request
# ~4 chars ≈ 1 token; we reserve ~3000 tokens for system prompt + history + answer
_MAX_DATA_CHARS = 8_000   # ≈ 2000 tokens for the data block


# ── Data helpers ──────────────────────────────────────────────────────────────
def _build_retrieval_payload(retrieved: dict) -> dict:
    """
    Build a token-safe payload for the LLM.

    Strategy:
      1. Always include the full summary object (aggregate stats, counts).
      2. Select only the columns relevant to the intent.
      3. Fit as many records as possible within _MAX_DATA_CHARS.
         If records are truncated, add a note so the LLM uses the summary.
    """
    records = retrieved.get("records", [])
    intent  = retrieved.get("intent", "general")

    # Pick the minimal column set for this intent
    cols = _INTENT_COLS.get(intent, _DEFAULT_COLS)
    slim_records = [
        {k: v for k, v in r.items() if k in cols}
        for r in records
    ]

    total = len(slim_records)
    included: list[dict] = []
    char_budget = _MAX_DATA_CHARS

    for rec in slim_records:
        serialised = json.dumps(rec)
        if char_budget - len(serialised) < 0:
            break
        included.append(rec)
        char_budget -= len(serialised)

    truncated = len(included) < total

    return {
        "intent":        intent,
        "entity":        retrieved.get("entity"),
        "summary":       retrieved.get("summary", {}),   # always full
        "record_count":  total,                          # true total
        "records_shown": len(included),
        "records":       included,
        "truncated":     truncated,
        "truncation_note": (
            f"Only {len(included)} of {total} records shown due to size limits. "
            "Use the summary object for accurate totals and averages."
        ) if truncated else None,
    }


def _cleanup_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _build_download_response(file_path: Path, media_type: str, filename: str) -> FileResponse:
    return FileResponse(
        str(file_path),
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(_cleanup_temp_file, file_path),
    )


# ── System prompts ────────────────────────────────────────────────────────────
def _build_system_prompt(role: str) -> str:
    base = (
        "You are Moodle AI, the academic assistant for NMIT (Nitte Meenakshi Institute of Technology). "
        "You have access to real student data from the college database. "
        "Always answer using the data provided in the user message — never say 'navigate to Moodle'. "
        "Keep responses clear, specific, and professional. "
        "You remember the full conversation history and can refer back to earlier messages.\n\n"
    )

    if role == "student":
        return base + (
            "The user is a STUDENT. Rules:\n"
            "- You MUST answer questions about their own data — CGPA, grades, attendance, "
            "backlogs, mentor, class teacher, contacts. This is always allowed.\n"
            "- The data payload contains ONLY this student's own record. Show it fully and directly.\n"
            "- If asked 'what is my CGPA', read it from the records and state it clearly.\n"
            "- If asked 'what are my grades', show them. Never refuse own-data queries.\n"
            "- Only refuse if they explicitly ask about a DIFFERENT student by name or USN.\n"
            "- Be friendly and personal — use their name.\n"
        )

    if role == "faculty":
        return base + (
            "The user is a FACULTY MEMBER. Rules:\n"
            "- Show their complete teaching dashboard: courses, sections, all their students.\n"
            "- The 'records' array contains ALL their students — list them completely when asked.\n"
            "- Show name, USN, attendance, CGPA, grade, backlogs for each student when listing.\n"
            "- For aggregate queries: use the summary object for stats.\n"
            "- Never show data from courses the faculty does not teach.\n"
            "- For 'show my students' or 'list students': format as a clean numbered list.\n"
        )

    if role == "admin":
        return base + (
            "The user is an ADMINISTRATOR. Rules:\n"
            "- Provide full campus-wide data, all courses, all students.\n"
            "- Format large lists cleanly with counts and summaries.\n"
            "- Include actionable details for administrative decisions.\n"
        )

    return base + "Answer general academic questions helpfully."


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


@app.get("/user-context/{user_id}")
async def user_context(user_id: str):
    identity = resolve_identity(user_id)
    context  = get_user_context(
        data_path=str(DATA_PATH),
        user_id=user_id,
        role=identity.role,
        assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
        identity=identity,
    )
    return JSONResponse(context)


@app.get("/")
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat/clear")
async def clear_history(payload: ClearHistoryRequest):
    """Clear the conversation memory for a specific user."""
    _chat_history.pop(payload.user_id, None)
    return JSONResponse({"message": f"History cleared for {payload.user_id}"})


@app.post("/mentor/assign")
async def mentor_assignment(payload: MentorAssignmentRequest):
    try:
        identity = resolve_identity(payload.actor_user_id)
        result   = assign_mentor(
            data_path=str(DATA_PATH),
            assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
            actor_role=identity.role,
            actor_user_id=payload.actor_user_id,
            student_id=payload.student_id,
            mentor_name=payload.mentor_name,
            mentor_email=payload.mentor_email,
            mentor_phone=payload.mentor_phone,
            identity=identity,
        )
        return JSONResponse({"role": identity.role, **result})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ask")
async def ask(payload: AskRequest):
    try:
        # ── 1. Resolve identity ───────────────────────────────────────────────
        identity = resolve_identity(payload.user_id)
        role     = identity.role

        # ── 2. Classify query ─────────────────────────────────────────────────
        classification = await classify_query(payload.query)
        logger.info("User=%s Role=%s Classification=%s",
                    payload.user_id, role, json.dumps(classification))

        query_type = classification.get("query_type", "general_query")
        intent     = classification.get("intent",     "general")
        entity     = classification.get("entity",     "general")

        # ── 3. RBAC permission gate (role-aware) ─────────────────────────────
        check_permission(identity, action_for_intent_and_role(intent, role))

        # ── 4. Build user message with data context ───────────────────────────
        if query_type == "organizational_query":
            # For faculty: always combine classifier entity + raw query so the
            # cross-scope guard can scan the full text for out-of-scope course names.
            if identity.role == "faculty":
                effective_entity = (
                    payload.query if entity.lower() in ("general", "")
                    else f"{entity} {payload.query}"
                )
            else:
                effective_entity = entity

            retrieved = retrieve_data(
                data_path=str(DATA_PATH),
                intent=intent,
                entity=effective_entity,
                role=role,
                user_id=payload.user_id,
                assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
                identity=identity,
            )
            data_payload = _build_retrieval_payload(retrieved)

            user_message = (
                f"[QUERY] {payload.query}\n\n"
                f"[DATA]\n{json.dumps(data_payload, indent=2)}"
            )

        else:
            # General/conceptual query — no DB lookup needed
            user_message = payload.query

        # ── 5. Retrieve conversation history for this user ────────────────────
        history = _chat_history[payload.user_id]

        # ── 6. Call LLM with full history ─────────────────────────────────────
        system_prompt = _build_system_prompt(role)
        answer = await ask_groq_with_history(
            system_prompt=system_prompt,
            history=history,
            new_user_message=user_message,
            model="llama-3.3-70b-versatile",
        )

        # ── 7. Append this turn to history (trimmed to MAX_HISTORY) ───────────
        # Store the original clean query (not the data-padded message)
        # so history stays readable without repeating the full data blob.
        # Truncate long assistant answers in history to save tokens —
        # keep first 400 chars which captures the key facts.
        history_answer = answer[:400] + "…" if len(answer) > 400 else answer
        history.append({"role": "user",      "content": payload.query})
        history.append({"role": "assistant", "content": history_answer})
        if len(history) > MAX_HISTORY * 2:
            # Keep only the most recent MAX_HISTORY turns
            _chat_history[payload.user_id] = history[-(MAX_HISTORY * 2):]

        # ── 8. Return response ────────────────────────────────────────────────
        if payload.format == "text":
            return JSONResponse({
                "answer":         format_text_response(answer),
                "role":           role,
                "classification": classification,
                "user_context":   get_user_context(
                    data_path=str(DATA_PATH),
                    user_id=payload.user_id,
                    role=role,
                    assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
                    identity=identity,
                ),
            })

        if payload.format == "txt":
            return _build_download_response(
                create_text_file(answer), "text/plain; charset=utf-8", "moodle_ai_response.txt"
            )
        if payload.format == "pdf":
            return _build_download_response(
                create_pdf(answer), "application/pdf", "moodle_ai_response.pdf"
            )
        if payload.format == "excel":
            return _build_download_response(
                create_excel(answer),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "moodle_ai_response.xlsx",
            )
        if payload.format == "word":
            return _build_download_response(
                create_word(answer),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "moodle_ai_response.docx",
            )

        raise HTTPException(status_code=400, detail="Invalid format")

    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=500, detail=f"Server error: {exc}")