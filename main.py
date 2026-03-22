import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Literal

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
from rbac import detect_role
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

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
MENTOR_ASSIGNMENTS_PATH = BASE_DIR / "data" / "mentor_assignments.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Moodle AI Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    format: Literal["text", "txt", "pdf", "excel", "word"] = "text"


class MentorAssignmentRequest(BaseModel):
    actor_user_id: str = Field(..., min_length=1)
    student_id: str = Field(..., min_length=1)
    mentor_name: str = Field(..., min_length=1)
    mentor_email: str = ""
    mentor_phone: str = ""


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or still using the placeholder value in .env")
    return Groq(api_key=api_key)


def _chat_sync(system_prompt: str, user_prompt: str, model: str) -> str:
    client = _get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (completion.choices[0].message.content or "").strip()


async def ask_groq(system_prompt: str, user_prompt: str, model: str) -> str:
    return await asyncio.to_thread(_chat_sync, system_prompt, user_prompt, model)


def _compact_retrieval_payload(retrieved: dict) -> dict:
    records = retrieved.get("records", [])
    sample_records = records[:12]
    return {
        "intent": retrieved.get("intent"),
        "entity": retrieved.get("entity"),
        "summary": retrieved.get("summary", {}),
        "record_count": len(records),
        "sample_records": sample_records,
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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/user-context/{user_id}")
async def user_context(user_id: str):
    role = detect_role(user_id)
    context = get_user_context(
        data_path=str(DATA_PATH),
        user_id=user_id,
        role=role,
        assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
    )
    return JSONResponse(context)


@app.get("/")
async def home():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.post("/mentor/assign")
async def mentor_assignment(payload: MentorAssignmentRequest):
    try:
        actor_role = detect_role(payload.actor_user_id)
        result = assign_mentor(
            data_path=str(DATA_PATH),
            assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
            actor_role=actor_role,
            actor_user_id=payload.actor_user_id,
            student_id=payload.student_id,
            mentor_name=payload.mentor_name,
            mentor_email=payload.mentor_email,
            mentor_phone=payload.mentor_phone,
        )
        return JSONResponse({"role": actor_role, **result})
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ask")
async def ask(payload: AskRequest):
    try:
        role = detect_role(payload.user_id)
        classification = await classify_query(payload.query)
        logger.info("Classification result: %s", json.dumps(classification))

        query_type = classification.get("query_type", "general_query")
        intent = classification.get("intent", "general")
        entity = classification.get("entity", "general")

        if query_type == "organizational_query":
            retrieved = retrieve_data(
                data_path=str(DATA_PATH),
                intent=intent,
                entity=entity,
                role=role,
                user_id=payload.user_id,
                assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
            )
            compact_payload = _compact_retrieval_payload(retrieved)
            system_prompt = (
                "You are Moodle AI Assistant for NMIT. "
                "Use the provided academic dataset and role context to answer clearly. "
                "When the user is a student, prefer a privacy-safe, student-specific answer. "
                "When the user is faculty or admin, include actionable administrative detail. "
                "If the query concerns mentors, class teachers, or contact details, present them cleanly. "
                "Use the structured summary first and only use sample records when needed. "
                "Do not mention missing raw data. "
                "Use a short WhatsApp-style tone when the user asks for chat/contact information."
            )
            user_prompt = (
                f"User role: {role}\n"
                f"User id: {payload.user_id}\n"
                f"Original query: {payload.query}\n"
                f"Classification: {classification}\n"
                f"Compact retrieval payload: {json.dumps(compact_payload)}\n"
                "Generate a clean natural language response."
            )
            answer = await ask_groq(system_prompt, user_prompt, "llama-3.3-70b-versatile")
        else:
            system_prompt = (
                "You are a helpful educational assistant. "
                "Answer in a clean, easy-to-read format."
            )
            user_prompt = f"User role: {role}\nQuestion: {payload.query}"
            answer = await ask_groq(system_prompt, user_prompt, "llama-3.3-70b-versatile")

        if payload.format == "text":
            return JSONResponse(
                {
                    "answer": format_text_response(answer),
                    "role": role,
                    "classification": classification,
                    "user_context": get_user_context(
                        data_path=str(DATA_PATH),
                        user_id=payload.user_id,
                        role=role,
                        assignments_path=str(MENTOR_ASSIGNMENTS_PATH),
                    ),
                }
            )

        if payload.format == "txt":
            file_path = create_text_file(answer)
            return _build_download_response(
                file_path=file_path,
                media_type="text/plain; charset=utf-8",
                filename="moodle_ai_response.txt",
            )

        if payload.format == "pdf":
            file_path = create_pdf(answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/pdf",
                filename="moodle_ai_response.pdf",
            )

        if payload.format == "excel":
            file_path = create_excel(answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename="moodle_ai_response.xlsx",
            )

        if payload.format == "word":
            file_path = create_word(answer)
            return _build_download_response(
                file_path=file_path,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename="moodle_ai_response.docx",
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
        logger.exception("Failed to process /ask")
        raise HTTPException(status_code=500, detail=f"Server error: {exc}")
