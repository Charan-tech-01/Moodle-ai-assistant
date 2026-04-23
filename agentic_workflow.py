"""
agentic_workflow.py

Multi-agent orchestrator for the Moodle AI Assistant.
Updated for database-backed data retrieval (no more CSV).

Agents:
  1. Role Guard   — detect user role from Moodle DB
  2. Context      — load role-scoped dashboard context
  3. Router       — classify query (LLM + heuristic fallback)
  4. Data         — retrieve academic data via SQL
  5. Executor     — structured answer for known intents (no LLM)
  6. Composer     — LLM fallback for complex/unknown intents
"""

import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List

from data_retriever import get_user_context, retrieve_data
from rbac import detect_role


ClassifierFn = Callable[[str], Awaitable[Dict[str, str]]]
ChatFn = Callable[[str, str, str], Awaitable[str]]


class _DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def _json_dumps(obj, **kwargs):
    return json.dumps(obj, cls=_DecimalEncoder, **kwargs)


@dataclass
class AgentStep:
    agent: str
    action: str
    status: str
    detail: str


@dataclass
class AgentResult:
    answer: str
    role: str
    classification: Dict[str, str]
    user_context: Dict[str, Any]
    trace: List[Dict[str, str]]


def _step(agent: str, action: str, status: str, detail: str) -> Dict[str, str]:
    return asdict(AgentStep(agent=agent, action=action, status=status, detail=detail))


def _compact_payload(retrieved: Dict[str, Any]) -> Dict[str, Any]:
    records = retrieved.get("records", [])
    return {
        "intent": retrieved.get("intent"),
        "entity": retrieved.get("entity"),
        "summary": retrieved.get("summary", {}),
        "record_count": len(records),
        "sample_records": records[:12],
        "requester_context": retrieved.get("requester_context", {}),
    }


def _safe(val):
    """Convert Decimal or None to a display-safe value."""
    if isinstance(val, Decimal):
        return float(val)
    return val if val is not None else "N/A"


def _structured_answer(query: str, role: str, payload: Dict[str, Any]) -> str:
    """
    Generate a direct answer from structured data for known intents.
    No LLM call needed — fast and deterministic.
    """
    intent = payload.get("intent", "general")
    summary = payload.get("summary", {})
    entity = payload.get("entity", "general")

    if intent == "student_count":
        return f"There are {summary.get('count', 0)} students in {summary.get('course', entity)}."

    if intent == "course_enrollment":
        # Student asking about their own courses
        if "enrolled_courses" in summary:
            courses = summary.get("enrolled_courses", [])
            names = ", ".join(c.get("fullname", "") for c in courses[:8])
            suffix = f"... and {len(courses) - 8} more" if len(courses) > 8 else ""
            return (
                f"{summary.get('student_name', 'You')} — enrolled in "
                f"{summary.get('count', 0)} courses: {names}{suffix}"
            )
        # Faculty/admin asking about a course's students
        students = summary.get("students", [])
        names = ", ".join(
            s.get("name", "") for s in students[:8] if s.get("name")
        )
        suffix = "..." if summary.get("count", 0) > 8 else ""
        return (
            f"{summary.get('count', 0)} students are enrolled in "
            f"{summary.get('course', entity)}. "
            f"Sample list: {names}{suffix}"
        )

    if intent == "faculty_list":
        faculty = ", ".join(summary.get("faculty", [])[:10])
        return (
            f"Faculty for {summary.get('course', entity)}: "
            f"{faculty or 'No faculty data found.'}"
        )

    if intent == "grades_average":
        if "average_grade" in summary:
            return (
                f"The average grade for {summary.get('course', entity)} is "
                f"{_safe(summary.get('average_grade', 0))} "
                f"(based on {summary.get('grade_count', 0)} graded students)."
            )
        averages = summary.get("course_averages", [])
        parts = [
            f"{item.get('course')}: avg {_safe(item.get('avg_grade', 0))}"
            for item in averages[:6]
        ]
        return "Course-wise grade averages:\n" + "\n".join(parts) if parts else "No grade data available."

    if intent == "attendance_report":
        # Individual student attendance
        if "student_name" in summary:
            return (
                f"Attendance for {summary.get('student_name')} in "
                f"{summary.get('course', 'all courses')}: "
                f"{_safe(summary.get('attendance_percent', 0))}% "
                f"({_safe(summary.get('present', 0))} present, "
                f"{_safe(summary.get('absent', 0))} absent, "
                f"{_safe(summary.get('late', 0))} late, "
                f"{_safe(summary.get('excused', 0))} excused "
                f"out of {summary.get('total_sessions', 0)} sessions)."
            )
        # Class-wide attendance
        return (
            f"Average attendance for {summary.get('course', entity)} is "
            f"{_safe(summary.get('average_attendance_percent', 0))}% "
            f"across {summary.get('student_count', 0)} students."
        )

    if intent == "student_profile":
        courses = summary.get("enrolled_courses", [])
        course_names = ", ".join(c.get("fullname", "") for c in courses[:5])
        suffix = f"... and {len(courses) - 5} more" if len(courses) > 5 else ""
        grades = summary.get("grades", [])
        grade_info = ", ".join(
            f"{g.get('course', '?')}: {_safe(g.get('grade', 0))}/{_safe(g.get('max_grade', 0))}"
            for g in grades[:5]
        )
        return (
            f"{summary.get('name')} (ID: {summary.get('student_id')}) — "
            f"{summary.get('department', 'N/A')} department.\n"
            f"Email: {summary.get('email', 'N/A')}\n"
            f"Enrolled in {summary.get('course_count', 0)} courses: {course_names}{suffix}\n"
            f"Average grade: {_safe(summary.get('average_grade_percent', 0))}%\n"
            f"Grade details: {grade_info or 'No grades recorded yet.'}"
        )

    if intent == "mentor_lookup":
        mentor = summary.get("mentor", {})
        return (
            f"The mentor for {summary.get('name')} is {mentor.get('name', 'Not assigned')}. "
            f"Email: {mentor.get('email', 'N/A')}, phone: {mentor.get('phone', 'N/A')}."
        )

    if intent == "class_teacher_info":
        # Now maps to faculty list for the course
        faculty = summary.get("faculty", [])
        note = summary.get("note", "")
        if faculty:
            return (
                f"Course teachers: {', '.join(faculty[:5])}. "
                f"{note}"
            )
        return "No class teacher information found."

    if intent == "backlog_report":
        students = summary.get("students", [])
        if not students:
            return "No students with backlogs were found in the current scope."
        rows = [
            f"{s.get('name')} (ID {s.get('student_id')}): "
            f"{s.get('backlog_count')} backlog(s) in {', '.join(s.get('backlog_courses', []))}"
            for s in students[:10]
        ]
        return (
            f"{summary.get('count_with_backlogs', 0)} students currently have backlogs.\n"
            + "\n".join(rows)
        )

    if intent == "contact_lookup":
        contact = summary.get("student_contact", {})
        return (
            f"Contact details for {summary.get('name')}: "
            f"email {contact.get('email', 'N/A')}, "
            f"phone {contact.get('phone', 'N/A')}."
        )

    return f"I found academic data for your query: {query}"


async def run_agentic_workflow(
    *,
    user_id: str,
    query: str,
    data_path: str = "",
    assignments_path: str = "",
    classify_query: ClassifierFn,
    ask_groq: ChatFn,
) -> AgentResult:
    trace: List[Dict[str, str]] = []

    # ── Role Guard Agent ──────────────────────────────────────────────────
    role = detect_role(user_id)
    trace.append(_step(
        "role-guard-agent", "detect_role", "completed",
        f"Resolved user role as '{role}' from Moodle database.",
    ))

    # ── Context Agent ─────────────────────────────────────────────────────
    user_context = get_user_context(
        user_id=user_id,
        role=role,
        assignments_path=assignments_path,
    )
    trace.append(_step(
        "context-agent", "load_user_context", "completed",
        "Loaded role-scoped dashboard context from database.",
    ))

    # ── Router Agent ──────────────────────────────────────────────────────
    classification = await classify_query(query)
    trace.append(_step(
        "router-agent", "classify_query", "completed",
        f"Routed as {classification.get('query_type')} — "
        f"intent={classification.get('intent')}, entity={classification.get('entity')}.",
    ))

    query_type = classification.get("query_type", "general_query")
    intent = classification.get("intent", "general")
    entity = classification.get("entity", "general")

    # ── General query → Knowledge Agent (direct LLM) ──────────────────────
    if query_type == "general_query":
        trace.append(_step(
            "knowledge-agent", "answer_general_query", "in_progress",
            "Sending conceptual query to Groq LLM.",
        ))
        answer = await ask_groq(
            "You are an educational assistant for NMIT. Answer clearly and accurately.",
            f"User role: {role}\nQuestion: {query}",
            "llama-3.3-70b-versatile",
        )
        trace[-1]["status"] = "completed"
        trace[-1]["detail"] = "Returned direct LLM answer for general knowledge query."
        return AgentResult(
            answer=answer, role=role, classification=classification,
            user_context=user_context, trace=trace,
        )

    # ── Data Agent ────────────────────────────────────────────────────────
    trace.append(_step(
        "data-agent", "retrieve_academic_data", "in_progress",
        f"Querying Moodle database for intent={intent}, entity={entity}.",
    ))
    retrieved = retrieve_data(
        intent=intent,
        entity=entity,
        role=role,
        user_id=user_id,
        assignments_path=assignments_path,
    )
    trace[-1]["status"] = "completed"
    trace[-1]["detail"] = "Academic records retrieved from database and summarized."

    compact_payload = _compact_payload(retrieved)

    # ── Executor Agent (known intents → no LLM) ──────────────────────────
    if intent in {
        "student_count", "course_enrollment", "faculty_list",
        "grades_average", "attendance_report", "student_profile",
        "mentor_lookup", "class_teacher_info", "backlog_report",
        "contact_lookup",
    }:
        trace.append(_step(
            "executor-agent", "compose_structured_answer", "completed",
            "Generated direct tool-based response (no LLM call).",
        ))
        answer = _structured_answer(query, role, compact_payload)

    # ── Composer Agent (complex/unknown → LLM fallback) ───────────────────
    else:
        trace.append(_step(
            "composer-agent", "compose_natural_language_response", "in_progress",
            "Sending database results to Groq LLM for natural language composition.",
        ))
        answer = await ask_groq(
            (
                "You are Moodle AI Assistant for NMIT. "
                "Use the provided structured data from the live Moodle database to answer clearly. "
                "Prefer the structured summary. Keep it concise and role-aware."
            ),
            (
                f"User role: {role}\n"
                f"Original query: {query}\n"
                f"Classification: {_json_dumps(classification)}\n"
                f"Database results: {_json_dumps(compact_payload)}\n"
                "Write the final answer."
            ),
            "llama-3.3-70b-versatile",
        )
        trace[-1]["status"] = "completed"
        trace[-1]["detail"] = "Natural-language answer composed from database results."

    return AgentResult(
        answer=answer, role=role, classification=classification,
        user_context=user_context, trace=trace,
    )
