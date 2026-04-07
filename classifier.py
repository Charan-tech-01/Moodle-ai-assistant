import asyncio
import json
import os
import re
from typing import Dict

from groq import Groq

ALLOWED_INTENTS = {
    "student_count",
    "course_enrollment",
    "faculty_list",
    "grades_average",
    "attendance_report",
    "student_profile",
    "mentor_lookup",
    "class_teacher_info",
    "backlog_report",
    "contact_lookup",
    "general",
}

COURSE_NAMES = [
    "Machine Learning",
    "Artificial Intelligence",
    "Data Structures",
    "Deep Learning",
    "Cloud Computing",
]

DEFAULT_CLASSIFICATION = {
    "query_type": "general_query",
    "intent": "general",
    "entity": "general",
}


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or still using the placeholder value.")
    return Groq(api_key=api_key)


def _strip_markdown_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _extract_entity(query: str) -> str:
    lowered = query.lower()
    for course in COURSE_NAMES:
        if course.lower() in lowered:
            return course

    student_id_match = re.search(r"\b\d[a-z0-9]{7,}\b", lowered, flags=re.IGNORECASE)
    if student_id_match:
        return student_id_match.group(0).upper()

    return "general"


def _heuristic_classify(query: str) -> Dict[str, str]:
    lowered = query.lower()
    entity = _extract_entity(query)

    if any(word in lowered for word in ["mentor assign", "assign mentor", "mentor for", "mentorship"]):
        intent = "mentor_lookup"
    elif any(word in lowered for word in ["class teacher", "homeroom", "ct info"]):
        intent = "class_teacher_info"
    elif any(word in lowered for word in ["mentor", "mentee"]):
        intent = "mentor_lookup"
    elif any(word in lowered for word in ["backlog", "arrear"]):
        intent = "backlog_report"
    elif any(word in lowered for word in ["contact", "phone", "email", "whatsapp"]):
        intent = "contact_lookup"
    elif any(word in lowered for word in ["profile", "my details", "student details", "usn", "cgpa", "interest"]):
        intent = "student_profile"
    elif any(word in lowered for word in ["attendance", "present percentage"]):
        intent = "attendance_report"
    elif any(word in lowered for word in ["average grade", "cgpa average", "grade average"]):
        intent = "grades_average"
    elif any(word in lowered for word in ["faculty", "professor", "teacher list"]):
        intent = "faculty_list"
    elif any(word in lowered for word in ["enrollment", "enrolled", "list students", "students in"]):
        intent = "course_enrollment"
    elif any(word in lowered for word in ["how many students", "student count", "count students"]):
        intent = "student_count"
    else:
        intent = "general"

    org_terms = [
        "student",
        "grade",
        "marks",
        "cgpa",
        "attendance",
        "faculty",
        "course",
        "mentor",
        "class teacher",
        "backlog",
        "usn",
        "semester",
        "enrollment",
        "whatsapp",
        "contact",
    ]
    query_type = "organizational_query" if any(term in lowered for term in org_terms) or entity != "general" else "general_query"
    return {"query_type": query_type, "intent": intent, "entity": entity}


def _classify_sync(query: str) -> Dict[str, str]:
    client = _get_client()

    system_prompt = (
        "You are a query classifier for an academic assistant. "
        "Return JSON only with keys query_type, intent, entity.\n"
        "query_type: general_query or organizational_query.\n"
        "intent: student_count, course_enrollment, faculty_list, grades_average, "
        "attendance_report, student_profile, mentor_lookup, class_teacher_info, "
        "backlog_report, contact_lookup, general.\n"
        "entity: course name, student USN, or general.\n"
        "Organizational queries mention students, grades, marks, attendance, faculty, "
        "mentors, class teachers, contacts, backlogs, semesters, enrollments, or course names.\n"
        "General queries are concept explanations, definitions, or how-to questions.\n"
        "Return valid JSON only."
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
    )

    raw = completion.choices[0].message.content or ""
    cleaned = _strip_markdown_fences(raw)

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return _heuristic_classify(query)

        query_type = parsed.get("query_type", DEFAULT_CLASSIFICATION["query_type"])
        intent = parsed.get("intent", DEFAULT_CLASSIFICATION["intent"])
        entity = parsed.get("entity", "general")

        if query_type not in {"general_query", "organizational_query"}:
            query_type = DEFAULT_CLASSIFICATION["query_type"]
        if intent not in ALLOWED_INTENTS:
            intent = DEFAULT_CLASSIFICATION["intent"]
        if not isinstance(entity, str) or not entity.strip():
            entity = _extract_entity(query)

        enriched = {
            "query_type": query_type,
            "intent": intent,
            "entity": entity.strip(),
        }

        if enriched["intent"] == "general" and _heuristic_classify(query)["intent"] != "general":
            return _heuristic_classify(query)
        return enriched
    except Exception:
        return _heuristic_classify(query)


async def classify_query(query: str) -> Dict[str, str]:
    return await asyncio.to_thread(_classify_sync, query)
