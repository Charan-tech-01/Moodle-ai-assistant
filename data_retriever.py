import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Import the full RBAC layer
from rbac import RoleIdentity, check_permission, faculty_scope, student_scope

GRADE_MAP = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
CORE_COLUMNS = {
    "student_id",
    "name",
    "course",
    "grade",
    "attendance_percent",
    "faculty",
    "department",
    "semester",
}
OPTIONAL_DEFAULTS = {
    "section": "",
    "class_teacher_name": "",
    "class_teacher_email": "",
    "class_teacher_phone": "",
    "mentor_name": "",
    "mentor_email": "",
    "mentor_phone": "",
    "phone": "",
    "college_email": "",
    "personal_email": "",
    "area_of_interest": "",
    "cgpa": 0.0,
    "aggregate_percent": 0.0,
    "backlog_count": 0,
    "backlog_subjects": "",
    "x_percent": 0.0,
    "xii_percent": 0.0,
    "year_gaps": 0,
    "gender": "",
    "dob": "",
}
FACULTY_DIRECTORY = [
    {"name": "Dr. Asha Rao", "email": "asha.rao@nmit.ac.in", "phone": "9845001101", "role": "Mentor"},
    {"name": "Prof. Vivek Shenoy", "email": "vivek.shenoy@nmit.ac.in", "phone": "9845001102", "role": "Mentor"},
    {"name": "Dr. Neha Kulkarni", "email": "neha.kulkarni@nmit.ac.in", "phone": "9845001103", "role": "Mentor"},
    {"name": "Prof. Kiran Bhat", "email": "kiran.bhat@nmit.ac.in", "phone": "9845001104", "role": "Mentor"},
    {"name": "Dr. Suma Pai", "email": "suma.pai@nmit.ac.in", "phone": "9845001105", "role": "Mentor"},
    {"name": "Prof. Harish Nayak", "email": "harish.nayak@nmit.ac.in", "phone": "9845001106", "role": "Mentor"},
]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_table(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python")


def _load_student_frame(data_path: str) -> pd.DataFrame:
    df = _read_table(data_path)
    df.columns = [str(column).strip() for column in df.columns]

    missing = CORE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    string_columns = [
        "student_id", "name", "course", "faculty", "section", "department",
        "class_teacher_name", "class_teacher_email", "class_teacher_phone",
        "mentor_name", "mentor_email", "mentor_phone",
        "phone", "college_email", "personal_email",
        "area_of_interest", "backlog_subjects",
    ]
    for column in string_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["student_id"] = df["student_id"].str.upper()
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce").fillna(0.0)
    df["aggregate_percent"] = pd.to_numeric(df["aggregate_percent"], errors="coerce").fillna(0.0)
    df["attendance_percent"] = pd.to_numeric(df["attendance_percent"], errors="coerce").fillna(0.0)
    df["backlog_count"] = pd.to_numeric(df["backlog_count"], errors="coerce").fillna(0).astype(int)
    return df


def _load_mentor_overrides(assignments_path: str) -> Dict[str, Dict[str, str]]:
    path = Path(assignments_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_mentor_overrides(assignments_path: str, overrides: Dict[str, Dict[str, str]]) -> None:
    Path(assignments_path).write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def _apply_mentor_overrides(df: pd.DataFrame, assignments_path: str | None) -> pd.DataFrame:
    if not assignments_path:
        return df
    overrides = _load_mentor_overrides(assignments_path)
    if not overrides:
        return df
    updated = df.copy()
    for student_id, mentor in overrides.items():
        mask = updated["student_id"] == student_id.upper()
        if mask.any():
            updated.loc[mask, "mentor_name"]  = mentor.get("mentor_name", "")
            updated.loc[mask, "mentor_email"] = mentor.get("mentor_email", "")
            updated.loc[mask, "mentor_phone"] = mentor.get("mentor_phone", "")
    return updated


def _extract_student_id(user_id: str | None) -> str | None:
    if not user_id:
        return None
    normalized = user_id.strip().upper()
    if normalized.startswith("STU"):
        normalized = re.sub(r"^STU[-_:]?", "", normalized)
    return normalized or None


def _find_student_row(df: pd.DataFrame, token: str | None) -> pd.Series | None:
    if not token or token == "GENERAL":
        return None
    normalized = token.strip().upper()
    direct = df[df["student_id"] == normalized]
    if not direct.empty:
        return direct.iloc[0]
    by_name = df[df["name"].str.upper().str.contains(normalized, na=False)]
    if not by_name.empty:
        return by_name.iloc[0]
    return None


def _find_course(df: pd.DataFrame, token: str | None) -> str | None:
    if not token or token.lower() == "general":
        return None
    lowered = token.lower()
    for course in df["course"].dropna().unique().tolist():
        if lowered in course.lower() or course.lower() in lowered:
            return course
    return None


def _record_to_profile(record: pd.Series) -> Dict[str, Any]:
    return {
        "student_id":       record["student_id"],
        "name":             record["name"],
        "department":       record["department"],
        "semester":         int(record["semester"]) if pd.notna(record["semester"]) else None,
        "section":          record["section"],
        "course":           record["course"],
        "grade":            record["grade"],
        "attendance_percent": float(record["attendance_percent"]),
        "cgpa":             float(record["cgpa"]),
        "aggregate_percent":float(record["aggregate_percent"]),
        "backlog_count":    int(record["backlog_count"]),
        "backlog_subjects": record["backlog_subjects"],
        "area_of_interest": record["area_of_interest"],
        "phone":            record["phone"],
        "college_email":    record["college_email"],
        "personal_email":   record["personal_email"],
        "class_teacher": {
            "name":  record["class_teacher_name"],
            "email": record["class_teacher_email"],
            "phone": record["class_teacher_phone"],
        },
        "mentor": {
            "name":  record["mentor_name"],
            "email": record["mentor_email"],
            "phone": record["mentor_phone"],
        },
        "course_faculty": {
            "name": record["faculty"],
        },
    }


# ── RBAC-aware data access ────────────────────────────────────────────────────

def _scoped_frame(df: pd.DataFrame, identity: RoleIdentity) -> pd.DataFrame:
    """
    Return the subset of df the identity is allowed to see.

      admin   → full frame
      faculty → rows where they are the course instructor or class teacher
      student → only their own row
      unknown → empty frame
    """
    if identity.role == "admin":
        return df
    if identity.role == "faculty":
        return faculty_scope(identity, df)
    if identity.role == "student":
        return student_scope(identity, df)
    # unknown
    return df.iloc[0:0]


# ── Public API ────────────────────────────────────────────────────────────────

def get_user_context(
    data_path: str,
    user_id: str,
    role: str,
    assignments_path: str | None = None,
    identity: RoleIdentity | None = None,
) -> Dict[str, Any]:
    """
    Build the UI context payload.

    Students only see their own profile.
    Faculty only see overview counts/courses scoped to their own sections.
    Admin sees everything.
    """
    from rbac import resolve_identity  # local import avoids circular at module level

    if identity is None:
        identity = resolve_identity(user_id)

    df = _apply_mentor_overrides(_load_student_frame(data_path), assignments_path)
    scoped = _scoped_frame(df, identity)

    # Profile: students get their own; faculty/admin get None (no personal profile card)
    profile: Dict[str, Any] | None = None
    if identity.role == "student":
        record = _find_student_row(df, identity.canonical)
        profile = _record_to_profile(record) if record is not None else None
    elif identity.role in {"faculty", "admin"}:
        # Faculty/admin don't have a student profile card
        profile = None

    # Mentor directory: only from scoped rows (faculty sees their own students' mentors)
    mentor_directory = sorted(
        {
            (row["mentor_name"], row["mentor_email"], row["mentor_phone"])
            for _, row in scoped.iterrows()
            if row["mentor_name"]
        }
    )

    return {
        "role":    identity.role,
        "user_id": user_id,
        "profile": profile,
        "overview": {
            "students": int(scoped.shape[0]),
            "courses":  sorted(scoped["course"].dropna().unique().tolist()),
            "sections": sorted(scoped["section"].dropna().unique().tolist()),
        },
        "mentor_directory": [
            {"name": name, "email": email, "phone": phone}
            for name, email, phone in mentor_directory
        ],
        "faculty_directory": FACULTY_DIRECTORY,
        "permissions": {
            "can_assign_mentor":    identity.can("assign_mentor"),
            "can_view_all_students":identity.can("view_all_students"),
        },
    }


def assign_mentor(
    data_path: str,
    assignments_path: str,
    actor_role: str,
    actor_user_id: str,
    student_id: str,
    mentor_name: str,
    mentor_email: str = "",
    mentor_phone: str = "",
    identity: RoleIdentity | None = None,
) -> Dict[str, Any]:
    """
    Assign a mentor to a student.

    Faculty can only assign mentors to students in their own courses/sections.
    Admin can assign to any student.
    """
    from rbac import resolve_identity, check_permission

    if identity is None:
        identity = resolve_identity(actor_user_id)

    # Top-level permission gate
    check_permission(identity, "assign_mentor")

    df = _load_student_frame(data_path)
    target = _find_student_row(df, student_id)
    if target is None:
        raise ValueError(f"Student '{student_id}' was not found in the dataset.")

    # Faculty scope check: the target student must be in the faculty's own courses
    if identity.role == "faculty":
        scoped = faculty_scope(identity, df)
        if scoped[scoped["student_id"] == target["student_id"]].empty:
            raise PermissionError(
                f"[FACULTY] You can only assign mentors to students in your own courses/sections. "
                f"Student '{student_id}' is not in your scope."
            )

    overrides = _load_mentor_overrides(assignments_path)
    overrides[target["student_id"]] = {
        "mentor_name":  mentor_name.strip(),
        "mentor_email": mentor_email.strip(),
        "mentor_phone": mentor_phone.strip(),
        "assigned_by":  actor_user_id.strip(),
    }
    _save_mentor_overrides(assignments_path, overrides)

    updated_df     = _apply_mentor_overrides(df, assignments_path)
    updated_record = _find_student_row(updated_df, target["student_id"])
    return {
        "message": f"Mentor updated for {target['name']}.",
        "student": _record_to_profile(updated_record),
    }


def retrieve_data(
    data_path: str,
    intent: str,
    entity: str,
    role: str = "unknown",
    user_id: str | None = None,
    assignments_path: str | None = None,
    identity: RoleIdentity | None = None,
) -> Dict[str, Any]:
    """
    Retrieve and filter data according to RBAC rules.

    - admin   : sees all rows, all intents allowed.
    - faculty : sees only rows in their own courses/sections.
                Accessing another faculty's course raises PermissionError.
    - student : can only query data that resolves to their own record.
                Any intent that would expose other students is blocked.
    - unknown : blocked for all organisational intents.
    """
    from rbac import resolve_identity, check_permission, action_for_intent

    if identity is None:
        identity = resolve_identity(user_id or "")

    # ── Permission gate ──────────────────────────────────────────────────────
    required_action = action_for_intent(intent)
    check_permission(identity, required_action)

    # ── Load & scope the base frame ──────────────────────────────────────────
    df_full  = _apply_mentor_overrides(_load_student_frame(data_path), assignments_path)
    df_scoped = _scoped_frame(df_full, identity)

    # Resolve entity references inside the SCOPED frame
    course           = _find_course(df_scoped, entity)
    requester_record = _find_student_row(df_full, _extract_student_id(user_id))

    # For student_profile / mentor / contact / class_teacher intents, the
    # "target" is always the requesting student (students cannot look up others).
    if identity.role == "student":
        target_record = requester_record
    else:
        # Faculty/admin: entity may name a specific student
        target_record = _find_student_row(df_scoped, entity) or requester_record

    # Further narrow by course if one was resolved
    filtered = df_scoped[df_scoped["course"] == course].copy() if course else df_scoped.copy()

    # ── Extra student guard ──────────────────────────────────────────────────
    # Students must never receive other students' data, regardless of intent.
    if identity.role == "student":
        if requester_record is not None:
            usn = requester_record["student_id"]
            filtered = filtered[filtered["student_id"] == usn]
        else:
            filtered = filtered.iloc[0:0]  # no matching record → empty

    # ── Extra faculty guard: cross-course block ──────────────────────────────
    # If the entity resolves to a specific course but the faculty has no rows
    # in that course, deny access entirely.
    if identity.role == "faculty" and course and filtered.empty:
        raise PermissionError(
            f"[FACULTY] You do not have access to course '{course}'. "
            "You can only view data for courses where you are the assigned faculty or class teacher."
        )

    # ── Build result payload ─────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "intent": intent,
        "entity": course or (
            target_record["student_id"]
            if target_record is not None and entity != "general"
            else "general"
        ),
        "records": filtered.to_dict(orient="records"),
        # raw_csv_data is scoped — never leaks data outside the identity's scope
        "raw_csv_data": df_scoped.to_dict(orient="records"),
    }

    # ── Intent-specific summaries ────────────────────────────────────────────

    if intent == "student_count":
        result["summary"] = {
            "count":  int(filtered.shape[0]),
            "course": course or "all_courses",
        }

    elif intent == "course_enrollment":
        result["summary"] = {
            "course":  course or "all_courses",
            "count":   int(filtered.shape[0]),
            "students": filtered[["student_id", "name", "section", "semester"]].head(50).to_dict(orient="records"),
        }

    elif intent == "faculty_list":
        faculty_rows = (
            filtered[["faculty", "class_teacher_name", "mentor_name"]]
            .fillna("").astype(str)
        )
        names = sorted(
            {
                name.strip()
                for name in faculty_rows.values.flatten().tolist()
                if name and name.strip() and name != "0"
            }
        )
        result["summary"] = {
            "course":  course or "all_courses",
            "faculty": names,
            "count":   len(names),
        }

    elif intent == "grades_average":
        tmp = filtered.copy()
        tmp["grade_points"] = tmp["grade"].map(GRADE_MAP).fillna(0)
        if course:
            result["summary"] = {
                "course":              course,
                "average_grade_point": round(float(tmp["grade_points"].mean()), 2),
                "average_cgpa":        round(float(tmp["cgpa"].mean()), 2),
            }
        else:
            grouped = (
                tmp.groupby("course", as_index=False)[["grade_points", "cgpa"]]
                .mean().round(2)
            )
            result["summary"] = {"course_averages": grouped.to_dict(orient="records")}

    elif intent == "attendance_report":
        result["summary"] = {
            "course":                    course or "all_courses",
            "average_attendance_percent":round(float(filtered["attendance_percent"].mean()), 2),
            "student_count":             int(filtered.shape[0]),
        }

    elif intent == "student_profile":
        if target_record is None:
            raise ValueError("No matching student profile was found.")
        result["summary"] = _record_to_profile(target_record)

    elif intent == "mentor_lookup":
        if target_record is None:
            raise ValueError("No matching student was found for mentor lookup.")
        result["summary"] = {
            "student_id": target_record["student_id"],
            "name":       target_record["name"],
            "mentor": {
                "name":  target_record["mentor_name"],
                "email": target_record["mentor_email"],
                "phone": target_record["mentor_phone"],
            },
        }

    elif intent == "class_teacher_info":
        if target_record is None:
            raise ValueError("No matching student was found for class teacher lookup.")
        result["summary"] = {
            "student_id":  target_record["student_id"],
            "name":        target_record["name"],
            "section":     target_record["section"],
            "class_teacher": {
                "name":  target_record["class_teacher_name"],
                "email": target_record["class_teacher_email"],
                "phone": target_record["class_teacher_phone"],
            },
        }

    elif intent == "backlog_report":
        target_frame = filtered
        if identity.role == "student" and requester_record is not None:
            # Student sees only their own backlogs
            target_frame = filtered[filtered["student_id"] == requester_record["student_id"]]
        result["summary"] = {
            "count_with_backlogs": int((target_frame["backlog_count"] > 0).sum()),
            "students": target_frame[target_frame["backlog_count"] > 0][
                ["student_id", "name", "backlog_count", "backlog_subjects"]
            ].head(30).to_dict(orient="records"),
        }

    elif intent == "contact_lookup":
        if target_record is None:
            raise ValueError("No matching student was found for contact lookup.")
        result["summary"] = {
            "student_id": target_record["student_id"],
            "name":       target_record["name"],
            "student_contact": {
                "phone":          target_record["phone"],
                "college_email":  target_record["college_email"],
                "personal_email": target_record["personal_email"],
            },
            "mentor_contact": {
                "name":  target_record["mentor_name"],
                "email": target_record["mentor_email"],
                "phone": target_record["mentor_phone"],
            },
            "class_teacher_contact": {
                "name":  target_record["class_teacher_name"],
                "email": target_record["class_teacher_email"],
                "phone": target_record["class_teacher_phone"],
            },
        }

    else:
        result["summary"] = {
            "count":   int(filtered.shape[0]),
            "message": "Organisational query matched the dataset but no specialised intent was triggered.",
        }

    if identity.role == "student" and requester_record is not None:
        result["requester_context"] = {
            "student_id": requester_record["student_id"],
            "course":     requester_record["course"],
            "section":    requester_record["section"],
        }

    return result