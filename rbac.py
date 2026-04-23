"""
rbac.py

Role-Based Access Control for the Moodle AI Assistant.
Detects user role by querying mdl_user_info_data (fieldid=1 = User_Role).
Accepts either a numeric Moodle user ID or a username string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from db import get_connection


# ---------------------------------------------------------------------------
# Permissions per role
# ---------------------------------------------------------------------------

_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "view_own_data",
        "view_student_data",
        "view_course_data",
        "assign_mentor",
        "general_query",
    },
    "faculty": {
        "view_course_data",
        "view_own_data",
        "assign_mentor",
        "general_query",
    },
    "student": {
        "view_own_data",
        "general_query",
    },
    "staff": {
        "view_course_data",
        "general_query",
    },
    "unknown": {
        "general_query",
    },
}

_DENIED: dict[str, str] = {
    "view_own_data": "Access denied.",
    "view_student_data": "You do not have permission to view student records.",
    "view_course_data": "You do not have permission to view course data.",
    "assign_mentor": "Only faculty and administrators can assign mentors.",
    "general_query": "Access denied.",
}


# ---------------------------------------------------------------------------
# Identity dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoleIdentity:
    role: str
    user_id: str
    moodle_id: int              # numeric mdl_user.id
    username: str               # mdl_user.username
    fullname: str               # firstname + lastname
    email: str
    permissions: set[str] = field(default_factory=set)

    def can(self, action: str) -> bool:
        return action in self.permissions


# ---------------------------------------------------------------------------
# Intent -> required action mapping
# ---------------------------------------------------------------------------

_SELF_INTENTS = {
    "student_profile", "mentor_lookup", "class_teacher_info",
    "contact_lookup", "grades_average", "attendance_report", "backlog_report",
    "course_enrollment", "student_count", "faculty_list",
}

_INTENT_ACTION = {
    "student_count": "view_course_data",
    "course_enrollment": "view_course_data",
    "faculty_list": "view_course_data",
    "grades_average": "view_own_data",
    "attendance_report": "view_own_data",
    "backlog_report": "view_own_data",
    "student_profile": "view_own_data",
    "mentor_lookup": "view_own_data",
    "class_teacher_info": "view_own_data",
    "contact_lookup": "view_own_data",
    "general": "general_query",
}


def action_for_intent_and_role(intent: str, role: str) -> str:
    if intent == "general":
        return "general_query"
    if role == "student" and intent in _SELF_INTENTS:
        return "view_own_data"
    if role in {"faculty", "admin"}:
        return "view_course_data"
    return _INTENT_ACTION.get(intent, "view_course_data")


# ---------------------------------------------------------------------------
# Core: resolve identity from DB
# ---------------------------------------------------------------------------

def _lookup_user(user_id: str) -> Optional[dict]:
    """
    Find a user by numeric ID or username.
    Returns dict with id, username, firstname, lastname, email or None.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Try numeric ID first
            try:
                numeric_id = int(user_id)
                cur.execute(
                    "SELECT id, username, firstname, lastname, email "
                    "FROM mdl_user WHERE id = %s AND deleted = 0",
                    (numeric_id,),
                )
                row = cur.fetchone()
                if row:
                    return row
            except (ValueError, TypeError):
                pass

            # Try username match
            cur.execute(
                "SELECT id, username, firstname, lastname, email "
                "FROM mdl_user WHERE username = %s AND deleted = 0",
                (user_id.strip(),),
            )
            return cur.fetchone()


def _lookup_role(moodle_id: int) -> str:
    """
    Get role from mdl_user_info_data (fieldid=1 = User_Role).
    Returns 'student', 'faculty', 'staff', or 'unknown'.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM mdl_user_info_data "
                "WHERE userid = %s AND fieldid = 1",
                (moodle_id,),
            )
            row = cur.fetchone()
            if not row:
                return "unknown"
            raw = row["data"].strip().lower()
            if raw == "student":
                return "student"
            if raw == "faculty":
                return "faculty"
            if raw == "staff":
                return "staff"
            return "unknown"


def resolve_identity(user_id: str) -> RoleIdentity:
    """
    Main entry point. Accepts a Moodle user ID (numeric) or username.
    Returns a RoleIdentity with role, permissions, and user info.
    """
    if not user_id or not user_id.strip():
        return RoleIdentity(
            role="unknown", user_id="", moodle_id=0,
            username="", fullname="", email="",
            permissions=set(_PERMISSIONS["unknown"]),
        )

    user = _lookup_user(user_id)
    if not user:
        return RoleIdentity(
            role="unknown", user_id=user_id, moodle_id=0,
            username="", fullname="", email="",
            permissions=set(_PERMISSIONS["unknown"]),
        )

    moodle_id = user["id"]
    role = _lookup_role(moodle_id)
    fullname = f"{user['firstname']} {user['lastname']}".strip()

    return RoleIdentity(
        role=role,
        user_id=user_id,
        moodle_id=moodle_id,
        username=user["username"],
        fullname=fullname,
        email=user["email"],
        permissions=set(_PERMISSIONS.get(role, _PERMISSIONS["unknown"])),
    )


def detect_role(user_id: str) -> str:
    """Backward-compatible wrapper — returns just the role string."""
    return resolve_identity(user_id).role


def check_permission(identity: RoleIdentity, action: str) -> None:
    if not identity.can(action):
        msg = _DENIED.get(action, "Access denied.")
        raise PermissionError(f"[{identity.role.upper()}] {msg}")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    test_id = sys.argv[1] if len(sys.argv) > 1 else "2"
    identity = resolve_identity(test_id)
    print(f"User ID:     {identity.user_id}")
    print(f"Moodle ID:   {identity.moodle_id}")
    print(f"Username:    {identity.username}")
    print(f"Full name:   {identity.fullname}")
    print(f"Email:       {identity.email}")
    print(f"Role:        {identity.role}")
    print(f"Permissions: {identity.permissions}")
