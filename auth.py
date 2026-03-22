"""
rbac.py — Role-Based Access Control for Moodle AI Assistant
============================================================
Roles
-----
  student   : can only access their own record (profile, grades, mentor,
              class teacher, contacts, backlogs).
  faculty   : can access data for courses/sections where they are the
              assigned faculty or class teacher. Cannot view other
              faculty's courses or arbitrary students.
  admin     : unrestricted access to all data and all actions.
  unknown   : read-only access to general (non-organisational) queries only.

Permission actions
------------------
  view_own_profile        – student views their own record
  view_student_record     – faculty/admin views any student record
  view_course_data        – faculty views their own courses; admin views all
  view_all_students       – admin only
  assign_mentor           – faculty (own course students) or admin
  view_backlog_report     – faculty (own course) or admin
  view_contact_details    – student (own) or faculty (own course) or admin
  general_query           – all roles including unknown
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── Immutable permission tables ──────────────────────────────────────────────

# Maps role → set of allowed actions (no context needed for these)
_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "view_own_profile",
        "view_student_record",
        "view_course_data",
        "view_all_students",
        "assign_mentor",
        "view_backlog_report",
        "view_contact_details",
        "general_query",
    },
    "faculty": {
        "view_course_data",        # scoped to own courses — enforced at data layer
        "view_student_record",     # scoped to own courses — enforced at data layer
        "assign_mentor",           # scoped to own courses — enforced at data layer
        "view_backlog_report",     # scoped to own courses — enforced at data layer
        "view_contact_details",    # scoped to own courses — enforced at data layer
        "general_query",
    },
    "student": {
        "view_own_profile",        # profile, mentor, class teacher, backlogs — own record only
        "view_contact_details",    # own contacts only — enforced at data layer
        "view_backlog_report",     # own backlogs only — enforced at data layer
        "general_query",
    },
    "unknown": {
        "general_query",
    },
}

# Human-readable denial messages per action
_DENIAL_MESSAGES: dict[str, str] = {
    "view_own_profile":    "Students can only view their own profile.",
    "view_student_record": "You do not have permission to view student records.",
    "view_course_data":    "You do not have permission to view course data.",
    "view_all_students":   "Only administrators can view all students.",
    "assign_mentor":       "Only faculty and administrators can assign mentors.",
    "view_backlog_report": "You do not have permission to view backlog reports.",
    "view_contact_details":"You do not have permission to view contact details.",
    "general_query":       "Access denied.",
}


# ── Role identity dataclass ──────────────────────────────────────────────────

@dataclass
class RoleIdentity:
    """
    Carries the resolved role and the canonical user identifier.

    Attributes
    ----------
    role        : One of 'student', 'faculty', 'admin', 'unknown'.
    user_id     : The raw user_id string as provided.
    canonical   : Normalised identifier (USN for students, ID for faculty/admin).
    permissions : Set of allowed action strings for this role.
    """
    role: str
    user_id: str
    canonical: str
    permissions: set[str] = field(default_factory=set)

    def can(self, action: str) -> bool:
        return action in self.permissions

    def __repr__(self) -> str:
        return f"RoleIdentity(role={self.role!r}, canonical={self.canonical!r})"


# ── Public API ───────────────────────────────────────────────────────────────

def detect_role(user_id: str) -> str:
    """
    Lightweight string-only role detection kept for backward compatibility
    with existing call sites in main.py that only need the role string.
    """
    return _resolve_identity(user_id).role


def resolve_identity(user_id: str) -> RoleIdentity:
    """
    Full identity resolution — returns a RoleIdentity with role, canonical
    ID, and the set of permitted actions.
    """
    return _resolve_identity(user_id)


def check_permission(identity: RoleIdentity, action: str) -> None:
    """
    Raise PermissionError if the identity is not allowed to perform action.
    Call this before serving any data.
    """
    if not identity.can(action):
        msg = _DENIAL_MESSAGES.get(action, f"Action '{action}' is not permitted for role '{identity.role}'.")
        raise PermissionError(f"[{identity.role.upper()}] {msg}")


def faculty_scope(identity: RoleIdentity, df: Any) -> Any:
    """
    Return the subset of df the faculty member is allowed to see.
    Matches on faculty_id / class_teacher_id columns first (preferred —
    these exist in the updated CSV), then falls back to name matching.

    Scope = rows where the faculty is the course instructor OR class teacher.
    Admin → full DataFrame. Non-faculty → empty DataFrame.
    """
    if identity.role == "admin":
        return df

    if identity.role != "faculty":
        return df.iloc[0:0]

    canonical = identity.canonical.upper()   # e.g. "FAC001"

    # ── Priority 1: match on *_id columns (new CSV format) ──────────────────
    has_fac_id = "faculty_id"        in df.columns
    has_ct_id  = "class_teacher_id"  in df.columns

    if has_fac_id or has_ct_id:
        masks = []
        if has_fac_id:
            masks.append(df["faculty_id"].str.strip().str.upper() == canonical)
        if has_ct_id:
            masks.append(df["class_teacher_id"].str.strip().str.upper() == canonical)
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        scoped = df[combined]
        if not scoped.empty:
            return scoped

    # ── Priority 2: match on name columns (old CSV / fallback) ──────────────
    faculty_col = df["faculty"].str.strip().str.upper()           if "faculty"            in df.columns else None
    ct_col      = df["class_teacher_name"].str.strip().str.upper() if "class_teacher_name" in df.columns else None

    if faculty_col is not None and ct_col is not None:
        mask = (faculty_col == canonical) | (ct_col == canonical)
    elif faculty_col is not None:
        mask = faculty_col == canonical
    elif ct_col is not None:
        mask = ct_col == canonical
    else:
        return df.iloc[0:0]

    scoped = df[mask]
    if not scoped.empty:
        return scoped

    # ── Priority 3: fuzzy name search using the raw user_id string ──────────
    return _faculty_scope_by_name(identity, df)


def student_scope(identity: RoleIdentity, df: Any) -> Any:
    """
    Return only the single row for the requesting student.
    """
    if identity.role == "admin":
        return df

    canonical = identity.canonical.upper()
    return df[df["student_id"].str.upper() == canonical]


# ── Intent → required action mapping ────────────────────────────────────────

#: Maps classifier intent strings to the RBAC action that must be permitted.
INTENT_ACTION_MAP: dict[str, str] = {
    # Faculty/admin only
    "student_count":     "view_course_data",
    "course_enrollment": "view_course_data",
    "faculty_list":      "view_course_data",
    "grades_average":    "view_course_data",
    "attendance_report": "view_course_data",
    # Self-access — students allowed on own record
    "student_profile":   "view_own_profile",
    "mentor_lookup":     "view_own_profile",
    "class_teacher_info":"view_own_profile",
    "backlog_report":    "view_backlog_report",
    "contact_lookup":    "view_contact_details",
    "general":           "general_query",
}


def action_for_intent(intent: str) -> str:
    """Return the RBAC action string that corresponds to a classifier intent."""
    return INTENT_ACTION_MAP.get(intent, "view_course_data")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_identity(user_id: str) -> RoleIdentity:
    if not user_id:
        return RoleIdentity(
            role="unknown",
            user_id="",
            canonical="",
            permissions=set(_ROLE_PERMISSIONS["unknown"]),
        )

    normalized = user_id.strip().upper()
    prefix = normalized[:3]

    if prefix == "ADM":
        role = "admin"
        canonical = normalized
    elif prefix == "FAC":
        role = "faculty"
        canonical = normalized
    elif prefix == "STU":
        role = "student"
        # Strip the STU prefix to get the actual USN
        canonical = re.sub(r"^STU[-_:]?", "", normalized) or normalized
    elif re.match(r"^\d[A-Z0-9]{6,}$", normalized):
        role = "student"
        canonical = normalized
    else:
        role = "unknown"
        canonical = normalized

    return RoleIdentity(
        role=role,
        user_id=user_id,
        canonical=canonical,
        permissions=set(_ROLE_PERMISSIONS.get(role, _ROLE_PERMISSIONS["unknown"])),
    )


def _faculty_scope_by_name(identity: RoleIdentity, df: Any) -> Any:
    """
    Fallback: try to match faculty by partial name rather than ID.
    This handles datasets where the 'faculty' column stores names, not IDs.
    Uses the part of the canonical ID after 'FAC' as a name hint.
    """
    # e.g. FAC001 → we have no name, so return empty
    # If the user_id itself looks like a name we can search; otherwise empty.
    raw = identity.user_id.strip()
    if raw.upper().startswith("FAC"):
        return df.iloc[0:0]

    name_hint = raw.upper()
    faculty_col = df["faculty"].str.strip().str.upper() if "faculty" in df.columns else None
    ct_col      = df["class_teacher_name"].str.strip().str.upper() if "class_teacher_name" in df.columns else None

    if faculty_col is not None and ct_col is not None:
        mask = faculty_col.str.contains(name_hint, na=False) | ct_col.str.contains(name_hint, na=False)
    elif faculty_col is not None:
        mask = faculty_col.str.contains(name_hint, na=False)
    elif ct_col is not None:
        mask = ct_col.str.contains(name_hint, na=False)
    else:
        return df.iloc[0:0]

    return df[mask]