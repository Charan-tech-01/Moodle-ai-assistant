"""
data_retriever.py

Data retrieval layer for the Moodle AI Assistant.
Queries the real Moodle MariaDB database instead of CSV files.

Each intent maps to one or more SQL queries against Moodle's schema:
  mdl_user, mdl_course, mdl_user_enrolments, mdl_enrol,
  mdl_grade_grades, mdl_grade_items, mdl_attendance_log,
  mdl_attendance_sessions, mdl_attendance_statuses,
  mdl_role_assignments, mdl_role, mdl_context, mdl_user_info_data
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from db import get_connection


def _clean(obj: Any) -> Any:
    """Recursively convert Decimal -> float so JSON serialization works."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Course search helper
# ---------------------------------------------------------------------------

def _find_course(entity: str) -> Optional[Dict[str, Any]]:
    """
    Find a course by fuzzy matching on fullname or shortname.
    Returns dict with id, shortname, fullname or None.
    """
    if not entity or entity.lower() == "general":
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Try exact match first
            cur.execute(
                "SELECT id, shortname, fullname FROM mdl_course "
                "WHERE fullname = %s OR shortname = %s LIMIT 1",
                (entity, entity),
            )
            row = cur.fetchone()
            if row:
                return row

            # Fuzzy match
            pattern = f"%{entity}%"
            cur.execute(
                "SELECT id, shortname, fullname FROM mdl_course "
                "WHERE fullname LIKE %s OR shortname LIKE %s "
                "ORDER BY id LIMIT 1",
                (pattern, pattern),
            )
            return cur.fetchone()


def _find_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Find a user by numeric ID, username, or name search."""
    if not user_id:
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Try numeric ID
            try:
                numeric = int(user_id)
                cur.execute(
                    "SELECT id, username, firstname, lastname, email, "
                    "phone1, phone2, department, institution "
                    "FROM mdl_user WHERE id = %s AND deleted = 0",
                    (numeric,),
                )
                row = cur.fetchone()
                if row:
                    return row
            except (ValueError, TypeError):
                pass

            # Try username
            cur.execute(
                "SELECT id, username, firstname, lastname, email, "
                "phone1, phone2, department, institution "
                "FROM mdl_user WHERE username = %s AND deleted = 0",
                (user_id.strip(),),
            )
            row = cur.fetchone()
            if row:
                return row

            # Try name search
            pattern = f"%{user_id}%"
            cur.execute(
                "SELECT id, username, firstname, lastname, email, "
                "phone1, phone2, department, institution "
                "FROM mdl_user WHERE deleted = 0 "
                "AND (CONCAT(firstname, ' ', lastname) LIKE %s) "
                "LIMIT 1",
                (pattern,),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------

def _student_count(course: Optional[Dict], entity: str) -> Dict[str, Any]:
    """Count students enrolled in a course (or all courses)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if course:
                cur.execute(
                    "SELECT COUNT(DISTINCT ue.userid) AS cnt "
                    "FROM mdl_user_enrolments ue "
                    "JOIN mdl_enrol e ON e.id = ue.enrolid "
                    "WHERE e.courseid = %s AND ue.status = 0",
                    (course["id"],),
                )
                count = cur.fetchone()["cnt"]
                return {
                    "count": count,
                    "course": course["fullname"],
                }
            else:
                # Count all unique students (users with Student role)
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM mdl_user_info_data "
                    "WHERE fieldid = 1 AND TRIM(data) = 'Student'",
                )
                count = cur.fetchone()["cnt"]
                return {"count": count, "course": "all_courses"}


def _course_enrollment(course: Optional[Dict], entity: str, user: Optional[Dict] = None) -> Dict[str, Any]:
    """List students enrolled in a course, or a student's own courses."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if course:
                cur.execute(
                    "SELECT u.id AS student_id, "
                    "CONCAT(u.firstname, ' ', u.lastname) AS name, "
                    "u.username, u.email "
                    "FROM mdl_user_enrolments ue "
                    "JOIN mdl_enrol e ON e.id = ue.enrolid "
                    "JOIN mdl_user u ON u.id = ue.userid "
                    "WHERE e.courseid = %s AND ue.status = 0 "
                    "AND u.deleted = 0 "
                    "ORDER BY u.lastname, u.firstname "
                    "LIMIT 50",
                    (course["id"],),
                )
                students = cur.fetchall()
                return {
                    "course": course["fullname"],
                    "count": len(students),
                    "students": students,
                }
            elif user:
                # Student asking about their own enrolled courses
                cur.execute(
                    "SELECT c.id, c.fullname, c.shortname "
                    "FROM mdl_user_enrolments ue "
                    "JOIN mdl_enrol e ON e.id = ue.enrolid "
                    "JOIN mdl_course c ON c.id = e.courseid "
                    "WHERE ue.userid = %s AND ue.status = 0 "
                    "ORDER BY c.fullname",
                    (user["id"],),
                )
                courses = cur.fetchall()
                fullname = f"{user['firstname']} {user['lastname']}".strip()
                return {
                    "course": "all_courses",
                    "student_name": fullname,
                    "count": len(courses),
                    "enrolled_courses": courses,
                }
            else:
                return {
                    "course": "all_courses",
                    "count": 0,
                    "students": [],
                    "message": "Please specify a course name.",
                }


def _faculty_list(course: Optional[Dict], entity: str) -> Dict[str, Any]:
    """List faculty (editing teachers + teachers) for a course."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = (
                "SELECT DISTINCT "
                "CONCAT(u.firstname, ' ', u.lastname) AS name, "
                "u.email, r.shortname AS role_type "
                "FROM mdl_role_assignments ra "
                "JOIN mdl_context ctx ON ctx.id = ra.contextid "
                "JOIN mdl_user u ON u.id = ra.userid "
                "JOIN mdl_role r ON r.id = ra.roleid "
                "WHERE ctx.contextlevel = 50 "  # course context
                "AND r.shortname IN ('editingteacher', 'teacher') "
                "AND u.deleted = 0 "
            )
            params = []
            if course:
                query += "AND ctx.instanceid = %s "
                params.append(course["id"])

            query += "ORDER BY name LIMIT 50"
            cur.execute(query, params)
            faculty = cur.fetchall()
            names = [f["name"] for f in faculty]
            return {
                "course": course["fullname"] if course else "all_courses",
                "faculty": names,
                "faculty_details": faculty,
                "count": len(names),
            }


def _grades_average(course: Optional[Dict], entity: str) -> Dict[str, Any]:
    """Average final grade for a course or across all courses."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if course:
                cur.execute(
                    "SELECT AVG(gg.finalgrade) AS avg_grade, "
                    "COUNT(gg.id) AS grade_count "
                    "FROM mdl_grade_grades gg "
                    "JOIN mdl_grade_items gi ON gi.id = gg.itemid "
                    "WHERE gi.courseid = %s "
                    "AND gi.itemtype = 'course' "
                    "AND gg.finalgrade IS NOT NULL",
                    (course["id"],),
                )
                row = cur.fetchone()
                return {
                    "course": course["fullname"],
                    "average_grade": round(float(row["avg_grade"] or 0), 2),
                    "grade_count": row["grade_count"],
                }
            else:
                # Per-course averages
                cur.execute(
                    "SELECT c.fullname AS course, "
                    "ROUND(AVG(gg.finalgrade), 2) AS avg_grade, "
                    "COUNT(gg.id) AS grade_count "
                    "FROM mdl_grade_grades gg "
                    "JOIN mdl_grade_items gi ON gi.id = gg.itemid "
                    "JOIN mdl_course c ON c.id = gi.courseid "
                    "WHERE gi.itemtype = 'course' "
                    "AND gg.finalgrade IS NOT NULL "
                    "GROUP BY c.id, c.fullname "
                    "ORDER BY avg_grade DESC "
                    "LIMIT 10",
                )
                averages = cur.fetchall()
                return {"course_averages": averages}


def _attendance_report(
    course: Optional[Dict],
    entity: str,
    user: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Attendance report from mdl_attendance_log.
    Can filter by course AND/OR specific student.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = (
                "SELECT "
                "u.id AS student_id, "
                "CONCAT(u.firstname, ' ', u.lastname) AS name, "
                "COUNT(al.id) AS total_sessions, "
                "SUM(CASE WHEN ast.acronym = 'P' THEN 1 ELSE 0 END) AS present, "
                "SUM(CASE WHEN ast.acronym = 'L' THEN 1 ELSE 0 END) AS late, "
                "SUM(CASE WHEN ast.acronym = 'A' THEN 1 ELSE 0 END) AS absent, "
                "SUM(CASE WHEN ast.acronym = 'E' THEN 1 ELSE 0 END) AS excused, "
                "ROUND("
                "  SUM(CASE WHEN ast.acronym IN ('P','L','E') THEN 1 ELSE 0 END) "
                "  * 100.0 / NULLIF(COUNT(al.id), 0), 2"
                ") AS attendance_percent "
                "FROM mdl_attendance_log al "
                "JOIN mdl_attendance_sessions asess ON asess.id = al.sessionid "
                "JOIN mdl_attendance a ON a.id = asess.attendanceid "
                "JOIN mdl_course c ON c.id = a.course "
                "JOIN mdl_user u ON u.id = al.studentid "
                "JOIN mdl_attendance_statuses ast ON ast.id = al.statusid "
                "WHERE u.deleted = 0 "
            )
            params: list = []

            if course:
                query += "AND c.id = %s "
                params.append(course["id"])

            if user:
                query += "AND u.id = %s "
                params.append(user["id"])

            query += "GROUP BY u.id, name ORDER BY name "

            # If asking about a specific student, get their row
            if user:
                cur.execute(query, params)
                rows = cur.fetchall()
                if rows:
                    r = rows[0]
                    return {
                        "course": course["fullname"] if course else "all_courses",
                        "student_id": r["student_id"],
                        "student_name": r["name"],
                        "attendance_percent": float(r["attendance_percent"] or 0),
                        "total_sessions": r["total_sessions"],
                        "present": r["present"],
                        "absent": r["absent"],
                        "late": r["late"],
                        "excused": r["excused"],
                        "student_count": 1,
                    }

            # Class-wide summary
            query += "LIMIT 50"
            cur.execute(query, params)
            rows = cur.fetchall()

            if rows:
                avg_pct = sum(float(r["attendance_percent"] or 0) for r in rows) / len(rows)
            else:
                avg_pct = 0

            return {
                "course": course["fullname"] if course else "all_courses",
                "average_attendance_percent": round(avg_pct, 2),
                "student_count": len(rows),
                "students": rows[:12],  # sample
            }


def _student_profile(user: Optional[Dict]) -> Dict[str, Any]:
    """Full profile for a student: info + enrolled courses + grades."""
    if not user:
        raise ValueError("No matching student profile was found.")

    moodle_id = user["id"]
    fullname = f"{user['firstname']} {user['lastname']}".strip()

    profile: Dict[str, Any] = {
        "student_id": moodle_id,
        "name": fullname,
        "username": user["username"],
        "email": user["email"],
        "phone": user.get("phone1", ""),
        "department": user.get("department", ""),
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Enrolled courses
            cur.execute(
                "SELECT c.id, c.fullname, c.shortname "
                "FROM mdl_user_enrolments ue "
                "JOIN mdl_enrol e ON e.id = ue.enrolid "
                "JOIN mdl_course c ON c.id = e.courseid "
                "WHERE ue.userid = %s AND ue.status = 0 "
                "ORDER BY c.fullname",
                (moodle_id,),
            )
            profile["enrolled_courses"] = cur.fetchall()
            profile["course_count"] = len(profile["enrolled_courses"])

            # Course total grades
            cur.execute(
                "SELECT c.fullname AS course, "
                "gg.finalgrade AS grade, gi.grademax AS max_grade "
                "FROM mdl_grade_grades gg "
                "JOIN mdl_grade_items gi ON gi.id = gg.itemid "
                "JOIN mdl_course c ON c.id = gi.courseid "
                "WHERE gg.userid = %s AND gi.itemtype = 'course' "
                "AND gg.finalgrade IS NOT NULL "
                "ORDER BY c.fullname",
                (moodle_id,),
            )
            grades = cur.fetchall()
            profile["grades"] = grades

            # Compute average grade percentage
            if grades:
                pcts = [
                    float(g["grade"]) / float(g["max_grade"]) * 100
                    for g in grades
                    if g["max_grade"] and float(g["max_grade"]) > 0
                ]
                profile["average_grade_percent"] = round(sum(pcts) / len(pcts), 2) if pcts else 0
            else:
                profile["average_grade_percent"] = 0

    return profile


def _mentor_lookup(user: Optional[Dict], assignments_path: str) -> Dict[str, Any]:
    """
    Mentor lookup — uses mentor_assignments.json overlay
    since mentor data isn't in Moodle core.
    """
    if not user:
        raise ValueError("No matching student was found for mentor lookup.")

    fullname = f"{user['firstname']} {user['lastname']}".strip()
    moodle_id = str(user["id"])

    # Check overlay file
    path = Path(assignments_path)
    mentor = {"name": "", "email": "", "phone": ""}
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
            if moodle_id in overrides:
                mentor = overrides[moodle_id]
        except json.JSONDecodeError:
            pass

    return {
        "student_id": user["id"],
        "name": fullname,
        "mentor": {
            "name": mentor.get("mentor_name", "Not assigned"),
            "email": mentor.get("mentor_email", ""),
            "phone": mentor.get("mentor_phone", ""),
        },
    }


def _backlog_report(course: Optional[Dict], user: Optional[Dict]) -> Dict[str, Any]:
    """
    Backlog = course total grade below passing threshold.
    We define 'backlog' as finalgrade < 40% of grademax.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = (
                "SELECT u.id AS student_id, "
                "CONCAT(u.firstname, ' ', u.lastname) AS name, "
                "c.fullname AS course, "
                "gg.finalgrade AS grade, gi.grademax AS max_grade "
                "FROM mdl_grade_grades gg "
                "JOIN mdl_grade_items gi ON gi.id = gg.itemid "
                "JOIN mdl_course c ON c.id = gi.courseid "
                "JOIN mdl_user u ON u.id = gg.userid "
                "WHERE gi.itemtype = 'course' "
                "AND gg.finalgrade IS NOT NULL "
                "AND gi.grademax > 0 "
                "AND (gg.finalgrade / gi.grademax) < 0.4 "
                "AND u.deleted = 0 "
            )
            params: list = []

            if course:
                query += "AND c.id = %s "
                params.append(course["id"])

            if user:
                query += "AND u.id = %s "
                params.append(user["id"])

            query += "ORDER BY name LIMIT 30"
            cur.execute(query, params)
            rows = cur.fetchall()

            students: list = []
            seen: dict = {}
            for r in rows:
                sid = r["student_id"]
                if sid not in seen:
                    seen[sid] = {
                        "student_id": sid,
                        "name": r["name"],
                        "backlog_count": 0,
                        "backlog_courses": [],
                    }
                seen[sid]["backlog_count"] += 1
                seen[sid]["backlog_courses"].append(r["course"])

            students = list(seen.values())

            return {
                "count_with_backlogs": len(students),
                "students": students,
            }


def _contact_lookup(user: Optional[Dict]) -> Dict[str, Any]:
    """Contact details for a student."""
    if not user:
        raise ValueError("No matching student was found for contact lookup.")

    fullname = f"{user['firstname']} {user['lastname']}".strip()
    return {
        "student_id": user["id"],
        "name": fullname,
        "student_contact": {
            "email": user.get("email", ""),
            "phone": user.get("phone1", ""),
            "phone2": user.get("phone2", ""),
        },
    }


# ---------------------------------------------------------------------------
# User context (dashboard data)
# ---------------------------------------------------------------------------

def get_user_context(
    data_path: str = "",
    user_id: str = "",
    role: str = "unknown",
    assignments_path: str = "",
) -> Dict[str, Any]:
    """
    Build a role-scoped dashboard context for the user.
    Replaces the CSV-based version.
    """
    user = _find_user(user_id)
    profile = None

    if user:
        fullname = f"{user['firstname']} {user['lastname']}".strip()
        profile = {
            "id": user["id"],
            "name": fullname,
            "username": user["username"],
            "email": user["email"],
            "department": user.get("department", ""),
        }

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Total students
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM mdl_user_info_data "
                "WHERE fieldid = 1 AND TRIM(data) = 'Student'",
            )
            student_count = cur.fetchone()["cnt"]

            # Course list (just names, limited)
            cur.execute(
                "SELECT fullname FROM mdl_course "
                "WHERE id > 1 ORDER BY fullname LIMIT 20",
            )
            courses = [r["fullname"] for r in cur.fetchall()]

    return _clean({
        "role": role,
        "user_id": user_id,
        "profile": profile,
        "overview": {
            "students": student_count,
            "courses": courses,
        },
        "permissions": {
            "can_assign_mentor": role in {"faculty", "admin"},
            "can_view_all_students": role in {"faculty", "admin"},
        },
    })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrieve_data(
    data_path: str = "",
    intent: str = "general",
    entity: str = "general",
    role: str = "unknown",
    user_id: str = "",
    assignments_path: str = "",
) -> Dict[str, Any]:
    """
    Main retrieval function. Routes by intent to the appropriate SQL query.
    Keeps the same signature as the CSV version so callers don't change.
    """
    course = _find_course(entity)
    user = _find_user(user_id)
    target_user = _find_user(entity) if entity and entity.lower() != "general" else None

    # For student-specific intents, prefer target_user, fall back to requester
    lookup_user = target_user or user

    result: Dict[str, Any] = {
        "intent": intent,
        "entity": course["fullname"] if course else (entity if entity != "general" else "general"),
        "records": [],       # kept for backward compat with _compact_payload
        "summary": {},
    }

    if intent == "student_count":
        result["summary"] = _student_count(course, entity)

    elif intent == "course_enrollment":
        result["summary"] = _course_enrollment(course, entity, user if role == "student" else None)

    elif intent == "faculty_list":
        result["summary"] = _faculty_list(course, entity)

    elif intent == "grades_average":
        result["summary"] = _grades_average(course, entity)

    elif intent == "attendance_report":
        # Pass the specific student if role is student (self-query)
        att_user = user if role == "student" else lookup_user
        result["summary"] = _attendance_report(course, entity, att_user if role == "student" else None)

    elif intent == "student_profile":
        result["summary"] = _student_profile(lookup_user)

    elif intent == "mentor_lookup":
        result["summary"] = _mentor_lookup(lookup_user, assignments_path)

    elif intent == "class_teacher_info":
        # Map to faculty list for the student's courses
        if lookup_user:
            result["summary"] = _faculty_list(course, entity)
            result["summary"]["note"] = "Showing course teachers (class teacher is not tracked separately in Moodle)."
        else:
            raise ValueError("No matching student was found for class teacher lookup.")

    elif intent == "backlog_report":
        result["summary"] = _backlog_report(course, lookup_user if role == "student" else None)

    elif intent == "contact_lookup":
        result["summary"] = _contact_lookup(lookup_user)

    else:
        result["summary"] = {
            "message": "Query matched the database but no specialized intent was triggered.",
        }

    # Add requester context for student role
    if role == "student" and user:
        result["requester_context"] = {
            "student_id": user["id"],
            "name": f"{user['firstname']} {user['lastname']}".strip(),
        }

    return _clean(result)


# ---------------------------------------------------------------------------
# Mentor assignment (kept from original — writes to JSON overlay)
# ---------------------------------------------------------------------------

def assign_mentor(
    data_path: str = "",
    assignments_path: str = "",
    actor_role: str = "",
    actor_user_id: str = "",
    student_id: str = "",
    mentor_name: str = "",
    mentor_email: str = "",
    mentor_phone: str = "",
) -> Dict[str, Any]:
    """Assign a mentor to a student. Writes to JSON overlay file."""
    if actor_role not in {"faculty", "admin"}:
        raise PermissionError("Only faculty or admin users can assign mentors.")

    student = _find_user(student_id)
    if not student:
        raise ValueError(f"Student '{student_id}' was not found.")

    path = Path(assignments_path)
    overrides: dict = {}
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    sid = str(student["id"])
    overrides[sid] = {
        "mentor_name": mentor_name.strip(),
        "mentor_email": mentor_email.strip(),
        "mentor_phone": mentor_phone.strip(),
        "assigned_by": actor_user_id.strip(),
    }
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    fullname = f"{student['firstname']} {student['lastname']}".strip()
    return {
        "message": f"Mentor updated for {fullname}.",
        "student_id": student["id"],
        "student_name": fullname,
        "mentor": overrides[sid],
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Student Count (all) ===")
    r = retrieve_data(intent="student_count", entity="general")
    print(r["summary"])

    print("\n=== Course Enrollment (first course match) ===")
    r = retrieve_data(intent="course_enrollment", entity="Operating Systems")
    print(r["summary"])

    print("\n=== Faculty List ===")
    r = retrieve_data(intent="faculty_list", entity="Operating Systems")
    print(r["summary"])

    print("\n=== Student Profile (user 9) ===")
    r = retrieve_data(intent="student_profile", entity="9", user_id="9")
    print(r["summary"])

    print("\n=== Attendance Report (user 9) ===")
    r = retrieve_data(intent="attendance_report", entity="general", role="student", user_id="9")
    print(r["summary"])
