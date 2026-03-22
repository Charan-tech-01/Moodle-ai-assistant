import re


def detect_role(user_id: str) -> str:
    if not user_id:
        return "unknown"

    normalized = user_id.strip().upper()
    prefix = normalized[:3]
    if prefix == "STU":
        return "student"
    if prefix == "FAC":
        return "faculty"
    if prefix == "ADM":
        return "admin"
    if re.match(r"^\d[A-Z0-9]{6,}$", normalized):
        return "student"
    return "unknown"
