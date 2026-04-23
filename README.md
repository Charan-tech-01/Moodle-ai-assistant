# Moodle AI Assistant — Setup Guide

## Prerequisites
- Python 3.10+
- Docker Desktop (running)
- Git

## 1. Clone the repo

```bash
git clone https://github.com/parkervijay/final.git
cd final
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Set up the database

### Start MariaDB in Docker

```bash
docker run -d --name moodle-db -e MARIADB_ROOT_PASSWORD=rootpass -e MARIADB_DATABASE=moodledb -p 3307:3306 mariadb:latest
```

Wait ~15 seconds, then verify it's running:

```bash
docker exec moodle-db mariadb -u root -prootpass -e "SELECT 1"
```

### Create the database user

```bash
docker exec moodle-db mariadb -u root -prootpass -e "CREATE USER 'moodleai'@'%' IDENTIFIED BY 'moodlepass'; GRANT ALL ON moodledb.* TO 'moodleai'@'%'; FLUSH PRIVILEGES;"
```

### Import the data

Get the `moodle_slim.sql` file from the team shared drive / WhatsApp group (not in the repo — it's 78 MB of real college data).

Place it in the project folder, then:

```bash
docker exec -i moodle-db mariadb -u root -prootpass moodledb < moodle_slim.sql
```

This takes ~1 minute. Verify:

```bash
docker exec moodle-db mariadb -u root -prootpass moodledb -e "SELECT 'users' AS tbl, COUNT(*) AS cnt FROM mdl_user UNION SELECT 'courses', COUNT(*) FROM mdl_course UNION SELECT 'enrolments', COUNT(*) FROM mdl_user_enrolments UNION SELECT 'grades', COUNT(*) FROM mdl_grade_grades UNION SELECT 'attendance_logs', COUNT(*) FROM mdl_attendance_log"
```

Expected output:
```
users           6,920
courses           869
enrolments     55,306
grades        561,066
attendance     40,261
```

## 4. Create your .env file

Create a file called `.env` in the project root with:

```
DB_HOST=127.0.0.1
DB_PORT=3307
DB_USER=moodleai
DB_PASSWORD=moodlepass
DB_NAME=moodledb
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key from https://console.groq.com/keys

### Test the database connection

```bash
python db.py
```

Should print row counts for all tables.

## 5. Run the server

```bash
python -m uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

### Test queries

Try these user IDs:
- `9` — Student (ANANYA P, ISE department)
- `2` — Faculty/Admin (LMS Administrator)

Sample queries:
- "show me my attendance"
- "what courses am I enrolled in"
- "show my student profile"
- "show faculty for Operating Systems"
- "how many students are there"

## Troubleshooting

**"uvicorn.exe was blocked by Device Guard"**
→ Use `python -m uvicorn main:app --reload` instead

**"Can't connect to MySQL server"**
→ Run `docker start moodle-db` — the container probably stopped

**"Access denied for user"**
→ Make sure DB_PORT is 3307 (not 3306) in your .env

**"'cryptography' package is required"**
→ Run `pip install cryptography`

## Architecture

```
main.py                 → FastAPI server, routes, LLM calls
agentic_workflow.py     → Multi-agent orchestrator (5 agents + trace)
classifier.py           → Query classification (LLM + heuristic)
data_retriever.py       → SQL queries against Moodle database
rbac.py                 → Role detection from mdl_user_info_data
db.py                   → Database connection (MariaDB)
auth.py                 → Legacy auth (kept for reference)
response_formatter.py   → PDF/Word/Excel/TXT export
static/                 → Frontend (HTML/CSS/JS)
```
