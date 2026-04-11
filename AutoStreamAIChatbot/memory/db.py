import sqlite3
from datetime import datetime

DB_PATH = "./memory/leads.db"


def init_db():
    """Initialize SQLite database with leads and sessions tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            platform TEXT NOT NULL,
            plan_interest TEXT,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            summary TEXT,
            turn_count INTEGER,
            lead_captured BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_lead(session_id: str, name: str, email: str, platform: str, plan_interest: str = None):
    """Insert a captured lead into the leads table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO leads (session_id,name,email,platform,plan_interest) "
        "VALUES (?,?,?,?,?)",
        (session_id, name, email, platform, plan_interest)
    )
    conn.commit()
    conn.close()


def save_session_summary(session_id: str, summary: str, turn_count: int, lead_captured: bool):
    """Upsert session metadata — insert or update on conflict."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO sessions (session_id,summary,turn_count,lead_captured)
        VALUES (?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
            summary=excluded.summary,
            turn_count=excluded.turn_count,
            lead_captured=excluded.lead_captured,
            updated_at=CURRENT_TIMESTAMP
    """, (session_id, summary, turn_count, lead_captured))
    conn.commit()
    conn.close()


def get_all_leads():
    """Fetch all captured leads from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM leads").fetchall()
    conn.close()
    return rows
