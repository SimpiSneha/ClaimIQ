"""
Database setup for ClaimIQ.
Uses SQLite for simplicity - good enough for a portfolio project.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "claimiq.db"


def get_connection():
    """Returns a new SQLite connection. Call this per-request, not globally."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db():
    """Creates the claims table if it doesn't exist yet. Safe to call every startup."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
            claimant_name TEXT NOT NULL,
            policy_number TEXT NOT NULL,
            description TEXT NOT NULL,
            photo_path TEXT,
            status TEXT DEFAULT 'submitted',
            fraud_score REAL,
            damage_severity TEXT,
            coverage_answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Run this file directly to initialize the DB: python database.py
    init_db()
    print(f"Database initialized at {DB_PATH}")