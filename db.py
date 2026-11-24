"""SQLite scan history (only the data needed to show a past result)."""
import json
import sqlite3
from datetime import datetime, timezone

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    thumb         TEXT NOT NULL,
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    top_json      TEXT NOT NULL,
    backend       TEXT,
    inference_ms  REAL,
    low_conf      INTEGER NOT NULL DEFAULT 0,
    quality_note  TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """Create the database file and tables if they don't exist."""
    with sqlite3.connect(app.config["DATABASE"]) as conn:
        conn.executescript(SCHEMA)
    app.teardown_appcontext(close_db)


def add_scan(filename, thumb, result, quality_note=None) -> int:
    cur = get_db().execute(
        "INSERT INTO scans (filename, thumb, label, confidence, top_json, backend, inference_ms, low_conf,"
        " quality_note, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (filename, thumb, result["label"], result["confidence"], json.dumps(result["top"]),
         result["backend"], result["inference_ms"], int(result["low_confidence"]), quality_note,
         datetime.now(timezone.utc).isoformat(timespec="seconds")))
    get_db().commit()
    return cur.lastrowid


def get_scan(scan_id):
    return get_db().execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()


def list_scans():
    return get_db().execute("SELECT * FROM scans ORDER BY created_at DESC, id DESC").fetchall()


def delete_scan(scan_id):
    row = get_scan(scan_id)
    if row:
        get_db().execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        get_db().commit()
    return row
