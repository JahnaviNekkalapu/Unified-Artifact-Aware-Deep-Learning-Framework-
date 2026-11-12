"""
SQLite-backed restoration history service for ReliefLink V5.

Records every completed restoration run.  Image arrays are NOT stored
in the database; only file references and metadata are persisted.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Database lives beside app.py
DB_PATH = Path(__file__).resolve().parents[1] / "Unified Artifcat_history.db"


# ──────────────────────────────────────────────────────────────────────────────
# Schema bootstrap
# ──────────────────────────────────────────────────────────────────────────────

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS restoration_runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    input_filename  TEXT,
    width           INTEGER,
    height          INTEGER,
    input_format    TEXT,
    duration_s      REAL,
    status          TEXT,
    artifact_label  TEXT,
    artifact_score  REAL,
    artifact_json   TEXT,
    psnr_db         REAL,
    ssim            REAL,
    mae             REAL,
    rmse            REAL,
    sam             REAL,
    output_filename TEXT
);
"""

_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON restoration_runs(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_status    ON restoration_runs(status);",
]


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _bootstrap() -> None:
    with _get_connection() as conn:
        conn.execute(_CREATE_SQL)
        for idx in _INDEXES_SQL:
            conn.execute(idx)
        conn.commit()


_bootstrap()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def new_run_id() -> str:
    return str(uuid.uuid4())


def record_run(
    *,
    run_id: str,
    input_filename: str,
    width: int,
    height: int,
    input_format: str,
    duration_s: float,
    status: str,                  # "success" | "failed"
    artifact_scores: dict[str, float] | None = None,
    metrics: dict[str, float] | None = None,
    output_filename: str | None = None,
) -> None:
    """Insert a completed run into the history database."""

    ts = datetime.now(timezone.utc).isoformat()

    # Top artifact
    artifact_label: str | None = None
    artifact_score: float | None = None
    artifact_json:  str | None = None

    if artifact_scores:
        artifact_json = json.dumps(artifact_scores)
        top = max(artifact_scores, key=artifact_scores.__getitem__)
        artifact_label = top
        artifact_score = artifact_scores[top]

    m = metrics or {}

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO restoration_runs
                (run_id, timestamp, input_filename, width, height,
                 input_format, duration_s, status,
                 artifact_label, artifact_score, artifact_json,
                 psnr_db, ssim, mae, rmse, sam, output_filename)
            VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id, ts, input_filename, width, height,
                input_format, duration_s, status,
                artifact_label, artifact_score, artifact_json,
                m.get("PSNR_dB"), m.get("SSIM"), m.get("MAE"),
                m.get("RMSE"),    m.get("SAM"),
                output_filename,
            ),
        )
        conn.commit()


def fetch_all_runs(
    limit: int = 500,
    search: str = "",
) -> list[dict[str, Any]]:
    """Return all runs, newest first, optionally filtered by filename."""
    with _get_connection() as conn:
        if search:
            like = f"%{search}%"
            rows = conn.execute(
                """
                SELECT * FROM restoration_runs
                WHERE input_filename LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM restoration_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def fetch_run(run_id: str) -> dict[str, Any] | None:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM restoration_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def summary_stats() -> dict[str, Any]:
    """Aggregate stats for the overview dashboard."""
    with _get_connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM restoration_runs"
        ).fetchone()[0]
        success = conn.execute(
            "SELECT COUNT(*) FROM restoration_runs WHERE status = 'success'"
        ).fetchone()[0]
        avg_dur = conn.execute(
            "SELECT AVG(duration_s) FROM restoration_runs WHERE status = 'success'"
        ).fetchone()[0]
        recent = conn.execute(
            """
            SELECT run_id, timestamp, input_filename, duration_s,
                   status, artifact_label, artifact_score
            FROM restoration_runs
            ORDER BY timestamp DESC
            LIMIT 5
            """
        ).fetchall()

    return {
        "total":         total,
        "success":       success,
        "avg_duration":  avg_dur,
        "recent":        [dict(r) for r in recent],
    }


def delete_run(run_id: str) -> None:
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM restoration_runs WHERE run_id = ?",
            (run_id,),
        )
        conn.commit()
