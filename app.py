"""FastAPI REST API exposing CodePlug-PB SQLite data.

This app is meant to sit on top of the SQLite database produced by `hpdb_to_sqlite.py`.

Run:
  uvicorn app:app --host 0.0.0.0 --port 16444

Or:
  python3 app.py

Environment:
  HPDB_PATH: path to the SQLite db (default: ./hpdb_default.sqlite)
  ZIP_CSV_PATH: ZIP metadata CSV (default: ./uszips.csv)
  HOST / PORT: used when running `python3 app.py`
"""

import math
import os
import sqlite3
import io
import shutil
import tempfile
import threading
import uuid
import hashlib
import time
import configparser
import hmac
import random
import requests
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("HPDB_PATH", "hpdb_default.sqlite")
ZIP_CSV_PATH = os.environ.get("ZIP_CSV_PATH", "uszips.csv")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
API_KEYS_PATH = os.environ.get("API_KEYS_PATH", "keys.ini")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def get_conn():
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"CodePlug-PB database not found at {DB_PATH!r}. Upload MasterHpdb.hp1 via /hpdb/admin/upload-master to initialize.",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _row_get(row: sqlite3.Row, key: str) -> Any:
    try:
        if hasattr(row, "keys") and key in row.keys():
            return row[key]
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def get_app_metadata():
    return {
        "db_path": DB_PATH,
        "db_exists": os.path.exists(DB_PATH),
        "api_keys_path": API_KEYS_PATH,
    }


@lru_cache(maxsize=1)
def load_zip_db():
    zips = {}
    if not os.path.exists(ZIP_CSV_PATH):
        return zips
    import csv

    with open(ZIP_CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            z = row.get("zip") or row.get("ZIP") or row.get("Zip")
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lng") or row.get("lon") or row.get("longitude")
            if not z or not lat or not lon:
                continue
            try:
                zips[z] = (float(lat), float(lon))
            except ValueError:
                continue
    return zips


@lru_cache(maxsize=1)
def load_zip_db_full():
    """
    ZIP metadata lookup keyed by ZIP code.
    Keeps more fields than load_zip_db() so we can map ZIP -> county/state for conventional frequencies.
    """
    zips = {}
    if not os.path.exists(ZIP_CSV_PATH):
        return zips
    import csv

    with open(ZIP_CSV_PATH, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            z = row.get("zip") or row.get("ZIP") or row.get("Zip")
            lat = row.get("lat") or row.get("latitude")
            lon = row.get("lng") or row.get("lon") or row.get("longitude")
            if not z or not lat or not lon:
                continue
            try:
                zips[str(z)] = {
                    "zip": str(z),
                    "lat": float(lat),
                    "lon": float(lon),
                    "county_name": (row.get("county_name") or "").strip() or None,
                    "state_id": (row.get("state_id") or "").strip() or None,
                    "state_name": (row.get("state_name") or "").strip() or None,
                }
            except ValueError:
                continue
    return zips


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def hz_to_mhz(freq_hz: int | None) -> float | None:
    if freq_hz is None:
        return None
    return float(freq_hz) / 1_000_000.0


@dataclass(frozen=True)
class CountyKey:
    state_id: int
    state_abbrev: str
    state_name: str
    county_id: int
    county_name: str


def resolve_county(db: sqlite3.Connection, state: str, county: str) -> CountyKey:
    q_state = norm(state)
    st_rows = db.execute(
        """
        SELECT StateId, name AS StateName, abbrev AS StateAbbrev
        FROM state_info
        WHERE lower(abbrev) = ?
           OR lower(name) = ?
           OR lower(name) LIKE ?
        ORDER BY CASE WHEN lower(abbrev)=? THEN 0 WHEN lower(name)=? THEN 1 ELSE 2 END, name
        """,
        (q_state, q_state, f"%{q_state}%", q_state, q_state),
    ).fetchall()
    if not st_rows:
        raise HTTPException(status_code=404, detail=f"State not found: {state!r}")
    st = st_rows[0]

    q_county = norm(county)
    c_rows = db.execute(
        """
        SELECT CountyId, name AS CountyName
        FROM county_info
        WHERE StateId = ?
          AND (lower(name) = ? OR lower(name) LIKE ?)
        ORDER BY CASE WHEN lower(name)=? THEN 0 ELSE 1 END, name
        """,
        (int(st["StateId"]), q_county, f"%{q_county}%", q_county),
    ).fetchall()
    if not c_rows:
        raise HTTPException(status_code=404, detail=f"County not found in {st['StateAbbrev']}: {county!r}")
    c = c_rows[0]

    return CountyKey(
        state_id=int(st["StateId"]),
        state_abbrev=str(st["StateAbbrev"] or ""),
        state_name=str(st["StateName"] or ""),
        county_id=int(c["CountyId"]),
        county_name=str(c["CountyName"] or ""),
    )


class CountyMatch(BaseModel):
    state_id: int
    state_abbrev: str
    state_name: str
    county_id: int
    county_name: str


class ConventionalRow(BaseModel):
    department: Optional[str] = None
    channel: Optional[str] = None
    frequency_hz: Optional[int] = None
    frequency_mhz: Optional[float] = None
    modulation: Optional[str] = None
    tone: Optional[str] = None
    nac: Optional[str] = None
    color_code: Optional[str] = None
    ran: Optional[str] = None
    number_tag: Optional[int] = None
    avoid: Optional[str] = None


class TalkgroupRow(BaseModel):
    trunk_id: int
    system_name: Optional[str] = None
    system_type: Optional[str] = None
    category: Optional[str] = None
    tid: int
    alpha_tag: Optional[str] = None
    talkgroup: Optional[str] = None
    service: Optional[str] = None
    number_tag: Optional[int] = None
    avoid: Optional[str] = None


class SiteFrequencyRow(BaseModel):
    frequency_hz: Optional[int] = None
    frequency_mhz: Optional[float] = None
    channel_name: Optional[str] = None
    lcn1: Optional[int] = None
    lcn2: Optional[int] = None
    ran: Optional[str] = None
    avoid: Optional[str] = None


class TrunkSite(BaseModel):
    site_id: int
    site_name: Optional[str] = None
    distance_miles: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    frequencies: List[SiteFrequencyRow] = []


class TrunkSystem(BaseModel):
    trunk_id: int
    system_name: Optional[str] = None
    system_type: Optional[str] = None
    sites: List[TrunkSite] = []


class CountyQueryResponse(BaseModel):
    county: CountyMatch
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_miles: Optional[float] = None
    conventional: List[ConventionalRow]
    trunked: List[TrunkSystem]
    talkgroups: List[TalkgroupRow]


app = FastAPI(title="CodePlug-PB REST API", version="0.1.0", description="REST wrapper over local CodePlug-PB SQLite")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _startup_init_state() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.state.jobs = {}
    app.state.jobs_lock = threading.Lock()
    app.state.db_swap_lock = threading.Lock()


def _load_enabled_api_keys() -> List[str]:
    """
    Loads enabled API keys from keys.ini.

    Format:
      [api_keys]
      SOME-KEY = enabled
      OTHER-KEY = disabled
    """
    if not os.path.exists(API_KEYS_PATH):
        raise HTTPException(status_code=500, detail=f"API keys file missing: {API_KEYS_PATH!r}")
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve case for API keys
    cp.read(API_KEYS_PATH)
    if "api_keys" not in cp:
        return []
    enabled = []
    for key, value in cp["api_keys"].items():
        if str(value).strip().lower() in ("1", "true", "yes", "enabled", "on"):
            # configparser lowercases keys by default; preserve exactness by reading raw file later if needed.
            enabled.append(key)
    return enabled


def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    keys = _load_enabled_api_keys()
    # Accept either exact match or case-insensitive match to be resilient to older keys.ini
    # that may have been written with lowercased keys.
    candidate = x_api_key.strip()
    for k in keys:
        k_norm = str(k).strip()
        if hmac.compare_digest(candidate, k_norm) or hmac.compare_digest(candidate.lower(), k_norm.lower()):
            return candidate
    raise HTTPException(status_code=403, detail="Invalid API key")


def validate_api_key_value(api_key: str) -> str:
    """
    Same validation as require_api_key(), but for an explicit value (e.g. HTML form field).
    """
    candidate = (api_key or "").strip()
    if not candidate:
        raise HTTPException(status_code=401, detail="Missing API key")
    keys = _load_enabled_api_keys()
    for k in keys:
        k_norm = str(k).strip()
        if hmac.compare_digest(candidate, k_norm) or hmac.compare_digest(candidate.lower(), k_norm.lower()):
            return candidate
    raise HTTPException(status_code=403, detail="Invalid API key")


class JobStatus(BaseModel):
    job_id: str
    kind: str
    status: str  # queued|running|success|error
    detail: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


def _set_job(job_key: str, **patch: Any) -> None:
    with app.state.jobs_lock:
        cur = app.state.jobs.get(job_key, {})
        cur.update(patch)
        app.state.jobs[job_key] = cur


def _convert_masterhp1_to_sqlite(master_hp1_path: str, out_sqlite_path: str) -> None:
    """
    Build a SQLite DB from an uploaded MasterHpdb.hp1 using the local converter.
    Writes to out_sqlite_path (which must be a file path, not a directory).
    """
    import hpdb_to_sqlite

    with tempfile.TemporaryDirectory(prefix="hpdb_import_") as tmpdir:
        tmp_master = os.path.join(tmpdir, "MasterHpdb.hp1")
        shutil.copy2(master_hp1_path, tmp_master)

        # If a local hpdb.cfg exists, include it (helps ensure reference tables exist even if missing).
        cfg_src = os.path.join(os.getcwd(), "hpdb.cfg")
        if os.path.exists(cfg_src):
            shutil.copy2(cfg_src, os.path.join(tmpdir, "hpdb.cfg"))

        rc = hpdb_to_sqlite.main(["--input", tmpdir, "--out", out_sqlite_path, "--overwrite"])
        if rc != 0:
            raise RuntimeError(f"hpdb_to_sqlite failed with exit code {rc}")


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_app_meta(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )


def _set_app_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    _ensure_app_meta(conn)
    conn.execute("INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)", (key, value))


def _get_app_meta(db_path: str, key: str) -> Optional[str]:
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='app_meta'").fetchone()
        if not row:
            return None
        r2 = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        return str(r2[0]) if r2 else None
    except Exception:
        return None
    finally:
        conn.close()


def _merge_update_into_new_db(*, active_path: str, update_path: str, merged_path: str) -> Dict[str, int]:
    """
    Merge (UPSERT-like) from update_path into a copy of active_path at merged_path, then VACUUM.
    Uses `INSERT OR REPLACE` for broad SQLite compatibility (avoids requiring newer UPSERT syntax).
    Returns change counts per table (inserts + replaces).
    """
    shutil.copy2(active_path, merged_path)
    conn = sqlite3.connect(merged_path)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("ATTACH DATABASE ? AS upd", (update_path,))

        inserted: Dict[str, int] = {}

        def do(sql: str, name: str) -> None:
            before = conn.total_changes
            conn.execute(sql)
            inserted[name] = conn.total_changes - before

        # Parents first
        do(
            """
            INSERT OR REPLACE INTO state_info (StateId, CountryId, name, abbrev, last_source_file_id)
            SELECT StateId, CountryId, name, abbrev, last_source_file_id FROM upd.state_info
            """,
            "state_info",
        )
        do(
            """
            INSERT OR REPLACE INTO county_info (CountyId, StateId, name, last_source_file_id)
            SELECT CountyId, StateId, name, last_source_file_id FROM upd.county_info
            """,
            "county_info",
        )
        do(
            """
            INSERT OR REPLACE INTO trunk (TrunkId, StateId, name, avoid, date_modified, system_type, nxdn_flavor, extras_json, last_source_file_id)
            SELECT TrunkId, StateId, name, avoid, date_modified, system_type, nxdn_flavor, extras_json, last_source_file_id FROM upd.trunk
            """,
            "trunk",
        )
        do(
            """
            INSERT OR REPLACE INTO site (SiteId, TrunkId, name, avoid, lat, lon, range, location_mode, site_type, bandwidth, shape, search_hint, extras_json, last_source_file_id)
            SELECT SiteId, TrunkId, name, avoid, lat, lon, range, location_mode, site_type, bandwidth, shape, search_hint, extras_json, last_source_file_id FROM upd.site
            """,
            "site",
        )
        do(
            """
            INSERT OR REPLACE INTO t_group (TGroupId, TrunkId, name, avoid, lat, lon, range, shape, extras_json, last_source_file_id)
            SELECT TGroupId, TrunkId, name, avoid, lat, lon, range, shape, extras_json, last_source_file_id FROM upd.t_group
            """,
            "t_group",
        )
        do(
            """
            INSERT OR REPLACE INTO tgid (Tid, TGroupId, alpha_tag, avoid, talkgroup_value, service, number_tag, extras_json, last_source_file_id)
            SELECT Tid, TGroupId, alpha_tag, avoid, talkgroup_value, service, number_tag, extras_json, last_source_file_id FROM upd.tgid
            """,
            "tgid",
        )
        do(
            """
            INSERT OR REPLACE INTO t_freq (TFreqId, SiteId, channel_name, avoid, frequency_hz, lcn1, lcn2, ran, area, extras_json, last_source_file_id)
            SELECT TFreqId, SiteId, channel_name, avoid, frequency_hz, lcn1, lcn2, ran, area, extras_json, last_source_file_id FROM upd.t_freq
            """,
            "t_freq",
        )
        do(
            """
            INSERT OR REPLACE INTO c_group (CGroupId, CountyId, AgencyId, name, avoid, lat, lon, range, shape, extra1, extra2, last_source_file_id)
            SELECT CGroupId, CountyId, AgencyId, name, avoid, lat, lon, range, shape, extra1, extra2, last_source_file_id FROM upd.c_group
            """,
            "c_group",
        )
        do(
            """
            INSERT OR REPLACE INTO c_freq (CFreqId, CGroupId, name, avoid, frequency_hz, modulation, tone, nac, color_code, ran, number_tag, extras_json, last_source_file_id)
            SELECT CFreqId, CGroupId, name, avoid, frequency_hz, modulation, tone, nac, color_code, ran, number_tag, extras_json, last_source_file_id FROM upd.c_freq
            """,
            "c_freq",
        )

        # Location mapping / helpers
        do(
            """
            INSERT OR REPLACE INTO lm (StateId, CountyId, TrunkId, SiteId, unk0, unk1, lat, lon, last_source_file_id)
            SELECT StateId, CountyId, TrunkId, SiteId, unk0, unk1, lat, lon, last_source_file_id FROM upd.lm
            """,
            "lm",
        )
        do(
            """
            INSERT OR REPLACE INTO lm_frequency (frequency_hz, unk0, code, last_source_file_id)
            SELECT frequency_hz, unk0, code, last_source_file_id FROM upd.lm_frequency
            """,
            "lm_frequency",
        )

        # These tables use surrogate PKs in our schema; do NOT copy IDs. Insert rows if the mapping tuple is new.
        do(
            """
            INSERT INTO area_state (StateId, CountyId, AgencyId, TrunkId, last_source_file_id)
            SELECT u.StateId, u.CountyId, u.AgencyId, u.TrunkId, u.last_source_file_id
            FROM upd.area_state u
            WHERE NOT EXISTS (
              SELECT 1 FROM area_state a
              WHERE a.StateId IS u.StateId
                AND a.CountyId IS u.CountyId
                AND a.AgencyId IS u.AgencyId
                AND a.TrunkId IS u.TrunkId
            )
            """,
            "area_state",
        )
        do(
            """
            INSERT INTO area_county (CountyId, AgencyId, TrunkId, last_source_file_id)
            SELECT u.CountyId, u.AgencyId, u.TrunkId, u.last_source_file_id
            FROM upd.area_county u
            WHERE NOT EXISTS (
              SELECT 1 FROM area_county a
              WHERE a.CountyId IS u.CountyId
                AND a.AgencyId IS u.AgencyId
                AND a.TrunkId IS u.TrunkId
            )
            """,
            "area_county",
        )

        # Optional decode helpers
        do(
            """
            INSERT OR REPLACE INTO bandplan_mot (SiteId, values_json, last_source_file_id)
            SELECT SiteId, values_json, last_source_file_id FROM upd.bandplan_mot
            """,
            "bandplan_mot",
        )
        do(
            """
            INSERT OR REPLACE INTO bandplan_p25 (SiteId, values_json, last_source_file_id)
            SELECT SiteId, values_json, last_source_file_id FROM upd.bandplan_p25
            """,
            "bandplan_p25",
        )
        do(
            """
            INSERT OR REPLACE INTO fleetmap (TrunkId, values_json, last_source_file_id)
            SELECT TrunkId, values_json, last_source_file_id FROM upd.fleetmap
            """,
            "fleetmap",
        )

        # Apply deletions for selected tables where the update master is the source of truth.
        # This keeps the active DB from accumulating removed records over time.
        do(
            """
            DELETE FROM tgid
            WHERE NOT EXISTS (SELECT 1 FROM upd.tgid u WHERE u.Tid IS tgid.Tid)
            """,
            "tgid_deleted",
        )
        do(
            """
            DELETE FROM t_freq
            WHERE NOT EXISTS (
              SELECT 1 FROM upd.t_freq u
              WHERE u.TFreqId IS t_freq.TFreqId
                AND u.SiteId IS t_freq.SiteId
                AND u.frequency_hz IS t_freq.frequency_hz
            )
            """,
            "t_freq_deleted",
        )
        do(
            """
            DELETE FROM c_freq
            WHERE NOT EXISTS (SELECT 1 FROM upd.c_freq u WHERE u.CFreqId IS c_freq.CFreqId)
            """,
            "c_freq_deleted",
        )
        do(
            """
            DELETE FROM t_group
            WHERE NOT EXISTS (SELECT 1 FROM upd.t_group u WHERE u.TGroupId IS t_group.TGroupId)
            """,
            "t_group_deleted",
        )
        do(
            """
            DELETE FROM site
            WHERE NOT EXISTS (SELECT 1 FROM upd.site u WHERE u.SiteId IS site.SiteId)
            """,
            "site_deleted",
        )
        do(
            """
            DELETE FROM trunk
            WHERE NOT EXISTS (SELECT 1 FROM upd.trunk u WHERE u.TrunkId IS trunk.TrunkId)
            """,
            "trunk_deleted",
        )
        do(
            """
            DELETE FROM bandplan_mot
            WHERE NOT EXISTS (SELECT 1 FROM upd.bandplan_mot u WHERE u.SiteId IS bandplan_mot.SiteId)
            """,
            "bandplan_mot_deleted",
        )
        do(
            """
            DELETE FROM bandplan_p25
            WHERE NOT EXISTS (SELECT 1 FROM upd.bandplan_p25 u WHERE u.SiteId IS bandplan_p25.SiteId)
            """,
            "bandplan_p25_deleted",
        )
        do(
            """
            DELETE FROM fleetmap
            WHERE NOT EXISTS (SELECT 1 FROM upd.fleetmap u WHERE u.TrunkId IS fleetmap.TrunkId)
            """,
            "fleetmap_deleted",
        )

        conn.commit()
        conn.execute("DETACH DATABASE upd")
        try:
            conn.execute("VACUUM")
            conn.commit()
        except sqlite3.OperationalError:
            # VACUUM may fail if the filesystem doesn't have enough temporary space.
            # The merged DB is still valid; it just won't be as compact.
            conn.commit()
        return inserted
    finally:
        conn.close()


def _compute_table_delta_simple(
    conn: sqlite3.Connection,
    *,
    table: str,
    key_cols: List[str],
    compare_cols: List[str],
    old_alias: str,
    new_alias: str,
) -> Dict[str, int]:
    """
    Compute added/removed/changed counts between old_alias.table and new_alias.table.
    Uses `IS` comparisons to treat NULLs as equal.
    """
    join_on = " AND ".join([f"o.\"{c}\" IS n.\"{c}\"" for c in key_cols])
    missing_side = f"o.\"{key_cols[0]}\" IS NULL"
    removed_side = f"n.\"{key_cols[0]}\" IS NULL"
    diff_where = " OR ".join([f"o.\"{c}\" IS NOT n.\"{c}\"" for c in compare_cols]) or "0"

    added = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {new_alias}.\"{table}\" n
        LEFT JOIN {old_alias}.\"{table}\" o
          ON {join_on}
        WHERE {missing_side}
        """
    ).fetchone()[0]
    removed = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {old_alias}.\"{table}\" o
        LEFT JOIN {new_alias}.\"{table}\" n
          ON {join_on}
        WHERE {removed_side}
        """
    ).fetchone()[0]
    changed = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {old_alias}.\"{table}\" o
        JOIN {new_alias}.\"{table}\" n
          ON {join_on}
        WHERE ({diff_where})
        """
    ).fetchone()[0]

    return {"added": int(added), "removed": int(removed), "changed": int(changed)}


def _compute_primary_update_delta(active_path: str, update_path: str) -> Dict[str, Dict[str, int]]:
    """
    Compute a human-friendly delta between the currently active DB and the newly uploaded master DB.
    Returned dict is keyed by table name with {added, removed, changed}.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("ATTACH DATABASE ? AS old", (active_path,))
        conn.execute("ATTACH DATABASE ? AS new", (update_path,))

        return {
            "trunk": _compute_table_delta_simple(
                conn,
                table="trunk",
                key_cols=["TrunkId"],
                compare_cols=[
                    "StateId",
                    "name",
                    "avoid",
                    "date_modified",
                    "system_type",
                    "nxdn_flavor",
                    "extras_json",
                ],
                old_alias="old",
                new_alias="new",
            ),
            "site": _compute_table_delta_simple(
                conn,
                table="site",
                key_cols=["SiteId"],
                compare_cols=[
                    "TrunkId",
                    "name",
                    "avoid",
                    "lat",
                    "lon",
                    "range",
                    "location_mode",
                    "site_type",
                    "bandwidth",
                    "shape",
                    "search_hint",
                    "extras_json",
                ],
                old_alias="old",
                new_alias="new",
            ),
            "tgid": _compute_table_delta_simple(
                conn,
                table="tgid",
                key_cols=["Tid"],
                compare_cols=[
                    "TGroupId",
                    "alpha_tag",
                    "avoid",
                    "talkgroup_value",
                    "service",
                    "number_tag",
                    "extras_json",
                ],
                old_alias="old",
                new_alias="new",
            ),
            "t_freq": _compute_table_delta_simple(
                conn,
                table="t_freq",
                key_cols=["TFreqId", "SiteId", "frequency_hz"],
                compare_cols=[
                    "channel_name",
                    "avoid",
                    "lcn1",
                    "lcn2",
                    "ran",
                    "area",
                    "extras_json",
                ],
                old_alias="old",
                new_alias="new",
            ),
            "c_freq": _compute_table_delta_simple(
                conn,
                table="c_freq",
                key_cols=["CFreqId"],
                compare_cols=[
                    "CGroupId",
                    "name",
                    "avoid",
                    "frequency_hz",
                    "modulation",
                    "tone",
                    "nac",
                    "color_code",
                    "ran",
                    "number_tag",
                    "extras_json",
                ],
                old_alias="old",
                new_alias="new",
            ),
        }
    finally:
        conn.close()


@app.get("/hpdb/admin/status")
def admin_status():
    return {
        **get_app_metadata(),
        "upload_dir": UPLOAD_DIR,
        "endpoints": {
            "upload_master": "/hpdb/admin/upload-master",
            "update_master": "/hpdb/admin/update-master",
            "jobs": "/hpdb/admin/jobs/{job_id}",
        },
    }


@app.get("/hpdb/admin/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    with app.state.jobs_lock:
        job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)


def _background_import_master(job_id: str, saved_hp1_path: str) -> None:
    _set_job(job_id, status="running")
    tmp_sqlite = os.path.join(UPLOAD_DIR, f"{job_id}.sqlite")
    try:
        sha = _sha256_file(saved_hp1_path)
        _convert_masterhp1_to_sqlite(saved_hp1_path, tmp_sqlite)
        # Stamp metadata into the new DB before swapping it into place.
        conn = sqlite3.connect(tmp_sqlite)
        try:
            _set_app_meta(conn, "masterhp1_sha256", sha)
            _set_app_meta(conn, "imported_at_unix", str(int(time.time())))
            conn.commit()
        finally:
            conn.close()
        with app.state.db_swap_lock:
            os.replace(tmp_sqlite, DB_PATH)
        _set_job(job_id, status="success", result={"db_path": DB_PATH})
    except Exception as e:
        _set_job(job_id, status="error", detail=str(e))
        try:
            if os.path.exists(tmp_sqlite):
                os.remove(tmp_sqlite)
        except Exception:
            pass


def _queue_upload_master(background: BackgroundTasks, *, job_id: str, saved_path: str) -> JobStatus:
    _set_job(job_id, **{"job_id": job_id, "kind": "upload-master", "status": "queued", "detail": None, "result": None})
    background.add_task(_background_import_master, job_id, saved_path)
    return JobStatus(**app.state.jobs[job_id])


@app.post("/hpdb/admin/upload-master", response_model=JobStatus)
def upload_master(
    background: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
    file: UploadFile = File(..., description="Upload MasterHpdb.hp1"),
):
    if os.path.exists(DB_PATH):
        raise HTTPException(status_code=409, detail="Database already exists; use /hpdb/admin/update-master instead.")

    job_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_MasterHpdb.hp1")
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    return _queue_upload_master(background, job_id=job_id, saved_path=saved_path)


def _background_update_master(job_id: str, saved_hp1_path: str) -> None:
    _set_job(job_id, status="running")
    update_sqlite = os.path.join(UPLOAD_DIR, f"{job_id}_hpdb_update.sqlite")
    merged_sqlite = os.path.join(UPLOAD_DIR, f"{job_id}_hpdb_merged.sqlite")
    try:
        sha = _sha256_file(saved_hp1_path)
        prev_sha = _get_app_meta(DB_PATH, "masterhp1_sha256")
        if prev_sha and prev_sha == sha:
            _set_job(
                job_id,
                status="success",
                result={
                    "skipped": True,
                    "reason": "uploaded MasterHpdb.hp1 matches active DB checksum",
                    "db_path": DB_PATH,
                },
            )
            return

        # Build update DB (never touching the active DB).
        _convert_masterhp1_to_sqlite(saved_hp1_path, update_sqlite)

        # Merge into a new DB based on a copy of the active DB, then swap atomically.
        with app.state.db_swap_lock:
            delta = _compute_primary_update_delta(DB_PATH, update_sqlite)
            changes = _merge_update_into_new_db(active_path=DB_PATH, update_path=update_sqlite, merged_path=merged_sqlite)
            changes_total = sum(changes.values())
            if changes_total > 0:
                # Stamp new checksum metadata into the merged DB before swapping.
                conn = sqlite3.connect(merged_sqlite)
                try:
                    _set_app_meta(conn, "masterhp1_sha256", sha)
                    _set_app_meta(conn, "updated_at_unix", str(int(time.time())))
                    conn.commit()
                finally:
                    conn.close()
                os.replace(merged_sqlite, DB_PATH)
                swapped = True
            else:
                swapped = False
                if os.path.exists(merged_sqlite):
                    os.remove(merged_sqlite)

        _set_job(
            job_id,
            status="success",
            result={
                "update_db": update_sqlite,
                "delta": delta,
                "changes": changes,
                "changes_total": changes_total,
                "swapped": swapped,
                "skipped": False,
                "db_path": DB_PATH,
            },
        )
    except Exception as e:
        _set_job(job_id, status="error", detail=str(e))
        for p in (merged_sqlite,):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.post("/hpdb/admin/update-master", response_model=JobStatus)
def update_master(
    background: BackgroundTasks,
    _api_key: str = Depends(require_api_key),
    file: UploadFile = File(..., description="Upload updated MasterHpdb.hp1"),
):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=409, detail="Database does not exist; use /hpdb/admin/upload-master first.")

    job_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_MasterHpdb.hp1")
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    _set_job(job_id, **{"job_id": job_id, "kind": "update-master", "status": "queued", "detail": None, "result": None})
    background.add_task(_background_update_master, job_id, saved_path)
    return JobStatus(**app.state.jobs[job_id])


@app.get("/", include_in_schema=False)
def root():
    if not os.path.exists(DB_PATH):
        return RedirectResponse(url="/initialize", status_code=302)
    if os.path.exists(os.path.join(STATIC_DIR, "map.html")):
        return RedirectResponse(url="/map", status_code=302)
    return RedirectResponse(url="/docs", status_code=302)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # serve an empty 1x1 gif to avoid 404 noise
    empty_gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    return Response(content=empty_gif, media_type="image/gif")


@app.get("/about", include_in_schema=False, response_class=HTMLResponse)
def about_page():
    about_path = os.path.join(STATIC_DIR, "about.html")
    if not os.path.exists(about_path):
        raise HTTPException(status_code=404, detail="about.html not found in static/")
    return HTMLResponse(Path(about_path).read_text(encoding="utf-8"))


def _pick_404_image() -> str:
    """
    Pick a random 404 image from static/ matching 404*.png, fallback to 404.png.
    """
    try:
        names = [
            fn
            for fn in os.listdir(STATIC_DIR)
            if fn.lower().startswith("404") and fn.lower().endswith(".png") and os.path.isfile(os.path.join(STATIC_DIR, fn))
        ]
    except FileNotFoundError:
        names = []
    if not names:
        return "404.png"
    return random.choice(names)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """
    Custom 404: return the branded HTML page when the client prefers HTML; otherwise JSON.
    """
    accept = (request.headers.get("accept") or "").lower()
    wants_html = "text/html" in accept or "*/*" in accept or not accept
    img_name = _pick_404_image()
    img_src = f"/static/{img_name}"
    if wants_html:
        template_path = os.path.join(STATIC_DIR, "404.html")
        if os.path.exists(template_path):
            body = (
                Path(template_path)
                .read_text(encoding="utf-8")
                .replace("STATIC_404_IMAGE", img_src)
                .replace("STATIC_404_NAME", img_name)
            )
        else:
            # minimal fallback that still shows the provided PNG
            body = f"""<!doctype html><html><head><meta charset='utf-8'><title>Not Found</title></head><body style='margin:0;background:#03060c;display:flex;align-items:center;justify-content:center;height:100vh;'><img src='{img_src}' alt='404 - Not Found'></body></html>"""
        return HTMLResponse(body, status_code=404)

    detail = exc.detail if isinstance(exc, HTTPException) and exc.detail else "Not Found"
    return JSONResponse({"detail": detail}, status_code=404)


@app.get("/map", include_in_schema=False, response_class=HTMLResponse)
def map_page():
    """
    Simple front-end that visualizes US states and links to county details.
    """
    map_path = os.path.join(STATIC_DIR, "map.html")
    if not os.path.exists(map_path):
        raise HTTPException(status_code=404, detail="map.html not found in static/")
    with open(map_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/map/state", include_in_schema=False, response_class=HTMLResponse)
def map_state_page():
    """
    County-level map for a single state. Expects ?state=FIPS (02-digit).
    """
    state_path = os.path.join(STATIC_DIR, "state.html")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=404, detail="state.html not found in static/")
    with open(state_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def _render_state_page_with_seo(state_abbrev: str | None = None, county_name: str | None = None) -> str:
    """
    Helper to inject data attributes for SEO-friendly county URLs into state.html.
    """
    state_path = os.path.join(STATIC_DIR, "state.html")
    if not os.path.exists(state_path):
        raise HTTPException(status_code=404, detail="state.html not found in static/")
    html = Path(state_path).read_text(encoding="utf-8")

    attrs = []
    if state_abbrev:
        attrs.append(f'data-state-abbr="{state_abbrev}"')
    if county_name:
        attrs.append(f'data-county-name="{county_name}"')
    attrs_str = " ".join(attrs)

    if "<body>" in html:
        html = html.replace("<body>", f"<body {attrs_str}>", 1)
    elif "<body " in html:
        # already has attributes; inject before closing bracket of first body tag
        html = html.replace("<body ", f"<body {attrs_str} ", 1)
    return html


def _serve_seo_county(state_abbrev: str, county_name: str) -> HTMLResponse:
    sa = (state_abbrev or "").strip().upper()
    cn = (county_name or "").strip()
    if len(sa) != 2 or not sa.isalpha():
        raise HTTPException(status_code=404, detail="Not Found")
    cn = cn.replace("-", " ").replace("_", " ")
    html = _render_state_page_with_seo(sa, cn)
    return HTMLResponse(html)


@app.get("/state/{state_abbrev}", include_in_schema=False, response_class=HTMLResponse)
def seo_state_page(state_abbrev: str):
    """
    SEO-friendly state URL: /state/NY (case-insensitive).
    """
    sa = (state_abbrev or "").strip().upper()
    if len(sa) != 2 or not sa.isalpha():
        raise HTTPException(status_code=404, detail="Not Found")
    html = _render_state_page_with_seo(sa, None)
    return HTMLResponse(html)


@app.get("/state/{state_abbrev}/{county_name}", include_in_schema=False, response_class=HTMLResponse)
def seo_county_page(state_abbrev: str, county_name: str):
    """
    SEO-friendly county URL: /state/NY/Suffolk (case-insensitive).
    """
    return _serve_seo_county(state_abbrev, county_name)


@app.get("/state/{state_abbrev}/{county_name}/", include_in_schema=False, response_class=HTMLResponse)
def seo_county_page_slash(state_abbrev: str, county_name: str):
    return _serve_seo_county(state_abbrev, county_name)


@app.get("/initialize", include_in_schema=False, response_class=HTMLResponse)
def initialize_page():
    if os.path.exists(DB_PATH):
        return RedirectResponse(url="/docs", status_code=302)

    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CodePlug-PB Initialization</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b1220; color: #e7eaf0; }
      .wrap { max-width: 860px; margin: 0 auto; padding: 56px 20px; }
      .card { background: #111a2e; border: 1px solid #243152; border-radius: 14px; padding: 22px; }
      h1 { margin: 10px 0 8px; font-size: 28px; }
      p { margin: 8px 0 0; color: #b8c1d9; line-height: 1.5; }
      code { background: #0c1427; border: 1px solid #2a3a62; padding: 2px 6px; border-radius: 6px; color: #cdd5ea; }
      label { display: block; margin-top: 14px; margin-bottom: 6px; color: #cdd5ea; font-weight: 650; }
      input[type="text"], input[type="file"] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #2a3a62; background: #0c1427; color: #e7eaf0; }
      button { margin-top: 16px; padding: 10px 14px; border-radius: 10px; border: 1px solid #2a3a62; background: #2b66f6; color: white; cursor: pointer; font-weight: 750; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .muted { color: #9fb0d6; font-size: 13px; margin-top: 6px; }
      .status { margin-top: 16px; padding: 12px; border-radius: 10px; background: #0c1427; border: 1px solid #2a3a62; color: #cdd5ea; white-space: pre-wrap; }
      .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #0c1427; border: 1px solid #2a3a62; color: #cdd5ea; font-size: 12px; }
      a { color: #8ab4ff; }
      .overlay { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(11,18,32,0.72); backdrop-filter: blur(4px); }
      .overlay-card { background: #111a2e; border: 1px solid #243152; border-radius: 14px; padding: 20px 18px; width: min(420px, 92vw); text-align: center; }
      .spinner { width: 44px; height: 44px; border: 4px solid #2a3a62; border-top-color: #2b66f6; border-radius: 50%; margin: 0 auto 12px; animation: spin 1s linear infinite; }
      @keyframes spin { to { transform: rotate(360deg); } }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="pill">Setup</div>
        <h1>Master Database Not Initialized</h1>
        <p>Please upload a <code>MasterHpdb.hp1</code> file to initialize the database.</p>
        <p class="muted">Uploads require an API key and run in the background.</p>

        <form id="uploadForm" method="post" action="/initialize/upload" enctype="multipart/form-data">
          <label for="api_key">API Key</label>
          <input id="api_key" name="api_key" type="text" placeholder="X-API-Key" autocomplete="off" required />
          <div class="muted">Generate a key with <code>python3 generate_api_key.py</code> and paste it here.</div>

          <label for="file">MasterHpdb.hp1</label>
          <input id="file" name="file" type="file" required />

          <button id="submitBtn" type="submit">Upload & Initialize</button>
        </form>

        <div id="status" class="status" style="display:none;"></div>
      </div>
    </div>

    <div id="overlay" class="overlay">
      <div class="overlay-card">
        <div class="spinner"></div>
        <div style="font-weight:750; margin-bottom:6px;">Processing the Database...</div>
        <div class="muted">This can take a minute. You’ll be redirected automatically when it’s ready.</div>
      </div>
    </div>

    <script>
      const form = document.getElementById('uploadForm');
      const statusEl = document.getElementById('status');
      const submitBtn = document.getElementById('submitBtn');
      const overlay = document.getElementById('overlay');

      function showStatus(obj) {
        statusEl.style.display = 'block';
        statusEl.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
      }

      async function poll(jobId) {
        while (true) {
          const res = await fetch(`/hpdb/admin/jobs/${jobId}`);
          const data = await res.json();
          showStatus(data);
          if (data.status === 'success') {
            window.location.href = '/docs';
            return;
          }
          if (data.status === 'error') return;
          await new Promise(r => setTimeout(r, 1000));
        }
      }

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        submitBtn.disabled = true;
        showStatus('Uploading...');
        try {
          const fd = new FormData(form);
          const res = await fetch('/initialize/upload', { method: 'POST', body: fd });
          const data = await res.json();
          showStatus(data);
          if (!res.ok) { submitBtn.disabled = false; return; }
          overlay.style.display = 'flex';
          await poll(data.job_id);
        } catch (err) {
          showStatus(String(err));
          overlay.style.display = 'none';
          submitBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
        """
    )


@app.post("/initialize/upload", include_in_schema=False)
def initialize_upload(
    background: BackgroundTasks,
    api_key: str = Form(...),
    file: UploadFile = File(...),
):
    if os.path.exists(DB_PATH):
        raise HTTPException(status_code=409, detail="Database already exists.")
    validate_api_key_value(api_key)

    job_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_MasterHpdb.hp1")
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    return _queue_upload_master(background, job_id=job_id, saved_path=saved_path)


@app.get("/update", include_in_schema=False, response_class=HTMLResponse)
def update_page():
    if not os.path.exists(DB_PATH):
        return RedirectResponse(url="/initialize", status_code=302)

    return HTMLResponse(
        """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CodePlug-PB Update</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #0b1220; color: #e7eaf0; }
      .wrap { max-width: 960px; margin: 0 auto; padding: 56px 20px; }
      .card { background: #111a2e; border: 1px solid #243152; border-radius: 14px; padding: 22px; }
      h1 { margin: 10px 0 8px; font-size: 28px; }
      p { margin: 8px 0 0; color: #b8c1d9; line-height: 1.5; }
      code { background: #0c1427; border: 1px solid #2a3a62; padding: 2px 6px; border-radius: 6px; color: #cdd5ea; }
      label { display: block; margin-top: 14px; margin-bottom: 6px; color: #cdd5ea; font-weight: 650; }
      input[type="text"], input[type="file"] { width: 100%; padding: 10px 12px; border-radius: 10px; border: 1px solid #2a3a62; background: #0c1427; color: #e7eaf0; }
      button { margin-top: 16px; padding: 10px 14px; border-radius: 10px; border: 1px solid #2a3a62; background: #2b66f6; color: white; cursor: pointer; font-weight: 750; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .muted { color: #9fb0d6; font-size: 13px; margin-top: 6px; }
      .status { margin-top: 16px; padding: 12px; border-radius: 10px; background: #0c1427; border: 1px solid #2a3a62; color: #cdd5ea; white-space: pre-wrap; }
      .pill { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #0c1427; border: 1px solid #2a3a62; color: #cdd5ea; font-size: 12px; }
      a { color: #8ab4ff; }
      .overlay { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(11,18,32,0.72); backdrop-filter: blur(4px); }
      .overlay-card { background: #111a2e; border: 1px solid #243152; border-radius: 14px; padding: 20px 18px; width: min(520px, 92vw); text-align: center; }
      .spinner { width: 44px; height: 44px; border: 4px solid #2a3a62; border-top-color: #2b66f6; border-radius: 50%; margin: 0 auto 12px; animation: spin 1s linear infinite; }
      @keyframes spin { to { transform: rotate(360deg); } }
      table { width: 100%; border-collapse: collapse; margin-top: 14px; }
      th, td { padding: 10px 10px; border-bottom: 1px solid #243152; text-align: left; }
      th { color: #cdd5ea; font-size: 13px; letter-spacing: 0.02em; text-transform: uppercase; }
      td { color: #e7eaf0; }
      .num { font-variant-numeric: tabular-nums; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <div class="pill">Admin</div>
        <h1>Update the Database</h1>
        <p>Upload a newer <code>MasterHpdb.hp1</code> to apply an update safely (builds an update DB, merges into a copy, then swaps).</p>

        <form id="updateForm" method="post" action="/update/upload" enctype="multipart/form-data">
          <label for="api_key">API Key</label>
          <input id="api_key" name="api_key" type="text" placeholder="X-API-Key" autocomplete="off" required />

          <label for="file">Newest MasterHpdb.hp1</label>
          <input id="file" name="file" type="file" required />

          <button id="submitBtn" type="submit">Upload & Update</button>
        </form>

        <div id="status" class="status" style="display:none;"></div>
        <div id="delta" style="display:none;"></div>
      </div>
    </div>

    <div id="overlay" class="overlay">
      <div class="overlay-card">
        <div class="spinner"></div>
        <div style="font-weight:750; margin-bottom:6px;">Processing the Update...</div>
        <div class="muted">This can take a minute. You’ll see a summary when it’s done.</div>
      </div>
    </div>

    <script>
      const form = document.getElementById('updateForm');
      const statusEl = document.getElementById('status');
      const deltaEl = document.getElementById('delta');
      const submitBtn = document.getElementById('submitBtn');
      const overlay = document.getElementById('overlay');

      function showStatus(obj) {
        statusEl.style.display = 'block';
        statusEl.textContent = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
      }

      function renderDelta(delta) {
        const order = ['trunk','site','tgid','t_freq','c_freq'];
        const labels = { trunk: 'trunk', site: 'site', tgid: 'tgid', t_freq: 't_freq', c_freq: 'c_freq' };
        let html = '<table><thead><tr><th>Table</th><th class=\"num\">Added</th><th class=\"num\">Removed</th><th class=\"num\">Changed</th></tr></thead><tbody>';
        for (const k of order) {
          if (!delta[k]) continue;
          html += `<tr><td><code>${labels[k]}</code></td>` +
                  `<td class=\"num\">${delta[k].added}</td>` +
                  `<td class=\"num\">${delta[k].removed}</td>` +
                  `<td class=\"num\">${delta[k].changed}</td></tr>`;
        }
        html += '</tbody></table>';
        html += '<div class=\"muted\" style=\"margin-top:10px;\">Tip: verify counts via <a href=\"/stats\">/stats</a> or run <code>python3 validate_hpdb_update.py</code> offline.</div>';
        deltaEl.style.display = 'block';
        deltaEl.innerHTML = html;
      }

      async function poll(jobId) {
        while (true) {
          const res = await fetch(`/hpdb/admin/jobs/${jobId}`);
          const data = await res.json();
          showStatus(data);
          if (data.status === 'success') {
            overlay.style.display = 'none';
            if (data.result && data.result.delta) renderDelta(data.result.delta);
            return;
          }
          if (data.status === 'error') { overlay.style.display = 'none'; return; }
          await new Promise(r => setTimeout(r, 1000));
        }
      }

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        submitBtn.disabled = true;
        deltaEl.style.display = 'none';
        showStatus('Uploading...');
        try {
          const fd = new FormData(form);
          const res = await fetch('/update/upload', { method: 'POST', body: fd });
          const data = await res.json();
          showStatus(data);
          if (!res.ok) { submitBtn.disabled = false; return; }
          overlay.style.display = 'flex';
          await poll(data.job_id);
          submitBtn.disabled = false;
        } catch (err) {
          showStatus(String(err));
          overlay.style.display = 'none';
          submitBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
        """
    )


@app.post("/update/upload", include_in_schema=False)
def update_upload(
    background: BackgroundTasks,
    api_key: str = Form(...),
    file: UploadFile = File(...),
):
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=409, detail="Database does not exist; initialize first.")
    validate_api_key_value(api_key)

    job_id = uuid.uuid4().hex
    saved_path = os.path.join(UPLOAD_DIR, f"{job_id}_MasterHpdb.hp1")
    with open(saved_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    _set_job(job_id, **{"job_id": job_id, "kind": "update-master", "status": "queued", "detail": None, "result": None})
    background.add_task(_background_update_master, job_id, saved_path)
    return JobStatus(**app.state.jobs[job_id])


@app.get("/hpdb/counties", response_model=List[CountyMatch])
def counties(
    state: str = Query(..., description="State name or abbreviation, e.g. 'MO' or 'Missouri'"),
    q: Optional[str] = Query(None, description="Optional county name search"),
    limit: int = Query(200, gt=0, le=5000),
    db: sqlite3.Connection = Depends(get_conn),
):
    q_state = norm(state)
    st = db.execute(
        """
        SELECT StateId, name AS StateName, abbrev AS StateAbbrev
        FROM state_info
        WHERE lower(abbrev) = ? OR lower(name) = ?
        ORDER BY CASE WHEN lower(abbrev)=? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (q_state, q_state, q_state),
    ).fetchone()
    if not st:
        raise HTTPException(status_code=404, detail=f"State not found: {state!r}")

    if q:
        qq = norm(q)
        rows = db.execute(
            """
            SELECT CountyId, name AS CountyName
            FROM county_info
            WHERE StateId = ? AND lower(name) LIKE ?
            ORDER BY name
            LIMIT ?
            """,
            (int(st["StateId"]), f"%{qq}%", limit),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT CountyId, name AS CountyName
            FROM county_info
            WHERE StateId = ?
            ORDER BY name
            LIMIT ?
            """,
            (int(st["StateId"]), limit),
        ).fetchall()

    return [
        CountyMatch(
            state_id=int(st["StateId"]),
            state_abbrev=str(st["StateAbbrev"] or ""),
            state_name=str(st["StateName"] or ""),
            county_id=int(r["CountyId"]),
            county_name=str(r["CountyName"] or ""),
        )
        for r in rows
    ]


@app.get("/hpdb/query", response_model=CountyQueryResponse)
def hpdb_query(
    state: str = Query(..., description="State name or abbreviation"),
    county: str = Query(..., description="County name"),
    include_talkgroups: bool = Query(True),
    include_trunk_sites: bool = Query(True),
    include_trunk_site_frequencies: bool = Query(True, description="Control channels are not marked; returns all site freqs"),
    limit_conventional: int = Query(2000, gt=0, le=200000),
    limit_talkgroups: int = Query(2000, gt=0, le=500000),
    limit_site_freq_rows: int = Query(2000, gt=0, le=500000),
    db: sqlite3.Connection = Depends(get_conn),
):
    return hpdb_query_impl(
        db=db,
        state=state,
        county=county,
        include_talkgroups=include_talkgroups,
        include_trunk_sites=include_trunk_sites,
        include_trunk_site_frequencies=include_trunk_site_frequencies,
        limit_conventional=limit_conventional,
        limit_talkgroups=limit_talkgroups,
        limit_site_freq_rows=limit_site_freq_rows,
    )


def hpdb_query_impl(
    *,
    db: sqlite3.Connection,
    state: str,
    county: str,
    include_talkgroups: bool = True,
    include_trunk_sites: bool = True,
    include_trunk_site_frequencies: bool = True,
    limit_conventional: int = 2000,
    limit_talkgroups: int = 2000,
    limit_site_freq_rows: int = 2000,
) -> CountyQueryResponse:
    ck = resolve_county(db, state=state, county=county)
    county_model = CountyMatch(
        state_id=ck.state_id,
        state_abbrev=ck.state_abbrev,
        state_name=ck.state_name,
        county_id=ck.county_id,
        county_name=ck.county_name,
    )

    conv_rows = db.execute(
        """
        SELECT DepartmentName, ChannelName, FrequencyHz, Modulation, Tone, NAC, ColorCode, RAN, NumberTag, Avoid
        FROM v_conventional_freqs_by_county
        WHERE StateId = ? AND CountyId = ?
        ORDER BY DepartmentName, FrequencyHz, ChannelName
        LIMIT ?
        """,
        (ck.state_id, ck.county_id, limit_conventional),
    ).fetchall()
    conventional = [
        ConventionalRow(
            department=r["DepartmentName"],
            channel=r["ChannelName"],
            frequency_hz=r["FrequencyHz"],
            frequency_mhz=hz_to_mhz(r["FrequencyHz"]),
            modulation=r["Modulation"],
            tone=r["Tone"],
            nac=r["NAC"],
            color_code=r["ColorCode"],
            ran=r["RAN"],
            number_tag=r["NumberTag"],
            avoid=r["Avoid"],
        )
        for r in conv_rows
    ]

    trunked: Dict[int, TrunkSystem] = {}
    if include_trunk_sites and include_trunk_site_frequencies:
        tf_rows = db.execute(
            """
            SELECT TrunkId, SystemName, SystemType, SiteId, SiteName, FrequencyHz, ChannelName, Lcn1, Lcn2, RAN, Avoid
            FROM v_trunk_sites_and_freqs_by_county
            WHERE StateId = ? AND CountyId = ?
            ORDER BY SystemName, SiteName, FrequencyHz
            LIMIT ?
            """,
            (ck.state_id, ck.county_id, limit_site_freq_rows),
        ).fetchall()

        # trunk_id -> site_id -> TrunkSite
        sites_by_trunk: Dict[int, Dict[int, TrunkSite]] = {}
        for r in tf_rows:
            trunk_id = int(r["TrunkId"])
            trunked.setdefault(
                trunk_id,
                TrunkSystem(
                    trunk_id=trunk_id,
                    system_name=r["SystemName"],
                    system_type=r["SystemType"],
                    sites=[],
                ),
            )
            sites_by_trunk.setdefault(trunk_id, {})
            site_id = int(r["SiteId"])
            sites_by_trunk[trunk_id].setdefault(
                site_id,
                TrunkSite(
                    site_id=site_id,
                    site_name=r["SiteName"],
                    lat=_row_get(r, "lat"),
                    lon=_row_get(r, "lon"),
                    frequencies=[],
                ),
            )
            sites_by_trunk[trunk_id][site_id].frequencies.append(
                SiteFrequencyRow(
                    frequency_hz=r["FrequencyHz"],
                    frequency_mhz=hz_to_mhz(r["FrequencyHz"]),
                    channel_name=r["ChannelName"],
                    lcn1=r["Lcn1"],
                    lcn2=r["Lcn2"],
                    ran=r["RAN"],
                    avoid=r["Avoid"],
                )
            )
        for trunk_id, site_map in sites_by_trunk.items():
            trunked[trunk_id].sites = list(site_map.values())
    elif include_trunk_sites:
        # Return systems without site frequencies (still useful to show system names/types).
        sys_rows = db.execute(
            """
            SELECT TrunkId, SystemName, SystemType
            FROM v_trunk_systems_by_county
            WHERE StateId = ? AND CountyId = ?
            ORDER BY SystemName
            """,
            (ck.state_id, ck.county_id),
        ).fetchall()
        for r in sys_rows:
            trunk_id = int(r["TrunkId"])
            trunked.setdefault(
                trunk_id,
                TrunkSystem(trunk_id=trunk_id, system_name=r["SystemName"], system_type=r["SystemType"], sites=[]),
            )

    talkgroups: List[TalkgroupRow] = []
    if include_talkgroups:
        tg_rows = db.execute(
            """
            SELECT TrunkId, SystemName, SystemType, TalkgroupCategory, Tid, AlphaTag, Talkgroup, Service, NumberTag, Avoid
            FROM v_trunk_talkgroups_by_county
            WHERE StateId = ? AND CountyId = ?
            ORDER BY SystemName, TalkgroupCategory, AlphaTag, Tid
            LIMIT ?
            """,
            (ck.state_id, ck.county_id, limit_talkgroups),
        ).fetchall()
        talkgroups = [
            TalkgroupRow(
                trunk_id=int(r["TrunkId"]),
                system_name=r["SystemName"],
                system_type=r["SystemType"],
                category=r["TalkgroupCategory"],
                tid=int(r["Tid"]),
                alpha_tag=r["AlphaTag"],
                talkgroup=r["Talkgroup"],
                service=r["Service"],
                number_tag=r["NumberTag"],
                avoid=r["Avoid"],
            )
            for r in tg_rows
        ]

    return CountyQueryResponse(
        county=county_model,
        center_lat=None,
        center_lon=None,
        radius_miles=None,
        conventional=conventional,
        trunked=list(trunked.values()),
        talkgroups=talkgroups,
    )


def bounding_box(lat: float, lon: float, radius_miles: float) -> Tuple[float, float, float, float]:
    # Approximate degrees per mile; good enough for prefiltering.
    lat_delta = radius_miles / 69.0
    lon_delta = radius_miles / (69.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta)


def _approx_zoom(lat: float, radius_miles: float, width_px: int = 640) -> int:
    """
    Pick a web-mercator zoom level so the given radius fits horizontally in width_px.
    """
    radius_m = max(radius_miles, 0.1) * 1609.344
    meters_per_px_target = max((radius_m * 2) / max(width_px, 64), 0.1)
    # meters/px at zoom 0: 156543.03392 * cos(lat)
    zoom_float = math.log2((156543.03392 * math.cos(math.radians(lat))) / meters_per_px_target)
    return int(min(17, max(2, round(zoom_float))))


def _latlon_to_pixel(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    # Web mercator x/y in pixel coords at given zoom
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n * 256.0
    return x, y


def _render_osm_static(center_lat: float, center_lon: float, zoom: int, width: int, height: int, markers: List[Tuple[float, float]]) -> bytes:
    """
    Compose a static PNG using OSM tiles and circle markers (no text labels for reliability).
    """
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pillow is required for static maps: {exc}")

    center_px, center_py = _latlon_to_pixel(center_lat, center_lon, zoom)
    half_w, half_h = width / 2.0, height / 2.0
    min_px = center_px - half_w
    min_py = center_py - half_h
    max_px = center_px + half_w
    max_py = center_py + half_h

    def tile_range(min_p: float, max_p: float) -> Tuple[int, int]:
        return int(math.floor(min_p / 256.0)), int(math.floor(max_p / 256.0))

    tx_min, tx_max = tile_range(min_px, max_px)
    ty_min, ty_max = tile_range(min_py, max_py)

    img = Image.new("RGBA", (width, height), (5, 7, 15, 255))
    draw = ImageDraw.Draw(img)

    ua = {"User-Agent": "CodePlug-PB/0.1 (static-map)"}
    tiles_drawn = 0
    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            if tx < 0 or ty < 0:
                continue
            tile_url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
            try:
                r = requests.get(tile_url, headers=ua, timeout=6)
                if r.status_code != 200:
                    continue
                tile = Image.open(io.BytesIO(r.content)).convert("RGBA")
            except Exception:
                continue
            px = int(tx * 256 - min_px)
            py = int(ty * 256 - min_py)
            img.paste(tile, (px, py))
            tiles_drawn += 1

    for lat, lon in markers:
        mx, my = _latlon_to_pixel(lat, lon, zoom)
        sx = int(round(mx - min_px))
        sy = int(round(my - min_py))
        r = 6
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(99, 179, 237, 255), outline=(10, 20, 40, 255), width=2)

    if tiles_drawn == 0:
        raise RuntimeError("No OSM tiles fetched")

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _render_staticmaps_with_labels(
    center_lat: float, center_lon: float, zoom: int, width: int, height: int, markers: List[Tuple[float, float, str]]
) -> bytes:
    """
    Render map via py-staticmaps (tiles + markers), then overlay lightweight labels with Pillow.
    """
    try:
        import staticmaps
        from PIL import Image, ImageDraw, ImageFont
        import tempfile
        import os
    except Exception as exc:
        raise RuntimeError(f"staticmaps or Pillow missing: {exc}")

    # Shim for Pillow 11 where textsize was removed; staticmaps still calls it.
    if not hasattr(ImageDraw.ImageDraw, "textsize"):
        def _textsize(self, text, font=None, *args, **kwargs):
            bbox = self.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        ImageDraw.ImageDraw.textsize = _textsize  # type: ignore

    def _make_label_icon(text: str) -> str:
        txt = (text or "").strip() or "Site"
        font = ImageFont.load_default()
        # Pillow 11 removed getsize; use textbbox for compatibility
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        w, h = tw + pad * 2, th + pad * 2
        img = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # bubble
        draw.rectangle((4, 4, 4 + w, 4 + h), fill=(0, 80, 200, 230), outline=(255, 255, 255, 220), width=1)
        draw.text((4 + pad, 4 + pad), txt, font=font, fill=(255, 255, 255, 255))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmp, format="PNG")
        tmp.close()
        return tmp.name

    ctx = staticmaps.Context()
    # Empty attribution to avoid renderer text issues; we handle labels separately.
    ctx.set_tile_provider(staticmaps.TileProvider("https://tile.openstreetmap.org/{z}/{x}/{y}.png", ""))
    ctx.set_center(staticmaps.create_latlng(center_lat, center_lon))
    ctx.set_zoom(zoom)
    tmp_files: List[str] = []
    try:
        for lat, lon, label in markers:
            icon_path = _make_label_icon(label)
            tmp_files.append(icon_path)
            pos = staticmaps.create_latlng(lat, lon)
            ctx.add_object(staticmaps.ImageMarker(pos, icon_path, origin_x=10, origin_y=10))

        img = ctx.render_pillow(width, height)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass


def _render_staticmaps_base(center_lat: float, center_lon: float, zoom: int, width: int, height: int) -> bytes:
    """
    Render only the base map via py-staticmaps (no markers). Used when we want our own overlays.
    """
    try:
        import staticmaps
    except Exception as exc:
        raise RuntimeError(f"staticmaps missing: {exc}")

    ctx = staticmaps.Context()
    ctx.set_tile_provider(staticmaps.TileProvider("https://tile.openstreetmap.org/{z}/{x}/{y}.png", ""))
    ctx.set_center(staticmaps.create_latlng(center_lat, center_lon))
    ctx.set_zoom(zoom)
    img = ctx.render_pillow(width, height)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _apply_osm_watermark(img) -> None:
    # Watermark intentionally disabled
    return None


def _finalize_png_bytes(buf: bytes) -> bytes:
    """
    Recompress PNG to keep size small (palette quantization).
    """
    try:
        from PIL import Image
    except Exception:
        return buf
    try:
        img = Image.open(io.BytesIO(buf))
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        pal = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        out = io.BytesIO()
        pal.save(out, format="PNG", optimize=True, compress_level=9)
        return out.getvalue()
    except Exception:
        return buf


def _draw_labels_on_image(
    base_png: bytes,
    center_lat: float,
    center_lon: float,
    zoom: int,
    width: int,
    height: int,
    labels: List[Tuple[float, float, str]],
    extent_width: Optional[int] = None,
    extent_height: Optional[int] = None,
) -> bytes:
    """
    Draw markers + labels on a transparent overlay, then composite onto the base map.
    extent_width/height lets us keep the geographic crop of a smaller canvas while rendering to a larger output.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return base_png

    def _encode_png_small(img: "Image.Image") -> bytes:
        """Encode PNG with aggressive compression/quantization."""
        try:
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            pal = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            buf = io.BytesIO()
            pal.save(buf, format="PNG", optimize=True, compress_level=9)
            return buf.getvalue()
        except Exception:
            return base_png

    extent_w = float(extent_width or width)
    extent_h = float(extent_height or height)
    scale_x = width / extent_w
    scale_y = height / extent_h

    try:
        base = Image.open(io.BytesIO(base_png)).convert("RGBA")
        if base.size != (width, height):
            base = base.resize((width, height), resample=Image.LANCZOS)
    except Exception:
        return base_png

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Pillow 11 compatibility: provide textsize if missing
    if not hasattr(draw, "textsize"):
        def _textsize(text, font=None):
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.textsize = _textsize  # type: ignore

    font = ImageFont.load_default()

    center_px, center_py = _latlon_to_pixel(center_lat, center_lon, zoom)
    half_w, half_h = extent_w / 2.0, extent_h / 2.0  # geographic extent
    min_px = center_px - half_w
    min_py = center_py - half_h

    # Precompute anchor points and label sizes
    nodes = []
    for lat, lon, label in labels:
        mx, my = _latlon_to_pixel(lat, lon, zoom)
        sx = int(round((mx - min_px) * scale_x))
        sy = int(round((my - min_py) * scale_y))
        txt = str(label)[:40]
        tw, th = draw.textsize(txt, font=font)
        pad = 3
        w = tw + pad * 2
        h = th + pad * 2
        nodes.append({"anchor": (sx, sy), "size": (w, h), "text": txt})

    # Greedy label placement with scored candidate offsets to reduce overlaps, then a light relaxation pass.
    density_scale = max(scale_x, scale_y)
    angle_deg = tuple(range(0, 360, 22))  # 16 directions
    radii = [
        32 * density_scale,
        64 * density_scale,
        110 * density_scale,
        160 * density_scale,
        220 * density_scale,
    ]

    def candidate_offsets():
        for r in radii:
            for a in angle_deg:
                rad = math.radians(a)
                yield (math.cos(rad) * r, math.sin(rad) * r)
        yield (0.0, 0.0)  # fallback on anchor

    def rect_for(anchor, size, offset):
        sx, sy = anchor
        ox, oy = offset
        w, h = size
        cx, cy = sx + ox, sy + oy
        return (cx, cy, cx + w, cy + h)

    placed_boxes: List[Tuple[float, float, float, float]] = []
    # Place longer labels first so they get more room
    nodes.sort(key=lambda n: len(n["text"]), reverse=True)
    for n in nodes:
        best = None
        best_score = float("inf")
        for off in candidate_offsets():
            box = rect_for(n["anchor"], n["size"], off)
            bx0, by0, bx1, by1 = box
            overlap_penalty = 0.0
            for pb in placed_boxes:
                # add padding so labels don't touch
                pad_sep = 6 * density_scale
                ox = min(bx1, pb[2] + pad_sep) - max(bx0, pb[0] - pad_sep)
                oy = min(by1, pb[3] + pad_sep) - max(by0, pb[1] - pad_sep)
                if ox > 0 and oy > 0:
                    overlap_penalty += ox * oy * 8.0
            # Penalize going out of bounds
            out_pen = 0.0
            if bx0 < 0:
                out_pen += abs(bx0) * 6
            if by0 < 0:
                out_pen += abs(by0) * 6
            if bx1 > width:
                out_pen += (bx1 - width) * 6
            if by1 > height:
                out_pen += (by1 - height) * 6
            anchor_pen = (off[0] ** 2 + off[1] ** 2) ** 0.5 * 0.5
            score = overlap_penalty + out_pen + anchor_pen
            if score < best_score:
                best_score = score
                best = (off, box)
        if best is None:
            n["offset"] = (10.0 * density_scale, 0.0)
        else:
            n["offset"], box = best
            placed_boxes.append(box)

    # Light relaxation: push overlapping boxes apart a few times
    for _ in range(14):
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                ni, nj = nodes[i], nodes[j]
                bi = rect_for(ni["anchor"], ni["size"], ni["offset"])
                bj = rect_for(nj["anchor"], nj["size"], nj["offset"])
                pad_sep = 8 * density_scale
                ox = min(bi[2], bj[2] + pad_sep) - max(bi[0], bj[0] - pad_sep)
                oy = min(bi[3], bj[3] + pad_sep) - max(bi[1], bj[1] - pad_sep)
                if ox > 0 and oy > 0:
                    shift_x = (ox / 2 + 4) * (1 if ni["anchor"][0] <= nj["anchor"][0] else -1)
                    shift_y = (oy / 2 + 4) * (1 if ni["anchor"][1] <= nj["anchor"][1] else -1)
                    ni["offset"] = (ni["offset"][0] - shift_x, ni["offset"][1] - shift_y)
                    nj["offset"] = (nj["offset"][0] + shift_x, nj["offset"][1] + shift_y)

    # Draw markers + labels with leader lines
    for n in nodes:
        sx, sy = n["anchor"]
        ox, oy = n["offset"]
        w, h = n["size"]
        txt = n["text"]

        r = 5
        dot_fill = (118, 180, 255, 235)
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=dot_fill, outline=(10, 30, 60, 255), width=2)

        bx0 = sx + ox
        by0 = sy + oy - h / 2
        bx1 = bx0 + w
        by1 = by0 + h
        draw.rectangle((bx0, by0, bx1, by1), fill=(12, 24, 48, 230), outline=(60, 120, 200, 220))
        draw.text((bx0 + 3, by0 + 3), txt, font=font, fill=(225, 238, 255, 255))
        # leader line
        draw.line((sx, sy, (bx0 + bx1) / 2, (by0 + by1) / 2), fill=(120, 180, 255, 200), width=1)

    # draw center bullseye
    cx, cy = width // 2, height // 2
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=(255, 60, 60, 230), width=2)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 60, 60, 230))

    img = Image.alpha_composite(base, overlay)
    return _encode_png_small(img)


def _render_overlay(center_lat: float, center_lon: float, zoom: int, width: int, height: int, markers: List[Tuple[float, float, str]]) -> bytes:
    """
    Transparent overlay with markers + labels (no base tiles).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pillow is required for static map overlays: {exc}")

    center_px, center_py = _latlon_to_pixel(center_lat, center_lon, zoom)
    half_w, half_h = width / 2.0, height / 2.0
    min_px = center_px - half_w
    min_py = center_py - half_h

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for lat, lon, label in markers:
        mx, my = _latlon_to_pixel(lat, lon, zoom)
        sx = int(round(mx - min_px))
        sy = int(round(my - min_py))
        r = 6
        draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(99, 179, 237, 220), outline=(10, 20, 40, 255), width=2)
        if label:
            txt = str(label)[:28]
            tw, th = draw.textsize(txt, font=font)
            pad = 3
            bx0, by0 = sx + r + 4, sy - th / 2 - pad
            bx1, by1 = bx0 + tw + pad * 2, by0 + th + pad * 2
            draw.rectangle((bx0, by0, bx1, by1), fill=(8, 12, 20, 230), outline=(30, 45, 70, 200))
            draw.text((bx0 + pad, by0 + pad), txt, font=font, fill=(220, 235, 245, 255))

    out = io.BytesIO()
    try:
        pal = img.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.Dither.NONE)  # type: ignore[attr-defined]
        pal.save(out, format="PNG", optimize=True, compress_level=9)
    except Exception:
        img.save(out, format="PNG")
    return out.getvalue()


def hpdb_query_near_point_impl(
    *,
    db: sqlite3.Connection,
    lat: float,
    lon: float,
    radius_miles: float,
    include_talkgroups: bool = True,
    include_trunk_sites: bool = True,
    include_trunk_site_frequencies: bool = True,
    limit_conventional: int = 2000,
    limit_talkgroups: int = 2000,
    limit_sites: int = 100,
    limit_site_freq_rows: int = 5000,
) -> CountyQueryResponse:
    """
    Location-based query using ZIP lat/lon and a radius.

    Notes:
      - Trunk control channels are not marked in this dataset; we return all site frequencies.
      - We restrict trunk sites by geographic distance (site.lat/lon) rather than county mapping.
      - Conventional groups are also selected by distance (c_group.lat/lon).
    """
    lat_min, lat_max, lon_min, lon_max = bounding_box(lat, lon, radius_miles)

    # Choose a representative county/state for the response header by picking the closest ZIP-mapped county.
    # If we don't have county data, fall back to the nearest county centroid candidate later.
    county_model = CountyMatch(state_id=0, state_abbrev="", state_name="", county_id=0, county_name="")

    # Conventional: get nearby departments (c_group), then join to c_freq.
    conv_groups = db.execute(
        """
        SELECT cg.CGroupId, cg.name AS DepartmentName, cg.lat, cg.lon
        FROM c_group cg
        WHERE cg.lat IS NOT NULL AND cg.lon IS NOT NULL
          AND cg.lat BETWEEN ? AND ?
          AND cg.lon BETWEEN ? AND ?
        """,
        (lat_min, lat_max, lon_min, lon_max),
    ).fetchall()

    group_ids: List[int] = []
    group_meta: Dict[int, Tuple[str, float]] = {}
    for r in conv_groups:
        try:
            d = haversine_miles(lat, lon, float(r["lat"]), float(r["lon"]))
        except Exception:
            continue
        if d <= radius_miles:
            gid = int(r["CGroupId"])
            group_ids.append(gid)
            group_meta[gid] = (str(r["DepartmentName"] or ""), float(d))
    group_ids = group_ids[: max(0, limit_conventional)]

    conventional: List[ConventionalRow] = []
    if group_ids:
        placeholders = ",".join(["?"] * len(group_ids))
        conv_rows = db.execute(
            f"""
            SELECT cf.CGroupId, cf.name AS ChannelName, cf.frequency_hz AS FrequencyHz,
                   cf.modulation AS Modulation, cf.tone AS Tone, cf.nac AS NAC,
                   cf.color_code AS ColorCode, cf.ran AS RAN, cf.number_tag AS NumberTag, cf.avoid AS Avoid
            FROM c_freq cf
            WHERE cf.CGroupId IN ({placeholders})
            ORDER BY cf.CGroupId, cf.frequency_hz, cf.name
            LIMIT ?
            """,
            (*group_ids, limit_conventional),
        ).fetchall()
        for r in conv_rows:
            gid = int(r["CGroupId"])
            dept, _dist = group_meta.get(gid, ("", 0.0))
            conventional.append(
                ConventionalRow(
                    department=dept,
                    channel=r["ChannelName"],
                    frequency_hz=r["FrequencyHz"],
                    frequency_mhz=hz_to_mhz(r["FrequencyHz"]),
                    modulation=r["Modulation"],
                    tone=r["Tone"],
                    nac=r["NAC"],
                    color_code=r["ColorCode"],
                    ran=r["RAN"],
                    number_tag=r["NumberTag"],
                    avoid=r["Avoid"],
                )
            )

    # Trunk sites: select nearby sites, then join t_freq, and later talkgroups by TrunkId.
    trunked: Dict[int, TrunkSystem] = {}
    talkgroups: List[TalkgroupRow] = []

    if include_trunk_sites:
        site_rows = db.execute(
            """
            SELECT s.SiteId, s.TrunkId, s.name AS SiteName, s.lat, s.lon, t.name AS SystemName, t.system_type AS SystemType
            FROM site s
            JOIN trunk t ON t.TrunkId = s.TrunkId
            WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
              AND s.lat BETWEEN ? AND ?
              AND s.lon BETWEEN ? AND ?
            """,
            (lat_min, lat_max, lon_min, lon_max),
        ).fetchall()

        selected_sites: List[Tuple[int, int, str, str, str, float]] = []
        for r in site_rows:
            try:
                d = haversine_miles(lat, lon, float(r["lat"]), float(r["lon"]))
            except Exception:
                continue
            if d <= radius_miles:
                selected_sites.append(
                    (
                        int(r["TrunkId"]),
                        int(r["SiteId"]),
                        str(r["SystemName"] or ""),
                        str(r["SystemType"] or ""),
                        str(r["SiteName"] or ""),
                        float(d),
                        float(r["lat"]),
                        float(r["lon"]),
                    )
                )
        selected_sites.sort(key=lambda x: x[5])
        selected_sites = selected_sites[:limit_sites]

        if include_trunk_site_frequencies and selected_sites:
            site_ids = [sid for _tid, sid, *_rest in selected_sites]
            placeholders = ",".join(["?"] * len(site_ids))
            tf_rows = db.execute(
                f"""
                SELECT tf.SiteId, tf.frequency_hz AS FrequencyHz, tf.channel_name AS ChannelName,
                       tf.lcn1 AS Lcn1, tf.lcn2 AS Lcn2, tf.ran AS RAN, tf.avoid AS Avoid
                FROM t_freq tf
                WHERE tf.SiteId IN ({placeholders})
                ORDER BY tf.SiteId, tf.frequency_hz
                LIMIT ?
                """,
                (*site_ids, limit_site_freq_rows),
            ).fetchall()
            freq_by_site: Dict[int, List[sqlite3.Row]] = {}
            for r in tf_rows:
                freq_by_site.setdefault(int(r["SiteId"]), []).append(r)

            sites_by_trunk: Dict[int, Dict[int, TrunkSite]] = {}
            for trunk_id, site_id, sys_name, sys_type, site_name, dist, site_lat, site_lon in selected_sites:
                trunked.setdefault(
                    trunk_id,
                    TrunkSystem(trunk_id=trunk_id, system_name=sys_name, system_type=sys_type, sites=[]),
                )
                sites_by_trunk.setdefault(trunk_id, {})
                ts = sites_by_trunk[trunk_id].setdefault(
                    site_id, TrunkSite(site_id=site_id, site_name=site_name, distance_miles=dist, lat=site_lat, lon=site_lon)
                )
                for fr in freq_by_site.get(site_id, []):
                    ts.frequencies.append(
                        SiteFrequencyRow(
                            frequency_hz=fr["FrequencyHz"],
                            frequency_mhz=hz_to_mhz(fr["FrequencyHz"]),
                            channel_name=fr["ChannelName"],
                            lcn1=fr["Lcn1"],
                            lcn2=fr["Lcn2"],
                            ran=fr["RAN"],
                            avoid=fr["Avoid"],
                        )
                    )
            for trunk_id, site_map in sites_by_trunk.items():
                trunked[trunk_id].sites = list(site_map.values())
        else:
            # Sites without frequencies: still return site metadata.
            sites_by_trunk: Dict[int, Dict[int, TrunkSite]] = {}
            for trunk_id, site_id, sys_name, sys_type, site_name, dist, site_lat, site_lon in selected_sites:
                trunked.setdefault(trunk_id, TrunkSystem(trunk_id=trunk_id, system_name=sys_name, system_type=sys_type, sites=[]))
                sites_by_trunk.setdefault(trunk_id, {})
                sites_by_trunk[trunk_id].setdefault(
                    site_id,
                    TrunkSite(site_id=site_id, site_name=site_name, distance_miles=dist, lat=site_lat, lon=site_lon),
                )
            for trunk_id, site_map in sites_by_trunk.items():
                trunked[trunk_id].sites = list(site_map.values())

        if include_talkgroups and trunked:
            trunk_ids = sorted(trunked.keys())
            placeholders = ",".join(["?"] * len(trunk_ids))
            tg_rows = db.execute(
                f"""
                SELECT t.TrunkId, t.name AS SystemName, t.system_type AS SystemType,
                       tg.TGroupId, tg.name AS TalkgroupCategory,
                       g.Tid, g.alpha_tag AS AlphaTag, g.talkgroup_value AS Talkgroup,
                       g.service AS Service, g.number_tag AS NumberTag, g.avoid AS Avoid
                FROM trunk t
                JOIN t_group tg ON tg.TrunkId = t.TrunkId
                JOIN tgid g ON g.TGroupId = tg.TGroupId
                WHERE t.TrunkId IN ({placeholders})
                ORDER BY t.name, tg.name, g.alpha_tag, g.Tid
                LIMIT ?
                """,
                (*trunk_ids, limit_talkgroups),
            ).fetchall()
            talkgroups = [
                TalkgroupRow(
                    trunk_id=int(r["TrunkId"]),
                    system_name=r["SystemName"],
                    system_type=r["SystemType"],
                    category=r["TalkgroupCategory"],
                    tid=int(r["Tid"]),
                    alpha_tag=r["AlphaTag"],
                    talkgroup=r["Talkgroup"],
                    service=r["Service"],
                    number_tag=r["NumberTag"],
                    avoid=r["Avoid"],
                )
                for r in tg_rows
            ]

    return CountyQueryResponse(
        county=county_model,
        center_lat=lat,
        center_lon=lon,
        radius_miles=radius_miles,
        conventional=conventional,
        trunked=list(trunked.values()),
        talkgroups=talkgroups,
    )


@app.get("/hpdb/query/by-zip", response_model=CountyQueryResponse)
def hpdb_query_by_zip(
    zip: str,
    radius_miles: float = Query(0.0, ge=0.0, description="Optional search radius in miles; 0 means ZIP's county only"),
    state_fallback: Optional[str] = Query(None, description="If ZIP is missing state_id, use this state abbrev"),
    include_talkgroups: bool = Query(True),
    include_trunk_sites: bool = Query(True),
    include_trunk_site_frequencies: bool = Query(True),
    limit_sites: int = Query(100, gt=0, le=5000, description="Max trunk sites to return when radius_miles > 0"),
    limit_talkgroups: int = Query(2000, gt=0, le=500000, description="Max talkgroups to return"),
    limit_site_freq_rows: int = Query(5000, gt=0, le=500000),
    db: sqlite3.Connection = Depends(get_conn),
):
    zipdb = load_zip_db_full()
    if zip not in zipdb:
        raise HTTPException(status_code=404, detail="ZIP not found in zip database")
    meta = zipdb[zip]
    lat = meta.get("lat")
    lon = meta.get("lon")
    county_name = meta.get("county_name")
    state_abbr = meta.get("state_id") or state_fallback
    if lat is None or lon is None:
        raise HTTPException(status_code=404, detail="ZIP missing lat/lon")

    if radius_miles and radius_miles > 0:
        return hpdb_query_near_point_impl(
            db=db,
            lat=float(lat),
            lon=float(lon),
            radius_miles=float(radius_miles),
            include_talkgroups=include_talkgroups,
            include_trunk_sites=include_trunk_sites,
            include_trunk_site_frequencies=include_trunk_site_frequencies,
            limit_sites=limit_sites,
            limit_talkgroups=limit_talkgroups,
            limit_site_freq_rows=limit_site_freq_rows,
        )

    if not county_name or not state_abbr:
        raise HTTPException(status_code=404, detail="ZIP missing county/state mapping")
    return hpdb_query_impl(
        db=db,
        state=str(state_abbr),
        county=str(county_name),
        include_talkgroups=include_talkgroups,
        include_trunk_sites=include_trunk_sites,
        include_trunk_site_frequencies=include_trunk_site_frequencies,
    )


@app.get("/hpdb/static-map/by-zip", include_in_schema=False)
def static_map_by_zip(
    zip: str,
    radius_miles: float = Query(25.0, ge=1.0, le=100.0, description="Radius in miles to show around the ZIP center"),
    max_markers: int = Query(50, ge=1, le=200, description="Max tower markers to place"),
    db: sqlite3.Connection = Depends(get_conn),
):
    """
    Generate a static PNG map (OpenStreetMap tiles) centered on the ZIP, with nearby tower markers.
    Output is locked to 1000x1000 pixels to avoid excessive tile requests.
    """
    width = height = 1000  # hard cap to avoid abusing OSM tiles

    # Simple ZIP+radius cache to minimize tile fetches
    cache_dir = os.path.join(STATIC_DIR, "cache", "zip-maps")
    os.makedirs(cache_dir, exist_ok=True)
    safe_zip = "".join(ch for ch in str(zip) if ch.isalnum())
    safe_radius = f"{float(radius_miles):.1f}".replace(".", "_")
    cache_path = os.path.join(cache_dir, f"{safe_zip}_{safe_radius}.png")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = f.read()
            # ensure watermark/compression even on legacy cache entries
            cached = _finalize_png_bytes(cached)
            try:
                with open(cache_path, "wb") as f:
                    f.write(cached)
            except Exception:
                pass
            return Response(content=cached, media_type="image/png")
        except Exception:
            pass
    zipdb = load_zip_db_full()
    if zip not in zipdb:
        raise HTTPException(status_code=404, detail="ZIP not found in zip database")
    meta = zipdb[zip]
    lat = meta.get("lat")
    lon = meta.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=404, detail="ZIP missing lat/lon")

    data = hpdb_query_near_point_impl(
        db=db,
        lat=float(lat),
        lon=float(lon),
        radius_miles=float(radius_miles),
        include_talkgroups=False,
        include_trunk_site_frequencies=False,
        limit_sites=max_markers,
    )

    markers: List[str] = []
    marker_points: List[Tuple[float, float]] = []
    marker_labels: List[Tuple[float, float, str]] = []
    seen = set()
    for sys in data.trunked or []:
        for site in sys.sites or []:
            if site.lat is None or site.lon is None:
                continue
            key = (round(site.lat, 5), round(site.lon, 5))
            if key in seen:
                continue
            seen.add(key)
            markers.append(f"{site.lat:.6f},{site.lon:.6f},lightblue1")
            marker_points.append((site.lat, site.lon))
            marker_labels.append((site.lat, site.lon, site.site_name or f"Site {site.site_id}"))
            if len(markers) >= max_markers:
                break
        if len(markers) >= max_markers:
            break

    zoom = _approx_zoom(float(lat), float(radius_miles), width_px=width)
    base_w, base_h = width, height  # locked extent and output size

    def _recompress_png_bytes(buf: bytes) -> bytes:
        return _finalize_png_bytes(buf)

    def overlay_if_ok(base_bytes: bytes) -> bytes | None:
        # treat tiny responses as failures
        if not base_bytes or len(base_bytes) < 1024:
            return None
        try:
            return _draw_labels_on_image(
                base_bytes,
                float(lat),
                float(lon),
                zoom,
                base_w,
                base_h,
                marker_labels,
                extent_width=base_w,
                extent_height=base_h,
            )
        except Exception:
            return None

    # Primary: render base via local OSM tiles, then overlay
    try:
        base = _render_osm_static(float(lat), float(lon), zoom, base_w, base_h, [])
        labeled = overlay_if_ok(base)
        if labeled:
            try:
                with open(cache_path, "wb") as f:
                    f.write(_recompress_png_bytes(labeled))
            except Exception:
                pass
            return Response(content=_recompress_png_bytes(labeled), media_type="image/png")
    except Exception:
        pass

    # Next: render base via py-staticmaps (no markers), then overlay
    try:
        base = _render_staticmaps_base(float(lat), float(lon), zoom, base_w, base_h)
        labeled = overlay_if_ok(base)
        if labeled:
            try:
                with open(cache_path, "wb") as f:
                    f.write(_recompress_png_bytes(labeled))
            except Exception:
                pass
            return Response(content=_recompress_png_bytes(labeled), media_type="image/png")
    except HTTPException:
        raise
    except Exception:
        pass

    # Next: fetch staticmap.de (no markers) and overlay
    url = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={base_w}x{base_h}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "CodePlug-PB/0.1 static-map"})
        if resp.status_code == 200 and resp.content:
            labeled = overlay_if_ok(resp.content)
            if labeled:
                try:
                    with open(cache_path, "wb") as f:
                        f.write(_recompress_png_bytes(labeled))
                except Exception:
                    pass
                return Response(content=_recompress_png_bytes(labeled), media_type="image/png")
            try:
                with open(cache_path, "wb") as f:
                    f.write(_recompress_png_bytes(resp.content))
            except Exception:
                pass
            return Response(content=_recompress_png_bytes(resp.content), media_type="image/png")
    except Exception:
        pass

    # Fallback: py-staticmaps with labels (full render)
    try:
        base = _render_staticmaps_with_labels(float(lat), float(lon), zoom, base_w, base_h, marker_labels)
        labeled = overlay_if_ok(base) or base
        if labeled:
            try:
                with open(cache_path, "wb") as f:
                    f.write(_recompress_png_bytes(labeled))
            except Exception:
                pass
            return Response(content=_recompress_png_bytes(labeled), media_type="image/png")
    except HTTPException:
        raise
    except Exception:
        pass

    # Final attempt: staticmap.de with markers (no overlay) so at least a map shows
    url_markers = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={base_w}x{base_h}"
    if markers:
        url_markers += "&markers=" + "|".join(markers)
    try:
        resp = requests.get(url_markers, timeout=10, headers={"User-Agent": "CodePlug-PB/0.1 static-map"})
        if resp.status_code == 200 and resp.content and len(resp.content) >= 4096:
            try:
                with open(cache_path, "wb") as f:
                    f.write(_recompress_png_bytes(resp.content))
            except Exception:
                pass
            return Response(content=_recompress_png_bytes(resp.content), media_type="image/png")
    except Exception:
        pass

    # Last resort: 1x1 transparent PNG so downstream ZIPs are never empty
    fallback_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15"
        b"\xc4\x89\x00\x00\x00\x0cIDAT\x08\xd7c```\x00\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(content=fallback_png, media_type="image/png")


@app.get("/hpdb/static-map-simple/by-zip", include_in_schema=False)
def static_map_simple_by_zip(
    zip: str,
    radius_miles: float = Query(25.0, ge=0.1, description="Radius in miles to show around the ZIP center"),
    width: int = Query(640, ge=128, le=2048),
    height: int = Query(640, ge=128, le=2048),
    max_markers: int = Query(50, ge=1, le=200),
    db: sqlite3.Connection = Depends(get_conn),
):
    """
    Old-style static map: OSM tiles + blue dots, no labels.
    """
    zipdb = load_zip_db_full()
    if zip not in zipdb:
        raise HTTPException(status_code=404, detail="ZIP not found in zip database")
    meta = zipdb[zip]
    lat = meta.get("lat")
    lon = meta.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=404, detail="ZIP missing lat/lon")

    data = hpdb_query_near_point_impl(
        db=db,
        lat=float(lat),
        lon=float(lon),
        radius_miles=float(radius_miles),
        include_talkgroups=False,
        include_trunk_site_frequencies=False,
        limit_sites=max_markers,
    )

    markers: List[str] = []
    marker_points: List[Tuple[float, float]] = []
    marker_labels: List[Tuple[float, float, str]] = []
    seen = set()
    for sys in data.trunked or []:
        for site in sys.sites or []:
            if site.lat is None or site.lon is None:
                continue
            key = (round(site.lat, 5), round(site.lon, 5))
            if key in seen:
                continue
            seen.add(key)
            markers.append(f"{site.lat:.6f},{site.lon:.6f},blue")
            marker_points.append((site.lat, site.lon))
            marker_labels.append((site.lat, site.lon, site.site_name or f"Site {site.site_id}"))
            if len(markers) >= max_markers:
                break
        if len(markers) >= max_markers:
            break

    zoom = _approx_zoom(float(lat), float(radius_miles), width_px=width)

    def overlay_if_ok(base_bytes: bytes) -> bytes | None:
        if not base_bytes or len(base_bytes) < 1024:
            return None
        try:
            return _draw_labels_on_image(base_bytes, float(lat), float(lon), zoom, width, height, marker_labels)
        except Exception:
            return None

    # Try local OSM tiles as base + labels
    try:
        base = _render_osm_static(float(lat), float(lon), zoom, width, height, [])
        if base and len(base) >= 1024:
            labeled = overlay_if_ok(base)
            if labeled:
                return Response(content=labeled, media_type="image/png")
    except HTTPException:
        raise
    except Exception:
        pass

    # Try staticmap.de base (no markers) and overlay
    url = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={width}x{height}"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "CodePlug-PB/0.1 static-map"})
        if resp.status_code == 200 and resp.content and len(resp.content) >= 4096:
            labeled = overlay_if_ok(resp.content)
            if labeled:
                return Response(content=labeled, media_type="image/png")
    except Exception:
        pass

    # Try staticmap.de with markers (no overlay) last
    url_markers = f"https://staticmap.openstreetmap.de/staticmap.php?center={lat},{lon}&zoom={zoom}&size={width}x{height}"
    if markers:
        url_markers += "&markers=" + "|".join(markers)
    try:
        resp = requests.get(url_markers, timeout=10, headers={"User-Agent": "CodePlug-PB/0.1 static-map"})
        if resp.status_code == 200 and resp.content and len(resp.content) >= 4096:
            return Response(content=resp.content, media_type="image/png")
    except Exception:
        pass

    # Last resort
    fallback_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15"
        b"\xc4\x89\x00\x00\x00\x0cIDAT\x08\xd7c```\x00\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(content=fallback_png, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", **get_app_metadata()}


@app.get("/stats")
def stats(db: sqlite3.Connection = Depends(get_conn)):
    # Basic counts from HPDB-derived tables.
    systems = db.execute("SELECT COUNT(*) FROM trunk").fetchone()[0]
    sites = db.execute("SELECT COUNT(*) FROM site").fetchone()[0]
    talkgroups = db.execute("SELECT COUNT(*) FROM tgid").fetchone()[0]
    site_freqs = db.execute("SELECT COUNT(*) FROM t_freq").fetchone()[0]
    conv_freqs = db.execute("SELECT COUNT(*) FROM c_freq").fetchone()[0]
    counties = db.execute("SELECT COUNT(*) FROM county_info").fetchone()[0]
    return {
        "trunk_systems": systems,
        "trunk_sites": sites,
        "trunk_site_frequencies": site_freqs,
        "talkgroups": talkgroups,
        "counties": counties,
        "conventional_frequencies": conv_freqs,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "16444"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
