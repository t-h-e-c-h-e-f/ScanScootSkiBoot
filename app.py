"""FastAPI REST API exposing HPDB SQLite data.

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
import shutil
import tempfile
import threading
import uuid
import hashlib
import time
import configparser
import hmac
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

DB_PATH = os.environ.get("HPDB_PATH", "hpdb_default.sqlite")
ZIP_CSV_PATH = os.environ.get("ZIP_CSV_PATH", "uszips.csv")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
API_KEYS_PATH = os.environ.get("API_KEYS_PATH", "keys.ini")


def get_conn():
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"HPDB database not found at {DB_PATH!r}. Upload MasterHpdb.hp1 via /hpdb/admin/upload-master to initialize.",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


app = FastAPI(title="HPDB REST API", version="0.1.0", description="REST wrapper over local HPDB SQLite")


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
    return RedirectResponse(url="/docs", status_code=302)


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
    <title>HPDB Initialization</title>
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
    <title>HPDB Update</title>
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
            sites_by_trunk[trunk_id].setdefault(site_id, TrunkSite(site_id=site_id, site_name=r["SiteName"], frequencies=[]))
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
            for trunk_id, site_id, sys_name, sys_type, site_name, dist in selected_sites:
                trunked.setdefault(
                    trunk_id,
                    TrunkSystem(trunk_id=trunk_id, system_name=sys_name, system_type=sys_type, sites=[]),
                )
                sites_by_trunk.setdefault(trunk_id, {})
                ts = sites_by_trunk[trunk_id].setdefault(site_id, TrunkSite(site_id=site_id, site_name=site_name, distance_miles=dist))
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
            # Sites without frequencies
            for trunk_id, _site_id, sys_name, sys_type, _site_name, _dist in selected_sites:
                trunked.setdefault(trunk_id, TrunkSystem(trunk_id=trunk_id, system_name=sys_name, system_type=sys_type, sites=[]))

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
