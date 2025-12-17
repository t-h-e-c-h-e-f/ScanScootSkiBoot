#!/usr/bin/env python3
"""
Convert Uniden-style HomePatrol database text files (hpdb.cfg / MasterHpdb.hp1 / s_*.hpd)
into a SQLite database.

This repo contains plain-text, tab-separated records. Each line begins with a record type
like "Conventional", "Trunk", "Site", "C-Freq", etc. After that come a mixture of:
  - key=value tokens (IDs and optional attributes)
  - positional fields (name, avoid, coordinates, etc), which vary by record type

This converter is intentionally conservative: it always stores the original line and
tokenization in a generic `records` table, and it also extracts common record types into
typed tables for easier querying.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


KV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ParsedLine:
    record_type: str
    kv: dict[str, str]
    fields: list[str]
    raw: str


def parse_hpdb_line(raw_line: str) -> ParsedLine | None:
    raw_line = raw_line.rstrip("\r\n")
    if raw_line == "":
        return None

    parts = raw_line.split("\t")
    record_type = parts[0]

    kv: dict[str, str] = {}
    fields: list[str] = []
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            if KV_KEY_RE.match(k):
                kv[k] = v
                continue
        fields.append(token)

    return ParsedLine(record_type=record_type, kv=kv, fields=fields, raw=raw_line)


def discover_input_files(root: Path, include_s_files: bool) -> list[Path]:
    files: list[Path] = []
    cfg = root / "hpdb.cfg"
    master = root / "MasterHpdb.hp1"
    if cfg.exists():
        files.append(cfg)
    if master.exists():
        files.append(master)

    if include_s_files:
        for p in sorted(root.glob("s_*.hpd")):
            files.append(p)
    return files


def classify_source(path: Path) -> tuple[str, int | None]:
    name = path.name
    if name == "hpdb.cfg":
        return ("cfg", None)
    if name == "MasterHpdb.hp1":
        return ("master", None)
    m = re.match(r"^s_(\d+)\.hpd$", name)
    if m:
        return ("update", int(m.group(1)))
    return ("unknown", None)


def init_db(conn: sqlite3.Connection, *, create_lossless_records: bool) -> None:
    conn.execute("PRAGMA foreign_keys=ON")

    # SQLite doesn't compress automatically. These pragmas reduce overhead during build and
    # can speed up inserts (and slightly reduce transient disk usage while loading).
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_files (
          source_file_id INTEGER PRIMARY KEY,
          path TEXT NOT NULL UNIQUE,
          kind TEXT NOT NULL,         -- cfg | master | update | unknown
          sequence INTEGER            -- for update files (s_000123.hpd -> 123)
        );
        """
    )

    if create_lossless_records:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
              record_id INTEGER PRIMARY KEY,
              source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
              line_no INTEGER NOT NULL,
              record_type TEXT NOT NULL,
              kv_json TEXT NOT NULL,
              fields_json TEXT NOT NULL,
              raw_line TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type);
            CREATE INDEX IF NOT EXISTS idx_records_source ON records(source_file_id);
            """
        )

    conn.executescript(
        """
        -- Reference / geography
        CREATE TABLE IF NOT EXISTS state_info (
          StateId INTEGER PRIMARY KEY,
          CountryId INTEGER,
          name TEXT,
          abbrev TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        CREATE TABLE IF NOT EXISTS county_info (
          CountyId INTEGER PRIMARY KEY,
          StateId INTEGER,
          name TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        -- Conventional
        CREATE TABLE IF NOT EXISTS conventional (
          -- There isn't an obvious stable primary key for these records, so use a synthetic key.
          conventional_id INTEGER PRIMARY KEY,
          StateId INTEGER,
          CountyId INTEGER,
          AgencyId INTEGER,
          name TEXT,
          avoid TEXT,
          date_modified TEXT,
          category TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_conventional_state ON conventional(StateId);
        CREATE INDEX IF NOT EXISTS idx_conventional_county ON conventional(CountyId);
        CREATE INDEX IF NOT EXISTS idx_conventional_agency ON conventional(AgencyId);

        -- Conventional groups and frequencies (used in conventional "Departments" / groups)
        CREATE TABLE IF NOT EXISTS c_group (
          CGroupId INTEGER PRIMARY KEY,
          CountyId INTEGER,
          AgencyId INTEGER,
          name TEXT,
          avoid TEXT,
          lat REAL,
          lon REAL,
          range REAL,
          shape TEXT,
          extra1 TEXT,
          extra2 TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_c_group_county ON c_group(CountyId);
        CREATE INDEX IF NOT EXISTS idx_c_group_agency ON c_group(AgencyId);

        CREATE TABLE IF NOT EXISTS c_freq (
          CFreqId INTEGER PRIMARY KEY,
          CGroupId INTEGER,
          name TEXT,
          avoid TEXT,
          frequency_hz INTEGER,
          modulation TEXT,
          tone TEXT,
          nac TEXT,
          color_code TEXT,
          ran TEXT,
          number_tag INTEGER,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_c_freq_group ON c_freq(CGroupId);
        CREATE INDEX IF NOT EXISTS idx_c_freq_freq ON c_freq(frequency_hz);

        -- Trunked systems, sites, talkgroup containers, talkgroups, and site frequencies
        CREATE TABLE IF NOT EXISTS trunk (
          TrunkId INTEGER PRIMARY KEY,
          StateId INTEGER,
          name TEXT,
          avoid TEXT,
          date_modified TEXT,
          system_type TEXT,
          nxdn_flavor TEXT,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_trunk_state ON trunk(StateId);

        CREATE TABLE IF NOT EXISTS site (
          SiteId INTEGER PRIMARY KEY,
          TrunkId INTEGER,
          name TEXT,
          avoid TEXT,
          lat REAL,
          lon REAL,
          range REAL,
          location_mode TEXT,
          site_type TEXT,
          bandwidth TEXT,
          shape TEXT,
          search_hint TEXT,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_site_trunk ON site(TrunkId);

        CREATE TABLE IF NOT EXISTS t_group (
          TGroupId INTEGER PRIMARY KEY,
          TrunkId INTEGER,
          name TEXT,
          avoid TEXT,
          lat REAL,
          lon REAL,
          range REAL,
          shape TEXT,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_t_group_trunk ON t_group(TrunkId);

        CREATE TABLE IF NOT EXISTS tgid (
          Tid INTEGER PRIMARY KEY,
          TGroupId INTEGER,
          alpha_tag TEXT,
          avoid TEXT,
          talkgroup_value TEXT,
          service TEXT,
          number_tag INTEGER,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_tgid_group ON tgid(TGroupId);

        CREATE TABLE IF NOT EXISTS t_freq (
          TFreqId INTEGER,
          SiteId INTEGER,
          channel_name TEXT,
          avoid TEXT,
          frequency_hz INTEGER,
          lcn1 INTEGER,
          lcn2 INTEGER,
          ran TEXT,
          area TEXT,
          extras_json TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
          PRIMARY KEY (TFreqId, SiteId, frequency_hz)
        );
        CREATE INDEX IF NOT EXISTS idx_t_freq_site ON t_freq(SiteId);
        CREATE INDEX IF NOT EXISTS idx_t_freq_freq ON t_freq(frequency_hz);

        -- Location / geography helpers
        CREATE TABLE IF NOT EXISTS area_state (
          area_state_id INTEGER PRIMARY KEY,
          StateId INTEGER,
          CountyId INTEGER,
          AgencyId INTEGER,
          TrunkId INTEGER,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_area_state_state ON area_state(StateId);
        CREATE INDEX IF NOT EXISTS idx_area_state_trunk ON area_state(TrunkId);

        CREATE TABLE IF NOT EXISTS area_county (
          area_county_id INTEGER PRIMARY KEY,
          CountyId INTEGER,
          AgencyId INTEGER,
          TrunkId INTEGER,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_area_county_county ON area_county(CountyId);
        CREATE INDEX IF NOT EXISTS idx_area_county_trunk ON area_county(TrunkId);

        CREATE TABLE IF NOT EXISTS rectangle (
          rectangle_id INTEGER PRIMARY KEY,
          CGroupId INTEGER,
          TGroupId INTEGER,
          top_lat REAL,
          left_lon REAL,
          bottom_lat REAL,
          right_lon REAL,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        CREATE TABLE IF NOT EXISTS lm (
          lm_id INTEGER PRIMARY KEY,
          StateId INTEGER,
          CountyId INTEGER,
          TrunkId INTEGER,
          SiteId INTEGER,
          unk0 TEXT,
          unk1 TEXT,
          lat REAL,
          lon REAL,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
          UNIQUE (StateId, CountyId, TrunkId, SiteId, unk0, unk1)
        );

        CREATE TABLE IF NOT EXISTS lm_frequency (
          lm_frequency_id INTEGER PRIMARY KEY,
          frequency_hz INTEGER,
          unk0 TEXT,
          code TEXT,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id),
          UNIQUE (frequency_hz, code)
        );

        -- Band plans / fleet maps are preserved in a structured but not deeply decoded way.
        CREATE TABLE IF NOT EXISTS bandplan_mot (
          SiteId INTEGER PRIMARY KEY,
          values_json TEXT NOT NULL,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        CREATE TABLE IF NOT EXISTS bandplan_p25 (
          SiteId INTEGER PRIMARY KEY,
          values_json TEXT NOT NULL,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        CREATE TABLE IF NOT EXISTS fleetmap (
          TrunkId INTEGER PRIMARY KEY,
          values_json TEXT NOT NULL,
          last_source_file_id INTEGER NOT NULL REFERENCES source_files(source_file_id)
        );

        -- Convenience views for common queries (state+county lookups).
        -- Recreate these views on every conversion so changes are picked up.
        DROP VIEW IF EXISTS v_trunk_freqs_by_county;
        DROP VIEW IF EXISTS v_trunk_sites_and_freqs_by_county;
        DROP VIEW IF EXISTS v_trunk_talkgroups_by_county;
        DROP VIEW IF EXISTS v_trunk_systems_by_county;
        DROP VIEW IF EXISTS v_conventional_freqs_by_county;
        DROP VIEW IF EXISTS v_counties;

        CREATE VIEW v_counties AS
        SELECT
          s.StateId AS StateId,
          s.name AS StateName,
          s.abbrev AS StateAbbrev,
          c.CountyId AS CountyId,
          c.name AS CountyName
        FROM county_info c
        JOIN state_info s ON s.StateId = c.StateId;

        CREATE VIEW v_conventional_freqs_by_county AS
        SELECT
          vc.StateId,
          vc.StateAbbrev,
          vc.StateName,
          vc.CountyId,
          vc.CountyName,
          cg.CGroupId,
          cg.name AS DepartmentName,
          cf.CFreqId,
          cf.name AS ChannelName,
          cf.frequency_hz AS FrequencyHz,
          cf.modulation AS Modulation,
          cf.tone AS Tone,
          cf.nac AS NAC,
          cf.color_code AS ColorCode,
          cf.ran AS RAN,
          cf.number_tag AS NumberTag,
          cf.avoid AS Avoid
        FROM v_counties vc
        JOIN c_group cg ON cg.CountyId = vc.CountyId
        JOIN c_freq cf ON cf.CGroupId = cg.CGroupId;

        -- Trunk mapping logic:
        --  - County-specific trunks: AreaCounty(CountyId=<county>)
        --  - Statewide/multi-county trunks: AreaCounty(CountyId=0) + AreaState(StateId=<state>)
        CREATE VIEW v_trunk_systems_by_county AS
        SELECT DISTINCT
          StateId,
          StateAbbrev,
          StateName,
          CountyId,
          CountyName,
          TrunkId,
          SystemName,
          SystemType,
          Avoid
        FROM (
          SELECT
            vc.StateId,
            vc.StateAbbrev,
            vc.StateName,
            vc.CountyId,
            vc.CountyName,
            t.TrunkId,
            t.name AS SystemName,
            t.system_type AS SystemType,
            t.avoid AS Avoid
          FROM v_counties vc
          JOIN area_county ac ON ac.CountyId = vc.CountyId
          JOIN trunk t ON t.TrunkId = ac.TrunkId
          WHERE ac.TrunkId IS NOT NULL

          UNION ALL

          SELECT
            vc.StateId,
            vc.StateAbbrev,
            vc.StateName,
            vc.CountyId,
            vc.CountyName,
            t.TrunkId,
            t.name AS SystemName,
            t.system_type AS SystemType,
            t.avoid AS Avoid
          FROM v_counties vc
          JOIN area_state ast ON ast.StateId = vc.StateId
          JOIN area_county ac0 ON ac0.TrunkId = ast.TrunkId AND ac0.CountyId = 0
          JOIN trunk t ON t.TrunkId = ast.TrunkId
          WHERE ast.TrunkId IS NOT NULL
        );

        CREATE VIEW v_trunk_talkgroups_by_county AS
        SELECT
          vc.StateId,
          vc.StateAbbrev,
          vc.StateName,
          vc.CountyId,
          vc.CountyName,
          t.TrunkId,
          t.name AS SystemName,
          t.system_type AS SystemType,
          tg.TGroupId,
          tg.name AS TalkgroupCategory,
          g.Tid,
          g.alpha_tag AS AlphaTag,
          g.talkgroup_value AS Talkgroup,
          g.service AS Service,
          g.number_tag AS NumberTag,
          g.avoid AS Avoid
        FROM v_counties vc
        JOIN v_trunk_systems_by_county vts
          ON vts.CountyId = vc.CountyId AND vts.StateId = vc.StateId
        JOIN trunk t ON t.TrunkId = vts.TrunkId
        JOIN t_group tg ON tg.TrunkId = t.TrunkId
        JOIN tgid g ON g.TGroupId = tg.TGroupId;

        CREATE VIEW v_trunk_sites_and_freqs_by_county AS
        SELECT
          vc.StateId,
          vc.StateAbbrev,
          vc.StateName,
          vc.CountyId,
          vc.CountyName,
          t.TrunkId,
          t.name AS SystemName,
          t.system_type AS SystemType,
          s.SiteId,
          s.name AS SiteName,
          tf.frequency_hz AS FrequencyHz,
          tf.channel_name AS ChannelName,
          tf.lcn1 AS Lcn1,
          tf.lcn2 AS Lcn2,
          tf.ran AS RAN,
          tf.avoid AS Avoid
        FROM v_counties vc
        JOIN v_trunk_systems_by_county vts
          ON vts.CountyId = vc.CountyId AND vts.StateId = vc.StateId
        JOIN trunk t ON t.TrunkId = vts.TrunkId
        JOIN lm
          ON lm.CountyId = vc.CountyId
         AND lm.TrunkId = t.TrunkId
        JOIN site s ON s.TrunkId = t.TrunkId
               AND s.SiteId = lm.SiteId
        JOIN t_freq tf ON tf.SiteId = s.SiteId;

        CREATE VIEW v_trunk_freqs_by_county AS
        SELECT DISTINCT
          CountyId,
          StateId,
          StateAbbrev,
          StateName,
          CountyName,
          TrunkId,
          SystemName,
          FrequencyHz
        FROM v_trunk_sites_and_freqs_by_county;
        """
    )


def upsert_state_info(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO state_info (StateId, CountryId, name, abbrev, last_source_file_id)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(StateId) DO UPDATE SET
          CountryId=excluded.CountryId,
          name=excluded.name,
          abbrev=excluded.abbrev,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("StateId")),
            to_int(kv.get("CountryId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            source_file_id,
        ),
    )


def upsert_county_info(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO county_info (CountyId, StateId, name, last_source_file_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(CountyId) DO UPDATE SET
          StateId=excluded.StateId,
          name=excluded.name,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("CountyId")),
            to_int(kv.get("StateId")),
            fields[0] if len(fields) > 0 else None,
            source_file_id,
        ),
    )


def insert_conventional(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO conventional (StateId, CountyId, AgencyId, name, avoid, date_modified, category, last_source_file_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_int(kv.get("StateId")),
            to_int(kv.get("CountyId")),
            to_int(kv.get("AgencyId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            fields[2] if len(fields) > 2 else None,
            fields[3] if len(fields) > 3 else None,
            source_file_id,
        ),
    )


def upsert_c_group(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO c_group
          (CGroupId, CountyId, AgencyId, name, avoid, lat, lon, range, shape, extra1, extra2, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(CGroupId) DO UPDATE SET
          CountyId=excluded.CountyId,
          AgencyId=excluded.AgencyId,
          name=excluded.name,
          avoid=excluded.avoid,
          lat=excluded.lat,
          lon=excluded.lon,
          range=excluded.range,
          shape=excluded.shape,
          extra1=excluded.extra1,
          extra2=excluded.extra2,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("CGroupId")),
            to_int(kv.get("CountyId")),
            to_int(kv.get("AgencyId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            to_float(fields[2]) if len(fields) > 2 else None,
            to_float(fields[3]) if len(fields) > 3 else None,
            to_float(fields[4]) if len(fields) > 4 else None,
            fields[5] if len(fields) > 5 else None,
            fields[6] if len(fields) > 6 else None,
            fields[7] if len(fields) > 7 else None,
            source_file_id,
        ),
    )


def _parse_c_freq_fields(fields: list[str], kv: dict[str, str]) -> tuple[str | None, str | None, int | None, str | None, str | None, int | None, list[str]]:
    name = fields[0] if len(fields) > 0 else None
    avoid = fields[1] if len(fields) > 1 else None
    frequency_hz = to_int(fields[2]) if len(fields) > 2 else None
    modulation = fields[3] if len(fields) > 3 else None

    # After modulation, the data is inconsistent across sources; keep a best-effort parse.
    rest = fields[4:] if len(fields) > 4 else []
    while rest and rest[-1] == "":
        rest = rest[:-1]

    tone_field: str | None = None
    number_tag: int | None = None
    extras_start = 0

    if rest:
        if rest[0] == "" and len(rest) >= 2 and rest[1].isdigit():
            tone_field = ""
            number_tag = int(rest[1])
            extras_start = 2
        elif rest[0].isdigit():
            number_tag = int(rest[0])
            extras_start = 1
        elif len(rest) >= 2 and rest[1].isdigit():
            tone_field = rest[0]
            number_tag = int(rest[1])
            extras_start = 2
        else:
            tone_field = rest[0]
            extras_start = 1

    tone = kv.get("TONE", tone_field)
    extras = rest[extras_start:] if extras_start < len(rest) else []

    return name, avoid, frequency_hz, modulation, tone, number_tag, extras


def upsert_c_freq(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    name, avoid, frequency_hz, modulation, tone, number_tag, extras = _parse_c_freq_fields(fields, kv)
    conn.execute(
        """
        INSERT INTO c_freq
          (CFreqId, CGroupId, name, avoid, frequency_hz, modulation, tone, nac, color_code, ran, number_tag, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(CFreqId) DO UPDATE SET
          CGroupId=excluded.CGroupId,
          name=excluded.name,
          avoid=excluded.avoid,
          frequency_hz=excluded.frequency_hz,
          modulation=excluded.modulation,
          tone=excluded.tone,
          nac=excluded.nac,
          color_code=excluded.color_code,
          ran=excluded.ran,
          number_tag=excluded.number_tag,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("CFreqId")),
            to_int(kv.get("CGroupId")),
            name,
            avoid,
            frequency_hz,
            modulation,
            tone,
            kv.get("NAC"),
            kv.get("ColorCode"),
            kv.get("RAN"),
            number_tag,
            json_dumps(extras),
            source_file_id,
        ),
    )


def upsert_trunk(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    extras = fields[4:] if len(fields) > 4 else []
    nxdn_flavor = fields[18] if len(fields) > 18 and fields[18] != "" else None
    conn.execute(
        """
        INSERT INTO trunk
          (TrunkId, StateId, name, avoid, date_modified, system_type, nxdn_flavor, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(TrunkId) DO UPDATE SET
          StateId=excluded.StateId,
          name=excluded.name,
          avoid=excluded.avoid,
          date_modified=excluded.date_modified,
          system_type=excluded.system_type,
          nxdn_flavor=excluded.nxdn_flavor,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("TrunkId")),
            to_int(kv.get("StateId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            fields[2] if len(fields) > 2 else None,
            fields[3] if len(fields) > 3 else None,
            nxdn_flavor,
            json_dumps(extras),
            source_file_id,
        ),
    )


def upsert_site(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    extras = fields[9:] if len(fields) > 9 else []
    search_hint = fields[14] if len(fields) > 14 and fields[14] != "" else None
    conn.execute(
        """
        INSERT INTO site
          (SiteId, TrunkId, name, avoid, lat, lon, range, location_mode, site_type, bandwidth, shape, search_hint, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(SiteId) DO UPDATE SET
          TrunkId=excluded.TrunkId,
          name=excluded.name,
          avoid=excluded.avoid,
          lat=excluded.lat,
          lon=excluded.lon,
          range=excluded.range,
          location_mode=excluded.location_mode,
          site_type=excluded.site_type,
          bandwidth=excluded.bandwidth,
          shape=excluded.shape,
          search_hint=excluded.search_hint,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("SiteId")),
            to_int(kv.get("TrunkId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            to_float(fields[2]) if len(fields) > 2 else None,
            to_float(fields[3]) if len(fields) > 3 else None,
            to_float(fields[4]) if len(fields) > 4 else None,
            fields[5] if len(fields) > 5 else None,
            fields[6] if len(fields) > 6 else None,
            fields[7] if len(fields) > 7 else None,
            fields[8] if len(fields) > 8 else None,
            search_hint,
            json_dumps(extras),
            source_file_id,
        ),
    )


def upsert_t_group(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    extras = fields[6:] if len(fields) > 6 else []
    conn.execute(
        """
        INSERT INTO t_group
          (TGroupId, TrunkId, name, avoid, lat, lon, range, shape, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(TGroupId) DO UPDATE SET
          TrunkId=excluded.TrunkId,
          name=excluded.name,
          avoid=excluded.avoid,
          lat=excluded.lat,
          lon=excluded.lon,
          range=excluded.range,
          shape=excluded.shape,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("TGroupId")),
            to_int(kv.get("TrunkId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            to_float(fields[2]) if len(fields) > 2 else None,
            to_float(fields[3]) if len(fields) > 3 else None,
            to_float(fields[4]) if len(fields) > 4 else None,
            fields[5] if len(fields) > 5 else None,
            json_dumps(extras),
            source_file_id,
        ),
    )


def upsert_tgid(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    extras = fields[5:] if len(fields) > 5 else []
    conn.execute(
        """
        INSERT INTO tgid
          (Tid, TGroupId, alpha_tag, avoid, talkgroup_value, service, number_tag, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Tid) DO UPDATE SET
          TGroupId=excluded.TGroupId,
          alpha_tag=excluded.alpha_tag,
          avoid=excluded.avoid,
          talkgroup_value=excluded.talkgroup_value,
          service=excluded.service,
          number_tag=excluded.number_tag,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("Tid")),
            to_int(kv.get("TGroupId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            fields[2] if len(fields) > 2 else None,
            fields[3] if len(fields) > 3 else None,
            to_int(fields[4]) if len(fields) > 4 else None,
            json_dumps(extras),
            source_file_id,
        ),
    )


def upsert_t_freq(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    extras = fields[5:] if len(fields) > 5 else []
    conn.execute(
        """
        INSERT INTO t_freq
          (TFreqId, SiteId, channel_name, avoid, frequency_hz, lcn1, lcn2, ran, area, extras_json, last_source_file_id)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(TFreqId, SiteId, frequency_hz) DO UPDATE SET
          channel_name=excluded.channel_name,
          avoid=excluded.avoid,
          lcn1=excluded.lcn1,
          lcn2=excluded.lcn2,
          ran=excluded.ran,
          area=excluded.area,
          extras_json=excluded.extras_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("TFreqId")) or 0,
            to_int(kv.get("SiteId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            to_int(fields[2]) if len(fields) > 2 else None,
            to_int(fields[3]) if len(fields) > 3 else None,
            to_int(fields[4]) if len(fields) > 4 else None,
            kv.get("RAN"),
            kv.get("Area"),
            json_dumps(extras),
            source_file_id,
        ),
    )


def insert_area_state(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO area_state (StateId, CountyId, AgencyId, TrunkId, last_source_file_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (to_int(kv.get("StateId")), to_int(kv.get("CountyId")), to_int(kv.get("AgencyId")), to_int(kv.get("TrunkId")), source_file_id),
    )


def insert_area_county(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str]) -> None:
    conn.execute(
        """
        INSERT INTO area_county (CountyId, AgencyId, TrunkId, last_source_file_id)
        VALUES (?, ?, ?, ?)
        """,
        (to_int(kv.get("CountyId")), to_int(kv.get("AgencyId")), to_int(kv.get("TrunkId")), source_file_id),
    )


def insert_rectangle(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO rectangle (CGroupId, TGroupId, top_lat, left_lon, bottom_lat, right_lon, last_source_file_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            to_int(kv.get("CGroupId")),
            to_int(kv.get("TGroupId")),
            to_float(fields[0]) if len(fields) > 0 else None,
            to_float(fields[1]) if len(fields) > 1 else None,
            to_float(fields[2]) if len(fields) > 2 else None,
            to_float(fields[3]) if len(fields) > 3 else None,
            source_file_id,
        ),
    )


def insert_lm(conn: sqlite3.Connection, source_file_id: int, kv: dict[str, str], fields: list[str]) -> None:
    conn.execute(
        """
        INSERT INTO lm (StateId, CountyId, TrunkId, SiteId, unk0, unk1, lat, lon, last_source_file_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(StateId, CountyId, TrunkId, SiteId, unk0, unk1) DO UPDATE SET
          lat=excluded.lat,
          lon=excluded.lon,
          last_source_file_id=excluded.last_source_file_id
        """,
        (
            to_int(kv.get("StateId")),
            to_int(kv.get("CountyId")),
            to_int(kv.get("TrunkId")),
            to_int(kv.get("SiteId")),
            fields[0] if len(fields) > 0 else None,
            fields[1] if len(fields) > 1 else None,
            to_float(fields[2]) if len(fields) > 2 else None,
            to_float(fields[3]) if len(fields) > 3 else None,
            source_file_id,
        ),
    )


def insert_lm_frequency(conn: sqlite3.Connection, source_file_id: int, fields: list[str]) -> None:
    frequency_hz = to_int(fields[0]) if len(fields) > 0 else None
    code = fields[2] if len(fields) > 2 else None
    conn.execute(
        """
        INSERT INTO lm_frequency (frequency_hz, unk0, code, last_source_file_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(frequency_hz, code) DO UPDATE SET
          unk0=excluded.unk0,
          last_source_file_id=excluded.last_source_file_id
        """,
        (frequency_hz, fields[1] if len(fields) > 1 else None, code, source_file_id),
    )


def upsert_bandplan(conn: sqlite3.Connection, table: str, source_file_id: int, site_id: int | None, fields: list[str]) -> None:
    if site_id is None:
        return
    conn.execute(
        f"""
        INSERT INTO {table} (SiteId, values_json, last_source_file_id)
        VALUES (?, ?, ?)
        ON CONFLICT(SiteId) DO UPDATE SET
          values_json=excluded.values_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (site_id, json_dumps(fields), source_file_id),
    )


def upsert_fleetmap(conn: sqlite3.Connection, source_file_id: int, trunk_id: int | None, fields: list[str]) -> None:
    if trunk_id is None:
        return
    conn.execute(
        """
        INSERT INTO fleetmap (TrunkId, values_json, last_source_file_id)
        VALUES (?, ?, ?)
        ON CONFLICT(TrunkId) DO UPDATE SET
          values_json=excluded.values_json,
          last_source_file_id=excluded.last_source_file_id
        """,
        (trunk_id, json_dumps(fields), source_file_id),
    )


def load_file(conn: sqlite3.Connection, path: Path, source_file_id: int, *, create_lossless_records: bool) -> None:
    with path.open("r", newline="") as f:
        for line_no, raw_line in enumerate(f, 1):
            parsed = parse_hpdb_line(raw_line)
            if parsed is None:
                continue

            if create_lossless_records:
                conn.execute(
                    """
                    INSERT INTO records (source_file_id, line_no, record_type, kv_json, fields_json, raw_line)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_file_id,
                        line_no,
                        parsed.record_type,
                        json_dumps(parsed.kv),
                        json_dumps(parsed.fields),
                        parsed.raw,
                    ),
                )

            rt = parsed.record_type
            kv = parsed.kv
            fields = parsed.fields

            if rt == "StateInfo":
                upsert_state_info(conn, source_file_id, kv, fields)
            elif rt == "CountyInfo":
                upsert_county_info(conn, source_file_id, kv, fields)
            elif rt == "Conventional":
                insert_conventional(conn, source_file_id, kv, fields)
            elif rt == "C-Group":
                upsert_c_group(conn, source_file_id, kv, fields)
            elif rt == "C-Freq":
                upsert_c_freq(conn, source_file_id, kv, fields)
            elif rt == "Trunk":
                upsert_trunk(conn, source_file_id, kv, fields)
            elif rt == "Site":
                upsert_site(conn, source_file_id, kv, fields)
            elif rt == "T-Group":
                upsert_t_group(conn, source_file_id, kv, fields)
            elif rt == "TGID":
                upsert_tgid(conn, source_file_id, kv, fields)
            elif rt == "T-Freq":
                upsert_t_freq(conn, source_file_id, kv, fields)
            elif rt == "AreaState":
                insert_area_state(conn, source_file_id, kv)
            elif rt == "AreaCounty":
                insert_area_county(conn, source_file_id, kv)
            elif rt == "Rectangle":
                insert_rectangle(conn, source_file_id, kv, fields)
            elif rt == "LM":
                insert_lm(conn, source_file_id, kv, fields)
            elif rt == "LM_Frequency":
                insert_lm_frequency(conn, source_file_id, fields)
            elif rt == "BandPlan_Mot":
                upsert_bandplan(conn, "bandplan_mot", source_file_id, to_int(kv.get("SiteId")), fields)
            elif rt == "BandPlan_P25":
                upsert_bandplan(conn, "bandplan_p25", source_file_id, to_int(kv.get("SiteId")), fields)
            elif rt == "FleetMap":
                upsert_fleetmap(conn, source_file_id, to_int(kv.get("TrunkId")), fields)


def insert_source_file(conn: sqlite3.Connection, path: Path) -> int:
    kind, sequence = classify_source(path)
    conn.execute(
        "INSERT OR IGNORE INTO source_files (path, kind, sequence) VALUES (?, ?, ?)",
        (str(path), kind, sequence),
    )
    row = conn.execute("SELECT source_file_id FROM source_files WHERE path=?", (str(path),)).fetchone()
    assert row is not None
    return int(row[0])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert HPDB text files to SQLite.")
    parser.add_argument("--input", default=".", help="Directory containing hpdb.cfg / MasterHpdb.hp1 / s_*.hpd")
    parser.add_argument("--out", default="hpdb.sqlite", help="Output SQLite file path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output DB if it already exists")
    parser.add_argument(
        "--include-s-files",
        action="store_true",
        help="Also load s_*.hpd files (often shards/duplicates of MasterHpdb.hp1)",
    )
    lossless_group = parser.add_mutually_exclusive_group()
    lossless_group.add_argument(
        "--lossless-records",
        action="store_true",
        help="Store the full lossless `records` table (larger DB; useful for debugging/auditing)",
    )
    lossless_group.add_argument(
        "--no-lossless-records",
        action="store_true",
        help="Do not store the full lossless `records` table (default; saves a lot of space)",
    )

    vacuum_group = parser.add_mutually_exclusive_group()
    vacuum_group.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM at the end to compact the DB file (default)",
    )
    vacuum_group.add_argument(
        "--no-vacuum",
        action="store_true",
        help="Skip VACUUM (faster build, larger DB)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.input).resolve()
    out_path = Path(args.out).resolve()

    if args.overwrite and out_path.exists():
        out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(out_path))
    try:
        # Defaults: compact DB (no lossless records) and VACUUM.
        create_lossless_records = bool(args.lossless_records)
        init_db(conn, create_lossless_records=create_lossless_records)

        files = discover_input_files(root, include_s_files=bool(args.include_s_files))
        if not files:
            raise SystemExit(f"No input files found in {root}")

        # Ensure update files are processed in numeric order.
        def sort_key(p: Path) -> tuple[int, int, str]:
            kind, seq = classify_source(p)
            kind_rank = {"cfg": 0, "master": 1, "update": 2}.get(kind, 9)
            return (kind_rank, seq or -1, p.name)

        files = sorted(files, key=sort_key)

        conn.execute("BEGIN")
        for path in files:
            source_file_id = insert_source_file(conn, path)
            load_file(conn, path, source_file_id, create_lossless_records=create_lossless_records)
        conn.commit()

        run_vacuum = not bool(args.no_vacuum)
        if run_vacuum:
            conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"Wrote SQLite DB: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
