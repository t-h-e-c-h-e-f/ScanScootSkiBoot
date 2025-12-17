#!/usr/bin/env python3
"""
Query helper for hpdb.sqlite produced by hpdb_to_sqlite.py.

Common use-case:
  Provide state + county => list conventional frequencies in that county,
  and list talkgroups for trunked systems mapped to that county.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def hz_to_mhz(freq_hz: int | None) -> str:
    if freq_hz is None:
        return ""
    return f"{freq_hz / 1_000_000:.6f}".rstrip("0").rstrip(".")


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


@dataclass(frozen=True)
class CountyKey:
    state_id: int
    state_abbrev: str
    state_name: str
    county_id: int
    county_name: str


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def find_state(conn: sqlite3.Connection, state_query: str) -> list[sqlite3.Row]:
    q = norm(state_query)
    return conn.execute(
        """
        SELECT StateId, name AS StateName, abbrev AS StateAbbrev
        FROM state_info
        WHERE lower(abbrev) = ?
           OR lower(name) = ?
           OR lower(name) LIKE ?
        ORDER BY CASE WHEN lower(abbrev)=? THEN 0 WHEN lower(name)=? THEN 1 ELSE 2 END, name
        """,
        (q, q, f"%{q}%", q, q),
    ).fetchall()


def find_county(conn: sqlite3.Connection, state_id: int, county_query: str) -> list[sqlite3.Row]:
    q = norm(county_query)
    return conn.execute(
        """
        SELECT CountyId, name AS CountyName
        FROM county_info
        WHERE StateId = ?
          AND (lower(name) = ? OR lower(name) LIKE ?)
        ORDER BY CASE WHEN lower(name)=? THEN 0 ELSE 1 END, name
        """,
        (state_id, q, f"%{q}%", q),
    ).fetchall()


def resolve_county(conn: sqlite3.Connection, state: str, county: str) -> CountyKey:
    states = find_state(conn, state)
    if not states:
        raise SystemExit(f"State not found: {state!r}")
    if len(states) > 1 and norm(states[0]["StateAbbrev"]) != norm(state) and norm(states[0]["StateName"]) != norm(state):
        choices = ", ".join(f"{r['StateAbbrev']} ({r['StateName']})" for r in states[:10])
        raise SystemExit(f"State is ambiguous: {state!r}. Matches: {choices}")

    st = states[0]
    counties = find_county(conn, int(st["StateId"]), county)
    if not counties:
        raise SystemExit(f"County not found in {st['StateAbbrev']}: {county!r}")
    if len(counties) > 1 and norm(counties[0]["CountyName"]) != norm(county):
        choices = ", ".join(f"{r['CountyName']} (CountyId={r['CountyId']})" for r in counties[:15])
        raise SystemExit(f"County is ambiguous in {st['StateAbbrev']}: {county!r}. Matches: {choices}")

    c = counties[0]
    return CountyKey(
        state_id=int(st["StateId"]),
        state_abbrev=str(st["StateAbbrev"] or ""),
        state_name=str(st["StateName"] or ""),
        county_id=int(c["CountyId"]),
        county_name=str(c["CountyName"] or ""),
    )


def rows_to_json(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def query_conventional(conn: sqlite3.Connection, county_id: int, limit: int | None) -> list[sqlite3.Row]:
    sql = """
      SELECT *
      FROM v_conventional_freqs_by_county
      WHERE CountyId = ?
      ORDER BY DepartmentName, FrequencyHz, ChannelName
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (county_id, limit)).fetchall()
    return conn.execute(sql, (county_id,)).fetchall()


def query_trunk_talkgroups(conn: sqlite3.Connection, county_id: int, limit: int | None) -> list[sqlite3.Row]:
    sql = """
      SELECT *
      FROM v_trunk_talkgroups_by_county
      WHERE CountyId = ?
      ORDER BY SystemName, TalkgroupCategory, AlphaTag, Tid
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (county_id, limit)).fetchall()
    return conn.execute(sql, (county_id,)).fetchall()


def query_trunk_site_freqs(conn: sqlite3.Connection, county_id: int, limit: int | None) -> list[sqlite3.Row]:
    sql = """
      SELECT *
      FROM v_trunk_sites_and_freqs_by_county
      WHERE CountyId = ?
      ORDER BY SystemName, SiteName, FrequencyHz
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (county_id, limit)).fetchall()
    return conn.execute(sql, (county_id,)).fetchall()


def group_site_freqs(rows: list[sqlite3.Row]) -> dict[str, dict[str, list[int]]]:
    """
    Returns:
      {system_name: {site_name: [frequency_hz, ...]}}
    """
    out: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        system = str(r["SystemName"] or "")
        site = str(r["SiteName"] or "")
        freq = r["FrequencyHz"]
        if freq is None:
            continue
        out.setdefault(system, {}).setdefault(site, [])
        if int(freq) not in out[system][site]:
            out[system][site].append(int(freq))

    for system in out:
        for site in out[system]:
            out[system][site].sort()
    return out


def print_header(county: CountyKey) -> None:
    print(f"{county.county_name}, {county.state_abbrev} (CountyId={county.county_id})")


def print_conventional_text(conv: list[sqlite3.Row]) -> None:
    print()
    print(f"Conventional frequencies: {len(conv)}")
    last_dept: str | None = None
    for r in conv:
        dept = r["DepartmentName"] or ""
        if dept != last_dept:
            print()
            print(f"[{dept}]")
            last_dept = dept
        freq = hz_to_mhz(r["FrequencyHz"])
        tone = r["Tone"] or r["NAC"] or r["ColorCode"] or r["RAN"] or ""
        tone_str = f" ({tone})" if tone else ""
        mod = (r["Modulation"] or "").strip()
        mod_str = f" {mod}" if mod else ""
        print(f"- {freq} MHz{mod_str}{tone_str} — {r['ChannelName'] or ''}")


def print_talkgroups_text(tgs: list[sqlite3.Row]) -> None:
    print()
    print(f"Trunked talkgroups: {len(tgs)}")
    last_system: str | None = None
    last_category: str | None = None
    for r in tgs:
        system = r["SystemName"] or ""
        category = r["TalkgroupCategory"] or ""
        if system != last_system:
            print()
            print(f"[{system}] ({r['SystemType'] or ''})")
            last_system = system
            last_category = None
        if category != last_category:
            print(f"  - {category}")
            last_category = category
        tg_value = r["Talkgroup"] or ""
        tid = r["Tid"]
        alpha = r["AlphaTag"] or ""
        svc = r["Service"] or ""
        svc_str = f" [{svc}]" if svc else ""
        tg_display = tg_value if tg_value else str(tid)
        print(f"    - {tg_display}: {alpha}{svc_str}")


def print_trunk_freqs_text(freq_rows: list[sqlite3.Row]) -> None:
    # The underlying HPDB text files do not explicitly label “control” vs “voice” in T-Freq.
    # In many cases Uniden/RRDB exports include only control/alternate channels; when the
    # full site frequency list is present, we print it all.
    grouped = group_site_freqs(freq_rows)
    total = sum(len(freqs) for sites in grouped.values() for freqs in sites.values())
    print()
    print(f"Trunked site frequencies (T-Freq): {total}")
    for system_name in sorted(grouped.keys()):
        print()
        print(f"[{system_name}]")
        for site_name in sorted(grouped[system_name].keys()):
            freqs = grouped[system_name][site_name]
            pretty = ", ".join(f"{hz_to_mhz(f)}" for f in freqs)
            print(f"- {site_name}: {pretty} MHz")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query hpdb.sqlite by state + county.")
    parser.add_argument("--db", default="hpdb.sqlite", help="Path to SQLite DB produced by hpdb_to_sqlite.py")
    parser.add_argument("--state", required=True, help="State abbrev or name (e.g., 'AL' or 'Alabama')")
    parser.add_argument("--county", required=True, help="County name (e.g., 'Autauga')")
    parser.add_argument("--limit-conventional", type=int, default=2000, help="Max conventional rows to print (default: 2000)")
    parser.add_argument("--limit-talkgroups", type=int, default=2000, help="Max talkgroup rows to print (default: 2000)")
    parser.add_argument("--limit-trunk-freqs", type=int, default=500, help="Max trunk site frequency rows (default: 500)")
    parser.add_argument("--no-trunk-freqs", action="store_true", help="Do not print trunk site frequencies")
    parser.add_argument("--all", action="store_true", help="Disable limits")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    conn = connect(db_path)
    try:
        county = resolve_county(conn, args.state, args.county)

        limit_conv = None if args.all else args.limit_conventional
        limit_tg = None if args.all else args.limit_talkgroups
        limit_tf = None if args.all else args.limit_trunk_freqs

        conv = query_conventional(conn, county.county_id, limit_conv)
        tgs = query_trunk_talkgroups(conn, county.county_id, limit_tg)
        trunk_freqs = [] if args.no_trunk_freqs else query_trunk_site_freqs(conn, county.county_id, limit_tf)

        if args.format == "json":
            payload = {
                "county": county.__dict__,
                "conventional": rows_to_json(conv),
                "talkgroups": rows_to_json(tgs),
                "trunk_site_frequencies": rows_to_json(trunk_freqs),
            }
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            print()
        else:
            print_header(county)
            print_conventional_text(conv)
            if not args.no_trunk_freqs:
                print_trunk_freqs_text(trunk_freqs)
            print_talkgroups_text(tgs)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
