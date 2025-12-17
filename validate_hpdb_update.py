#!/usr/bin/env python3
"""
Validate HPDB update imports by comparing SQLite databases.

Typical workflow:
  1) You have an existing active DB (hpdb_default.sqlite).
  2) You have a new MasterHpdb.hp1 (e.g. Newest-MasterHpdb.hp1).
  3) Build a fresh "expected" SQLite DB from that master and compare it to active.

This script compares stable-key tables (those with real primary keys / natural keys).
It reports:
  - row counts in each DB
  - keys missing in active (should be 0 after a successful update)
  - keys extra in active (indicates deletions not applied)
  - rows with differing non-key columns

Notes:
  - We ignore `last_source_file_id` because it can differ between DBs and does not
    reflect substantive record content.
  - Tables with synthetic surrogate PKs (e.g. `conventional`, `rectangle`) are not
    compared row-by-row here because the keys are not stable across rebuilds.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TABLE_KEYS: Dict[str, Sequence[str]] = {
    "state_info": ("StateId",),
    "county_info": ("CountyId",),
    "trunk": ("TrunkId",),
    "site": ("SiteId",),
    "t_group": ("TGroupId",),
    "tgid": ("Tid",),
    "t_freq": ("TFreqId", "SiteId", "frequency_hz"),
    "c_group": ("CGroupId",),
    "c_freq": ("CFreqId",),
    "bandplan_mot": ("SiteId",),
    "bandplan_p25": ("SiteId",),
    "fleetmap": ("TrunkId",),
    # These have stable unique constraints in our schema, but not stable integer PKs.
    "lm": ("StateId", "CountyId", "TrunkId", "SiteId", "unk0", "unk1"),
    "lm_frequency": ("frequency_hz", "code"),
    # Mapping tables with synthetic ids; compare by natural key.
    "area_state": ("StateId", "CountyId", "AgencyId", "TrunkId"),
    "area_county": ("CountyId", "AgencyId", "TrunkId"),
}


# Columns that are expected to differ across rebuilds/merges and are not "content".
IGNORED_COLUMNS = {
    "last_source_file_id",
    # Synthetic PKs for tables where we compare via a natural key instead.
    "lm_id",
    "lm_frequency_id",
    "area_state_id",
    "area_county_id",
}


@dataclass(frozen=True)
class TableDiff:
    table: str
    active_count: int
    expected_count: int
    missing_in_active: int
    extra_in_active: int
    differing_rows: int


@dataclass(frozen=True)
class TableDelta:
    table: str
    old_count: int
    new_count: int
    added: int
    removed: int
    changed: int


def build_sqlite_from_master(*, master_hp1: Path, hpdb_cfg: Optional[Path], out_db: Path) -> None:
    import hpdb_to_sqlite

    with tempfile.TemporaryDirectory(prefix="hpdb_validate_") as tmpdir:
        tmpdir_p = Path(tmpdir)
        (tmpdir_p / "MasterHpdb.hp1").write_bytes(master_hp1.read_bytes())
        if hpdb_cfg and hpdb_cfg.exists():
            (tmpdir_p / "hpdb.cfg").write_bytes(hpdb_cfg.read_bytes())
        hpdb_to_sqlite.main(["--input", str(tmpdir_p), "--out", str(out_db), "--overwrite"])


def q_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_table_columns(conn: sqlite3.Connection, db_alias: str, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA {db_alias}.table_info({q_ident(table)})").fetchall()
    return [r[1] for r in rows]  # name


def sql_join_on(keys: Sequence[str], left: str, right: str) -> str:
    return " AND ".join([f"{left}.{q_ident(k)} IS {right}.{q_ident(k)}" for k in keys])


def sql_key_null_check(keys: Sequence[str], alias: str) -> str:
    # For "missing" checks we look at the first key column being NULL on the joined side.
    return f"{alias}.{q_ident(keys[0])} IS NULL"


def sql_diff_where(keys: Sequence[str], cols: Sequence[str], left: str, right: str) -> str:
    parts = []
    for c in cols:
        if c in keys or c in IGNORED_COLUMNS:
            continue
        parts.append(f"{left}.{q_ident(c)} IS NOT {right}.{q_ident(c)}")
    if not parts:
        return "0"  # no diffable columns
    return "(" + " OR ".join(parts) + ")"


def compute_table_diff(conn: sqlite3.Connection, table: str, keys: Sequence[str]) -> TableDiff:
    # ensure table exists in both
    active_cols = list_table_columns(conn, "active", table)
    exp_cols = list_table_columns(conn, "exp", table)
    if not active_cols or not exp_cols:
        raise RuntimeError(f"Table missing in one DB: {table}")

    # Only compare common columns
    common_cols = [c for c in active_cols if c in set(exp_cols)]

    active_count = conn.execute(f"SELECT COUNT(*) FROM active.{q_ident(table)}").fetchone()[0]
    expected_count = conn.execute(f"SELECT COUNT(*) FROM exp.{q_ident(table)}").fetchone()[0]

    join_on = sql_join_on(keys, "a", "e")
    missing = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM exp.{q_ident(table)} e
        LEFT JOIN active.{q_ident(table)} a
          ON {join_on}
        WHERE {sql_key_null_check(keys, "a")}
        """
    ).fetchone()[0]

    extra = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM active.{q_ident(table)} a
        LEFT JOIN exp.{q_ident(table)} e
          ON {join_on}
        WHERE {sql_key_null_check(keys, "e")}
        """
    ).fetchone()[0]

    diff_where = sql_diff_where(keys, common_cols, "a", "e")
    differing = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM active.{q_ident(table)} a
        JOIN exp.{q_ident(table)} e
          ON {join_on}
        WHERE {diff_where}
        """
    ).fetchone()[0]

    return TableDiff(
        table=table,
        active_count=int(active_count),
        expected_count=int(expected_count),
        missing_in_active=int(missing),
        extra_in_active=int(extra),
        differing_rows=int(differing),
    )


def compute_table_delta(
    conn: sqlite3.Connection,
    table: str,
    keys: Sequence[str],
    *,
    old_alias: str,
    new_alias: str,
) -> TableDelta:
    old_cols = list_table_columns(conn, old_alias, table)
    new_cols = list_table_columns(conn, new_alias, table)
    if not old_cols or not new_cols:
        raise RuntimeError(f"Table missing in one DB: {table}")

    common_cols = [c for c in old_cols if c in set(new_cols)]

    old_count = conn.execute(f"SELECT COUNT(*) FROM {old_alias}.{q_ident(table)}").fetchone()[0]
    new_count = conn.execute(f"SELECT COUNT(*) FROM {new_alias}.{q_ident(table)}").fetchone()[0]

    join_on = sql_join_on(keys, "o", "n")

    added = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {new_alias}.{q_ident(table)} n
        LEFT JOIN {old_alias}.{q_ident(table)} o
          ON {join_on}
        WHERE {sql_key_null_check(keys, "o")}
        """
    ).fetchone()[0]

    removed = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {old_alias}.{q_ident(table)} o
        LEFT JOIN {new_alias}.{q_ident(table)} n
          ON {join_on}
        WHERE {sql_key_null_check(keys, "n")}
        """
    ).fetchone()[0]

    diff_where = sql_diff_where(keys, common_cols, "o", "n")
    changed = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {old_alias}.{q_ident(table)} o
        JOIN {new_alias}.{q_ident(table)} n
          ON {join_on}
        WHERE {diff_where}
        """
    ).fetchone()[0]

    return TableDelta(
        table=table,
        old_count=int(old_count),
        new_count=int(new_count),
        added=int(added),
        removed=int(removed),
        changed=int(changed),
    )


def print_table(diffs: List[TableDiff]) -> None:
    headers = ["table", "active", "expected", "missing", "extra", "diff"]
    rows = [
        [d.table, str(d.active_count), str(d.expected_count), str(d.missing_in_active), str(d.extra_in_active), str(d.differing_rows)]
        for d in diffs
    ]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))
    fmt = "  ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def print_delta(deltas: List[TableDelta]) -> None:
    headers = ["table", "old", "new", "added", "removed", "changed"]
    rows = [[d.table, str(d.old_count), str(d.new_count), str(d.added), str(d.removed), str(d.changed)] for d in deltas]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))
    fmt = "  ".join("{:" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate hpdb_default.sqlite against a MasterHpdb.hp1-derived expected DB.")
    parser.add_argument("--active-db", default="hpdb_default.sqlite", help="Path to active SQLite DB to validate")
    parser.add_argument("--master", required=True, help="Path to MasterHpdb.hp1 to treat as expected truth")
    parser.add_argument("--baseline-master", help="Optional old MasterHpdb.hp1 to compare against --master")
    parser.add_argument("--hpdb-cfg", default="hpdb.cfg", help="Optional hpdb.cfg (helps include reference tables)")
    parser.add_argument("--tables", nargs="*", default=list(DEFAULT_TABLE_KEYS.keys()), help="Tables to compare")
    parser.add_argument("--keep", action="store_true", help="Keep the generated expected DB next to the master file")
    args = parser.parse_args(list(argv) if argv is not None else None)

    active_db = Path(args.active_db)
    master = Path(args.master)
    baseline_master = Path(args.baseline_master) if args.baseline_master else None
    cfg = Path(args.hpdb_cfg) if args.hpdb_cfg else None
    if not active_db.exists():
        raise SystemExit(f"Active DB not found: {active_db}")
    if not master.exists():
        raise SystemExit(f"Master file not found: {master}")

    out_expected = Path(tempfile.mkstemp(prefix="hpdb_expected_", suffix=".sqlite")[1])
    out_baseline = Path(tempfile.mkstemp(prefix="hpdb_baseline_", suffix=".sqlite")[1]) if baseline_master else None
    try:
        build_sqlite_from_master(master_hp1=master, hpdb_cfg=cfg if cfg and cfg.exists() else None, out_db=out_expected)
        if baseline_master:
            if not baseline_master.exists():
                raise SystemExit(f"Baseline master not found: {baseline_master}")
            build_sqlite_from_master(
                master_hp1=baseline_master,
                hpdb_cfg=cfg if cfg and cfg.exists() else None,
                out_db=out_baseline,  # type: ignore[arg-type]
            )

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("ATTACH DATABASE ? AS active", (str(active_db),))
            conn.execute("ATTACH DATABASE ? AS exp", (str(out_expected),))
            if out_baseline:
                conn.execute("ATTACH DATABASE ? AS base", (str(out_baseline),))

            diffs: List[TableDiff] = []
            for t in args.tables:
                keys = DEFAULT_TABLE_KEYS.get(t)
                if not keys:
                    continue
                diffs.append(compute_table_diff(conn, t, keys))

            print_table(diffs)

            if out_baseline:
                print()
                print("Baseline vs new master (added/removed/changed):")
                deltas: List[TableDelta] = []
                for t in args.tables:
                    keys = DEFAULT_TABLE_KEYS.get(t)
                    if not keys:
                        continue
                    deltas.append(compute_table_delta(conn, t, keys, old_alias="base", new_alias="exp"))
                print_delta(deltas)

            # Exit non-zero if we detect missing rows or differing rows.
            missing_total = sum(d.missing_in_active for d in diffs)
            differing_total = sum(d.differing_rows for d in diffs)
            if missing_total > 0 or differing_total > 0:
                print()
                print(f"Summary: missing_in_active={missing_total}, differing_rows={differing_total}")
                print("If extra_in_active > 0, it usually means deletions are not applied during updates.")
                return 2
            return 0
        finally:
            conn.close()
    finally:
        if args.keep:
            kept = master.with_suffix(".expected.sqlite")
            shutil.move(str(out_expected), str(kept))
            print(f"Kept expected DB: {kept}")
        else:
            try:
                os.remove(out_expected)
            except Exception:
                pass
        if out_baseline:
            try:
                os.remove(out_baseline)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
