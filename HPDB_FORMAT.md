# HPDB folder: what the files are, and how to convert to SQLite

This directory contains a HomePatrol-style Uniden scanner database in *plain text*.
All files are tab-separated (`\t`) with CRLF line endings.

## What each file is

- `hpdb.cfg`
  - Contains **reference / support tables** used by the scanner for location + geography:
    - `StateInfo` (states/provinces)
    - `CountyInfo` (counties/regions within a state)
    - `LM` and `LM_Frequency` (location mapping helpers; see below)
  - Also includes header metadata lines: `TargetModel`, `FormatVersion`, `DateModified`.

- `MasterHpdb.hp1`
  - Appears to be the **full, flattened database snapshot**: conventional channels, trunked systems, talkgroups, plus the same reference tables (`StateInfo`, `CountyInfo`, `LM`, `LM_Frequency`).
  - Most “real” radio data is here: `Conventional`, `C-Group`, `C-Freq`, `Trunk`, `Site`, `T-Group`, `TGID`, `T-Freq`, etc.

- `s_*.hpd`
  - These files contain **shards/subsets** of the same database records that also appear in `MasterHpdb.hp1` (matching IDs like `CFreqId=...`, `SiteId=...`, `Tid=...`).
  - The main difference is formatting: the `s_*.hpd` lines are typically **padded with extra trailing tabs** (reserved/empty fields).
  - These are *not “diffs”* in this dataset; they look like a way to distribute/load the database in chunks.

## Line format (how to read it)

Each line begins with a **record type**, then a mixture of:

1. `key=value` tokens (IDs and optional attributes)
2. positional fields (strings/numbers in a fixed order for that record type)

Example (conventional frequency):

```
C-Freq   CFreqId=24589   CGroupId=6206   Fire: Dispatch - East/West   Off   154355000   NFM   TONE=C100.0   3
```

## Record types (what they represent)

This is the set of record types observed in `MasterHpdb.hp1`, with short descriptions:

- **Metadata**
  - `TargetModel`: which scanner model this export targets (here: `BCDx36HP`)
  - `FormatVersion`: file format version (here: `1.00`)
  - `DateModified`: build time for the snapshot
  - `File`: a label string (e.g., `HomePatrol Database`)

- **Geography / reference**
  - `StateInfo`: `StateId`, `CountryId`, plus state/province name and abbreviation.
  - `CountyInfo`: `CountyId`, `StateId`, plus county/region name.

- **Conventional (non-trunked)**
  - `Conventional`: top-level container for a county/agency’s conventional channels.
  - `C-Group`: a “department/group” inside a conventional container; includes name + location (lat/lon/range/shape).
  - `C-Freq`: an actual conventional channel entry (frequency + modulation, optional tone/NAC/color code/RAN, etc).

- **Trunked systems**
  - `Trunk`: a trunked radio system (has `TrunkId`).
  - `Site`: a site within a trunked system (`SiteId`, `TrunkId`), with location fields.
  - `T-Group`: a talkgroup container/category for a trunked system (`TGroupId`, `TrunkId`), with location fields.
  - `TGID`: a specific talkgroup (`Tid`, `TGroupId`) with alpha tag and other attributes.
  - `T-Freq`: site frequencies (`SiteId`, optional `RAN`, etc). The exact meaning of the positional integers varies by system type.
  - `BandPlan_Mot`, `BandPlan_P25`, `FleetMap`: system/site decoding helpers (stored as numeric arrays in these files).

- **Location / coverage helpers**
- `AreaState`, `AreaCounty`: mapping records that associate a system/group with a state/county/agency.
- `Rectangle`: bounding box entries (often used when the “shape” is `Rectangles`).
- `LM`: links a `TrunkId`+`SiteId` to a state/county (used to determine which trunk sites belong to which county).
- `LM_Frequency`: a frequency list used by the location mapping logic (exact semantics not decoded here).

## Converting to SQLite

Use `hpdb_to_sqlite.py` (added in this repo). By default it loads `hpdb.cfg` and `MasterHpdb.hp1`.

Create a DB:

```
python3 hpdb_to_sqlite.py --input . --out hpdb.sqlite --overwrite
```

If you *also* want to ingest `s_*.hpd` (often redundant shards), add:

```
python3 hpdb_to_sqlite.py --include-s-files --out hpdb.sqlite --overwrite
```

### Reducing SQLite size

SQLite does not compress data by default. The converter’s default output is larger than `MasterHpdb.hp1` mainly because:

- it stores the original text lines *again* in a lossless `records` table (`raw_line` plus JSON tokenizations)
- it adds indexes to keep queries fast

This converter now defaults to a compact DB:

- `VACUUM` is run at the end
- the lossless `records` table is *not* stored

To keep the DB small explicitly (equivalent to defaults):

```
python3 hpdb_to_sqlite.py --out hpdb.sqlite --overwrite
```

To force keeping lossless line-by-line storage (larger DB):

```
python3 hpdb_to_sqlite.py --out hpdb.sqlite --overwrite --lossless-records
```

To skip the final compaction pass (faster build, larger DB):

```
python3 hpdb_to_sqlite.py --out hpdb.sqlite --overwrite --no-vacuum
```

### Output tables

The SQLite DB contains:

- `records`: **lossless** storage of every parsed line (`record_type`, `kv_json`, `fields_json`, `raw_line`). This table is omitted if you use `--no-lossless-records`.
- Typed tables for common record types:
  - `state_info`, `county_info`
  - `conventional`, `c_group`, `c_freq`
  - `trunk`, `site`, `t_group`, `tgid`, `t_freq`
  - `area_state`, `area_county`, `rectangle`, `lm`, `lm_frequency`
  - `bandplan_mot`, `bandplan_p25`, `fleetmap`

Some positional fields are stored as `extras_json` when their meaning is unclear or varies by system type.

## Querying by state + county

The converter also creates a few convenience SQLite views:

- `v_counties`: `(StateId, StateAbbrev, StateName, CountyId, CountyName)`
- `v_conventional_freqs_by_county`: conventional channels (joins `c_group` + `c_freq`)
- `v_trunk_systems_by_county`: trunk systems mapped to a county (via `area_county`)
- `v_trunk_talkgroups_by_county`: talkgroups for trunk systems mapped to a county
- `v_trunk_sites_and_freqs_by_county`: site frequencies for trunk systems mapped to a county (uses `LM` to pick sites in that county)

There’s also a small helper script to run the most common query pattern:

```
python3 query_hpdb.py --db hpdb.sqlite --state AL --county Autauga
```

This prints conventional frequencies, trunk talkgroups, and trunk site frequencies (from `T-Freq`).
Use `--limit-trunk-freqs` to cap the trunk frequency output (default: 500), or `--no-trunk-freqs` to disable it.

Note: some trunk systems (especially statewide ones like MOSWIN) are not linked to specific counties via `AreaCounty`.
For those, the views treat `AreaCounty CountyId=0` plus `AreaState StateId=<state>` as “available in the state”, and
`LM` is used to decide which *sites* (towers) are in a specific county.

For machine-readable output:

```
python3 query_hpdb.py --db hpdb.sqlite --state AL --county Autauga --format json
```
