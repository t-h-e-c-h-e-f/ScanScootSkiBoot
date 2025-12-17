# Quickstart

## 1) Prereqs

- Python 3.10+ (tested with 3.12)
- `pip` (or conda env with pip)

Install runtime deps:

```bash
python3 -m pip install fastapi uvicorn
```

## 2) Generate an API key

This repo ships with a default key in `keys.ini`:

- `YOU-REALLY-GOTTA-CHANGE-THIS`

Replace it immediately:

```bash
python3 generate_api_key.py --comment "local admin key"
```

The command prints the new key. Save it somewhere safe (password manager, secrets store, etc).

## 3) Start the server

Recommended:

```bash
uvicorn app:app --host 0.0.0.0 --port 16444
```

Or:

```bash
python3 app.py
```

Environment variables (optional):

- `HPDB_PATH` (default: `./hpdb_default.sqlite`)
- `API_KEYS_PATH` (default: `./keys.ini`)
- `UPLOAD_DIR` (default: `./uploads`)
- `ZIP_CSV_PATH` (default: `./uszips.csv`)

## 4) Initialize the database (first run)

### Browser (recommended)

- Open: `http://<host>:16444/`
- You’ll be redirected to `/initialize` if the DB doesn’t exist.
- Enter your API key and upload `MasterHpdb.hp1`.

![Initialization UI](init.png)

### cURL

```bash
curl \\
  -H "X-API-Key: <your key here>" \\
  -F "file=@MasterHpdb.hp1" \\
  "http://localhost:16444/hpdb/admin/upload-master"
```

Then poll:

```bash
curl "http://localhost:16444/hpdb/admin/jobs/<job_id>"
```

## 5) Query

State + county:

```bash
curl "http://localhost:16444/hpdb/query?state=MO&county=Nodaway"
```

ZIP (county mapping):

```bash
curl "http://localhost:16444/hpdb/query/by-zip?zip=64468"
```

ZIP + radius search:

```bash
curl "http://localhost:16444/hpdb/query/by-zip?zip=64468&radius_miles=25"
```

## 6) Update to a newer MasterHpdb

### Browser

- Open: `http://<host>:16444/update`
- Enter API key and upload the newer `MasterHpdb.hp1`
- When complete, the page shows an added/removed/changed summary for key tables.

![Update UI](update.png)

### cURL

```bash
curl \\
  -H "X-API-Key: <your key here>" \\
  -F "file=@Newest-MasterHpdb.hp1" \\
  "http://localhost:16444/hpdb/admin/update-master"
```

Then poll:

```bash
curl "http://localhost:16444/hpdb/admin/jobs/<job_id>"
```

## 7) Validate updates (offline)

Compare the active DB against a master snapshot:

```bash
python3 validate_hpdb_update.py --active-db hpdb_default.sqlite --master Newest-MasterHpdb.hp1
```

Compare old vs new master snapshots:

```bash
python3 validate_hpdb_update.py --active-db hpdb_default.sqlite --baseline-master MasterHpdb.hp1 --master Newest-MasterHpdb.hp1
```
