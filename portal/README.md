# Portal Website MVP (Phase 7.3)

This folder contains the portal website for Phase 7.3.

Default local mode uses **SQLite** (no separate DB server needed). Production
target remains **ASP.NET Core + SQL Server**.

It consumes edge uploader payloads from:

- `POST /api/runs/upsert`
- `POST /api/events/upsert`
- `POST /api/events/{event_uid}/thumbnail`

Then provides:

- Dashboard summary
- Event browser with filters
- Review queue (`Qualified Yes/No` + notes + keyboard shortcuts)
- CSV export of reviewed events
- Minimal login gate (cookie auth)

## Stack

- ASP.NET Core MVC (`net8.0`)
- EF Core (`Sqlite` for local dev, `SqlServer` for deployment)
- Cookie authentication (MVP login gate)

## Configuration

Edit `appsettings.json`:

- `Database:Provider`: `Sqlite` (local) or `SqlServer` (deployment)
- `ConnectionStrings:PortalDb`: provider-specific connection string
- `Portal:ApiKey`: API key expected on uploader endpoints (`X-API-Key`)
- `Portal:EvidenceRootPath`: disk folder for thumbnails (relative or absolute)
- `LoginGate:Username/Password/DisplayName`: MVP login credentials

## Run (Local SQLite, recommended)

Use **Windows PowerShell** because the project path is on Windows drive.

### 1) Start portal

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
dotnet restore
dotnet build
dotnet run
```

Or use helper script (background + log files):

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
.\scripts\start-portal.ps1 -Port 5000
```

Expected startup line:

```text
Now listening on: http://localhost:5000
```

Login page:

- `http://localhost:5000/Account/Login`

Default credentials:

- username: `admin`
- password: `admin123`

With default `Sqlite` config, no SQL Server setup is required.

### 2) Stop/close portal

- In the same terminal: press `Ctrl + C`.
- If terminal was closed and process is still running:

```powershell
$pid = (Get-NetTCPConnection -LocalPort 5000 -State Listen).OwningProcess
Stop-Process -Id $pid -Force
```

Or use helper script:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
.\scripts\stop-portal.ps1 -Port 5000 -Force
```

### 3) Run on a different port (optional)

```powershell
dotnet run --urls "http://localhost:5001"
```

If port changes, update uploader URL (`--api-base-url`) to same port.

### 4) Logs

By default, logs are shown in terminal only.

To save logs into file:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
New-Item -ItemType Directory -Force logs | Out-Null
dotnet run *>&1 | Tee-Object -FilePath ".\logs\portal-$(Get-Date -Format yyyyMMdd-HHmmss).log"
```

Log folder:

- `portal/logs/`
- Helper script output:
  - `portal/logs/portal-<timestamp>-stdout.log`
  - `portal/logs/portal-<timestamp>-stderr.log`

### 5) Common startup issue

Error:

```text
Failed to bind to address ... address already in use
```

Fix:

- Stop process using that port (command above), or run with another port.

### 6) Local files created by portal

- SQLite DB: `portal/portal.db`
- SQLite WAL files: `portal/portal.db-shm`, `portal/portal.db-wal`
- Uploaded evidence thumbnails: `portal/evidence/`
- Optional saved logs: `portal/logs/`

## Database setup

### Local (SQLite, default)

No manual setup. `dotnet run` creates `portal/portal.db` automatically.

### SQL Server mode (deployment/optional local)

Set `appsettings.json`:

```json
"Database": { "Provider": "SqlServer" },
"ConnectionStrings": {
  "PortalDb": "Server=.\\SQLEXPRESS;Database=PedestrianLinePortal;Trusted_Connection=True;TrustServerCertificate=True;Encrypt=True"
}
```

Then choose one setup path.

Preferred with migrations:

```bash
cd portal
dotnet ef migrations add InitialPortal
dotnet ef database update
```

If migrations tooling is not available yet, apply:

- `portal/sql/001_init.sql`

against your SQL Server database first, then run the app.

### Quick SQL Server setup on Windows (Express)

Run in **Administrator PowerShell**:

```powershell
winget install -e --id Microsoft.SQLServer.2022.Express --accept-package-agreements --accept-source-agreements
winget install -e --id Microsoft.Sqlcmd --accept-package-agreements --accept-source-agreements
```

Then create schema:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
sqlcmd -S ".\SQLEXPRESS" -E -d PedestrianLinePortal -i .\sql\001_init.sql
```

If your SQL instance is different, update both:

- `sqlcmd -S <server>`
- `ConnectionStrings:PortalDb` in `portal/appsettings.json`

## Uploader integration

Use existing edge uploader:

```bash
python3 -m pedestrian_line_counter.portal_uploader \
  --spool-dir /path/to/spool \
  --api-base-url http://localhost:5000 \
  --api-key change-this-api-key
```

Set uploader API key to match `Portal:ApiKey`.

### Uploader watch mode

```bash
python3 -m pedestrian_line_counter.portal_uploader \
  --spool-dir /path/to/spool \
  --api-base-url http://localhost:5000 \
  --api-key change-this-api-key \
  --watch \
  --poll-interval-s 10
```

Per-run uploader state file:

- `<run_dir>/.portal_upload_state.json`

## Tests

Focused Phase 7.3 portal integration tests live in:

- `portal/tests/Portal.Web.Tests/`

Run:

```powershell
cd "D:\RZQ\Coding\Python\Projects\Pedestrian Line\portal"
dotnet test .\tests\Portal.Web.Tests\Portal.Web.Tests.csproj
```

## Route map

Pages:

- `/Account/Login`
- `/` (dashboard)
- `/Events`
- `/Events/ReviewQueue`
- `/Events/Detail/{eventUid}`
- `/Events/ExportCsv`

API:

- `POST /api/runs/upsert` (`X-API-Key` required)
- `POST /api/events/upsert` (`X-API-Key` required)
- `POST /api/events/{event_uid}/thumbnail` (`X-API-Key` required)
- `POST /api/events/{event_uid}/review` (authenticated reviewer)
- `GET /api/events` (authenticated reviewer)
- `GET /api/events/{event_uid}/thumbnail` (authenticated reviewer)
- `GET /api/dashboard/summary` (authenticated reviewer)
