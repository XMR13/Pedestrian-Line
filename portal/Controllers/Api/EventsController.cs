using System.Text.Json;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Portal.Web.Contracts;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using Portal.Web.Models;

namespace Portal.Web.Controllers.Api;

[ApiController]
[Route("api/events")]
public sealed class EventsController(
    PortalDbContext db,
    IWebHostEnvironment env,
    IOptions<PortalOptions> portalOptions) : ControllerBase
{
    [HttpPost("upsert")]
    [ApiKeyAuthorize]
    public async Task<IActionResult> Upsert([FromBody] EventUpsertBatchRequest request, CancellationToken ct)
    {
        if (!IsContractValid(request.ContractVersion))
        {
            return BadRequest(new { error = "unsupported_contract_version" });
        }

        if (request.Events is null || request.Events.Count == 0)
        {
            return BadRequest(new { error = "empty_events" });
        }

        var now = DateTimeOffset.UtcNow;
        var normalized = request.Events
            .Where(x => !string.IsNullOrWhiteSpace(x.EventUid) && !string.IsNullOrWhiteSpace(x.RunUid))
            .ToList();

        if (normalized.Count == 0)
        {
            return BadRequest(new { error = "missing_required_fields", required = new[] { "event_uid", "run_uid" } });
        }

        var runIds = normalized.Select(x => x.RunUid!.Trim()).Distinct(StringComparer.Ordinal).ToArray();
        var eventIds = normalized.Select(x => x.EventUid!.Trim()).Distinct(StringComparer.Ordinal).ToArray();

        var runs = await db.Runs.Where(x => runIds.Contains(x.RunUid)).ToDictionaryAsync(x => x.RunUid, ct);
        var eventsById = await db.Events
            .Include(x => x.Review)
            .Where(x => eventIds.Contains(x.EventUid))
            .ToDictionaryAsync(x => x.EventUid, ct);

        var inserted = 0;
        var updated = 0;

        foreach (var req in normalized)
        {
            var runUid = req.RunUid!.Trim();
            var eventUid = req.EventUid!.Trim();
            var siteId = req.SiteId?.Trim() ?? string.Empty;
            var cameraId = req.CameraId?.Trim() ?? string.Empty;

            if (!runs.TryGetValue(runUid, out var run))
            {
                run = new RunRecord
                {
                    RunUid = runUid,
                    SiteId = siteId,
                    CameraId = cameraId,
                    UpdatedAtUtc = now,
                };
                runs[runUid] = run;
                db.Runs.Add(run);
            }
            else
            {
                if (!string.IsNullOrWhiteSpace(siteId))
                {
                    run.SiteId = siteId;
                }

                if (!string.IsNullOrWhiteSpace(cameraId))
                {
                    run.CameraId = cameraId;
                }

                run.UpdatedAtUtc = now;
            }

            if (!eventsById.TryGetValue(eventUid, out var row))
            {
                row = new EventRecord
                {
                    EventUid = eventUid,
                    RunUid = runUid,
                    SiteId = siteId,
                    CameraId = cameraId,
                    UpdatedAtUtc = now,
                };
                db.Events.Add(row);
                eventsById[eventUid] = row;
                inserted += 1;
            }
            else
            {
                updated += 1;
            }

            row.RunUid = runUid;
            row.SiteId = string.IsNullOrWhiteSpace(siteId) ? run.SiteId : siteId;
            row.CameraId = string.IsNullOrWhiteSpace(cameraId) ? run.CameraId : cameraId;
            row.OccurredAtUtc = req.OccurredAtUtc;
            row.FrameIndex = req.FrameIndex;
            row.VideoTimeS = req.VideoTimeS;
            row.Direction = NormalizeDirection(req.Direction);
            row.TrackId = req.TrackId;
            row.ClassId = req.ClassId;
            row.ClassName = EmptyToNull(req.ClassName);
            row.Confidence = req.Confidence;
            row.BboxJson = NormalizeBbox(req.BboxXyxy);
            row.LineMode = EmptyToNull(req.LineMode);
            row.OccurredAtUtcSource = EmptyToNull(req.OccurredAtUtcSource);
            row.UpdatedAtUtc = now;

            if (!string.IsNullOrWhiteSpace(req.ThumbRelpath) && string.IsNullOrWhiteSpace(row.ThumbPath))
            {
                row.ThumbPath = NormalizeStoragePath(req.ThumbRelpath!);
            }

            if (!string.IsNullOrWhiteSpace(req.SceneRelpath) && string.IsNullOrWhiteSpace(row.ScenePath))
            {
                row.ScenePath = NormalizeStoragePath(req.SceneRelpath!);
            }

            if (row.Review is null)
            {
                row.Review = new EventReview
                {
                    EventUid = row.EventUid,
                    ReviewStatus = ReviewStatuses.Pending,
                    UpdatedAtUtc = now,
                };
            }
        }

        await db.SaveChangesAsync(ct);

        return Ok(new
        {
            status = "ok",
            inserted,
            updated,
            total = normalized.Count,
        });
    }

    [HttpPost("{eventUid}/thumbnail")]
    [ApiKeyAuthorize]
    [RequestSizeLimit(20_000_000)]
    public async Task<IActionResult> UploadThumbnail(string eventUid, [FromForm] IFormFile file, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(eventUid))
        {
            return BadRequest(new { error = "missing_event_uid" });
        }

        var row = await db.Events.FirstOrDefaultAsync(x => x.EventUid == eventUid, ct);
        if (row is null)
        {
            return NotFound(new { error = "event_not_found" });
        }

        if (file is null || file.Length == 0)
        {
            return BadRequest(new { error = "empty_file" });
        }

        var kind = Request.Headers["X-Evidence-Kind"].ToString().Trim().ToLowerInvariant();
        if (kind is not ("scene" or "object"))
        {
            kind = "object";
        }

        var extension = Path.GetExtension(file.FileName);
        if (string.IsNullOrWhiteSpace(extension))
        {
            extension = ".jpg";
        }

        var root = EnsureEvidenceRoot();
        var relativePath = NormalizeStoragePath(Path.Combine(kind, eventUid + extension));
        var absolutePath = Path.Combine(root, relativePath.Replace('/', Path.DirectorySeparatorChar));

        var directory = Path.GetDirectoryName(absolutePath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var shouldWrite = true;
        if (System.IO.File.Exists(absolutePath))
        {
            var existingInfo = new FileInfo(absolutePath);
            if (existingInfo.Length == file.Length)
            {
                shouldWrite = false;
            }
        }

        if (shouldWrite)
        {
            await using var stream = System.IO.File.Create(absolutePath);
            await file.CopyToAsync(stream, ct);
        }

        if (kind == "scene")
        {
            row.ScenePath = relativePath;
        }
        else
        {
            row.ThumbPath = relativePath;
        }

        row.UpdatedAtUtc = DateTimeOffset.UtcNow;
        await db.SaveChangesAsync(ct);

        return Ok(new
        {
            status = "ok",
            event_uid = eventUid,
            path = relativePath,
            wrote = shouldWrite,
        });
    }

    [HttpPost("{eventUid}/review")]
    [Authorize]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> Review(string eventUid, [FromForm] EventReviewRequest request, CancellationToken ct)
    {
        var row = await db.Events.Include(x => x.Review).FirstOrDefaultAsync(x => x.EventUid == eventUid, ct);
        if (row is null)
        {
            return NotFound(new { error = "event_not_found" });
        }

        var status = ResolveReviewStatus(request);
        if (status is null)
        {
            return BadRequest(new { error = "invalid_review_status" });
        }

        var now = DateTimeOffset.UtcNow;
        row.Review ??= new EventReview
        {
            EventUid = row.EventUid,
            UpdatedAtUtc = now,
        };

        row.Review.ReviewStatus = status;
        row.Review.Notes = EmptyToNull(request.Notes);
        row.Review.UpdatedAtUtc = now;

        if (status == ReviewStatuses.Pending)
        {
            row.Review.ReviewedAtUtc = null;
            row.Review.ReviewedBy = null;
        }
        else
        {
            row.Review.ReviewedAtUtc = now;
            row.Review.ReviewedBy = User?.Identity?.Name ?? "reviewer";
        }

        await db.SaveChangesAsync(ct);

        if (Request.Headers.Accept.ToString().Contains("text/html", StringComparison.OrdinalIgnoreCase))
        {
            var returnUrl = Request.Form["returnUrl"].ToString();
            if (!string.IsNullOrWhiteSpace(returnUrl) && Url.IsLocalUrl(returnUrl))
            {
                return Redirect(returnUrl);
            }

            return RedirectToAction("ReviewQueue", "Events");
        }

        return Ok(new
        {
            status = "ok",
            event_uid = row.EventUid,
            review_status = row.Review.ReviewStatus,
        });
    }

    [HttpGet]
    [Authorize]
    public async Task<IActionResult> List([FromQuery] EventQueryRequest request, CancellationToken ct)
    {
        var options = portalOptions.Value;
        var pageSize = ClampPageSize(request.PageSize, options.DefaultPageSize, options.MaxPageSize);
        var page = Math.Max(request.Page, 1);
        var pageOffset = (page - 1) * pageSize;
        var isSqlServer = db.Database.IsSqlServer();

        var baseQuery = db.Events.PortalQueryable()
            .ApplyEventFilters(request, includeDateFilter: isSqlServer);

        List<EventListRow> rows;
        int total;
        if (isSqlServer)
        {
            total = await baseQuery.CountAsync(ct);
            rows = await baseQuery
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Skip(pageOffset)
                .Take(pageSize)
                .Select(x => new EventListRow
                {
                    EventUid = x.EventUid,
                    RunUid = x.RunUid,
                    SiteId = x.SiteId,
                    CameraId = x.CameraId,
                    OccurredAtUtc = x.OccurredAtUtc,
                    Direction = x.Direction,
                    ClassName = x.ClassName,
                    TrackId = x.TrackId,
                    ThumbPath = x.ThumbPath,
                    ScenePath = x.ScenePath,
                    ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                    Notes = x.Review != null ? x.Review.Notes : null,
                })
                .ToListAsync(ct);
        }
        else
        {
            rows = (await baseQuery
                .Select(x => new EventListRow
                {
                    EventUid = x.EventUid,
                    RunUid = x.RunUid,
                    SiteId = x.SiteId,
                    CameraId = x.CameraId,
                    OccurredAtUtc = x.OccurredAtUtc,
                    Direction = x.Direction,
                    ClassName = x.ClassName,
                    TrackId = x.TrackId,
                    ThumbPath = x.ThumbPath,
                    ScenePath = x.ScenePath,
                    ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                    Notes = x.Review != null ? x.Review.Notes : null,
                })
                .ToListAsync(ct))
                .ApplyLocalDateRangeInMemory(request, x => x.OccurredAtUtc)
                .ToList();

            total = rows.Count;
            rows = rows
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Skip(pageOffset)
                .Take(pageSize)
                .ToList();
        }

        return Ok(new
        {
            page,
            page_size = pageSize,
            total,
            items = rows.Select(x => new
            {
                event_uid = x.EventUid,
                run_uid = x.RunUid,
                site_id = x.SiteId,
                camera_id = x.CameraId,
                occurred_at_utc = x.OccurredAtUtc,
                direction = x.Direction,
                class_name = x.ClassName,
                track_id = x.TrackId,
                review_status = x.ReviewStatus,
                notes = x.Notes,
                thumbnail_url = !string.IsNullOrWhiteSpace(x.ScenePath)
                    ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                    : (string.IsNullOrWhiteSpace(x.ThumbPath) ? null : $"/api/events/{x.EventUid}/thumbnail"),
            }),
        });
    }

    [HttpGet("{eventUid}/thumbnail")]
    [Authorize]
    public async Task<IActionResult> Thumbnail(string eventUid, [FromQuery] string? kind, CancellationToken ct)
    {
        var row = await db.Events
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.EventUid == eventUid, ct);

        if (row is null)
        {
            return NotFound();
        }

        var root = EnsureEvidenceRoot();
        var wantScene = string.Equals((kind ?? string.Empty).Trim(), "scene", StringComparison.OrdinalIgnoreCase);
        var candidates = wantScene
            ? new[] { row.ScenePath, row.ThumbPath }
            : new[] { row.ThumbPath, row.ScenePath };

        foreach (var rel in candidates)
        {
            if (string.IsNullOrWhiteSpace(rel))
            {
                continue;
            }

            var absolutePath = Path.Combine(root, rel.Replace('/', Path.DirectorySeparatorChar));
            if (System.IO.File.Exists(absolutePath))
            {
                return PhysicalFile(absolutePath, "image/jpeg", enableRangeProcessing: true);
            }
        }

        return NotFound();
    }

    private string EnsureEvidenceRoot()
    {
        var configured = portalOptions.Value.EvidenceRootPath?.Trim();
        var path = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(env.ContentRootPath, "evidence")
            : configured;

        if (!Path.IsPathRooted(path))
        {
            path = Path.Combine(env.ContentRootPath, path);
        }

        Directory.CreateDirectory(path);
        return path;
    }

    private static List<T> OrderByEventUids<T>(IEnumerable<T> rows, IReadOnlyList<string> orderedEventUids, Func<T, string> getEventUid)
    {
        var rowMap = rows.ToDictionary(getEventUid, StringComparer.Ordinal);
        var ordered = new List<T>(orderedEventUids.Count);
        foreach (var eventUid in orderedEventUids)
        {
            if (rowMap.TryGetValue(eventUid, out var row))
            {
                ordered.Add(row);
            }
        }

        return ordered;
    }

    private sealed class EventListRow
    {
        public string EventUid { get; set; } = string.Empty;
        public string RunUid { get; set; } = string.Empty;
        public string SiteId { get; set; } = string.Empty;
        public string CameraId { get; set; } = string.Empty;
        public DateTimeOffset? OccurredAtUtc { get; set; }
        public string? Direction { get; set; }
        public string? ClassName { get; set; }
        public int? TrackId { get; set; }
        public string? ThumbPath { get; set; }
        public string? ScenePath { get; set; }
        public string ReviewStatus { get; set; } = ReviewStatuses.Pending;
        public string? Notes { get; set; }
    }

    private static int ClampPageSize(int pageSize, int defaultValue, int maxValue)
    {
        var fallback = defaultValue > 0 ? defaultValue : 50;
        var cap = maxValue > 0 ? maxValue : 200;
        var value = pageSize > 0 ? pageSize : fallback;
        return Math.Min(Math.Max(1, value), cap);
    }

    private static string? ResolveReviewStatus(EventReviewRequest request)
    {
        if (!string.IsNullOrWhiteSpace(request.ReviewStatus))
        {
            var normalized = request.ReviewStatus.Trim().ToUpperInvariant();
            if (ReviewStatuses.All.Contains(normalized))
            {
                return normalized;
            }

            return null;
        }

        if (!request.Qualified.HasValue)
        {
            return ReviewStatuses.Pending;
        }

        return request.Qualified.Value
            ? ReviewStatuses.Qualified
            : ReviewStatuses.NotQualified;
    }

    private static bool IsContractValid(string? contractVersion)
    {
        var value = (contractVersion ?? string.Empty).Trim();
        return string.Equals(value, "v1", StringComparison.OrdinalIgnoreCase);
    }

    private static string? NormalizeDirection(string? value)
    {
        var normalized = EmptyToNull(value)?.ToUpperInvariant();
        return normalized;
    }

    private static string? EmptyToNull(string? value)
    {
        var trimmed = value?.Trim();
        return string.IsNullOrWhiteSpace(trimmed) ? null : trimmed;
    }

    private static string? NormalizeBbox(List<int>? bbox)
    {
        if (bbox is null || bbox.Count != 4)
        {
            return null;
        }

        return JsonSerializer.Serialize(new[] { bbox[0], bbox[1], bbox[2], bbox[3] });
    }

    private static string NormalizeStoragePath(string relativePath)
    {
        var normalized = relativePath.Replace('\\', '/').TrimStart('/').Trim();
        return normalized;
    }
}
