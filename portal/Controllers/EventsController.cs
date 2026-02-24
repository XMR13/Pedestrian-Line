using System.Text;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Portal.Web.Contracts;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using Portal.Web.Models;
using Portal.Web.ViewModels;

namespace Portal.Web.Controllers;

[Authorize]
public sealed class EventsController(
    PortalDbContext db,
    IOptions<PortalOptions> portalOptions) : Controller
{
    [HttpGet]
    public async Task<IActionResult> Index([FromQuery] EventQueryRequest request, CancellationToken ct)
    {
        var options = portalOptions.Value;
        var pageSize = ClampPageSize(request.PageSize, options.DefaultPageSize, options.MaxPageSize);
        var page = Math.Max(request.Page, 1);
        var pageOffset = (page - 1) * pageSize;
        var isSqlServer = db.Database.IsSqlServer();

        var filtered = db.Events.PortalQueryable().ApplyEventFilters(request);
        var total = await filtered.CountAsync(ct);

        List<EventListItemViewModel> items;
        if (isSqlServer)
        {
            items = await filtered
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Skip(pageOffset)
                .Take(pageSize)
                .Select(x => new EventListItemViewModel
                {
                    EventUid = x.EventUid,
                    RunUid = x.RunUid,
                    SiteId = x.SiteId,
                    CameraId = x.CameraId,
                    OccurredAtUtc = x.OccurredAtUtc,
                    Direction = x.Direction ?? "-",
                    ClassName = x.ClassName ?? "unknown",
                    ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                    Notes = x.Review != null ? x.Review.Notes : null,
                    ThumbnailUrl = x.ScenePath != null
                        ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                        : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                })
                .ToListAsync(ct);
        }
        else
        {
            var orderedEventUids = (await filtered
                .Select(x => new
                {
                    x.EventUid,
                    x.OccurredAtUtc,
                })
                .ToListAsync(ct))
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Skip(pageOffset)
                .Take(pageSize)
                .Select(x => x.EventUid)
                .ToList();

            if (orderedEventUids.Count == 0)
            {
                items = [];
            }
            else
            {
                var rows = await filtered
                    .Where(x => orderedEventUids.Contains(x.EventUid))
                    .Select(x => new EventListItemViewModel
                    {
                        EventUid = x.EventUid,
                        RunUid = x.RunUid,
                        SiteId = x.SiteId,
                        CameraId = x.CameraId,
                        OccurredAtUtc = x.OccurredAtUtc,
                        Direction = x.Direction ?? "-",
                        ClassName = x.ClassName ?? "unknown",
                        ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                        Notes = x.Review != null ? x.Review.Notes : null,
                        ThumbnailUrl = x.ScenePath != null
                            ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                            : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                    })
                    .ToListAsync(ct);

                items = OrderByEventUids(rows, orderedEventUids, x => x.EventUid);
            }
        }

        var vm = new EventListPageViewModel
        {
            SiteId = request.SiteId,
            CameraId = request.CameraId,
            Date = request.Date,
            Direction = request.Direction,
            ClassName = request.ClassName,
            ReviewStatus = request.ReviewStatus,
            Page = page,
            PageSize = pageSize,
            Total = total,
            TotalPages = total == 0 ? 1 : (int)Math.Ceiling(total / (double)pageSize),
            Items = items,
        };

        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> ReviewQueue(CancellationToken ct)
    {
        var pendingBaseQuery = db.Events.PortalQueryable()
            .Where(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending)
            .AsQueryable();

        var isSqlServer = db.Database.IsSqlServer();
        List<EventDetailViewModel> pendingRows;
        if (isSqlServer)
        {
            pendingRows = await pendingBaseQuery
                .OrderBy(x => x.OccurredAtUtc ?? DateTimeOffset.MaxValue)
                .ThenBy(x => x.EventUid)
                .Take(7)
                .Select(x => new EventDetailViewModel
                {
                    EventUid = x.EventUid,
                    RunUid = x.RunUid,
                    SiteId = x.SiteId,
                    CameraId = x.CameraId,
                    OccurredAtUtc = x.OccurredAtUtc,
                    FrameIndex = x.FrameIndex,
                    VideoTimeS = x.VideoTimeS,
                    Direction = x.Direction,
                    TrackId = x.TrackId,
                    ClassName = x.ClassName,
                    Confidence = x.Confidence,
                    ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                    Notes = x.Review != null ? x.Review.Notes : null,
                    ThumbnailUrl = x.ScenePath != null
                        ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                        : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                })
                .ToListAsync(ct);
        }
        else
        {
            var orderedPendingUids = (await pendingBaseQuery
                .Select(x => new
                {
                    x.EventUid,
                    x.OccurredAtUtc,
                })
                .ToListAsync(ct))
                .OrderBy(x => x.OccurredAtUtc ?? DateTimeOffset.MaxValue)
                .ThenBy(x => x.EventUid)
                .Take(7)
                .Select(x => x.EventUid)
                .ToList();

            if (orderedPendingUids.Count == 0)
            {
                pendingRows = [];
            }
            else
            {
                var rows = await pendingBaseQuery
                    .Where(x => orderedPendingUids.Contains(x.EventUid))
                    .Select(x => new EventDetailViewModel
                    {
                        EventUid = x.EventUid,
                        RunUid = x.RunUid,
                        SiteId = x.SiteId,
                        CameraId = x.CameraId,
                        OccurredAtUtc = x.OccurredAtUtc,
                        FrameIndex = x.FrameIndex,
                        VideoTimeS = x.VideoTimeS,
                        Direction = x.Direction,
                        TrackId = x.TrackId,
                        ClassName = x.ClassName,
                        Confidence = x.Confidence,
                        ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                        Notes = x.Review != null ? x.Review.Notes : null,
                        ThumbnailUrl = x.ScenePath != null
                            ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                            : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                    })
                    .ToListAsync(ct);

                pendingRows = OrderByEventUids(rows, orderedPendingUids, x => x.EventUid);
            }
        }

        var current = pendingRows.FirstOrDefault();

        if (current is not null)
        {
            var criteria = await db.CameraCriteria
                .AsNoTracking()
                .FirstOrDefaultAsync(x => x.SiteId == current.SiteId && x.CameraId == current.CameraId, ct);

            if (criteria is not null)
            {
                current.CriteriaTitle = criteria.CriteriaTitle;
                current.CriteriaDescription = criteria.CriteriaDescription;
            }
        }

        var upcoming = pendingRows
            .Skip(1)
            .Take(6)
            .Select(x => new EventListItemViewModel
            {
                EventUid = x.EventUid,
                RunUid = x.RunUid,
                SiteId = x.SiteId,
                CameraId = x.CameraId,
                OccurredAtUtc = x.OccurredAtUtc,
                Direction = x.Direction ?? "-",
                ClassName = x.ClassName ?? "unknown",
                ReviewStatus = ReviewStatuses.Pending,
                ThumbnailUrl = x.ThumbnailUrl,
            })
            .ToList();

        var reviewTotals = await db.EventReviews
            .AsNoTracking()
            .GroupBy(_ => 1)
            .Select(g => new
            {
                Pending = g.Sum(x => x.ReviewStatus == ReviewStatuses.Pending ? 1 : 0),
                Qualified = g.Sum(x => x.ReviewStatus == ReviewStatuses.Qualified ? 1 : 0),
                NotQualified = g.Sum(x => x.ReviewStatus == ReviewStatuses.NotQualified ? 1 : 0),
            })
            .FirstOrDefaultAsync(ct);

        var vm = new ReviewQueueViewModel
        {
            Current = current,
            Upcoming = upcoming,
            PendingCount = reviewTotals?.Pending ?? 0,
            QualifiedCount = reviewTotals?.Qualified ?? 0,
            NotQualifiedCount = reviewTotals?.NotQualified ?? 0,
        };

        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> Detail(string eventUid, CancellationToken ct)
    {
        var row = await db.Events
            .AsNoTracking()
            .Include(x => x.Review)
            .FirstOrDefaultAsync(x => x.EventUid == eventUid, ct);

        if (row is null)
        {
            return NotFound();
        }

        var vm = new EventDetailViewModel
        {
            EventUid = row.EventUid,
            RunUid = row.RunUid,
            SiteId = row.SiteId,
            CameraId = row.CameraId,
            OccurredAtUtc = row.OccurredAtUtc,
            FrameIndex = row.FrameIndex,
            VideoTimeS = row.VideoTimeS,
            Direction = row.Direction,
            TrackId = row.TrackId,
            ClassName = row.ClassName,
            Confidence = row.Confidence,
            ReviewStatus = row.Review != null ? row.Review.ReviewStatus : ReviewStatuses.Pending,
            Notes = row.Review?.Notes,
            ThumbnailUrl = BuildEvidenceUrl(row.EventUid, row.ThumbPath, row.ScenePath),
        };

        var criteria = await db.CameraCriteria
            .AsNoTracking()
            .FirstOrDefaultAsync(x => x.SiteId == vm.SiteId && x.CameraId == vm.CameraId, ct);

        if (criteria is not null)
        {
            vm.CriteriaTitle = criteria.CriteriaTitle;
            vm.CriteriaDescription = criteria.CriteriaDescription;
        }

        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> ExportCsv([FromQuery] EventQueryRequest request, CancellationToken ct)
    {
        request.ReviewStatus ??= string.Empty;

        var filtered = db.Events.PortalQueryable().ApplyEventFilters(request)
            .Where(x => x.Review != null && x.Review.ReviewStatus != ReviewStatuses.Pending);

        var rows = await filtered
            .Select(x => new
            {
                x.EventUid,
                x.RunUid,
                x.SiteId,
                x.CameraId,
                x.OccurredAtUtc,
                x.Direction,
                x.ClassName,
                x.TrackId,
                ReviewStatus = x.Review!.ReviewStatus,
                x.Review!.ReviewedAtUtc,
                x.Review!.ReviewedBy,
                x.Review!.Notes,
            })
            .ToListAsync(ct);

        var orderedRows = rows
            .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
            .ThenByDescending(x => x.EventUid);

        var sb = new StringBuilder();
        sb.AppendLine("event_uid,run_uid,site_id,camera_id,occurred_at_utc,direction,class_name,track_id,review_status,reviewed_at_utc,reviewed_by,notes");

        foreach (var row in orderedRows)
        {
            sb
                .Append(Csv(row.EventUid)).Append(',')
                .Append(Csv(row.RunUid)).Append(',')
                .Append(Csv(row.SiteId)).Append(',')
                .Append(Csv(row.CameraId)).Append(',')
                .Append(Csv(row.OccurredAtUtc?.ToString("O"))).Append(',')
                .Append(Csv(row.Direction)).Append(',')
                .Append(Csv(row.ClassName)).Append(',')
                .Append(Csv(row.TrackId?.ToString())).Append(',')
                .Append(Csv(row.ReviewStatus)).Append(',')
                .Append(Csv(row.ReviewedAtUtc?.ToString("O"))).Append(',')
                .Append(Csv(row.ReviewedBy)).Append(',')
                .Append(Csv(row.Notes))
                .AppendLine();
        }

        var fileName = $"reviewed-events-{DateTime.UtcNow:yyyyMMdd-HHmmss}.csv";
        return File(Encoding.UTF8.GetBytes(sb.ToString()), "text/csv", fileName);
    }

    private static int ClampPageSize(int pageSize, int defaultValue, int maxValue)
    {
        var fallback = defaultValue > 0 ? defaultValue : 50;
        var cap = maxValue > 0 ? maxValue : 200;
        var value = pageSize > 0 ? pageSize : fallback;
        return Math.Min(Math.Max(1, value), cap);
    }

    private static string? BuildEvidenceUrl(string eventUid, string? thumbPath, string? scenePath)
    {
        if (!string.IsNullOrWhiteSpace(scenePath))
        {
            return $"/api/events/{eventUid}/thumbnail?kind=scene";
        }

        if (!string.IsNullOrWhiteSpace(thumbPath))
        {
            return $"/api/events/{eventUid}/thumbnail";
        }

        return null;
    }

    private static string Csv(string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            return string.Empty;
        }

        var escaped = value.Replace("\"", "\"\"");
        if (escaped.Contains(',') || escaped.Contains('"') || escaped.Contains('\n') || escaped.Contains('\r'))
        {
            return $"\"{escaped}\"";
        }

        return escaped;
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
}
