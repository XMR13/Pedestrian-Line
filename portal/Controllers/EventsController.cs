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

        var filtered = db.Events.PortalQueryable().ApplyEventFilters(request);
        var total = await filtered.CountAsync(ct);

        var items = (await filtered
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
                ThumbnailUrl = x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null,
            })
            .ToListAsync(ct))
            .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
            .ThenByDescending(x => x.EventUid)
            .Skip((page - 1) * pageSize)
            .Take(pageSize)
            .ToList();

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
        var pendingRows = (await db.Events.PortalQueryable()
            .Where(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending)
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
                ThumbnailUrl = x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null,
            })
            .ToListAsync(ct))
            .OrderBy(x => x.OccurredAtUtc ?? DateTimeOffset.MaxValue)
            .ThenBy(x => x.EventUid)
            .ToList();

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

        var vm = new ReviewQueueViewModel
        {
            Current = current,
            Upcoming = upcoming,
            PendingCount = await db.EventReviews.CountAsync(x => x.ReviewStatus == ReviewStatuses.Pending, ct),
            QualifiedCount = await db.EventReviews.CountAsync(x => x.ReviewStatus == ReviewStatuses.Qualified, ct),
            NotQualifiedCount = await db.EventReviews.CountAsync(x => x.ReviewStatus == ReviewStatuses.NotQualified, ct),
        };

        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> Detail(string eventUid, CancellationToken ct)
    {
        var row = await db.Events
            .PortalQueryable()
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
            ThumbnailUrl = row.ThumbPath != null ? $"/api/events/{row.EventUid}/thumbnail" : null,
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
}
