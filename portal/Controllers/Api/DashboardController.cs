using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Portal.Web.Contracts;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using Portal.Web.Models;

namespace Portal.Web.Controllers.Api;

[ApiController]
[Route("api/dashboard")]
[Authorize]
public sealed class DashboardController(PortalDbContext db) : ControllerBase
{
    [HttpGet("summary")]
    public async Task<IActionResult> Summary([FromQuery] EventQueryRequest request, CancellationToken ct)
    {
        var isSqlServer = db.Database.IsSqlServer();
        var selectedSingleDate = request.ResolveSingleLocalDate();
        var filtered = db.Events.PortalQueryable()
            .ApplyEventFilters(request, includeDateFilter: isSqlServer);

        int totalAToB;
        int totalBToA;
        int pending;
        int qualified;
        int notQualified;
        List<DashboardTrendEventRow> trendRows;
        List<object> recent;
        if (isSqlServer)
        {
            var totals = await filtered
                .GroupBy(_ => 1)
                .Select(g => new
                {
                    TotalAToB = g.Sum(x => x.Direction == "A_TO_B" ? 1 : 0),
                    TotalBToA = g.Sum(x => x.Direction == "B_TO_A" ? 1 : 0),
                    Pending = g.Sum(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending ? 1 : 0),
                    Qualified = g.Sum(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.Qualified ? 1 : 0),
                    NotQualified = g.Sum(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.NotQualified ? 1 : 0),
                })
                .FirstOrDefaultAsync(ct);

            totalAToB = totals?.TotalAToB ?? 0;
            totalBToA = totals?.TotalBToA ?? 0;
            pending = totals?.Pending ?? 0;
            qualified = totals?.Qualified ?? 0;
            notQualified = totals?.NotQualified ?? 0;

            trendRows = await DashboardTrendBuilder.QueryRowsAsync(filtered, selectedSingleDate, ct);
            recent = (await filtered
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Take(8)
                .Select(x => new
                {
                    event_uid = x.EventUid,
                    occurred_at_utc = x.OccurredAtUtc,
                    site_id = x.SiteId,
                    camera_id = x.CameraId,
                    class_name = x.ClassName,
                    direction = x.Direction,
                    review_status = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                })
                .ToListAsync(ct))
                .Cast<object>()
                .ToList();
        }
        else
        {
            var rows = (await filtered
                .Select(x => new DashboardSummaryRow
                {
                    EventUid = x.EventUid,
                    OccurredAtUtc = x.OccurredAtUtc,
                    SiteId = x.SiteId,
                    CameraId = x.CameraId,
                    ClassName = x.ClassName,
                    Direction = x.Direction,
                    ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
                })
                .ToListAsync(ct))
                .ApplyLocalDateRangeInMemory(request, x => x.OccurredAtUtc)
                .ToList();

            totalAToB = rows.Count(x => string.Equals(x.Direction, "A_TO_B", StringComparison.OrdinalIgnoreCase));
            totalBToA = rows.Count(x => string.Equals(x.Direction, "B_TO_A", StringComparison.OrdinalIgnoreCase));
            qualified = rows.Count(x => string.Equals(x.ReviewStatus, ReviewStatuses.Qualified, StringComparison.OrdinalIgnoreCase));
            notQualified = rows.Count(x => string.Equals(x.ReviewStatus, ReviewStatuses.NotQualified, StringComparison.OrdinalIgnoreCase));
            pending = rows.Count - qualified - notQualified;

            trendRows = rows
                .Where(x => x.OccurredAtUtc.HasValue)
                .Select(x => new DashboardTrendEventRow
                {
                    OccurredAtUtc = x.OccurredAtUtc ?? DateTimeOffset.MinValue,
                    Direction = x.Direction,
                    ReviewStatus = x.ReviewStatus ?? ReviewStatuses.Pending,
                })
                .ToList();

            recent = rows
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Take(8)
                .Select(x => new
                {
                    event_uid = x.EventUid,
                    occurred_at_utc = x.OccurredAtUtc,
                    site_id = x.SiteId,
                    camera_id = x.CameraId,
                    class_name = x.ClassName,
                    direction = x.Direction,
                    review_status = x.ReviewStatus ?? ReviewStatuses.Pending,
                })
                .Cast<object>()
                .ToList();
        }

        var reviewed = qualified + notQualified;
        var trend = DashboardTrendBuilder.Build(trendRows, selectedSingleDate, TimeZoneInfo.Local);

        return Ok(new
        {
            totals = new
            {
                a_to_b = totalAToB,
                b_to_a = totalBToA,
                pending,
                reviewed,
                qualified,
                not_qualified = notQualified,
            },
            trend = new
            {
                bucket = trend.Bucket,
                timezone = trend.TimeZoneId,
                range_label = trend.RangeLabel,
                points = trend.Points.Select(x => new
                {
                    bucket_start_local = x.BucketStartLocal.ToString("yyyy-MM-ddTHH:mm:ss"),
                    label = x.Label,
                    tooltip = x.TooltipLabel,
                    a_to_b = x.AToB,
                    b_to_a = x.BToA,
                    reviewed = x.Reviewed,
                    pending = x.Pending,
                }),
            },
            recent,
        });
    }

    private sealed class DashboardSummaryRow
    {
        public string EventUid { get; set; } = string.Empty;
        public DateTimeOffset? OccurredAtUtc { get; set; }
        public string SiteId { get; set; } = string.Empty;
        public string CameraId { get; set; } = string.Empty;
        public string? ClassName { get; set; }
        public string? Direction { get; set; }
        public string? ReviewStatus { get; set; }
    }
}
