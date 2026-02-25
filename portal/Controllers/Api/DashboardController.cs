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
        var baseQuery = isSqlServer
            ? db.Events.PortalQueryable()
            : db.Events.ApplySqliteDateRangeFilter(request);
        var filtered = baseQuery
            .ApplyEventFilters(request, includeDateFilter: isSqlServer);

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

        var totalAToB = totals?.TotalAToB ?? 0;
        var totalBToA = totals?.TotalBToA ?? 0;
        var pending = totals?.Pending ?? 0;
        var qualified = totals?.Qualified ?? 0;
        var notQualified = totals?.NotQualified ?? 0;

        var trendRows = await DashboardTrendBuilder.QueryRowsAsync(filtered, selectedSingleDate, ct);
        var recent = (await filtered
            .ApplyDefaultSortDesc(isSqlServer)
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

}
