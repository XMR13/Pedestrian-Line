using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Portal.Web.Contracts;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using Portal.Web.Models;
using Portal.Web.ViewModels;

namespace Portal.Web.Controllers;

[Authorize]
public sealed class HomeController(PortalDbContext db) : Controller
{
    [HttpGet]
    public async Task<IActionResult> Index([FromQuery] EventQueryRequest request, CancellationToken ct)
    {
        var isSqlServer = db.Database.IsSqlServer();
        var selectedSingleDate = request.ResolveSingleLocalDate();
        var filtered = db.Events.PortalQueryable()
            .ApplyEventFilters(request, includeDateFilter: isSqlServer);

        var vm = new DashboardViewModel
        {
            SiteId = request.SiteId,
            CameraId = request.CameraId,
            Date = request.Date,
            DateFrom = request.DateFrom,
            DateTo = request.DateTo,
        };

        if (isSqlServer)
        {
            var totals = await filtered
                .GroupBy(_ => 1)
                .Select(g => new
                {
                    TotalAToB = g.Sum(x => x.Direction == "A_TO_B" ? 1 : 0),
                    TotalBToA = g.Sum(x => x.Direction == "B_TO_A" ? 1 : 0),
                    TotalPending = g.Sum(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending ? 1 : 0),
                    TotalQualified = g.Sum(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.Qualified ? 1 : 0),
                    TotalNotQualified = g.Sum(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.NotQualified ? 1 : 0),
                })
                .FirstOrDefaultAsync(ct);

            vm.TotalAToB = totals?.TotalAToB ?? 0;
            vm.TotalBToA = totals?.TotalBToA ?? 0;
            vm.TotalPending = totals?.TotalPending ?? 0;
            vm.TotalQualified = totals?.TotalQualified ?? 0;
            vm.TotalNotQualified = totals?.TotalNotQualified ?? 0;
            vm.TotalReviewed = vm.TotalQualified + vm.TotalNotQualified;

            var trendRows = await DashboardTrendBuilder.QueryRowsAsync(filtered, selectedSingleDate, ct);
            var trend = DashboardTrendBuilder.Build(trendRows, selectedSingleDate, TimeZoneInfo.Local);
            vm.TrendBucket = trend.Bucket;
            vm.TrendTimezone = trend.TimeZoneId;
            vm.TrendRangeLabel = trend.RangeLabel;
            vm.TrendPoints = trend.Points
                .Select(x => new DashboardTrendPointViewModel
                {
                    BucketStartLocal = x.BucketStartLocal,
                    Label = x.Label,
                    TooltipLabel = x.TooltipLabel,
                    AToB = x.AToB,
                    BToA = x.BToA,
                    Reviewed = x.Reviewed,
                    Pending = x.Pending,
                })
                .ToList();

            vm.RecentEvents = await filtered
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Take(8)
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
                    ThumbnailUrl = x.ScenePath != null
                        ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                        : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                })
                .ToListAsync(ct);
        }
        else
        {
            var rows = (await filtered
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
                    ThumbnailUrl = x.ScenePath != null
                        ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                        : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                })
                .ToListAsync(ct))
                .ApplyLocalDateRangeInMemory(request, x => x.OccurredAtUtc)
                .ToList();

            vm.TotalAToB = rows.Count(x => string.Equals(x.Direction, "A_TO_B", StringComparison.OrdinalIgnoreCase));
            vm.TotalBToA = rows.Count(x => string.Equals(x.Direction, "B_TO_A", StringComparison.OrdinalIgnoreCase));
            vm.TotalQualified = rows.Count(x => string.Equals(x.ReviewStatus, ReviewStatuses.Qualified, StringComparison.OrdinalIgnoreCase));
            vm.TotalNotQualified = rows.Count(x => string.Equals(x.ReviewStatus, ReviewStatuses.NotQualified, StringComparison.OrdinalIgnoreCase));
            vm.TotalReviewed = vm.TotalQualified + vm.TotalNotQualified;
            vm.TotalPending = rows.Count - vm.TotalReviewed;

            var trendRows = rows
                .Where(x => x.OccurredAtUtc.HasValue)
                .Select(x => new DashboardTrendEventRow
                {
                    OccurredAtUtc = x.OccurredAtUtc ?? DateTimeOffset.MinValue,
                    Direction = x.Direction,
                    ReviewStatus = x.ReviewStatus,
                })
                .ToList();
            var trend = DashboardTrendBuilder.Build(trendRows, selectedSingleDate, TimeZoneInfo.Local);
            vm.TrendBucket = trend.Bucket;
            vm.TrendTimezone = trend.TimeZoneId;
            vm.TrendRangeLabel = trend.RangeLabel;
            vm.TrendPoints = trend.Points
                .Select(x => new DashboardTrendPointViewModel
                {
                    BucketStartLocal = x.BucketStartLocal,
                    Label = x.Label,
                    TooltipLabel = x.TooltipLabel,
                    AToB = x.AToB,
                    BToA = x.BToA,
                    Reviewed = x.Reviewed,
                    Pending = x.Pending,
                })
                .ToList();

            vm.RecentEvents = rows
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Take(8)
                .ToList();
        }

        return View(vm);
    }

    [AllowAnonymous]
    [HttpGet]
    public IActionResult Error()
    {
        return View();
    }
}
