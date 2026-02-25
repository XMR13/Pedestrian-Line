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
        var baseQuery = isSqlServer
            ? db.Events.PortalQueryable()
            : db.Events.ApplySqliteDateRangeFilter(request);
        var filtered = baseQuery
            .ApplyEventFilters(request, includeDateFilter: isSqlServer);

        var vm = new DashboardViewModel
        {
            SiteId = request.SiteId,
            CameraId = request.CameraId,
            Date = request.Date,
            DateFrom = request.DateFrom,
            DateTo = request.DateTo,
        };

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
            .ApplyDefaultSortDesc(isSqlServer)
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
                ThumbnailUrl = x.ThumbPath != null
                    ? $"/api/events/{x.EventUid}/thumbnail"
                    : (x.ScenePath != null ? $"/api/events/{x.EventUid}/thumbnail?kind=scene" : null),
            })
            .ToListAsync(ct);

        return View(vm);
    }

    [AllowAnonymous]
    [HttpGet]
    public IActionResult Error()
    {
        return View();
    }
}
