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
        var filtered = db.Events.PortalQueryable().ApplyEventFilters(request);
        var isSqlServer = db.Database.IsSqlServer();

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

        var vm = new DashboardViewModel
        {
            SiteId = request.SiteId,
            CameraId = request.CameraId,
            Date = request.Date,
            TotalAToB = totals?.TotalAToB ?? 0,
            TotalBToA = totals?.TotalBToA ?? 0,
            TotalPending = totals?.TotalPending ?? 0,
            TotalQualified = totals?.TotalQualified ?? 0,
            TotalNotQualified = totals?.TotalNotQualified ?? 0,
        };

        vm.TotalReviewed = vm.TotalQualified + vm.TotalNotQualified;

        if (isSqlServer)
        {
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
            var orderedEventUids = (await filtered
                .Select(x => new
                {
                    x.EventUid,
                    x.OccurredAtUtc,
                })
                .ToListAsync(ct))
                .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
                .ThenByDescending(x => x.EventUid)
                .Take(8)
                .Select(x => x.EventUid)
                .ToList();

            if (orderedEventUids.Count == 0)
            {
                vm.RecentEvents = [];
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
                        ThumbnailUrl = x.ScenePath != null
                            ? $"/api/events/{x.EventUid}/thumbnail?kind=scene"
                            : (x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null),
                    })
                    .ToListAsync(ct);

                var rowMap = rows.ToDictionary(x => x.EventUid, StringComparer.Ordinal);
                vm.RecentEvents = orderedEventUids
                    .Where(rowMap.ContainsKey)
                    .Select(eventUid => rowMap[eventUid])
                    .ToList();
            }
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
