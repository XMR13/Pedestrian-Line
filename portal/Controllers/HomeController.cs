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

        var vm = new DashboardViewModel
        {
            SiteId = request.SiteId,
            CameraId = request.CameraId,
            Date = request.Date,
            TotalAToB = await filtered.CountAsync(x => x.Direction == "A_TO_B", ct),
            TotalBToA = await filtered.CountAsync(x => x.Direction == "B_TO_A", ct),
            TotalPending = await filtered.CountAsync(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending, ct),
            TotalQualified = await filtered.CountAsync(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.Qualified, ct),
            TotalNotQualified = await filtered.CountAsync(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.NotQualified, ct),
        };

        vm.TotalReviewed = vm.TotalQualified + vm.TotalNotQualified;

        vm.RecentEvents = (await filtered
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
                ThumbnailUrl = x.ThumbPath != null ? $"/api/events/{x.EventUid}/thumbnail" : null,
            })
            .ToListAsync(ct))
            .OrderByDescending(x => x.OccurredAtUtc ?? DateTimeOffset.MinValue)
            .ThenByDescending(x => x.EventUid)
            .Take(8)
            .ToList();

        return View(vm);
    }

    [AllowAnonymous]
    [HttpGet]
    public IActionResult Error()
    {
        return View();
    }
}
