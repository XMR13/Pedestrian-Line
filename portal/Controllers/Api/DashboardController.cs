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
        var filtered = db.Events.PortalQueryable().ApplyEventFilters(request);

        var totalAToB = await filtered.CountAsync(x => x.Direction == "A_TO_B", ct);
        var totalBToA = await filtered.CountAsync(x => x.Direction == "B_TO_A", ct);

        var pending = await filtered.CountAsync(x => x.Review == null || x.Review.ReviewStatus == ReviewStatuses.Pending, ct);
        var qualified = await filtered.CountAsync(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.Qualified, ct);
        var notQualified = await filtered.CountAsync(x => x.Review != null && x.Review.ReviewStatus == ReviewStatuses.NotQualified, ct);

        var reviewed = qualified + notQualified;

        var recent = (await filtered
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
            .OrderByDescending(x => x.occurred_at_utc ?? DateTimeOffset.MinValue)
            .ThenByDescending(x => x.event_uid)
            .Take(8)
            .ToList();

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
            recent,
        });
    }
}
