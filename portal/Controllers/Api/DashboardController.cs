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
        var isSqlServer = db.Database.IsSqlServer();

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

        var reviewed = qualified + notQualified;

        List<object> recent;
        if (isSqlServer)
        {
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
                recent = [];
            }
            else
            {
                var rows = await filtered
                    .Where(x => orderedEventUids.Contains(x.EventUid))
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
                    .ToListAsync(ct);

                var rowMap = rows.ToDictionary(x => x.event_uid, StringComparer.Ordinal);
                recent = orderedEventUids
                    .Where(rowMap.ContainsKey)
                    .Select(eventUid => (object)rowMap[eventUid])
                    .ToList();
            }
        }

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
