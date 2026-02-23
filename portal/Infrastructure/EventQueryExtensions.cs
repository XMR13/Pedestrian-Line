using Microsoft.EntityFrameworkCore;
using Portal.Web.Contracts;
using Portal.Web.Models;

namespace Portal.Web.Infrastructure;

public static class EventQueryExtensions
{
    public static IQueryable<EventRecord> ApplyEventFilters(this IQueryable<EventRecord> query, EventQueryRequest req)
    {
        if (!string.IsNullOrWhiteSpace(req.SiteId))
        {
            var siteId = req.SiteId.Trim();
            query = query.Where(x => x.SiteId == siteId);
        }

        if (!string.IsNullOrWhiteSpace(req.CameraId))
        {
            var cameraId = req.CameraId.Trim();
            query = query.Where(x => x.CameraId == cameraId);
        }

        if (req.Date.HasValue)
        {
            var start = req.Date.Value.Date;
            var end = start.AddDays(1);
            query = query.Where(x => x.OccurredAtUtc >= start && x.OccurredAtUtc < end);
        }

        if (!string.IsNullOrWhiteSpace(req.Direction))
        {
            var direction = req.Direction.Trim();
            query = query.Where(x => x.Direction == direction);
        }

        if (!string.IsNullOrWhiteSpace(req.ClassName))
        {
            var className = req.ClassName.Trim();
            query = query.Where(x => x.ClassName == className);
        }

        if (!string.IsNullOrWhiteSpace(req.ReviewStatus))
        {
            var status = req.ReviewStatus.Trim().ToUpperInvariant();
            query = query.Where(x => x.Review != null && x.Review.ReviewStatus == status);
        }

        return query;
    }

    public static string? ToThumbnailUrl(this EventRecord e)
    {
        if (string.IsNullOrWhiteSpace(e.ThumbPath))
        {
            return null;
        }

        return $"/api/events/{e.EventUid}/thumbnail";
    }

    public static IQueryable<EventRecord> PortalQueryable(this DbSet<EventRecord> events)
    {
        return events
            .AsNoTracking()
            .AsQueryable();
    }
}
