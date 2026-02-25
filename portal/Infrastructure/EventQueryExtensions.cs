using Microsoft.EntityFrameworkCore;
using Portal.Web.Contracts;
using Portal.Web.Models;

namespace Portal.Web.Infrastructure;

public static class EventQueryExtensions
{
    public static IQueryable<EventRecord> ApplyEventFilters(
        this IQueryable<EventRecord> query,
        EventQueryRequest req,
        bool includeDateFilter = true)
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

        if (includeDateFilter && TryResolveLocalDateRange(req, out var localStartDate, out var localEndDateExclusive))
        {
            var startUtc = LocalDateStartToUtc(localStartDate);
            var endUtc = LocalDateStartToUtc(localEndDateExclusive);
            query = query.Where(x => x.OccurredAtUtc >= startUtc && x.OccurredAtUtc < endUtc);
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

    public static DateTime? ResolveSingleLocalDate(this EventQueryRequest req)
    {
        if (req.Date.HasValue)
        {
            return req.Date.Value.Date;
        }

        if (req.DateFrom.HasValue && req.DateTo.HasValue)
        {
            var from = req.DateFrom.Value.Date;
            var to = req.DateTo.Value.Date;
            return from == to ? from : null;
        }

        return null;
    }

    public static IEnumerable<T> ApplyLocalDateRangeInMemory<T>(
        this IEnumerable<T> source,
        EventQueryRequest req,
        Func<T, DateTimeOffset?> occurredAtSelector)
    {
        if (!TryResolveLocalDateRange(req, out var localStartDate, out var localEndDateExclusive))
        {
            return source;
        }

        var startUtc = LocalDateStartToUtc(localStartDate);
        var endUtc = LocalDateStartToUtc(localEndDateExclusive);
        return source.Where(x =>
        {
            var occurredAtUtc = occurredAtSelector(x);
            return occurredAtUtc.HasValue
                && occurredAtUtc.Value >= startUtc
                && occurredAtUtc.Value < endUtc;
        });
    }

    public static string? ToThumbnailUrl(this EventRecord e)
    {
        if (!string.IsNullOrWhiteSpace(e.ScenePath))
        {
            return $"/api/events/{e.EventUid}/thumbnail?kind=scene";
        }

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

    private static bool TryResolveLocalDateRange(
        EventQueryRequest req,
        out DateTime localStartDate,
        out DateTime localEndDateExclusive)
    {
        if (req.DateFrom.HasValue || req.DateTo.HasValue)
        {
            var start = (req.DateFrom ?? req.DateTo)!.Value.Date;
            var end = (req.DateTo ?? req.DateFrom)!.Value.Date;
            if (end < start)
            {
                (start, end) = (end, start);
            }

            localStartDate = start;
            localEndDateExclusive = end.AddDays(1);
            return true;
        }

        if (req.Date.HasValue)
        {
            localStartDate = req.Date.Value.Date;
            localEndDateExclusive = localStartDate.AddDays(1);
            return true;
        }

        localStartDate = default;
        localEndDateExclusive = default;
        return false;
    }

    private static DateTimeOffset LocalDateStartToUtc(DateTime localDate)
    {
        var localUnspecified = DateTime.SpecifyKind(localDate.Date, DateTimeKind.Unspecified);
        var utc = TimeZoneInfo.ConvertTimeToUtc(localUnspecified, TimeZoneInfo.Local);
        return new DateTimeOffset(utc, TimeSpan.Zero);
    }
}
