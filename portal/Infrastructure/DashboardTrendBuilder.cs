using Microsoft.EntityFrameworkCore;
using Portal.Web.Models;

namespace Portal.Web.Infrastructure;

public static class DashboardTrendBuilder
{
    public const int RecentDailyWindowDays = 14;

    public static DashboardTrendResult Build(
        IReadOnlyList<DashboardTrendEventRow> rows,
        DateTime? selectedDate,
        TimeZoneInfo timeZone)
    {
        return selectedDate.HasValue
            ? BuildHourlyForDate(rows, selectedDate.Value.Date, timeZone)
            : BuildRecentDaily(rows, timeZone, RecentDailyWindowDays);
    }

    public static async Task<List<DashboardTrendEventRow>> QueryRowsAsync(
        IQueryable<EventRecord> filtered,
        DateTime? selectedDate,
        CancellationToken ct)
    {
        var trendQuery = filtered.Where(x => x.OccurredAtUtc != null);
        var rows = await trendQuery
            .Select(x => new DashboardTrendEventRow
            {
                OccurredAtUtc = x.OccurredAtUtc ?? DateTimeOffset.MinValue,
                Direction = x.Direction,
                ReviewStatus = x.Review != null ? x.Review.ReviewStatus : ReviewStatuses.Pending,
            })
            .ToListAsync(ct);

        if (!selectedDate.HasValue && rows.Count > 0)
        {
            var maxOccurredAtUtc = rows.Max(x => x.OccurredAtUtc);
            var cutoff = maxOccurredAtUtc.AddDays(-(RecentDailyWindowDays - 1));
            rows = rows
                .Where(x => x.OccurredAtUtc >= cutoff)
                .ToList();
        }

        return rows;
    }

    public static DashboardTrendResult BuildHourlyForDate(
        IReadOnlyList<DashboardTrendEventRow> rows,
        DateTime localDate,
        TimeZoneInfo timeZone)
    {
        var date = localDate.Date;
        var buckets = Enumerable.Range(0, 24)
            .ToDictionary(
                i => date.AddHours(i),
                _ => new DashboardTrendAccumulator());

        foreach (var row in rows)
        {
            var local = TimeZoneInfo.ConvertTime(row.OccurredAtUtc, timeZone).DateTime;
            if (local.Date != date)
            {
                continue;
            }

            var key = new DateTime(local.Year, local.Month, local.Day, local.Hour, 0, 0, DateTimeKind.Unspecified);
            if (!buckets.TryGetValue(key, out var acc))
            {
                continue;
            }

            ApplyRow(acc, row);
        }

        return new DashboardTrendResult
        {
            Bucket = "hour",
            TimeZoneId = timeZone.Id,
            RangeLabel = $"{date:yyyy-MM-dd} (hourly)",
            Points = buckets
                .OrderBy(x => x.Key)
                .Select(x => new DashboardTrendPoint
                {
                    BucketStartLocal = x.Key,
                    Label = x.Key.ToString("HH:mm"),
                    TooltipLabel = x.Key.ToString("yyyy-MM-dd HH:mm"),
                    AToB = x.Value.AToB,
                    BToA = x.Value.BToA,
                    Reviewed = x.Value.Reviewed,
                    Pending = x.Value.Pending,
                })
                .ToList(),
        };
    }

    public static DashboardTrendResult BuildRecentDaily(
        IReadOnlyList<DashboardTrendEventRow> rows,
        TimeZoneInfo timeZone,
        int recentDays)
    {
        var maxLocalDate = rows.Count == 0
            ? DateTime.Now.Date
            : rows.Max(x => TimeZoneInfo.ConvertTime(x.OccurredAtUtc, timeZone).DateTime.Date);

        var startLocalDate = maxLocalDate.AddDays(-(recentDays - 1));
        var buckets = Enumerable.Range(0, recentDays)
            .ToDictionary(
                i => startLocalDate.AddDays(i),
                _ => new DashboardTrendAccumulator());

        foreach (var row in rows)
        {
            var localDate = TimeZoneInfo.ConvertTime(row.OccurredAtUtc, timeZone).DateTime.Date;
            if (localDate < startLocalDate || localDate > maxLocalDate)
            {
                continue;
            }

            if (!buckets.TryGetValue(localDate, out var acc))
            {
                continue;
            }

            ApplyRow(acc, row);
        }

        return new DashboardTrendResult
        {
            Bucket = "day",
            TimeZoneId = timeZone.Id,
            RangeLabel = $"{startLocalDate:yyyy-MM-dd} to {maxLocalDate:yyyy-MM-dd} (daily)",
            Points = buckets
                .OrderBy(x => x.Key)
                .Select(x => new DashboardTrendPoint
                {
                    BucketStartLocal = x.Key,
                    Label = x.Key.ToString("MM-dd"),
                    TooltipLabel = x.Key.ToString("yyyy-MM-dd"),
                    AToB = x.Value.AToB,
                    BToA = x.Value.BToA,
                    Reviewed = x.Value.Reviewed,
                    Pending = x.Value.Pending,
                })
                .ToList(),
        };
    }

    private static void ApplyRow(DashboardTrendAccumulator acc, DashboardTrendEventRow row)
    {
        if (string.Equals(row.Direction, "A_TO_B", StringComparison.OrdinalIgnoreCase))
        {
            acc.AToB++;
        }
        else if (string.Equals(row.Direction, "B_TO_A", StringComparison.OrdinalIgnoreCase))
        {
            acc.BToA++;
        }

        if (IsReviewed(row.ReviewStatus))
        {
            acc.Reviewed++;
        }
        else
        {
            acc.Pending++;
        }
    }

    private static bool IsReviewed(string? reviewStatus)
    {
        return string.Equals(reviewStatus, ReviewStatuses.Qualified, StringComparison.OrdinalIgnoreCase)
            || string.Equals(reviewStatus, ReviewStatuses.NotQualified, StringComparison.OrdinalIgnoreCase);
    }
}

public sealed class DashboardTrendEventRow
{
    public DateTimeOffset OccurredAtUtc { get; set; }
    public string? Direction { get; set; }
    public string ReviewStatus { get; set; } = ReviewStatuses.Pending;
}

public sealed class DashboardTrendResult
{
    public string Bucket { get; set; } = "hour";
    public string TimeZoneId { get; set; } = TimeZoneInfo.Local.Id;
    public string RangeLabel { get; set; } = string.Empty;
    public List<DashboardTrendPoint> Points { get; set; } = [];
}

public sealed class DashboardTrendPoint
{
    public DateTime BucketStartLocal { get; set; }
    public string Label { get; set; } = string.Empty;
    public string TooltipLabel { get; set; } = string.Empty;
    public int AToB { get; set; }
    public int BToA { get; set; }
    public int Reviewed { get; set; }
    public int Pending { get; set; }
}

internal sealed class DashboardTrendAccumulator
{
    public int AToB { get; set; }
    public int BToA { get; set; }
    public int Reviewed { get; set; }
    public int Pending { get; set; }
}
