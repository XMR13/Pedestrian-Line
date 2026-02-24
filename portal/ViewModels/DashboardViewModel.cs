namespace Portal.Web.ViewModels;

public sealed class DashboardViewModel
{
    public string? SiteId { get; set; }
    public string? CameraId { get; set; }
    public DateTime? Date { get; set; }

    public int TotalAToB { get; set; }
    public int TotalBToA { get; set; }
    public int TotalPending { get; set; }
    public int TotalReviewed { get; set; }
    public int TotalQualified { get; set; }
    public int TotalNotQualified { get; set; }

    public string TrendBucket { get; set; } = "hour";
    public string TrendTimezone { get; set; } = TimeZoneInfo.Local.Id;
    public string TrendRangeLabel { get; set; } = string.Empty;
    public List<DashboardTrendPointViewModel> TrendPoints { get; set; } = [];

    public List<EventListItemViewModel> RecentEvents { get; set; } = new();
}
