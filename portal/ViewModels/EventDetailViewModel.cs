namespace Portal.Web.ViewModels;

public sealed class EventDetailViewModel
{
    public string EventUid { get; set; } = string.Empty;
    public string RunUid { get; set; } = string.Empty;
    public string SiteId { get; set; } = string.Empty;
    public string CameraId { get; set; } = string.Empty;
    public DateTimeOffset? OccurredAtUtc { get; set; }
    public int? FrameIndex { get; set; }
    public double? VideoTimeS { get; set; }
    public string? Direction { get; set; }
    public int? TrackId { get; set; }
    public string? ClassName { get; set; }
    public double? Confidence { get; set; }
    public string ReviewStatus { get; set; } = "PENDING";
    public string? Notes { get; set; }
    public string? ThumbnailUrl { get; set; }

    public string CriteriaTitle { get; set; } = "Qualified Criteria";
    public string CriteriaDescription { get; set; } = "Use your site SOP for qualification decisions.";
}
