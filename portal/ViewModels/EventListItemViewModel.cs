namespace Portal.Web.ViewModels;

public sealed class EventListItemViewModel
{
    public string EventUid { get; set; } = string.Empty;
    public string RunUid { get; set; } = string.Empty;
    public string SiteId { get; set; } = string.Empty;
    public string CameraId { get; set; } = string.Empty;
    public DateTimeOffset? OccurredAtUtc { get; set; }
    public string Direction { get; set; } = string.Empty;
    public string ClassName { get; set; } = string.Empty;
    public string ReviewStatus { get; set; } = string.Empty;
    public string? Notes { get; set; }
    public string? ThumbnailUrl { get; set; }
}
