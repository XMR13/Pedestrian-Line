namespace Portal.Web.Contracts;

public sealed class EventQueryRequest
{
    public string? SiteId { get; set; }
    public string? CameraId { get; set; }
    public DateTime? Date { get; set; }
    public string? Direction { get; set; }
    public string? ClassName { get; set; }
    public string? ReviewStatus { get; set; }

    public int Page { get; set; } = 1;
    public int PageSize { get; set; } = 50;
}
