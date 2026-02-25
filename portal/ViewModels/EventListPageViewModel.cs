namespace Portal.Web.ViewModels;

public sealed class EventListPageViewModel
{
    public string? SiteId { get; set; }
    public string? CameraId { get; set; }
    public DateTime? Date { get; set; }
    public DateTime? DateFrom { get; set; }
    public DateTime? DateTo { get; set; }
    public string? Direction { get; set; }
    public string? ClassName { get; set; }
    public string? ReviewStatus { get; set; }

    public int Page { get; set; }
    public int PageSize { get; set; }
    public int Total { get; set; }
    public int TotalPages { get; set; }

    public IReadOnlyList<EventListItemViewModel> Items { get; set; } = Array.Empty<EventListItemViewModel>();
}
