namespace Portal.Web.ViewModels;

public sealed class ReviewQueueViewModel
{
    public EventDetailViewModel? Current { get; set; }
    public IReadOnlyList<EventListItemViewModel> Upcoming { get; set; } = Array.Empty<EventListItemViewModel>();
    public int PendingCount { get; set; }
    public int QualifiedCount { get; set; }
    public int NotQualifiedCount { get; set; }
}
