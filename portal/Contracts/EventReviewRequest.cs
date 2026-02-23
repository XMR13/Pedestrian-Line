namespace Portal.Web.Contracts;

public sealed class EventReviewRequest
{
    public bool? Qualified { get; set; }
    public string? ReviewStatus { get; set; }
    public string? Notes { get; set; }
}
