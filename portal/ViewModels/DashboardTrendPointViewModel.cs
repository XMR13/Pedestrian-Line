namespace Portal.Web.ViewModels;

public sealed class DashboardTrendPointViewModel
{
    public DateTime BucketStartLocal { get; set; }
    public string Label { get; set; } = string.Empty;
    public string TooltipLabel { get; set; } = string.Empty;
    public int AToB { get; set; }
    public int BToA { get; set; }
    public int Reviewed { get; set; }
    public int Pending { get; set; }
}
