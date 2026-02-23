namespace Portal.Web.Contracts;

public sealed class RunUpsertRequest
{
    public string? ContractVersion { get; set; }
    public string? RunUid { get; set; }
    public string? SiteId { get; set; }
    public string? CameraId { get; set; }

    public DateTimeOffset? StartedAtUtc { get; set; }
    public DateTimeOffset? EndedAtUtc { get; set; }

    public string? SourceType { get; set; }
    public string? SourceValue { get; set; }

    public string? ModelVersion { get; set; }
    public string? CfgVersion { get; set; }

    public string? LineMode { get; set; }
    public string? LineId { get; set; }

    public double? Fps { get; set; }
    public int? FrameWidth { get; set; }
    public int? FrameHeight { get; set; }

    public object? HealthSummaryJson { get; set; }
    public string? ReportCsvRelpath { get; set; }
}
