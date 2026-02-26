namespace Portal.Web.ViewModels;

public sealed class DashboardHeadlessStatusViewModel
{
    public string RunUid { get; set; } = string.Empty;
    public string SiteId { get; set; } = string.Empty;
    public string CameraId { get; set; } = string.Empty;

    public string LifecycleStatus { get; set; } = "UNKNOWN";
    public bool IsRunning { get; set; }
    public bool IsStale { get; set; }

    public DateTimeOffset? StartedAtUtc { get; set; }
    public DateTimeOffset? EndedAtUtc { get; set; }
    public DateTimeOffset? StatusUpdatedAtUtc { get; set; }
    public DateTimeOffset PortalUpdatedAtUtc { get; set; }

    public int? FramesTotal { get; set; }
    public int? FramesProcessed { get; set; }
    public int? EventsEmittedTotal { get; set; }
    public int? CountAToB { get; set; }
    public int? CountBToA { get; set; }
    public double? EffectiveFps { get; set; }
    public double? ProcessedFps { get; set; }
    public int? ReconnectCycles { get; set; }
    public int? ReaderDroppedFrames { get; set; }
    public string? QueuePolicy { get; set; }
    public int? QueueSize { get; set; }
    public DateTimeOffset? PortalUploadLastSuccessAtUtc { get; set; }
    public string? PortalUploadLastError { get; set; }
}
