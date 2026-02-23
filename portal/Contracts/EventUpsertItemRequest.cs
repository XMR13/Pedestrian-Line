namespace Portal.Web.Contracts;

public sealed class EventUpsertItemRequest
{
    public string? ContractVersion { get; set; }
    public string? EventUid { get; set; }
    public string? RunUid { get; set; }
    public string? SiteId { get; set; }
    public string? CameraId { get; set; }

    public DateTimeOffset? OccurredAtUtc { get; set; }
    public int? FrameIndex { get; set; }
    public double? VideoTimeS { get; set; }

    public string? Direction { get; set; }
    public int? TrackId { get; set; }

    public int? ClassId { get; set; }
    public string? ClassName { get; set; }
    public double? Confidence { get; set; }
    public List<int>? BboxXyxy { get; set; }

    public string? LineMode { get; set; }
    public string? OccurredAtUtcSource { get; set; }

    public string? ThumbRelpath { get; set; }
    public string? SceneRelpath { get; set; }
}
