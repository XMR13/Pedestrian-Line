using System.Text.Json;
using Portal.Web.Models;

namespace Portal.Web.Infrastructure;

public sealed class HeadlessStatusSnapshot
{
    public string RunUid { get; init; } = string.Empty;
    public string SiteId { get; init; } = string.Empty;
    public string CameraId { get; init; } = string.Empty;
    public string LifecycleStatus { get; init; } = "UNKNOWN";
    public bool IsRunning { get; init; }
    public bool IsStale { get; init; }

    public DateTimeOffset? StartedAtUtc { get; init; }
    public DateTimeOffset? EndedAtUtc { get; init; }
    public DateTimeOffset? StatusUpdatedAtUtc { get; init; }
    public DateTimeOffset PortalUpdatedAtUtc { get; init; }

    public int? FramesTotal { get; init; }
    public int? FramesProcessed { get; init; }
    public int? EventsEmittedTotal { get; init; }
    public int? CountAToB { get; init; }
    public int? CountBToA { get; init; }
    public double? EffectiveFps { get; init; }
    public double? ProcessedFps { get; init; }

    public int? ReconnectCycles { get; init; }
    public int? ReaderDroppedFrames { get; init; }
    public string? QueuePolicy { get; init; }
    public int? QueueSize { get; init; }

    public DateTimeOffset? PortalUploadLastSuccessAtUtc { get; init; }
    public string? PortalUploadLastError { get; init; }
}

public static class HeadlessStatusSnapshotMapper
{
    private static readonly TimeSpan RunningStaleThreshold = TimeSpan.FromMinutes(2);

    public static HeadlessStatusSnapshot? Build(RunRecord? run, DateTimeOffset nowUtc)
    {
        if (run is null)
        {
            return null;
        }

        using var doc = ParseJson(run.HealthSummaryJson);
        var health = doc?.RootElement;
        var lifecycleRaw = GetString(health, "lifecycle_status");
        var lifecycle = string.IsNullOrWhiteSpace(lifecycleRaw)
            ? (run.EndedAtUtc.HasValue ? "STOPPED" : "RUNNING")
            : lifecycleRaw!.Trim().ToUpperInvariant();
        var isRunning = string.Equals(lifecycle, "RUNNING", StringComparison.OrdinalIgnoreCase)
            || (!run.EndedAtUtc.HasValue && !string.Equals(lifecycle, "STOPPED", StringComparison.OrdinalIgnoreCase));

        var statusUpdatedAtUtc = GetDateTimeOffset(health, "status_updated_at_utc");
        var staleAnchor = statusUpdatedAtUtc ?? run.UpdatedAtUtc;
        var isStale = isRunning && (nowUtc - staleAnchor) > RunningStaleThreshold;

        var portalUpload = GetObject(health, "portal_upload_runtime");
        return new HeadlessStatusSnapshot
        {
            RunUid = run.RunUid,
            SiteId = run.SiteId,
            CameraId = run.CameraId,
            LifecycleStatus = lifecycle,
            IsRunning = isRunning,
            IsStale = isStale,
            StartedAtUtc = run.StartedAtUtc,
            EndedAtUtc = run.EndedAtUtc ?? GetDateTimeOffset(health, "ended_at_utc"),
            StatusUpdatedAtUtc = statusUpdatedAtUtc,
            PortalUpdatedAtUtc = run.UpdatedAtUtc,
            FramesTotal = GetInt(health, "frames_total"),
            FramesProcessed = GetInt(health, "frames_processed"),
            EventsEmittedTotal = GetInt(health, "events_emitted_total"),
            CountAToB = GetInt(health, "count_a_to_b"),
            CountBToA = GetInt(health, "count_b_to_a"),
            EffectiveFps = GetDouble(health, "effective_fps"),
            ProcessedFps = GetDouble(health, "processed_fps"),
            ReconnectCycles = GetInt(health, "reconnect_cycles"),
            ReaderDroppedFrames = GetInt(health, "reader_dropped_frames"),
            QueuePolicy = GetString(health, "queue_policy"),
            QueueSize = GetInt(health, "queue_size"),
            PortalUploadLastSuccessAtUtc = GetDateTimeOffset(portalUpload, "last_success_at_utc"),
            PortalUploadLastError = GetString(portalUpload, "last_error"),
        };
    }

    private static JsonDocument? ParseJson(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        try
        {
            return JsonDocument.Parse(value);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static JsonElement? GetObject(JsonElement? parent, string name)
    {
        if (parent is null || parent.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        return parent.Value.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.Object
            ? element
            : null;
    }

    private static string? GetString(JsonElement? parent, string name)
    {
        if (parent is null || parent.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!parent.Value.TryGetProperty(name, out var element))
        {
            return null;
        }

        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number => element.GetRawText(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => null,
        };
    }

    private static int? GetInt(JsonElement? parent, string name)
    {
        if (parent is null || parent.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!parent.Value.TryGetProperty(name, out var element))
        {
            return null;
        }

        if (element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out var n))
        {
            return n;
        }

        if (element.ValueKind == JsonValueKind.String && int.TryParse(element.GetString(), out var parsed))
        {
            return parsed;
        }

        return null;
    }

    private static double? GetDouble(JsonElement? parent, string name)
    {
        if (parent is null || parent.Value.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!parent.Value.TryGetProperty(name, out var element))
        {
            return null;
        }

        if (element.ValueKind == JsonValueKind.Number && element.TryGetDouble(out var n))
        {
            return n;
        }

        if (element.ValueKind == JsonValueKind.String && double.TryParse(element.GetString(), out var parsed))
        {
            return parsed;
        }

        return null;
    }

    private static DateTimeOffset? GetDateTimeOffset(JsonElement? parent, string name)
    {
        var value = GetString(parent, name);
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        return DateTimeOffset.TryParse(value, out var parsed) ? parsed : null;
    }
}
