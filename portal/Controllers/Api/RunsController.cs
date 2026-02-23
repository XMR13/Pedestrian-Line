using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Portal.Web.Contracts;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using Portal.Web.Models;

namespace Portal.Web.Controllers.Api;

[ApiController]
[Route("api/runs")]
public sealed class RunsController(PortalDbContext db) : ControllerBase
{
    [HttpPost("upsert")]
    [ApiKeyAuthorize]
    public async Task<IActionResult> Upsert([FromBody] RunUpsertRequest request, CancellationToken ct)
    {
        if (!IsContractValid(request.ContractVersion))
        {
            return BadRequest(new { error = "unsupported_contract_version" });
        }

        var runUid = request.RunUid?.Trim();
        var siteId = request.SiteId?.Trim();
        var cameraId = request.CameraId?.Trim();

        if (string.IsNullOrWhiteSpace(runUid) || string.IsNullOrWhiteSpace(siteId) || string.IsNullOrWhiteSpace(cameraId))
        {
            return BadRequest(new { error = "missing_required_fields", required = new[] { "run_uid", "site_id", "camera_id" } });
        }

        var now = DateTimeOffset.UtcNow;
        var row = await db.Runs.FirstOrDefaultAsync(x => x.RunUid == runUid, ct);
        if (row is null)
        {
            row = new RunRecord
            {
                RunUid = runUid,
                SiteId = siteId,
                CameraId = cameraId,
            };
            db.Runs.Add(row);
        }

        row.SiteId = siteId;
        row.CameraId = cameraId;
        row.StartedAtUtc = request.StartedAtUtc;
        row.EndedAtUtc = request.EndedAtUtc;
        row.SourceType = EmptyToNull(request.SourceType);
        row.SourceValue = EmptyToNull(request.SourceValue);
        row.ModelVersion = EmptyToNull(request.ModelVersion);
        row.CfgVersion = EmptyToNull(request.CfgVersion);
        row.LineMode = EmptyToNull(request.LineMode);
        row.LineId = EmptyToNull(request.LineId);
        row.Fps = request.Fps;
        row.FrameWidth = request.FrameWidth;
        row.FrameHeight = request.FrameHeight;
        row.HealthSummaryJson = request.HealthSummaryJson is null
            ? null
            : JsonSerializer.Serialize(request.HealthSummaryJson);
        row.ReportCsvRelpath = EmptyToNull(request.ReportCsvRelpath);
        row.UpdatedAtUtc = now;

        await db.SaveChangesAsync(ct);

        return Ok(new
        {
            status = "ok",
            run_uid = row.RunUid,
            updated_at_utc = row.UpdatedAtUtc,
        });
    }

    private static bool IsContractValid(string? contractVersion)
    {
        var value = (contractVersion ?? string.Empty).Trim();
        return string.Equals(value, "v1", StringComparison.OrdinalIgnoreCase);
    }

    private static string? EmptyToNull(string? value)
    {
        var trimmed = value?.Trim();
        return string.IsNullOrWhiteSpace(trimmed) ? null : trimmed;
    }
}
