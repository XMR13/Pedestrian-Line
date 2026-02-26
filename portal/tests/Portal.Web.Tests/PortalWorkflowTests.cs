using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Portal.Web.Data;
using Portal.Web.Models;
using Xunit;

namespace Portal.Web.Tests;

public sealed class PortalWorkflowTests
{
    private const string TestUsername = "reviewer";
    private const string TestPassword = "secret123";
    private const string TestDisplayName = "Test Reviewer";

    [Fact]
    public async Task Login_WithValidCredentials_RedirectsToDashboard()
    {
        await using var app = new PortalTestApp();
        await app.InitializeAsync();
        using var client = app.CreatePortalClient();

        var loginResponse = await PostLoginAsync(client);

        Assert.Equal(HttpStatusCode.Found, loginResponse.StatusCode);
        Assert.Equal("/", loginResponse.Headers.Location?.OriginalString);

        var dashboardResponse = await client.GetAsync("/");
        Assert.Equal(HttpStatusCode.OK, dashboardResponse.StatusCode);
    }

    [Fact]
    public async Task DashboardSummary_ReturnsExpectedTotals()
    {
        await using var app = new PortalTestApp();
        await app.InitializeAsync();
        using var client = app.CreatePortalClient();
        await PostLoginAsync(client);

        var response = await client.GetAsync("/api/dashboard/summary");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var totals = doc.RootElement.GetProperty("totals");
        var trend = doc.RootElement.GetProperty("trend");
        var headless = doc.RootElement.GetProperty("headless_status");

        Assert.Equal(2, totals.GetProperty("a_to_b").GetInt32());
        Assert.Equal(1, totals.GetProperty("b_to_a").GetInt32());
        Assert.Equal(1, totals.GetProperty("pending").GetInt32());
        Assert.Equal(2, totals.GetProperty("reviewed").GetInt32());
        Assert.Equal(1, totals.GetProperty("qualified").GetInt32());
        Assert.Equal(1, totals.GetProperty("not_qualified").GetInt32());

        Assert.Equal("day", trend.GetProperty("bucket").GetString());
        var points = trend.GetProperty("points").EnumerateArray().ToList();
        Assert.Equal(14, points.Count);
        Assert.Equal(2, points.Sum(x => x.GetProperty("a_to_b").GetInt32()));
        Assert.Equal(1, points.Sum(x => x.GetProperty("b_to_a").GetInt32()));
        Assert.Equal(2, points.Sum(x => x.GetProperty("reviewed").GetInt32()));
        Assert.Equal(1, points.Sum(x => x.GetProperty("pending").GetInt32()));

        Assert.Equal("run-001", headless.GetProperty("run_uid").GetString());
        Assert.Equal("STOPPED", headless.GetProperty("lifecycle_status").GetString());
        Assert.False(headless.GetProperty("is_running").GetBoolean());
        Assert.Equal(3, headless.GetProperty("events_emitted_total").GetInt32());
    }

    [Fact]
    public async Task EventsIndex_DateAndDateRangeFilters_ReturnExpectedTotals()
    {
        await using var app = new PortalTestApp();
        await app.InitializeAsync();
        using var client = app.CreatePortalClient();
        await PostLoginAsync(client);

        var seededEventLocalDate = DateTimeOffset.Parse("2026-02-23T00:00:00+00:00")
            .AddMinutes(-25)
            .ToLocalTime()
            .ToString("yyyy-MM-dd");

        var singleDateResponse = await client.GetAsync($"/Events?date={seededEventLocalDate}");
        Assert.Equal(HttpStatusCode.OK, singleDateResponse.StatusCode);
        var singleDateHtml = await singleDateResponse.Content.ReadAsStringAsync();
        Assert.Contains("3 total event(s) matched", singleDateHtml, StringComparison.Ordinal);

        var rangeResponse = await client.GetAsync($"/Events?dateFrom={seededEventLocalDate}&dateTo={seededEventLocalDate}");
        Assert.Equal(HttpStatusCode.OK, rangeResponse.StatusCode);
        var rangeHtml = await rangeResponse.Content.ReadAsStringAsync();
        Assert.Contains("3 total event(s) matched", rangeHtml, StringComparison.Ordinal);

        var missingDate = DateTime.Parse(seededEventLocalDate).AddDays(2).ToString("yyyy-MM-dd");
        var missingRangeResponse = await client.GetAsync($"/Events?dateFrom={missingDate}&dateTo={missingDate}");
        Assert.Equal(HttpStatusCode.OK, missingRangeResponse.StatusCode);
        var missingRangeHtml = await missingRangeResponse.Content.ReadAsStringAsync();
        Assert.Contains("0 total event(s) matched", missingRangeHtml, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ReviewQueue_SubmitQualifiedYes_UpdatesReviewAndRedirects()
    {
        await using var app = new PortalTestApp();
        await app.InitializeAsync();
        using var client = app.CreatePortalClient();
        await PostLoginAsync(client);

        var queueResponse = await client.GetAsync("/Events/ReviewQueue");
        Assert.Equal(HttpStatusCode.OK, queueResponse.StatusCode);
        var queueHtml = await queueResponse.Content.ReadAsStringAsync();

        var token = ExtractAntiforgeryToken(queueHtml);
        var eventUid = ExtractCurrentReviewEventUid(queueHtml);

        using var submitRequest = new HttpRequestMessage(HttpMethod.Post, $"/api/events/{eventUid}/review")
        {
            Content = new FormUrlEncodedContent(new Dictionary<string, string>
            {
                ["__RequestVerificationToken"] = token,
                ["reviewStatus"] = ReviewStatuses.Qualified,
                ["notes"] = "confirmed by reviewer",
                ["returnUrl"] = "/Events/ReviewQueue",
            }),
        };
        submitRequest.Headers.TryAddWithoutValidation("Accept", "text/html");

        var submitResponse = await client.SendAsync(submitRequest);

        Assert.Equal(HttpStatusCode.Found, submitResponse.StatusCode);
        Assert.Equal("/Events/ReviewQueue", submitResponse.Headers.Location?.OriginalString);

        await using (var scope = app.Services.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<PortalDbContext>();
            var review = await db.EventReviews.FirstAsync(x => x.EventUid == eventUid);
            Assert.Equal(ReviewStatuses.Qualified, review.ReviewStatus);
            Assert.Equal("confirmed by reviewer", review.Notes);
            Assert.Equal(TestDisplayName, review.ReviewedBy);
            Assert.NotNull(review.ReviewedAtUtc);
        }
    }

    [Fact]
    public async Task ExportCsv_IncludesReviewedOnly()
    {
        await using var app = new PortalTestApp();
        await app.InitializeAsync();
        using var client = app.CreatePortalClient();
        await PostLoginAsync(client);

        var exportResponse = await client.GetAsync("/Events/ExportCsv");

        Assert.Equal(HttpStatusCode.OK, exportResponse.StatusCode);
        Assert.Equal("text/csv", exportResponse.Content.Headers.ContentType?.MediaType);

        var csv = await exportResponse.Content.ReadAsStringAsync();
        Assert.Contains("event_uid,run_uid,site_id,camera_id", csv, StringComparison.Ordinal);
        Assert.Contains("evt-002", csv, StringComparison.Ordinal);
        Assert.Contains("evt-003", csv, StringComparison.Ordinal);
        Assert.DoesNotContain("evt-001", csv, StringComparison.Ordinal);
    }

    private static async Task<HttpResponseMessage> PostLoginAsync(HttpClient client, string returnUrl = "/")
    {
        var loginGetResponse = await client.GetAsync($"/Account/Login?returnUrl={Uri.EscapeDataString(returnUrl)}");
        Assert.Equal(HttpStatusCode.OK, loginGetResponse.StatusCode);

        var html = await loginGetResponse.Content.ReadAsStringAsync();
        var token = ExtractAntiforgeryToken(html);

        var form = new Dictionary<string, string>
        {
            ["__RequestVerificationToken"] = token,
            ["Username"] = TestUsername,
            ["Password"] = TestPassword,
            ["ReturnUrl"] = returnUrl,
        };

        return await client.PostAsync("/Account/Login", new FormUrlEncodedContent(form));
    }

    private static string ExtractAntiforgeryToken(string html)
    {
        var match = Regex.Match(
            html,
            "name=\"__RequestVerificationToken\"[^>]*value=\"([^\"]+)\"",
            RegexOptions.IgnoreCase);

        Assert.True(match.Success, "Antiforgery token was not found in HTML response.");
        return WebUtility.HtmlDecode(match.Groups[1].Value);
    }

    private static string ExtractCurrentReviewEventUid(string html)
    {
        var match = Regex.Match(html, "action=\"/api/events/([^\"/]+)/review\"", RegexOptions.IgnoreCase);
        Assert.True(match.Success, "Review form action with event uid was not found.");
        return WebUtility.HtmlDecode(match.Groups[1].Value);
    }

    private sealed class PortalTestApp : WebApplicationFactory<Program>
    {
        private readonly string _projectRoot = ResolvePortalProjectPath();
        private readonly string _dbPath = Path.Combine(Path.GetTempPath(), $"portal-tests-{Guid.NewGuid():N}.db");
        private readonly string _evidenceRoot = Path.Combine(Path.GetTempPath(), $"portal-evidence-{Guid.NewGuid():N}");
        private bool _disposed;

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.UseEnvironment("Development");
            builder.UseContentRoot(_projectRoot);
            builder.ConfigureAppConfiguration((_, configBuilder) =>
            {
                configBuilder.AddInMemoryCollection(new Dictionary<string, string?>
                {
                    ["Database:Provider"] = "Sqlite",
                    ["ConnectionStrings:PortalDb"] = $"Data Source={_dbPath}",
                    ["Portal:ApiKey"] = "test-api-key",
                    ["Portal:EvidenceRootPath"] = _evidenceRoot,
                    ["Portal:DefaultPageSize"] = "25",
                    ["Portal:MaxPageSize"] = "100",
                    ["LoginGate:Username"] = TestUsername,
                    ["LoginGate:Password"] = TestPassword,
                    ["LoginGate:DisplayName"] = TestDisplayName,
                });
            });
        }

        public HttpClient CreatePortalClient()
        {
            return CreateClient(new WebApplicationFactoryClientOptions
            {
                AllowAutoRedirect = false,
                BaseAddress = new Uri("https://localhost"),
            });
        }

        public async Task InitializeAsync()
        {
            await using var scope = Services.CreateAsyncScope();
            var db = scope.ServiceProvider.GetRequiredService<PortalDbContext>();
            await db.Database.EnsureDeletedAsync();
            await db.Database.EnsureCreatedAsync();
            SeedDatabase(db);
            await db.SaveChangesAsync();
        }

        public override async ValueTask DisposeAsync()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            await base.DisposeAsync();

            try
            {
                if (Directory.Exists(_evidenceRoot))
                {
                    Directory.Delete(_evidenceRoot, recursive: true);
                }

                foreach (var suffix in new[] { string.Empty, "-shm", "-wal" })
                {
                    var path = _dbPath + suffix;
                    if (File.Exists(path))
                    {
                        File.Delete(path);
                    }
                }
            }
            catch (IOException)
            {
                // Ignore cleanup issues from file locking on test failure.
            }
        }

        private static string ResolvePortalProjectPath()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null)
            {
                if (File.Exists(Path.Combine(dir.FullName, "Portal.Web.csproj")))
                {
                    return dir.FullName;
                }

                dir = dir.Parent;
            }

            throw new InvalidOperationException("Unable to locate portal project root (Portal.Web.csproj).");
        }

        private static void SeedDatabase(PortalDbContext db)
        {
            var now = DateTimeOffset.Parse("2026-02-23T00:00:00+00:00");

            db.Runs.Add(new RunRecord
            {
                RunUid = "run-001",
                SiteId = "subang",
                CameraId = "cam_01",
                StartedAtUtc = now.AddMinutes(-30),
                EndedAtUtc = now.AddMinutes(-1),
                HealthSummaryJson = JsonSerializer.Serialize(new
                {
                    lifecycle_status = "stopped",
                    status_updated_at_utc = now.ToString("O"),
                    frames_total = 1000,
                    frames_processed = 750,
                    events_emitted_total = 3,
                    count_a_to_b = 2,
                    count_b_to_a = 1,
                    effective_fps = 29.5,
                    processed_fps = 12.5,
                    reconnect_cycles = 1,
                    reader_dropped_frames = 4,
                    queue_policy = "drop_oldest",
                    queue_size = 3,
                    portal_upload_runtime = new
                    {
                        last_success_at_utc = now.ToString("O"),
                        last_error = (string?)null,
                    },
                }),
                UpdatedAtUtc = now,
            });

            db.Events.AddRange(
                new EventRecord
                {
                    EventUid = "evt-001",
                    RunUid = "run-001",
                    SiteId = "subang",
                    CameraId = "cam_01",
                    OccurredAtUtc = now.AddMinutes(-25),
                    Direction = "A_TO_B",
                    ClassName = "truck",
                    TrackId = 11,
                    UpdatedAtUtc = now,
                },
                new EventRecord
                {
                    EventUid = "evt-002",
                    RunUid = "run-001",
                    SiteId = "subang",
                    CameraId = "cam_01",
                    OccurredAtUtc = now.AddMinutes(-20),
                    Direction = "B_TO_A",
                    ClassName = "pickup",
                    TrackId = 12,
                    UpdatedAtUtc = now,
                },
                new EventRecord
                {
                    EventUid = "evt-003",
                    RunUid = "run-001",
                    SiteId = "subang",
                    CameraId = "cam_01",
                    OccurredAtUtc = now.AddMinutes(-15),
                    Direction = "A_TO_B",
                    ClassName = "bus",
                    TrackId = 13,
                    UpdatedAtUtc = now,
                });

            db.EventReviews.AddRange(
                new EventReview
                {
                    EventUid = "evt-001",
                    ReviewStatus = ReviewStatuses.Pending,
                    UpdatedAtUtc = now,
                },
                new EventReview
                {
                    EventUid = "evt-002",
                    ReviewStatus = ReviewStatuses.Qualified,
                    ReviewedAtUtc = now.AddMinutes(-5),
                    ReviewedBy = "seed-user",
                    Notes = "meets criteria",
                    UpdatedAtUtc = now,
                },
                new EventReview
                {
                    EventUid = "evt-003",
                    ReviewStatus = ReviewStatuses.NotQualified,
                    ReviewedAtUtc = now.AddMinutes(-4),
                    ReviewedBy = "seed-user",
                    Notes = "does not meet criteria",
                    UpdatedAtUtc = now,
                });
        }
    }
}
