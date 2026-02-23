namespace Portal.Web.Infrastructure;

public sealed class PortalOptions
{
    public string ApiKey { get; set; } = string.Empty;
    public string EvidenceRootPath { get; set; } = "evidence";
    public int DefaultPageSize { get; set; } = 50;
    public int MaxPageSize { get; set; } = 200;
}
