namespace Portal.Web.Infrastructure;

public sealed class LoginGateOptions
{
    public string Username { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public string DisplayName { get; set; } = "Portal Reviewer";
}
