namespace Portal.Web.Infrastructure;

public sealed class LoginGateOptions
{
    public string Username { get; set; } = "admin";
    public string Password { get; set; } = "admin123";
    public string DisplayName { get; set; } = "Portal Reviewer";
}
