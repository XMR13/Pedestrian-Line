using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.Extensions.Options;

namespace Portal.Web.Infrastructure;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
public sealed class ApiKeyAuthorizeAttribute : Attribute, IAsyncActionFilter
{
    private const string ApiKeyHeader = "X-API-Key";

    public async Task OnActionExecutionAsync(ActionExecutingContext context, ActionExecutionDelegate next)
    {
        var options = context.HttpContext.RequestServices
            .GetRequiredService<IOptions<PortalOptions>>()
            .Value;

        var configuredKey = options.ApiKey?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(configuredKey))
        {
            context.Result = new StatusCodeResult(StatusCodes.Status500InternalServerError);
            return;
        }

        var suppliedKey = context.HttpContext.Request.Headers[ApiKeyHeader].ToString().Trim();
        if (!FixedTimeEquals(suppliedKey, configuredKey))
        {
            context.Result = new UnauthorizedObjectResult(new
            {
                error = "invalid_api_key"
            });
            return;
        }

        await next();
    }

    private static bool FixedTimeEquals(string left, string right)
    {
        var leftBytes = System.Text.Encoding.UTF8.GetBytes(left ?? string.Empty);
        var rightBytes = System.Text.Encoding.UTF8.GetBytes(right ?? string.Empty);
        return System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(leftBytes, rightBytes);
    }
}
