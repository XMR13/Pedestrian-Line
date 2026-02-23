using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.EntityFrameworkCore;
using Portal.Web.Data;
using Portal.Web.Infrastructure;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddControllersWithViews()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
        options.JsonSerializerOptions.DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower;
        options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
    });
builder.Services.AddHttpContextAccessor();

builder.Services.Configure<PortalOptions>(builder.Configuration.GetSection("Portal"));
builder.Services.Configure<LoginGateOptions>(builder.Configuration.GetSection("LoginGate"));

builder.Services
    .AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme)
    .AddCookie(options =>
    {
        options.LoginPath = "/Account/Login";
        options.AccessDeniedPath = "/Account/Login";
        options.SlidingExpiration = true;
        options.ExpireTimeSpan = TimeSpan.FromHours(8);
    });

builder.Services.AddAuthorization();

var dbProvider = (builder.Configuration["Database:Provider"] ?? "Sqlite").Trim();
var useSqlite = string.Equals(dbProvider, "Sqlite", StringComparison.OrdinalIgnoreCase);
var connectionString = builder.Configuration.GetConnectionString("PortalDb")
    ?? (useSqlite ? "Data Source=portal.db" : throw new InvalidOperationException("Missing ConnectionStrings:PortalDb"));

builder.Services.AddDbContext<PortalDbContext>(options =>
{
    if (useSqlite)
    {
        options.UseSqlite(connectionString);
        return;
    }

    options.UseSqlServer(connectionString);
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();
app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<PortalDbContext>();
    if (useSqlite)
    {
        db.Database.EnsureCreated();
    }
    else
    {
        db.Database.Migrate();
    }
}

app.Run();
