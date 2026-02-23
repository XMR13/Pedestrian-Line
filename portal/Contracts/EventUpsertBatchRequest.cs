namespace Portal.Web.Contracts;

public sealed class EventUpsertBatchRequest
{
    public string? ContractVersion { get; set; }
    public List<EventUpsertItemRequest> Events { get; set; } = new();
}
