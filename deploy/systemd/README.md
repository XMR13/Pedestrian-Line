# Jetson systemd examples

These files are tracked examples for the deployment shape documented in
`docs/jetson_deployment_runbook.md`.

Do not commit the real Jetson environment files. Keep camera URLs, local IPs,
passwords, and API keys in `/etc/vehicle_count/*.env` on the Jetson.

Recommended install layout:

```text
/etc/vehicle_count/single_loop.env
/etc/vehicle_count/edge_service.env
/etc/systemd/system/single_loop.service
/etc/systemd/system/edge_service.service
```

Suggested copy flow on the Jetson:

```bash
sudo mkdir -p /etc/vehicle_count
sudo cp deploy/systemd/pedestrian-single-loop.env.example /etc/vehicle_count/single_loop.env
sudo cp deploy/systemd/pedestrian-edge-service.env.example /etc/vehicle_count/edge_service.env
sudo cp deploy/systemd/pedestrian-single-loop.service.example /etc/systemd/system/single_loop.service
sudo cp deploy/systemd/pedestrian-edge-service.service.example /etc/systemd/system/edge_service.service
sudoedit /etc/vehicle_count/single_loop.env
sudoedit /etc/vehicle_count/edge_service.env
sudo systemctl daemon-reload
sudo systemctl enable --now edge_service.service
sudo systemctl enable --now single_loop.service
```

Check runtime logs:

```bash
journalctl -u single_loop.service -f
journalctl -u edge_service.service -f
```
