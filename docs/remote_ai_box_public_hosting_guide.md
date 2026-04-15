# Remote AI Box Public Hosting Guide

This guide explains how to host the project website directly on the **remote AI box**
so users in the office can access it over the internet.

This guide matches the real project flow:

- the **AI box** and **camera** are installed at a remote site,
- the remote site has its own modem/router and internet connection,
- the AI box runs the detector pipeline and saves results locally,
- the AI box also runs the local FastAPI operator service that reads those local results,
- office users connect remotely through a public domain.

The goal is to make that setup work in a safer and more production-friendly way.

## 1. What We Are Building

We are **not** exposing FastAPI directly to the internet.

We are building this flow instead:

1. `pedestrian_line_counter.main` runs on the AI box.
2. It writes spool data and evidence locally on the AI box.
3. `pedestrian_line_counter.service` runs on the same AI box.
4. That FastAPI service reads the local spool and serves the dashboard/review pages.
5. A reverse proxy on the AI box handles public HTTPS traffic.
6. The remote modem/router forwards public traffic to the AI box.
7. Office users open a public domain and reach the website remotely.

## 2. Final Architecture

```text
Office Users
    |
    |  https://traffic.yourdomain.com
    v
Internet
    |
    v
Remote Site Public IP
    |
    v
Remote Site Modem/Router
    |  Port forward 80/443
    v
AI Box
    |
    |-- Caddy/Nginx on :80 and :443
    |       |
    |       v
    |   FastAPI on 127.0.0.1:8080
    |       |
    |       v
    |   Reads local spool data
    |
    |-- Detector pipeline writes spool locally
```

## 3. Important Networking Rule

Because the AI box is located at the remote site, the required port forwarding is
done on the **remote site modem/router**, not the office router.

The office router is irrelevant for inbound access to the remote AI box.

If users in the office want to open the site, traffic must go to:

- the public IP of the **remote site**,
- then the remote modem/router forwards to the AI box.

## 4. Before You Start

You should have:

- SSH or terminal access to the AI box,
- admin access to the **remote site modem/router**,
- a domain name or subdomain you can manage,
- the AI box already running the project locally,
- a known spool directory,
- a known AI box local IP at the remote site,
- permission to open ports on the remote site network.

Example values used in this guide:

- Domain: `traffic.example.com`
- AI box LAN IP: `192.168.1.50`
- FastAPI local port: `8080`
- Public HTTPS port: `443`
- Public HTTP port: `80`

Replace them with your real values.

## 5. Step 1: Check Whether Public Exposure Is Possible

Before touching DNS or port forwarding, verify the remote site internet connection.

### 5.1 Check the public IP seen from the AI box

Run on the AI box:

```bash
curl ifconfig.me
```

Or:

```bash
curl https://ifconfig.me
```

Write down the IP returned.

### 5.2 Check the WAN IP shown by the remote modem/router

Open the modem/router admin page and find its WAN/public IP.

### 5.3 Compare the two IPs

If the IP from the AI box and the WAN IP on the modem/router are the same:

- normal port forwarding should work.

If they are different:

- the site may be behind CGNAT or upstream NAT,
- normal port forwarding may fail,
- you may need:
  - a public/static IP from the ISP, or
  - a tunnel/VPN solution later.

Do not skip this check. It prevents a lot of wasted time.

## 6. Step 2: Give the AI Box a Stable Local IP

The router must always know where to forward traffic.

The AI box should therefore keep the same LAN IP.

Recommended approach:

- create a DHCP reservation in the remote modem/router for the AI box MAC address.

Example:

- AI box always gets `192.168.1.50`

Alternative:

- configure a static IP on the AI box itself.

DHCP reservation is usually easier and safer.

## 7. Step 3: Keep FastAPI Private on the AI Box

The FastAPI app should **not** bind to `0.0.0.0` for public access.

It should bind only to:

- `127.0.0.1:8080`

That way:

- FastAPI is reachable only inside the AI box itself,
- the reverse proxy becomes the only public entry point,
- you avoid exposing Uvicorn/FastAPI directly.

### 7.1 Recommended environment file

Create or update:

`/etc/vehicle_count/edge_service.env`

Example:

```env
PLC_SPOOL_DIR=/var/lib/pedline/traffic_runs

PLC_SERVICE_HOST=127.0.0.1
PLC_SERVICE_PORT=8080
PLC_SERVICE_EXPOSURE=loopback
PLC_SERVICE_DOCS=0

EDGE_UI_USERNAME=admin
EDGE_UI_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD
EDGE_SERVICE_API_KEY=CHANGE_THIS_TO_A_LONG_RANDOM_KEY

PLC_SERVICE_RETENTION_ENABLED=1
PLC_SERVICE_RETENTION_MAX_AGE_DAYS=90
PLC_SERVICE_RETENTION_AUTO_INTERVAL_S=3600
```

### 7.2 Restart the service

```bash
sudo systemctl restart edge_service.service
sudo systemctl status edge_service.service
```

### 7.3 Verify FastAPI is listening locally

Run:

```bash
curl http://127.0.0.1:8080/healthz
```

You should get a healthy JSON response.

If this fails, do not continue to public hosting yet.

## 8. Step 4: Install the Reverse Proxy on the AI Box

Use a reverse proxy to serve HTTPS publicly.

Recommended choice:

- **Caddy**

Why Caddy:

- simpler HTTPS setup,
- automatic certificate management,
- less manual TLS configuration,
- good fit for a single public host.

Nginx is also valid, but Caddy is easier for this deployment.

### 8.1 Install Caddy on Debian/Ubuntu

```bash
sudo apt update
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install -y caddy
```

### 8.2 Confirm Caddy is installed

```bash
caddy version
sudo systemctl status caddy
```

## 9. Step 5: Configure Caddy

Edit:

`/etc/caddy/Caddyfile`

Example:

```caddy
traffic.example.com {
    encode gzip

    reverse_proxy 127.0.0.1:8080

    header {
        X-Frame-Options "DENY"
        X-Content-Type-Options "nosniff"
        Referrer-Policy "strict-origin-when-cross-origin"
    }
}
```

### 9.1 Reload Caddy

```bash
sudo systemctl reload caddy
sudo systemctl status caddy
```

### 9.2 What this does

This configuration means:

- public users connect to `https://traffic.example.com`,
- Caddy listens on ports `80` and `443`,
- Caddy handles TLS certificates,
- Caddy forwards traffic internally to `127.0.0.1:8080`.

FastAPI stays private.

## 10. Step 6: Point the Domain to the Remote Site

In your DNS provider, create an `A` record:

- Name: `traffic`
- Value: the **remote site public IP**

Example:

- `traffic.example.com -> 36.88.123.45`

### 10.1 Verify DNS

From any machine:

```bash
nslookup traffic.example.com
```

or:

```bash
dig traffic.example.com
```

The result should match the remote site public IP.

## 11. Step 7: Configure Port Forwarding on the Remote Site Modem/Router

This step is done on the **remote site modem/router**.

Create port forward rules:

- WAN `80` -> `192.168.1.50:80`
- WAN `443` -> `192.168.1.50:443`

### Important

Do **not** forward:

- `8080`
- raw FastAPI
- raw Uvicorn
- any unnecessary internal service

Only expose:

- `80` for HTTP challenge/redirect
- `443` for HTTPS

## 12. Step 8: Open the Local Firewall on the AI Box

If the AI box uses `ufw`:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

If another firewall is used, allow inbound:

- TCP `80`
- TCP `443`

## 13. Step 9: Test in the Correct Order

Test from smallest scope to largest scope.

### 13.1 Test FastAPI locally on the AI box

```bash
curl http://127.0.0.1:8080/healthz
```

### 13.2 Test Caddy locally on the AI box

```bash
curl -I http://127.0.0.1
```

If DNS is already ready:

```bash
curl -I https://traffic.example.com
```

### 13.3 Test from the remote site LAN

From a laptop or phone on the same remote-site network, open:

- `https://traffic.example.com`

### 13.4 Test from outside the remote site

Use:

- office network, or
- mobile hotspot

Open:

- `https://traffic.example.com`

This confirms real public access.

## 14. Step 10: Verify the Core Application Flow

Once the website is reachable, make sure the actual project flow still works:

1. detector pipeline writes spool data,
2. FastAPI reads that spool data,
3. dashboard shows recent runs/events,
4. event detail page can show evidence,
5. review actions work,
6. remote users can log in and browse the UI.

This is the real success condition for your project, not just getting a page to open.

## 15. Daily Operations Checklist

When the remote deployment is live, these are the basic checks:

### Check detector service

```bash
sudo systemctl status single_loop.service
journalctl -u single_loop.service -f
```

### Check FastAPI service

```bash
sudo systemctl status edge_service.service
journalctl -u edge_service.service -f
```

### Check Caddy

```bash
sudo systemctl status caddy
journalctl -u caddy -f
```

### Check spool growth

```bash
du -sh /var/lib/pedline/traffic_runs
```

### Check local health endpoint

```bash
curl http://127.0.0.1:8080/healthz
```

## 16. Troubleshooting Guide

### Problem: The domain does not resolve

Check:

```bash
nslookup traffic.example.com
```

Likely causes:

- wrong DNS record,
- DNS not propagated yet,
- typo in domain name.

### Problem: FastAPI works locally but public access fails

Likely causes:

- router port forwarding not configured,
- wrong AI box LAN IP in router,
- firewall blocking ports `80/443`,
- Caddy not listening properly.

### Problem: Port forwarding is configured but site still unreachable

Likely causes:

- CGNAT or ISP NAT,
- ISP blocks inbound connections,
- router WAN IP is not truly public.

Check by comparing:

- router WAN IP,
- `curl ifconfig.me`

If they differ, normal port forwarding may not work.

### Problem: HTTPS certificate is not issued

Likely causes:

- DNS points to wrong IP,
- ports `80/443` are not reachable from the internet,
- Caddy cannot complete ACME challenge.

### Problem: FastAPI is public on `8080`

This is misconfiguration.

Fix:

- set FastAPI host to `127.0.0.1`,
- remove any router forwarding to `8080`,
- use only Caddy on `80/443`.

## 17. Security Guidance for This Setup

Hosting directly on the AI box is possible, but you should follow these rules:

### Always do this

- keep FastAPI on `127.0.0.1`,
- expose only `80/443`,
- use a reverse proxy,
- use strong passwords,
- keep the OS updated,
- monitor logs,
- keep SSH locked down.

### Avoid this

- exposing `8080` directly,
- exposing raw Uvicorn,
- forwarding many unnecessary ports,
- using weak passwords,
- leaving debug/docs endpoints publicly enabled.

## 18. Current Codebase Hardening Gaps

This repo is close to a workable operator-facing system, but before public internet
exposure you should assume it still needs hardening.

Important codebase concerns to address:

- the current evidence serving path should be reviewed carefully before public launch,
- read endpoints should not stay broadly open for internet users,
- session cookies should be hardened for HTTPS-only deployment,
- review/auth flows should get stronger protection,
- security headers and request protections should be improved.

That means:

- this hosting guide is the correct infrastructure path,
- but the app itself should still be hardened before full public rollout.

## 19. Recommended Production Shape

On the remote AI box:

- `single_loop.service`
  - detector pipeline
  - writes spool locally

- `edge_service.service`
  - FastAPI dashboard/review/event UI
  - reads spool locally
  - binds to `127.0.0.1:8080`

- `caddy`
  - public HTTPS entry point
  - reverse proxies to FastAPI

On the remote modem/router:

- port forward `80` and `443` to the AI box

For office users:

- open `https://traffic.example.com`

## 20. Deployment Summary

If you want this project hosted directly from the AI box, the correct setup is:

1. run detector on the AI box,
2. run FastAPI on the AI box,
3. keep FastAPI local-only,
4. install Caddy on the AI box,
5. point DNS to the remote site public IP,
6. port forward on the **remote site modem/router**,
7. expose only `80/443`,
8. test from outside the remote site.

That is the right infrastructure shape for your current project flow.

## 21. Next Recommended Work

After this document, the practical next steps are:

1. prepare the real `edge_service.env` for your AI box,
2. prepare the real `Caddyfile` using your actual domain,
3. verify whether the remote site internet supports port forwarding,
4. harden the FastAPI app for safer public use,
5. test access from the office.
