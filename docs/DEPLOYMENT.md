# QuantAgentLab — Deployment Guide

> Target environment: EU VPS running Ubuntu 22.04 LTS | Docker + docker-compose

---

## 1. VPS Specifications

Minimum recommended specifications for running the full stack (Qlib + LangGraph + Streamlit):

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 2 GB | **4 GB** |
| vCPU | 1 | **2** |
| Storage | 20 GB | **50 GB SSD NVMe** |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Network | 100 Mbps | 1 Gbps |
| Swap | 2 GB | 4 GB |

The daily pipeline processes up to 5,000 equity symbols through Qlib, which peaks at ~2 GB
resident memory. The Streamlit dashboard adds ~200 MB. 4 GB RAM with 4 GB swap is the safe
operating point.

---

## 2. Provider Comparison

| Provider | Plan | vCPU | RAM | Storage | Monthly Cost | Notes |
|---|---|---|---|---|---|---|
| **Hetzner** | CX22 | 2 | 4 GB | 40 GB NVMe | **€4.15** | Best price/performance; EU datacentre |
| DigitalOcean | Basic | 1 | 2 GB | 50 GB SSD | $6.00 | Well-documented; good API |
| OVH | VPS Starter | 1 | 2 GB | 20 GB NVMe | €3.59 | Cheapest; limited support |
| Linode (Akamai) | Nanode | 1 | 1 GB | 25 GB SSD | $5.00 | Only 1 GB RAM; not recommended |
| Vultr | Cloud Compute | 1 | 2 GB | 55 GB NVMe | $6.00 | Good performance; US/EU locations |

**Recommendation: Hetzner CX22** — best price/performance ratio in the EU, NVMe storage,
Falkenstein (Germany) or Helsinki datacentre options, hourly billing so you can scale up
during backtesting and scale back down.

---

## 3. Deployment Steps

### Step 1: Create the VPS

1. Log in to [console.hetzner.cloud](https://console.hetzner.cloud)
2. Create new server: **CX22**, Ubuntu 22.04, Falkenstein DC (or Helsinki for lower latency
   to US markets)
3. In the SSH Keys section, paste your public key (`~/.ssh/id_ed25519.pub`)
4. Note the server's public IPv4 address (referred to as `SERVER_IP` below)

```bash
# On your local machine — verify SSH access
ssh root@SERVER_IP
```

### Step 2: System Update

```bash
apt update && apt upgrade -y
apt install -y git curl wget vim htop unzip logrotate fail2ban ufw
```

### Step 3: Install Docker and docker-compose

```bash
# Install Docker via official script
curl -fsSL https://get.docker.com | sh

# Add a non-root user (created in Step 7)
usermod -aG docker aiquant

# Install docker-compose v2
apt install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

### Step 4: Clone the Repository

```bash
cd /opt
git clone https://github.com/youruser/QuantAgentLab
cd QuantAgentLab
```

### Step 5: Create the Environment File

```bash
# Copy the template
cp config/.env.example config/.env

# Edit with your actual API keys
vim config/.env
```

At minimum, set:

```bash
AIQUANT_ANTHROPIC_API_KEY=sk-ant-...
AIQUANT_TELEGRAM_TOKEN=1234567890:AAF...
AIQUANT_TELEGRAM_CHAT_ID=-100123456789
```

The `.env` file must never be committed to git. Verify with:

```bash
grep -q ".env" .gitignore && echo "OK: .env is git-ignored" || echo "WARNING: .env not in .gitignore"
```

### Step 6: Start Services

```bash
# Build and start all services in detached mode
docker compose up -d --build

# Follow logs during first startup
docker compose logs -f --tail=50
```

Expected services after startup:

- `aiquant-pipeline` — daily pipeline runner
- `aiquant-dashboard` — Streamlit on port 8501
- `aiquant-scheduler` — APScheduler daemon

### Step 7: Verify Services

```bash
# Check all containers are healthy
docker compose ps

# Expected output:
# NAME                   STATUS          PORTS
# aiquant-pipeline       Up              -
# aiquant-dashboard      Up              0.0.0.0:8501->8501/tcp
# aiquant-scheduler      Up              -

# Run a quick health check
docker compose exec aiquant-pipeline aiquant monitor
```

If the dashboard is accessible at `http://SERVER_IP:8501` the deployment is successful.

---

## 4. Log Rotation

By default, loguru writes to `logs/aiquant_YYYY-MM-DD.log` with 10 MB file rotation and 7-day
retention (configured in `src/utils/logger.py`). For system-level rotation, also create a
logrotate config:

```bash
cat > /etc/logrotate.d/aiquant << 'EOF'
/opt/QuantAgentLab/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 root root
    sharedscripts
    postrotate
        docker compose -f /opt/QuantAgentLab/docker-compose.yml kill -s HUP aiquant-pipeline 2>/dev/null || true
    endscript
}
EOF

# Test the config
logrotate --debug /etc/logrotate.d/aiquant
```

---

## 5. Systemd Watchdog

Create a systemd unit so Docker and the services restart automatically on server reboot:

```bash
cat > /etc/systemd/system/aiquant.service << 'EOF'
[Unit]
Description=QuantAgentLab docker-compose stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/QuantAgentLab
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable aiquant.service
systemctl start aiquant.service

# Verify
systemctl status aiquant.service
```

Test the watchdog by rebooting the server and verifying containers come back up automatically:

```bash
reboot
# (reconnect after ~60 seconds)
docker compose ps
```

---

## 6. Monitoring Cron

An hourly health-check cron job sends a Telegram alert if any container is down:

```bash
cat > /opt/QuantAgentLab/scripts/health_check.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/QuantAgentLab

FAILED=$(docker compose ps --status exited --quiet 2>/dev/null | wc -l)
if [ "$FAILED" -gt 0 ]; then
    TOKEN=$(grep AIQUANT_TELEGRAM_TOKEN config/.env | cut -d= -f2)
    CHAT=$(grep AIQUANT_TELEGRAM_CHAT_ID config/.env | cut -d= -f2)
    MSG="[QuantAgentLab] ALERT: $FAILED container(s) exited on $(hostname). Check: docker compose ps"
    curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
         -d chat_id="$CHAT" \
         -d text="$MSG" > /dev/null
    docker compose up -d  # attempt auto-restart
fi
EOF

chmod +x /opt/QuantAgentLab/scripts/health_check.sh

# Add to crontab (run every hour)
(crontab -l 2>/dev/null; echo "0 * * * * /opt/QuantAgentLab/scripts/health_check.sh >> /var/log/aiquant_cron.log 2>&1") | crontab -

# Verify cron entry
crontab -l
```

---

## 7. Security Hardening

### Create a Non-Root User

```bash
useradd -m -s /bin/bash aiquant
usermod -aG docker aiquant
usermod -aG sudo aiquant

# Copy SSH keys from root
mkdir -p /home/aiquant/.ssh
cp ~/.ssh/authorized_keys /home/aiquant/.ssh/
chown -R aiquant:aiquant /home/aiquant/.ssh
chmod 700 /home/aiquant/.ssh
chmod 600 /home/aiquant/.ssh/authorized_keys

# Transfer ownership of the project
chown -R aiquant:aiquant /opt/QuantAgentLab
```

### UFW Firewall

```bash
# Set default policies
ufw default deny incoming
ufw default allow outgoing

# Allow SSH (do this BEFORE enabling to avoid locking yourself out)
ufw allow 22/tcp comment "SSH"

# Allow Streamlit dashboard (consider restricting to your IP in production)
ufw allow 8501/tcp comment "Streamlit dashboard"

# Enable firewall
ufw --force enable
ufw status verbose
```

If you add an nginx reverse proxy with SSL, also allow 443:

```bash
ufw allow 443/tcp comment "HTTPS reverse proxy"
ufw deny 8501/tcp  # close direct Streamlit port once behind nginx
```

### Fail2ban for SSH

```bash
cat > /etc/fail2ban/jail.local << 'EOF'
[sshd]
enabled = true
port = ssh
filter = sshd
maxretry = 5
bantime = 3600
findtime = 600
EOF

systemctl restart fail2ban
fail2ban-client status sshd
```

### Disable Root SSH Login

```bash
sed -i 's/^PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#PermitRootLogin/PermitRootLogin no #/' /etc/ssh/sshd_config
systemctl restart sshd
```

Make sure you can log in as the `aiquant` user before disconnecting.

### Automatic Security Updates

```bash
apt install -y unattended-upgrades
dpkg-reconfigure --priority=low unattended-upgrades
```

---

## 8. Cost Estimate

Monthly total cost breakdown for a production deployment:

| Item | Provider | Cost/month |
|---|---|---|
| VPS (Hetzner CX22) | Hetzner | €4.15 |
| Anthropic API (claude-sonnet-4-6 at €2/day cap) | Anthropic | ~€5–10 |
| FinnHub API | FinnHub | €0 (free tier) |
| SEC EDGAR | EDGAR | €0 (free) |
| CoinGecko | CoinGecko | €0 (free tier) |
| Backups (Hetzner automatic) | Hetzner | €0.83 (20% of VPS cost) |
| Domain (optional, for dashboard) | Namecheap | ~€1/month |
| **Total (low estimate)** | | **~€10–12/month** |
| **Total (high estimate)** | | **~€15–18/month** |

Cost is dominated by LLM API usage. Running 20 tickers/day with claude-sonnet-4-6 at 2 debate
rounds costs approximately $1.50–2.50/day. Setting `AIQUANT_DAILY_BUDGET_USD=2.0` caps this.

To reduce further, switch to `claude-haiku-3` for sentiment/technical nodes
(`AIQUANT_LLM_MODEL=claude-haiku-3`) and keep claude-sonnet-4-6 only for the strategist. This
typically reduces cost by 60–70% with minimal quality degradation.

---

## 9. Troubleshooting

### Container exits immediately on startup

```bash
# Check logs of the failing container
docker compose logs aiquant-pipeline --tail=100

# Common causes:
# - Missing API key in config/.env
# - Qlib data directory not initialised (run aiquant setup)
# - Port 8501 already in use
```

### Out of memory (OOM kill)

```bash
# Check if containers were OOM-killed
dmesg | grep -i "out of memory"
docker stats --no-stream

# Fix: add swap space
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### Pipeline runs but produces no signals

```bash
# Check Qlib data availability
docker compose exec aiquant-pipeline python -c "
from src.core.data_pipeline import DataPipeline
dp = DataPipeline()
print(dp.get_features(['AAPL'], '2025-01-01', '2025-03-01').head())
"

# If Qlib returns empty, yfinance fallback should activate.
# Check logs for: "Falling back to yfinance"
```

### Telegram alerts not arriving

```bash
# Test the bot directly
TOKEN="your-token"
CHAT="your-chat-id"
curl -s "https://api.telegram.org/bot${TOKEN}/getMe"
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
     -d chat_id="$CHAT" -d text="test from server"
```

### Disk fills up

```bash
# Check disk usage
df -h
du -sh /opt/QuantAgentLab/outputs/*
du -sh /opt/QuantAgentLab/logs/*

# Docker layer cache
docker system df
docker system prune --volumes  # WARNING: removes stopped containers and unused volumes

# Rotate logs immediately
logrotate --force /etc/logrotate.d/aiquant
```

### Clock drift (market data timestamps wrong)

```bash
# Install and enable NTP
apt install -y chrony
systemctl enable chrony
chronyc tracking
```

### Upgrading the application

```bash
cd /opt/QuantAgentLab
git pull origin master
docker compose build --no-cache
docker compose up -d
docker compose logs -f --tail=50
```

Always run `pytest` in the container before deploying a new version to production:

```bash
docker compose exec aiquant-pipeline python -m pytest tests/ -q --ignore=tests/test_integration.py
```
