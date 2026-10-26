---
title: Standardize Z420 storage mounts in config.py, fstab UUIDs, and Antigravity IDE venv/cron bindings
kind: preference
author: Tom & Bailey
date: 2026-10-22
touches:
  - /etc/fstab
  - /mnt/storage_8tb_linux/python_projects/scripts/config.py
  - /mnt/nvme_storage/ai_projects/tom_and_bailey_coding/run_analyzer.sh
  - src/subsystems.hpp
---
## Preference
- Maintain all Z420 Python path references through `/mnt/storage_8tb_linux/python_projects/scripts/config.py` (25/25 paths validated by `verify_paths.py`) and mount drives by UUID with `noatime,nofail` in `/etc/fstab`:
  - `/` : 512 GB SATA Boot SSD (Ubuntu 26.04 LTS)
  - `/mnt/samsung_storage` : 500 GB Samsung 850 EVO SSD (`/mnt/samsung_storage/containers` / `/mnt/samsung_storage/docker` data-root via `/etc/docker/daemon.json` to protect NVMe write endurance)
  - `/mnt/nvme_storage` : 2 TB HP EX950 NVMe SSD in Slot 4 (`devvault.db`, active GGUFs, `/mnt/nvme_storage/swapfile`)
  - `/mnt/storage_8tb_linux` : 8 TB WD Enterprise ext4 HDD (`ollama_models`, cold model library, 47,783 sorted media files, and daily 2:00 AM `rsync` system config mirror)
  - `/mnt/exos_mirror_18tb` -> `/mnt/exos_raid10_36tb` : 2x -> 4x 18 TB Seagate Exos array (with `StorageGuard` 66% capacity alerts + `verify-array`)
  - `/mnt/esp4b_nvme` / `/mnt/turbo_vault` : Minisforum ESP4B / HP Z Turbo Quad Pro 4x M.2 NVMe switch card (onboard Avago/Broadcom PEX8747 hardware PCIe switch requiring no motherboard bifurcation; move J2 jumper if installing a second HP Z Turbo card)
- Bind Antigravity IDE Python execution to `/mnt/nvme_storage/ai_projects/tom_and_bailey_coding/venv/bin/python` (`Hypo30285/tom_and_bailey_coding`) and run `run_analyzer.sh` via cron (Fridays at 4:00 PM, Sundays at midnight, and `@reboot`) emailing `weekly_disk_report.txt` to Bailey (`Bburnsed1995@gmail.com`).
