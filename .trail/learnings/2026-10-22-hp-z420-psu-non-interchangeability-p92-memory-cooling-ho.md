---
title: HP Z420 PSU non-interchangeability, P92 memory cooling, Honeywell PTM heatsink, and BIOS v03.96 ReBarUEFI flash rules
kind: guardrail
author: Tom
date: 2026-10-22
touches:
  - HP_Z420_Master_Upgrade_and_Agent_Plan.md
---
## Guardrail
- PSU Non-Interchangeability: Never attempt to install an HP Z620 800W PSU (`623194-001`, blind-mate blade brick with zero cables) in a Z420 chassis, and never plug an HP Z440 700W PSU (`719795-001`, uses `+12V_SB` standby instead of Z420 `+11V_SB` with shifted 18-pin wiring) directly into a Z420 motherboard without repinning/adapting. Stay within `<= 270W` combined PCIe draw on the stock 600W Gold PSU or use the backup 750W Corsair PSU with a verified 24-pin ATX to 18-pin HP Z420 adapter.
- Thermal & Heatsink Rules: Keep the snap-on Memory Airflow Guide and Fan connected to header `P92` over the 8x 8GB ECC DDR3 DIMMs, keep rear exhaust `P95 CHFAN2` clear, and mount the lapped Z440 heatsink with Honeywell phase-change material (6-pin fan connector on 5-pin header with overhang), verifying even core temperatures (`30-40C` idle) via `watch -n 1 sensors`.
- BIOS `J61 v03.96` SPI Injection (`CH341A`): Confirm the 2013 V2 boot block for native Xeon E5-2697 v2 support; read the stock 03.96 ROM twice with identical SHA-256 hashes before injecting `NvmExpressDxe.ffs` and `ReBarDxe.ffs` (unlocking Above 4G Decoding and 32GB/64GB ReBAR apertures required for Intel Arc Pro B60/B70 and headless Tesla P40/V100 cards; note `E15` Crisis Recovery jumper).
