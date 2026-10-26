---
title: Match scavenged Dell 1U server CPUs and 768GB DDR4 ECC RAM to HP Z840 or Z8 G4 — never HP Z4 G4
kind: guardrail
author: Tom & Bailey
date: 2026-10-22
touches:
  - Z420_FINAL_BUILD_AND_SERVER_2027.md
---
## Guardrail
- When scavenging 20+ core Xeon CPUs and 768 GB ECC DDR4 RAM from a donor Dell 1U server for the King Kong Server build (intended to host Dual Intel Arc Pro B70 32GB ECC GPUs):
  - Dell PowerEdge R630 (Xeon E5-2600 v3/v4, Socket LGA 2011-v3) must go into an HP Z840 (dual socket, 16 DIMM slots \= 512 GB using 32 GB sticks or 768 GB using 64 GB sticks) or HP Z440 (8 DIMM slots).
  - Dell PowerEdge R640 (Xeon Scalable Gen 1/2, Socket LGA 3647) must go into an HP Z8 G4 (dual socket, 24 DIMM slots \= full 768 GB) or HP Z6 G4 (12 DIMM slots).
  - Never buy an HP Z4 G4 as a donor chassis for scavenged Dell 1U server CPUs, because the Z4 G4 uses Socket LGA 2066 (Xeon W-2100/W-2200) and is physically incompatible with both LGA 2011-v3 and LGA 3647 processors.
