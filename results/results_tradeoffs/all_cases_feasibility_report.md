# Spacecraft Trade-Off Cases Feasibility and Validation Report

| Case ID | Case Title | Platform | Payload | Dry Mass (kg) | Wet Mass (kg) | Feasible? | Failed Checks |
|---|---|---|---|---|---|---|---|
| CASE_01 | 16U CubeSat with HyperScape100 | PLT-007 | PAY-003 | 18.25 | 23.25 | **Feasible: No** | Battery SoC dropped below min limit (70.0%) during cycle execution |
| CASE_02 | SSTL-MICRO with MultiScape200 (external 400 Wh EPS) | PLT-017 | PAY-005 | 56.42 | 66.42 | **Feasible: No** | Battery SoC dropped below min limit (73.0%) during cycle execution |
| CASE_03 | SSTL-MICRO with DragonEye (external 800 Wh EPS) | PLT-017 | PAY-011 | 69.42 | 84.42 | **Feasible** | None |
| CASE_04 | The Frame with HyperScape100 (integrated 1100 Wh EPS) | PLT-033 | PAY-003 | 113.06 | 128.06 | **Feasible** | None |
| CASE_05 | The Frame with MultiScape200 (integrated 1100 Wh EPS) | PLT-033 | PAY-005 | 124.11 | 144.11 | **Feasible** | None |
| CASE_06 | The Frame with Raptor (integrated 1100 Wh EPS) | PLT-033 | PAY-012 | 171.80 | 191.80 | **Feasible** | None |
| CASE_07 | MOOG Meteorite with Raptor (external 800 Wh EPS) | PLT-042 | PAY-012 | 211.42 | 241.42 | **Feasible** | None |
| CASE_08 | MOOG Meteorite with SAR-C (external 1200 Wh EPS) | PLT-042 | PAY-015 | 336.50 | 366.50 | **Feasible: No** | Payload active power (250.0 W) exceeds platform avg payload power (150.0 W); Payload peak power (4000.0 W) exceeds platform peak payload power (2000.0 W); Payload peak power (4000.0 W) exceeds battery max continuous power (2073.6 W) |
