# Spacecraft Trade-Off Cases Feasibility and Validation Report

| Case ID | Case Title | Platform | Payload | Dry Mass (kg) | Wet Mass (kg) | Feasible? | Failed Checks |
|---|---|---|---|---|---|---|---|
| CASE_01 | 16U CubeSat with HyperScape100 | PLT-007 | PAY-003 | 16.89 | 21.89 | **Feasible: No** | Battery SoC dropped below min limit (70.0%) during cycle execution |
| CASE_02 | SSTL-MICRO with MultiScape200 (external 400 Wh EPS) | PLT-017 | PAY-005 | 55.06 | 65.06 | **Feasible: No** | Battery SoC dropped below min limit (73.0%) during cycle execution |
| CASE_03 | SSTL-MICRO with DragonEye (external 800 Wh EPS) | PLT-017 | PAY-011 | 68.06 | 83.06 | **Feasible: No** | None |
| CASE_04 | The Frame with HyperScape100 (integrated 1100 Wh EPS) | PLT-033 | PAY-003 | 111.70 | 126.70 | **Feasible: No** | None |
| CASE_05 | The Frame with MultiScape200 (integrated 1100 Wh EPS) | PLT-033 | PAY-005 | 122.75 | 142.75 | **Feasible: No** | None |
| CASE_06 | The Frame with Raptor (integrated 1100 Wh EPS) | PLT-033 | PAY-012 | 170.44 | 190.44 | **Feasible: No** | None |
| CASE_07 | MOOG Meteorite with Raptor (external 800 Wh EPS) | PLT-042 | PAY-012 | 210.06 | 240.06 | **Feasible: No** | None |
| CASE_08 | MOOG Meteorite with SAR-C (external 1200 Wh EPS) | PLT-042 | PAY-015 | 335.14 | 365.14 | **Feasible: No** | Payload active power (250.0 W) exceeds platform avg payload power (150.0 W); Payload peak power (4000.0 W) exceeds platform peak payload power (2000.0 W); Payload peak power (4000.0 W) exceeds battery max continuous power (2073.6 W) |
