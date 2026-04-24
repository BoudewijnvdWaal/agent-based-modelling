# Sensitivity Analysis

Deze map bevat de sensitivity analysis voor de agent-based GSE simulatie.

## Uitvoeren

```bash
cd sensitivity_analysis
python sensitivity_analysis.py
```

Dit genereert:
- sensitivity_gse_count.csv en .png
- sensitivity_alpha_beta.csv en _heatmap.png

## Parameters

- GSE Count: 4 tot 12 (stappen van 1)
- Alpha: 0.1 tot 0.9 (stappen van 0.1)
- Beta: 0.1 tot 0.9 (stappen van 0.1)

Simulatie duur: 24 uur voor volledige resultaten.