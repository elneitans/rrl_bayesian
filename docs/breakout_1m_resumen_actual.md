# Breakout a 1M: resumen de metadatos descargados

Modelos con 1M y diez evaluaciones completas: **38/40**.

Este resumen valida hashes, identidades y contratos compartidos. No verifica los blobs ni las cadenas completas de logs; los reportes en compute siguen siendo necesarios.

| Semilla | Baseline | CART | ExtraTrees | BART |
|---|---:|---:|---:|---:|
| 710601 | 101 | 0 | 0 | 4 |
| 710602 | 3 | 11 | 19 | 40 |
| 710603 | 25 | 1 | 3 | 3 |
| 710604 | 29 | 0 | 0 | 9 |
| 710605 | 0 | 0 | 7 | 2 |
| 710606 | 25 | 11 | 3 | 3 |
| 710607 | 55 | 12 | 3 | 0 (10/10 trunc.) |
| 710608 | 1 (10/10 trunc.) | 1 | 4 | pendiente (3/10 trunc.) |
| 710609 | 37 | 52 | 11 | 0 (10/10 trunc.) |
| 710610 | 3 | 3 | 3 | pendiente |

La unidad de comparación es la semilla de entrenamiento. Diez episodios por modelo no equivalen a diez agentes independientes. Los retornos truncados corresponden al horizonte declarado. No extraer una clasificación final de subconjuntos incompletos.

| Método | Modelos evaluados | Media entre modelos disponibles | Mediana |
|---|---:|---:|---:|
| corrected_common | 10/10 | 27.900 | 25.000 |
| cart | 10/10 | 9.100 | 2.000 |
| extra_trees | 10/10 | 5.300 | 3.000 |
| bart | 8/10 | 7.625 | 3.000 |
