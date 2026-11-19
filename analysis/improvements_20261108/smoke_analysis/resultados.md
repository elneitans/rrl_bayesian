# Resultados de mejoras RRTL

Retornos incrementales desde snapshots; no son puntajes de partidas completas. Comparación exploratoria, sin test reservado.
Continuaciones completas: 14. Transiciones verificadas: 14,000.

| Criterio | Seed | Rol | Variante | Media final | Mediana final | Podas | Reemplazos | Refits |
|---|---:|---|---|---:|---:|---:|---:|---:|
|historical_f|710603|development|Baseline|3.00|3.00|0|0|0|
|historical_f|710603|development|TD lento|3.00|3.00|0|0|0|
|historical_f|710603|development|Objetivo Double|3.00|3.00|0|0|0|
|historical_f|710603|development|TD lento + Double|3.00|3.00|0|0|0|
|historical_f|710603|development|Revisión protegida|3.00|3.00|1|0|0|
|historical_f|710603|development|TD estable + revisión|3.00|3.00|0|0|0|
|historical_f|710603|development|Reestimación sin poda|2.50|2.50|0|0|4|
|anova|710603|development|Baseline|3.50|3.50|0|0|0|
|anova|710603|development|TD lento|3.00|3.00|0|0|0|
|anova|710603|development|Objetivo Double|1.50|1.50|0|0|0|
|anova|710603|development|TD lento + Double|3.00|3.00|0|0|0|
|anova|710603|development|Revisión protegida|3.50|3.50|2|0|0|
|anova|710603|development|TD estable + revisión|3.00|3.00|3|0|0|
|anova|710603|development|Reestimación sin poda|3.50|3.50|0|0|3|

Los arranques de un mismo seed son dependientes. Los grupos de confirmación no se usaron para desarrollar estas revisiones, pero no son un benchmark ciego. Una media alta aislada no demuestra superioridad general.

![Curvas](continuaciones.png)
