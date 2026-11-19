# Resultados de mejoras RRTL

Retornos incrementales desde snapshots; no son puntajes de partidas completas. Comparación exploratoria, sin test reservado.
Continuaciones completas: 70. Transiciones verificadas: 7,000,000.

| Criterio | Seed | Rol | Variante | Media final | Mediana final | Podas | Reemplazos | Refits |
|---|---:|---|---|---:|---:|---:|---:|---:|
|historical_f|710602|confirmation|Baseline|1.83|2.00|0|0|0|
|historical_f|710602|confirmation|TD lento|2.08|2.00|0|0|0|
|historical_f|710602|confirmation|Objetivo Double|2.08|2.00|0|0|0|
|historical_f|710602|confirmation|TD lento + Double|2.58|2.50|0|0|0|
|historical_f|710602|confirmation|Revisión protegida|3.17|3.00|1|0|0|
|historical_f|710602|confirmation|TD estable + revisión|2.58|2.50|0|0|0|
|historical_f|710602|confirmation|Reestimación sin poda|2.08|2.00|0|0|4|
|historical_f|710603|development|Baseline|70.33|41.00|0|0|0|
|historical_f|710603|development|TD lento|78.17|47.00|0|0|0|
|historical_f|710603|development|Objetivo Double|6.75|4.50|0|0|0|
|historical_f|710603|development|TD lento + Double|41.92|43.00|0|0|0|
|historical_f|710603|development|Revisión protegida|42.50|38.00|3|0|0|
|historical_f|710603|development|TD estable + revisión|41.92|43.00|1|0|0|
|historical_f|710603|development|Reestimación sin poda|41.92|43.00|0|0|4|
|historical_f|710607|development|Baseline|62.75|64.00|0|0|0|
|historical_f|710607|development|TD lento|62.75|64.00|0|0|0|
|historical_f|710607|development|Objetivo Double|11.17|10.50|0|0|0|
|historical_f|710607|development|TD lento + Double|23.33|22.50|0|0|0|
|historical_f|710607|development|Revisión protegida|2.08|2.00|1|0|0|
|historical_f|710607|development|TD estable + revisión|24.25|22.50|3|0|0|
|historical_f|710607|development|Reestimación sin poda|39.75|42.00|0|0|4|
|historical_f|710608|confirmation|Baseline|2.08|2.00|0|0|0|
|historical_f|710608|confirmation|TD lento|2.92|3.00|0|0|0|
|historical_f|710608|confirmation|Objetivo Double|2.75|2.50|0|0|0|
|historical_f|710608|confirmation|TD lento + Double|2.08|2.00|0|0|0|
|historical_f|710608|confirmation|Revisión protegida|2.08|2.00|0|0|0|
|historical_f|710608|confirmation|TD estable + revisión|2.08|2.00|0|0|0|
|historical_f|710608|confirmation|Reestimación sin poda|1.08|1.00|0|0|4|
|historical_f|710610|development|Baseline|2.83|3.00|0|0|0|
|historical_f|710610|development|TD lento|2.33|2.00|0|0|0|
|historical_f|710610|development|Objetivo Double|0.58|0.00|0|0|0|
|historical_f|710610|development|TD lento + Double|2.75|2.50|0|0|0|
|historical_f|710610|development|Revisión protegida|1.83|2.00|0|1|0|
|historical_f|710610|development|TD estable + revisión|1.00|1.00|1|1|0|
|historical_f|710610|development|Reestimación sin poda|1.58|2.00|0|0|4|
|anova|710602|confirmation|Baseline|21.75|13.50|0|0|0|
|anova|710602|confirmation|TD lento|53.67|65.50|0|0|0|
|anova|710602|confirmation|Objetivo Double|2.00|1.00|0|0|0|
|anova|710602|confirmation|TD lento + Double|53.67|65.50|0|0|0|
|anova|710602|confirmation|Revisión protegida|28.42|30.50|1|0|0|
|anova|710602|confirmation|TD estable + revisión|73.42|38.00|1|0|0|
|anova|710602|confirmation|Reestimación sin poda|40.25|46.00|0|0|4|
|anova|710603|development|Baseline|23.08|22.50|0|0|0|
|anova|710603|development|TD lento|3.58|3.00|0|0|0|
|anova|710603|development|Objetivo Double|4.92|3.00|0|0|0|
|anova|710603|development|TD lento + Double|2.25|2.00|0|0|0|
|anova|710603|development|Revisión protegida|16.92|13.00|2|1|0|
|anova|710603|development|TD estable + revisión|20.50|11.00|4|0|0|
|anova|710603|development|Reestimación sin poda|17.00|8.00|0|0|4|
|anova|710607|development|Baseline|28.17|30.00|0|0|0|
|anova|710607|development|TD lento|80.75|43.00|0|0|0|
|anova|710607|development|Objetivo Double|4.08|4.00|0|0|0|
|anova|710607|development|TD lento + Double|81.00|44.00|0|0|0|
|anova|710607|development|Revisión protegida|42.00|35.50|1|0|0|
|anova|710607|development|TD estable + revisión|67.42|68.00|1|0|0|
|anova|710607|development|Reestimación sin poda|69.75|83.50|0|0|4|
|anova|710608|confirmation|Baseline|43.08|34.50|0|0|0|
|anova|710608|confirmation|TD lento|45.42|44.50|0|0|0|
|anova|710608|confirmation|Objetivo Double|13.25|6.50|0|0|0|
|anova|710608|confirmation|TD lento + Double|31.50|35.00|0|0|0|
|anova|710608|confirmation|Revisión protegida|25.50|30.00|2|0|0|
|anova|710608|confirmation|TD estable + revisión|31.50|35.00|0|0|0|
|anova|710608|confirmation|Reestimación sin poda|176.17|94.50|0|0|3|
|anova|710610|development|Baseline|23.83|21.00|0|0|0|
|anova|710610|development|TD lento|9.25|9.00|0|0|0|
|anova|710610|development|Objetivo Double|4.58|4.50|0|0|0|
|anova|710610|development|TD lento + Double|10.83|13.50|0|0|0|
|anova|710610|development|Revisión protegida|21.58|22.50|0|1|0|
|anova|710610|development|TD estable + revisión|10.83|13.50|0|0|0|
|anova|710610|development|Reestimación sin poda|11.67|10.00|0|0|2|

Los arranques de un mismo seed son dependientes. Los grupos de confirmación no se usaron para desarrollar estas revisiones, pero no son un benchmark ciego. Una media alta aislada no demuestra superioridad general.

![Curvas](continuaciones.png)
