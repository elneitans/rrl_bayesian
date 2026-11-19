# Resultados: revisión de árboles y diagnóstico de aprendizaje

Comparación exploratoria de continuaciones desde snapshots. Estos retornos no son puntajes de partidas completas.
Campaña completa: **True**. Rollouts congelados: 324. Continuaciones: 30.
Diseño efectivo: 100,000 decisiones nuevas por agente; horizonte de evaluación 3000 decisiones; 6 snapshots y 3 semillas por criterio.

## Evaluación congelada

| Criterio | Semilla | Checkpoint | ε | Retorno medio desde snapshot | Truncadas |
|---|---:|---:|---:|---:|---:|
|anova|710603|100000|0.0|10.33|0/6|
|anova|710603|100000|0.01|10.00|0/6|
|anova|710603|100000|0.1|11.00|0/6|
|anova|710603|500000|0.0|23.67|0/6|
|anova|710603|500000|0.01|24.00|0/6|
|anova|710603|500000|0.1|11.50|0/6|
|anova|710603|1000000|0.0|4.83|0/6|
|anova|710603|1000000|0.01|5.50|0/6|
|anova|710603|1000000|0.1|8.83|0/6|
|anova|710607|100000|0.0|75.00|0/6|
|anova|710607|100000|0.01|59.00|0/6|
|anova|710607|100000|0.1|38.33|0/6|
|anova|710607|500000|0.0|205.00|0/6|
|anova|710607|500000|0.01|157.00|0/6|
|anova|710607|500000|0.1|43.00|0/6|
|anova|710607|1000000|0.0|124.83|0/6|
|anova|710607|1000000|0.01|110.83|0/6|
|anova|710607|1000000|0.1|49.83|0/6|
|anova|710610|100000|0.0|15.83|0/6|
|anova|710610|100000|0.01|19.17|0/6|
|anova|710610|100000|0.1|19.50|0/6|
|anova|710610|500000|0.0|3.00|0/6|
|anova|710610|500000|0.01|3.17|0/6|
|anova|710610|500000|0.1|4.67|0/6|
|anova|710610|1000000|0.0|6.67|0/6|
|anova|710610|1000000|0.01|6.67|0/6|
|anova|710610|1000000|0.1|5.33|0/6|
|historical_f|710603|100000|0.0|6.00|0/6|
|historical_f|710603|100000|0.01|6.00|0/6|
|historical_f|710603|100000|0.1|6.33|0/6|
|historical_f|710603|500000|0.0|65.33|0/6|
|historical_f|710603|500000|0.01|65.17|0/6|
|historical_f|710603|500000|0.1|38.50|0/6|
|historical_f|710603|1000000|0.0|35.00|0/6|
|historical_f|710603|1000000|0.01|48.67|0/6|
|historical_f|710603|1000000|0.1|22.67|0/6|
|historical_f|710607|100000|0.0|11.83|0/6|
|historical_f|710607|100000|0.01|14.00|0/6|
|historical_f|710607|100000|0.1|15.50|0/6|
|historical_f|710607|500000|0.0|2.33|0/6|
|historical_f|710607|500000|0.01|2.33|0/6|
|historical_f|710607|500000|0.1|2.33|0/6|
|historical_f|710607|1000000|0.0|63.33|0/6|
|historical_f|710607|1000000|0.01|61.17|0/6|
|historical_f|710607|1000000|0.1|54.33|0/6|
|historical_f|710610|100000|0.0|1.67|0/6|
|historical_f|710610|100000|0.01|1.67|0/6|
|historical_f|710610|100000|0.1|1.50|0/6|
|historical_f|710610|500000|0.0|2.00|2/6|
|historical_f|710610|500000|0.01|2.00|0/6|
|historical_f|710610|500000|0.1|2.67|0/6|
|historical_f|710610|1000000|0.0|3.33|0/6|
|historical_f|710610|1000000|0.01|3.33|0/6|
|historical_f|710610|1000000|0.1|3.33|0/6|

## Continuaciones finales frente al control normal

Diferencias pareadas por los mismos arranques. Los arranques son casos diagnósticos relacionados, no semillas de entrenamiento independientes.

| Criterio | Semilla | Modalidad | Diferencia media de retorno |
|---|---:|---|---:|
|anova|710603|normal|+0.00|
|anova|710603|frozen|-16.17|
|anova|710603|prune|+62.67|
|anova|710603|prune_regrow|-17.00|
|anova|710603|replace|-17.17|
|anova|710607|normal|+0.00|
|anova|710607|frozen|-1.67|
|anova|710607|prune|-1.67|
|anova|710607|prune_regrow|+0.00|
|anova|710607|replace|-22.67|
|anova|710610|normal|+0.00|
|anova|710610|frozen|-19.00|
|anova|710610|prune|-19.00|
|anova|710610|prune_regrow|+0.00|
|anova|710610|replace|+72.00|
|historical_f|710603|normal|+0.00|
|historical_f|710603|frozen|-50.17|
|historical_f|710603|prune|-49.00|
|historical_f|710603|prune_regrow|-49.00|
|historical_f|710603|replace|-42.33|
|historical_f|710607|normal|+0.00|
|historical_f|710607|frozen|+0.00|
|historical_f|710607|prune|-60.83|
|historical_f|710607|prune_regrow|-60.83|
|historical_f|710607|replace|-52.67|
|historical_f|710610|normal|+0.00|
|historical_f|710610|frozen|+0.00|
|historical_f|710610|prune|-2.83|
|historical_f|710610|prune_regrow|-2.83|
|historical_f|710610|replace|-2.67|

## Operaciones de estructura

| Criterio | Semilla | Modalidad | Divisiones activas | Crecimientos | Podas | Reemplazos |
|---|---:|---|---:|---:|---:|---:|
|historical_f|710603|normal|16|1|0|0|
|historical_f|710603|frozen|15|0|0|0|
|historical_f|710603|prune|10|0|4|0|
|historical_f|710603|prune_regrow|13|3|4|0|
|historical_f|710603|replace|15|1|0|2|
|historical_f|710607|normal|10|0|0|0|
|historical_f|710607|frozen|10|0|0|0|
|historical_f|710607|prune|6|0|4|0|
|historical_f|710607|prune_regrow|6|0|4|0|
|historical_f|710607|replace|12|1|0|4|
|historical_f|710610|normal|7|0|0|0|
|historical_f|710610|frozen|7|0|0|0|
|historical_f|710610|prune|2|0|4|0|
|historical_f|710610|prune_regrow|3|1|4|0|
|historical_f|710610|replace|13|6|0|4|
|anova|710603|normal|108|4|0|0|
|anova|710603|frozen|104|0|0|0|
|anova|710603|prune|96|0|4|0|
|anova|710603|prune_regrow|100|10|4|0|
|anova|710603|replace|93|18|0|2|
|anova|710607|normal|117|4|0|0|
|anova|710607|frozen|113|0|0|0|
|anova|710607|prune|113|0|0|0|
|anova|710607|prune_regrow|117|4|0|0|
|anova|710607|replace|104|7|0|2|
|anova|710610|normal|99|5|0|0|
|anova|710610|frozen|94|0|0|0|
|anova|710610|prune|94|0|0|0|
|anova|710610|prune_regrow|99|5|0|0|
|anova|710610|replace|90|8|0|2|

## Alcance y límites

- Diagnostic snapshot continuations, not complete-game benchmark scores.
- Three selected seed case studies do not estimate population superiority.
- Block-bootstrap gates are exploratory and repeated selection is not calibrated.
- Bellman loss improvements do not guarantee return improvement.
- No reserved test executed.
