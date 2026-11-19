# Resultados: revisión de árboles y diagnóstico de aprendizaje

Comparación exploratoria de continuaciones desde snapshots. Estos retornos no son puntajes de partidas completas.
Campaña completa: **True**. Rollouts congelados: 4. Continuaciones: 10.
**Prueba de integración reducida: estos resultados verifican ejecución, no eficacia de las variantes.**
Diseño efectivo: 1,000 decisiones nuevas por agente; horizonte de evaluación 128 decisiones; 2 snapshots y 1 semillas por criterio.

## Evaluación congelada

| Criterio | Semilla | Checkpoint | ε | Retorno medio desde snapshot | Truncadas |
|---|---:|---:|---:|---:|---:|
|anova|710603|500000|0.0|3.50|2/2|
|historical_f|710603|500000|0.0|3.50|2/2|

## Continuaciones finales frente al control normal

Diferencias pareadas por los mismos arranques. Los arranques son casos diagnósticos relacionados, no semillas de entrenamiento independientes.

| Criterio | Semilla | Modalidad | Diferencia media de retorno |
|---|---:|---|---:|
|anova|710603|normal|+0.00|
|anova|710603|frozen|+0.00|
|anova|710603|prune|+0.00|
|anova|710603|prune_regrow|+0.00|
|anova|710603|replace|+0.00|
|historical_f|710603|normal|+0.00|
|historical_f|710603|frozen|+0.00|
|historical_f|710603|prune|-1.50|
|historical_f|710603|prune_regrow|-1.50|
|historical_f|710603|replace|-2.00|

## Operaciones de estructura

| Criterio | Semilla | Modalidad | Divisiones activas | Crecimientos | Podas | Reemplazos |
|---|---:|---|---:|---:|---:|---:|
|historical_f|710603|normal|15|0|0|0|
|historical_f|710603|frozen|15|0|0|0|
|historical_f|710603|prune|8|0|4|0|
|historical_f|710603|prune_regrow|8|0|4|0|
|historical_f|710603|replace|18|0|0|2|
|anova|710603|normal|104|0|0|0|
|anova|710603|frozen|104|0|0|0|
|anova|710603|prune|97|0|2|0|
|anova|710603|prune_regrow|97|0|2|0|
|anova|710603|replace|96|0|0|1|

## Alcance y límites

- Diagnostic snapshot continuations, not complete-game benchmark scores.
- Three selected seed case studies do not estimate population superiority.
- Block-bootstrap gates are exploratory and repeated selection is not calibrated.
- Bellman loss improvements do not guarantee return improvement.
- No reserved test executed.
