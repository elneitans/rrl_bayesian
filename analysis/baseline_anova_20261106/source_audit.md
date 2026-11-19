# Auditoría de fuentes congeladas: baseline y ANOVA

6 de noviembre de 2026. Auditoría estática de bytes y protocolos; sin nuevas ejecuciones ni cambios de fuentes. Evidencia estructurada: [source_audit.json](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/analysis/baseline_anova_20261106/source_audit.json).

## Identidad y alcance

Se verificaron los SHA256 de los payloads de ambos manifiestos. El manifiesto histórico declara88 fuentes y ANOVA90. Hay **13 diferencias:11 archivos modificados y2 añadidos**. Cada fuente antigua modificada se recuperó de un snapshot cuyo SHA coincide exactamente con el manifiesto; cada fuente nueva coincide exactamente con el commit `c2b4faa460904e67ec5ffd5203e54d9df51485bf`. No se infiere identidad a partir de la fecha o del nombre del snapshot.

- Histórico: [/Users/natanzamorano/Documents/Universidad/capstone/tmp/baseline_analysis_20261101/remote/campaign.json](/Users/natanzamorano/Documents/Universidad/capstone/tmp/baseline_analysis_20261101/remote/campaign.json).
- ANOVA: [/Users/natanzamorano/Documents/Universidad/capstone/tmp/anova_analysis_20261106/remote/corrected_common/campaign.json](/Users/natanzamorano/Documents/Universidad/capstone/tmp/anova_analysis_20261106/remote/corrected_common/campaign.json).

El runtime completo es idéntico. Los protocolos solo difieren en nombre y `split_test="anova"`; las diez celdas solo añaden ese campo en estimator y cambian sus hashes derivados. Las semillas, acciones, presupuesto, recompensa, exploración, representación, horizontes y calendario son iguales.

## Diferencias verificadas

| Archivo | Efecto de la diferencia | Snapshot antiguo exacto |
|---|---|---|
| `bayesian_rrtl/baseline.py` | Selecciona ANOVANode; no cambia fórmula TD, siguiente estado, terminales ni orden TD/estadísticas/split. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline.py) |
| `bayesian_rrtl/baseline_adapter.py` | Añade split_test, validaciones y clase de nodos; conserva normalización, catálogo y política. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline_adapter.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline_adapter.py) |
| `bayesian_rrtl/baseline_anova.py` | Nueva regla ANOVA sobre Q actualizado, con validación de momentos, degeneraciones, selección y crecimiento. | Archivo nuevo |
| `bayesian_rrtl/baseline_checkpoint.py` | Añade fuentes/tipos ANOVA y comprueba coherencia de momentos al validar checkpoints; sin modificar Q, estructura o RNG válidos. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline_checkpoint.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/baseline_checkpoint.py) |
| `bayesian_rrtl/config.py` | Añade split_test y rechaza combinaciones incompatibles; valor predeterminado historical_f. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/config.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/config.py) |
| `bayesian_rrtl/evaluation.py` | Presenta costes incompletos como desconocidos y conserva sumas registradas como límite inferior. | [historical/atari_common_C_20261014_code/bayesian_rrtl/evaluation.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/atari_common_C_20261014_code/bayesian_rrtl/evaluation.py) |
| `bayesian_rrtl/online/final_test.py` | Reporte de test reservado distingue unidades pendientes/en ejecución mediante lock; no cambia simulación. Sin reservado ejecutado en esta comparación. | [historical/atari_common_C_20261014_code/bayesian_rrtl/online/final_test.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/atari_common_C_20261014_code/bayesian_rrtl/online/final_test.py) |
| `bayesian_rrtl/online/learners.py` | Extrae baseline_config con los mismos argumentos compartidos; añade split_test a métricas. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/online/learners.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/online/learners.py) |
| `bayesian_rrtl/online/protocol.py` | Excluye tiempos/RSS desconocidos de proyecciones y registra conteos incompletos. | [historical/atari_common_C_20261014_code/bayesian_rrtl/online/protocol.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/atari_common_C_20261014_code/bayesian_rrtl/online/protocol.py) |
| `bayesian_rrtl/online/session.py` | Al restaurar baseline contrasta configuración y valida árbol; no cambia step, selección ni epsilon. | [historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/online/session.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/pre_rrtl_anova_20261101_code/bayesian_rrtl/online/session.py) |
| `experiments/audit_pong_representation.py` | Auditoría offline compara evolución a igual estado y acción y distingue diferencias geométricas; no cambia campañas. | [historical/atari_common_C_20261014_code/experiments/audit_pong_representation.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/atari_common_C_20261014_code/experiments/audit_pong_representation.py) |
| `experiments/compare_pong_progressive.py` | Inicializa metadata de intentos y reporta costes incompletos; GreedyController y simulación de evaluación sin cambios. | [historical/atari_common_C_20261014_code/experiments/compare_pong_progressive.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/historical/atari_common_C_20261014_code/experiments/compare_pong_progressive.py) |
| `experiments/report_rrtl_split_tests.py` | Nuevo reporte pareado de artefactos; exige fuentes y runtime idénticos, por lo que rechaza estas dos campañas. | Archivo nuevo |

Las validaciones nuevas de restauración examinan configuración, tipos, conteos y momentos; no alteran un árbol/Q/RNG válido. Pueden rechazar un artefacto inválido. Los cambios de costes pueden modificar el diagnóstico operativo de completitud de mediciones, no la recompensa aprendida ni el puntaje simulado.

## Núcleo idéntico por SHA256

El JSON conserva los hashes completos de las siguientes fuentes exactamente iguales:

- `relational_regresion_tree.py`, `preprocessing.py`, `targets.py`, `encoding.py`.
- `online/environments.py`, `ocatari_backend.py`, `games/breakout.py`, `games/slot_relations.py`, `online/storage.py`.

Por tanto, el cambio científico identificado es la regla de crecimiento y su selección explícita: TD, recompensa, dinámica/reset, extracción, relaciones y simulación de evaluación no contienen una intervención simultánea detectada. La selección ANOVA incluye sus grados de libertad, categorías observadas, degeneraciones y desempates; no representa solo un cambio de etiqueta.

## Límite de la comparación

La comparación descriptiva por semillas está bien alineada y la auditoría permite especificar qué cambió. Aun así, [el exportador estricto](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/experiments/report_rrtl_split_tests.py:31) exige igualdad literal de fuentes y runtime, y rechaza estas campañas por las fuentes. No se ha pasado esa validación ni se debe saltar. Una réplica de histórico con el checkout ANOVA confirmaría el contraste bajo software idéntico. Esta auditoría por sí sola no demuestra que un split cause una caída de retorno ni sustituye una intervención controlada.
