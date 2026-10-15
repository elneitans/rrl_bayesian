# Preparación del bloque B — 14 de octubre de 2026

Implementación B8–B12 terminada, con ExtraTrees opcional integrado. El bloque C no se inició. La comparación larga queda inactiva hasta resolver parámetros, costo Linux y autorización. No se enviaron jobs ni se ejecutó test reservado.

Entregas separadas: [B8](pong_online/B8.md), [B9](pong_online/B9.md), [B10](pong_online/B10.md), [B11](pong_online/B11.md), [B12](pong_online/B12.md). [Guía y comandos exactos](pong_online_guide.md). [Reporte estructurado](pong_online_preparacion.json).

## Recuperación comprobada

La campaña descartable `results/pong_online_recovery_sim_20261014_01` completó las seis celdas obligatorias. En cada una: checkpoint, USR1 real, continuación en otro proceso, igualdad exacta de decisiones/modelos/RNG respecto a continuo, ausencia de transiciones duplicadas, segundo writer rechazado, evaluación completa reutilizada y reporte reconstruido. Los nombres visual/ocatari identifican celdas; se usó un entorno simulado en ambas, sin validar extractores reales en esta medición.

El reporte íntegro está en `results/pong_online_recovery_sim_20261014_01/recovery_report.json`. El JSON de preparación incorpora su checksum, resultados y microbenchmark. La copia histórica conserva sus 40 fuentes originales verificadas y está en `historical/pong_pilot_linux_01_code`; transferirla separadamente si se mueve el piloto.

## Costo observado local: 64 decisiones por celda

| Celda | Entrenamiento (s) | Workers evaluación (s) | Logs (bytes) | Checkpoints totales (bytes) | RSS máximo (MiB) |
|---|---:|---:|---:|---:|---:|
| visual/corrected_common/501501 | 0.1581 | 0.2598 | 27675 | 29241 | 207.8 |
| visual/cart/501501 | 0.1677 | 0.2814 | 28027 | 73011 | 232.6 |
| visual/bart/501501 | 0.1937 | 0.3292 | 40064 | 141247 | 215.6 |
| ocatari/corrected_common/501501 | 0.1577 | 0.2595 | 27675 | 29256 | 236.3 |
| ocatari/cart/501501 | 0.1716 | 0.2754 | 28027 | 73026 | 232.7 |
| ocatari/bart/501501 | 0.1905 | 0.3012 | 40065 | 141262 | 235.2 |

Cada celda tiene cuatro episodios evaluativos planificados y se repitió su comando completo para verificar reutilización. El costo de worker incluye lectura/verificación de checkpoints y ambos comandos; el presupuesto por episodio mide la ejecución interna separadamente. Entrenamiento excluye arranque/importaciones y validación de código previa al presupuesto. Los tiempos son consumo local, no wall time paralelo ni estimación para Atari.

La instrumentación por fases, rangos de bytes por checkpoint y presupuesto evaluativo están en el JSON. Los logs se agregan por segmentos hasta 16 filas en smoke; no se reescribe todo el historial. Memoria de replay y caché están acotadas por configuración; las estructuras/modelos y el filesystem siguen requiriendo medición.

## Benchmark de predicción: referencia y caché

200 predicciones del mismo modelo/lote de dos filas, comprobadas con igualdad exacta en cada repetición. Incluye el costo de esas comprobaciones. No representa la tasa de estados repetidos ni el costo del emulador.

| Celda | Referencia (s) | Caché (s) | Hits/misses | Bytes contabilizados |
|---|---:|---:|---:|---:|
| visual/cart/501501 | 0.008874 | 0.002906 | 199/1 | 1188 |
| visual/bart/501501 | 0.042738 | 0.003067 | 199/1 | 1188 |
| ocatari/cart/501501 | 0.008703 | 0.002949 | 199/1 | 1188 |
| ocatari/bart/501501 | 0.044182 | 0.003146 | 199/1 | 1188 |

## Protocolos y escala pendiente

- Smoke: 6 celdas, 64 decisiones, 5 checkpoints por celda y 4 ajustes por learner batch. BART m=2, burn-in=2, draws=3, profundidad 2; solo evidencia técnica.
- Baseline progresivo: 6 celdas y máximo congelado 2 millones por celda, hitos 100k/500k/2M, tres semillas, ambos pipelines A3. Presupuesto inicial por unidad: 3.600 s de entrenamiento y 600 s por episodio evaluativo; se amplía con ID/motivo, sin reiniciar contadores. El máximo exige hasta 12 millones de decisiones de entrenamiento, 240 episodios evaluativos y 3.911 checkpoints por celda antes de retención. Estos son conteos, no presupuestos de tiempo aprobados.
- Larga: template de 30 celdas (tres learners, dos pipelines, cinco semillas). Decisiones, calendario definitivo, BART y presupuesto permanecen pendientes. ExtraTrees se habilita solo en una variante explícita.
- Exploración: epsilon 1→.1 exponencial en 50.000 decisiones acumuladas. Baseline mantiene eta=.025, gamma=.99, significancia .0001, soporte 128 y profundidad 4. Recompensas, TD/F-test y representación A3 permanecen iguales.

No hay una proyección válida de tiempo, RSS o disco para la campaña larga Linux. `dry-run --cost-reference` rechaza estas mediciones simuladas y exige coincidencia de contrato, catálogo, código, configuración, runtime y horizontes. Reporta rangos de tasas observadas con incertidumbre; no promete un máximo. Retención protege calendario evaluativo, final, último y respaldo; limpieza únicamente explícita tras inventario.

## Pendientes experimentales antes de lanzar la larga

1. Ejecutar la recuperación Atari pequeña en el Linux previsto, con ambos extractores. Comandos preparados en la guía; no enviados.
2. Completar A6/A7 y registrar conclusiones y hashes. Elegir la configuración definitiva de BART a partir de esa evidencia.
3. Medir costo Linux compatible por learner/pipeline, incluyendo evaluación y almacenamiento; resolver máximo, calendario y presupuestos en una configuración nueva.
4. Congelar y autorizar explícitamente el hito de ejecución, respaldado por esas mediciones. Una política que termine al piso se reporta sin concluir equivalencia.

## Verificación final

162 pruebas correctas, una omitida (paridad del emulador real, opt-in) y 11 excluidas por requerir Atari/OCAtari. Incluye las regresiones de B y las pruebas históricas portables de comparación, checkpoint, análisis y Slurm. Ruff sin hallazgos y sintaxis/comentario verificados en los ocho batches nuevos.

Las pruebas de ExtraTrees declaran explícitamente la versión local disponible (1.5.0) en su configuración descartable; no cambian el default de producción 1.9.1. Validar ese entorno fijado en el preflight Linux antes de usar el control opcional. Las dependencias no se instalaron ni actualizaron. Para evitar lecturas bloqueadas de caché se ejecutaron fuentes instaladas con rutas temporales de importación; esto no constituye una validación del runtime Atari previsto.
