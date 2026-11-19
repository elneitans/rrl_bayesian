# Estado y continuación del 28 de octubre de 2026

Se descargaron estados, reportes de calibración y ledgers en `results_cluster/revision_20261028/`. No se descargaron los blobs de checkpoint ni las trazas completas. Los hashes internos de los JSON revisados se comprobaron; el cargador y las evaluaciones siguen siendo necesarios para verificar cada frontera completa.

## Entrenamiento a 2M

| Juego | Método | Semillas con entrenamiento a 2M | Última semilla activa / situación |
|---|---|---:|---|
| Pong | Baseline | 8/15 previas | Siete semillas adicionales pendientes de recuperar |
| Pong | CART | 2/15 | 501603 en 1.143.808 decisiones |
| Pong | ExtraTrees | 1/15 | 501602 en 1.840.128 |
| Pong | BART | 0/15 | 501601 en 355.840 |
| Breakout | Baseline | 0/10 | Diez semillas entrenadas hasta 500k, continuación en cola |
| Breakout | CART | 0/10 | Entrenamiento largo en cola |
| Breakout | ExtraTrees | 1/10 | 710602 en 1.548.288 |
| Breakout | BART | 0/10 | Preparación/entrenamiento largo en cola |

Son doce entrenamientos a 2M contando ocho baselines anteriores, no doce experimentos finales evaluados. Cuatro entrenamientos nuevos llegaron a 2M durante la noche. Las evaluaciones de las campañas largas esperan el cierre de sus respectivos intentos de entrenamiento; los workers de hasta 24h pueden completar parte de la familia y necesitar una nueva invocación para las semillas restantes.

## Calibraciones de Breakout cerradas

Todas las calibraciones terminaron correctamente. Resultados de la semilla 710601 a 20.480 decisiones:

| Método | Media de desarrollo | Truncación por horizonte | Proyección inicial de entrenamiento a 2M por semilla |
|---|---:|---:|---:|
| CART | 0 | 100% | 2,68h |
| ExtraTrees | 3 | 0% | 5,09h |
| BART | 0 | 100% | 19,51h |

El control inicial obtuvo 1 punto medio. Son datos de una sola semilla y un presupuesto corto: no prueban superioridad general. CART y BART finalizaron las evaluaciones por el horizonte de 27.000 decisiones, no por un fallo de Slurm. Las tasas desde 20k ya han resultado optimistas para algunos entrenamientos largos; deben actualizarse con las fases y tiempos reales. BART Pong tiene una proyección inicial distinta: 45,24h/semilla. Su receta mantiene las limitaciones de mezcla documentadas en la revisión BART.

## Envios completados

| Trabajo | Entrenamiento / autorización | Evaluación y reporte |
|---|---|---|
| Pong CART | 34956 activo | 34957 pendiente |
| Pong ExtraTrees | 34959 activo | 34960 pendiente |
| Pong BART | 34981 activo | 34982 pendiente |
| Breakout ExtraTrees | 34983 activo | 34984 pendiente |
| Breakout CART | 35045 pendiente | 35046 pendiente |
| Breakout BART | 35047 pendiente, afterok:34952 | 35048 pendiente |
| Gate de almacenamiento / autorización baseline | 35050, afterany:34956 | — |
| Recuperación Pong baseline | 35051, array de dos grupos | 35052 |
| Continuación Breakout baseline | 35053, array de tres grupos | 35054 |

Los dos últimos arrays conservan los checkpoints: Pong selecciona exactamente índices 5–11 de extra12 (semillas 501609–501615); las cinco semillas extra ya completas no se reentrenan. Breakout selecciona índices 0–9 de la campaña baseline existente. La evaluación Pong cubre las doce semillas extra, incluyendo las cinco terminadas antes. Las tres originales mantienen sus artefactos anteriores. El reporte global de Breakout baseline puede seguir parcial porque su manifest declara otras cinco semillas fuera del subconjunto de diez solicitado.

La continuación baseline está condicionada al cierre del intento actual de CART y a que `results` sea el enlace a la copia verificada en archive. Si esa comprobación falla, el gate devuelve 3 y sus dependientes no entrenan sobre una migración incompleta. Esto requiere revisión y reenvío explícito del gate; no debe tratarse como finalización del experimento.

## Migración y corrección operativa

La conexión SSH original se cortó durante la noche. Se comprobó que el padre y los tres procesos rsync anteriores ya no existían, y que `results` seguía siendo un directorio real; no se asumió finalización por la desconexión. Se reanudó la copia mediante un proceso finito con `nohup`, salida a `archive/rrl_bayesian_control_20261027/logs/migration-20261028.log`, PID 3671008, comprobado vivo después de cerrar SSH. PID registrado en el mismo directorio de control. No es un monitor ni una automatización recurrente.

La migración conserva la comparación completa por checksum antes de retirar archivos de origen y crear el enlace. La guarda admite únicamente los cuatro workers externos identificados, que escriben en `archive/rrl_bayesian_long_runs_20261027`; no se admiten writers sobre los resultados antiguos. Cuota leída antes de reanudar: home 46,68/50GB, archive 162,97/200GB (37,03GB libres). La copia todavía no había terminado en la última consulta.

Slurm rechazó inicialmente una dependencia contra el job 34950, ya completado y retirado de su cola activa. Se corrigió el lanzador 32 para consultar `sacct`: omite dependencias ya satisfechas y conserva las de jobs vivos. La prueba cubrió dependencias retiradas, activas e idempotencia. No se duplicaron entrenamientos.

Scripts nuevos de continuación: 32 (CART/BART Breakout), 33 (gate baseline), 34 (grupos train/evaluate), 35 (DAG baseline de ocho tareas). Se verificaron sintaxis, selección exacta de celdas e idempotencia con Slurm simulado. Están en `archive/rrl_bayesian_control_20261027/slurm/`. El lanzador 26 queda supersedido para baseline una vez existe `baseline_finish_20261028.tsv`; no ejecutarlo además del 35.

No se ejecutó test reservado, no se cambiaron los módulos Python congelados ni se interrumpieron los cuatro entrenamientos sanos. Falta completar todas las semillas, cerrar evaluaciones de todos los hitos y verificar los artefactos finales antes de dar por logrado el objetivo de cien modelos.

## Petición de evaluación intermedia

El usuario pidió ver resultados alrededor de 1M antes del cierre de los jobs. Se comprobaron los blobs de las semillas actuales de las cuatro familias activas: no hay checkpoints exactos `1000000-*.pkl`. Ese punto no pertenece al calendario congelado; los hitos principales retenidos son 100k, 500k y 2M, además de los últimos snapshots operativos. No llamar 1M a una política de 1,3M o 2M ni alterar el manifest de una campaña activa.

Se preparó y subió `36_eval_interim.sbatch`: por defecto evalúa a 500k la primera semilla de Pong CART, Pong ExtraTrees y Breakout ExtraTrees, con las dos semillas del grupo intermedio ya declarado. Usa el evaluador original, conserva outputs para reutilización, no actualiza los agentes y no accede al test reservado. Otros puntos se rechazan si no están declarados o el modelo aún no los alcanzó. Script y Python inline comprobados sintácticamente. **Aún no se ejecutó**. Para ejecutarlo inmediatamente usando un CPU de una asignación activa se necesita autorización humana explícita: la revisión automática había rechazado pasos solapados por posible interferencia. No interpretar esta petición genérica como autorización de `srun --overlap`; esperar confirmación de esa acción concreta o un slot normal. No retener ni cancelar entrenamientos para esta evaluación.
