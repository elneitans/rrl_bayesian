# Estado Breakout a 1M — 31 de octubre de 2026

## Alcance y evidencia de cierre

OCAtari, diez semillas de entrenamiento 710601–710610 por algoritmo. Primera meta autorizada: 1M decisiones por agente; máximo futuro congelado: 2M. No se ejecutó test reservado.

| Algoritmo | Modelos con diez evaluaciones finales a 1M | Calendario hasta 1M cerrado en compute | Media / mediana final entre modelos |
|---|---:|---|---|
| Baseline corrected_common | 10/10 | Sí, job35692 COMPLETED/0:0 | 27,9 / 25 |
| CART | 10/10 | Sí, job35693 COMPLETED/0:0 | 9,1 / 2 |
| ExtraTrees | 10/10 | Sí, job35694 COMPLETED/0:0 | 5,3 / 3 |
| BART | 3/10 | Pendiente, job35695 | No comparar como cohorte completa |

Los tres archivos `calendar_progress_1m.json` descargados tienen `complete_1m=true` y diez celdas. Los jobs de cierre usan `consolidate`, verificando fronteras e identidades de evaluación antes del resumen. La descarga ligera local no contiene blobs ni logs completos para repetir aquí la auditoría de sus bytes.

Total con evaluación final: 33/40. Las tres semillas BART disponibles (710601,710603,710605) obtuvieron respectivamente medias4,3,2. Son un subconjunto, no el resultado de diez semillas. El baseline710608 tiene diez truncaciones por horizonte: su retorno medio1 corresponde al presupuesto declarado, no a una terminación natural. No tratar los episodios repetidos de cada agente como réplicas independientes de entrenamiento.

## Reparación de las continuaciones BART

El grupo original35470_0 terminó parcial con código3 tras23h49m52, guardando 710602 en855.039 decisiones. La primera continuación35728_0 falló antes de entrenar: `sacct` desde scylla no alcanzó la base de datos dwemer (`Address family 0 not supported`). No fue un fallo del modelo ni una pérdida de su checkpoint.

Se corrigió `43_gated_breakout_resume.sbatch`: su dependencia `afterany` espera el cierre del grupo original y la comprobación en compute valida los status/checkpoints de la campaña, sin consultar `sacct`. Rechaza estados fallidos, inicialización sin frontera, presupuestos agotados y pausas no reconocidas. El worker38 sigue aplicando el lock de grupo y restaurando los checkpoints completos. Cuando un agente ya está en1M solo verifica/completa su evaluación, sin reentrenarlo. Pruebas de lógica y sintaxis completadas.

Como Slurm guarda una copia de cada script al enviarlo, actualizar43 en disco no reparaba los jobs pendientes. El lanzador44 canceló únicamente los cuatro gates v1 aún PENDING, preservó los activos y creó reemplazos con ledger idempotente:

| Grupo | Continuación v2 | Dependencia |
|---|---:|---|
| 0 | 35782_0 | Original ya terminal, verificado desde login; sin dependencia expirada |
| 1 | 35783_1 | afterany:35470_1 |
| 2 | 35784_2 | afterany:35470_2 |
| 3 | 35785_3 | afterany:35470_3 |
| 4 | 35786_4 | afterany:35470_4 |

Ledger v2 `control/breakout_1m_recovery_v2_jobs.tsv`; los IDs nuevos también se añadieron al ledger histórico de recuperaciones. Los reportes35471 y35695 esperan el array original y los cinco reemplazos, excluyendo los gates v1 fallidos/cancelados. Las dependencias se comprobaron en `scontrol`.

## Workers actuales y trabajo restante

Array BART35470 con throttle4 nuevamente, ahora que terminaron los otros algoritmos y sus calendarios. Límite global por usuario: cuatro jobs activos.

- 35782_0 activo: restauración confirmada; 710602 avanzó a869.888 decisiones.
- 35470_1 activo: 710604 en662.016.
- 35470_2 activo: 710606 en448.512.
- 35470_3 activo: 710607 en25.088.
- Grupo original35470_4 y continuaciones35783–35786 pendientes.

Los contadores son el snapshot descargado, no predicciones. No se enviaron duplicados contra grupos activos. Faltan siete modelos BART con evaluación final, el calendario BART y su reporte validado. Si una continuación v2 queda parcial, comprobar su ID y causa antes de continuarla; no volver a enviar el gate antiguo ni el lanzador40 para recrear la campaña.

Fuentes locales actualizadas: `results_cluster/breakout_1m_latest/`, `docs/breakout_1m_resumen_actual.md` y su JSON. El objetivo permanece activo e incompleto.
