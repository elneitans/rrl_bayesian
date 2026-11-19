# Revisión de resultados: 28 de octubre de 2026, tarde

Consulta directa a Slurm y descarga en `results_cluster/revision_20261028_tarde/`. Se mantuvo la instrucción de priorizar el otro proyecto: no se enviaron ni reanudaron jobs. Los diecisiete pendientes cancelados siguen fuera del flujo activo.

## Estado observado

| Campaña OCAtari | Semillas entrenadas a 2M | Frontera de la siguiente semilla | Job |
|---|---:|---|---|
| Pong CART | 4/15, 501601–501604 | 501605: 596.479 | 34956, cerrado tras 23:50:11 |
| Pong ExtraTrees | 3/15, 501601–501603 | 501604: 462.847 | 34959, cerrado tras 23:50:10 |
| Pong BART | 0/15 | 501601: 561.663 | 34981, cerrado tras 18:58:05 |
| Breakout ExtraTrees | 3/10, 710601–710603 | 710604: 945.152 en snapshot descargado | 34983, ejecutándose, ~21h47 |

Son diez entrenamientos nuevos terminados a 2M, sin evaluaciones finales nuevas. Las ocho semillas Pong baseline previamente completas siguen siendo evidencia histórica separada; esta consulta no revalidó sus blobs. Breakout baseline conserva las diez semillas a 500k previamente documentadas. CART y BART Breakout largos no se iniciaron.

Se verificaron los hashes internos de los catorce `status.json` descargados. Esto comprueba su integridad interna, no la cadena completa de logs, los blobs de checkpoints ni la calidad de las políticas. No se descargaron esos blobs.

## Causa de los cierres

Slurm registra FAILED, ExitCode 3:0 para los tres jobs Pong. En estos scripts el código 3 representa trabajo parcial. No se observó un traceback en la búsqueda de errores de los logs consultados.

CART y ExtraTrees cerraron cerca de las 24 horas, con el margen de salida de 600 segundos de `28_long_cell.sbatch`. Sus últimas fronteras publicadas indican `insufficient_time_before_fit`: guardaron antes de un ajuste que no cabía en el tiempo restante. No completaron todas las semillas de sus campañas.

BART registró exactamente cuarenta pausas `insufficient_time_before_fit`. El bucle de `28_long_cell.sbatch` admite cuarenta intentos de hasta 1.740 segundos; al agotarlos devuelve 3 aunque todavía quede tiempo de la asignación. Por eso se detuvo tras ~19 horas, antes del límite de 24h. La restauración desde su frontera guardada permite continuar; hacerlo requiere respetar la prioridad vigente del usuario. Antes de futuras tandas conviene que el límite de intentos no corte prematuramente un worker sano, sin cambiar la receta científica congelada.

## Señales de recompensa recientes

Para cada celda se extrajeron las transiciones con `episode_end=true` de sus últimos cien archivos de logs ordenados por contador. La extracción se hizo mientras Breakout seguía activo: es una muestra reciente, no una frontera atómica ni una validación completa de logs. Datos en `control/recent_training_episodes.txt` dentro de la descarga. Las partidas usan exploración epsilon=0,1; no son evaluaciones de la política sin exploración.

| Juego/método | Semilla | Partidas recientes completas | Retorno bruto medio | Rango |
|---|---:|---:|---:|---|
| Pong CART, 2M | 501601 | 57 | −20,579 | −21 a −18 |
| Pong CART, 2M | 501602 | 56 | −20,500 | −21 a −18 |
| Pong CART, 2M | 501603 | 54 | −20,333 | −21 a −16 |
| Pong CART, 2M | 501604 | 56 | −20,589 | −21 a −18 |
| Pong ExtraTrees, 2M | 501601 | 56 | −20,536 | −21 a −18 |
| Pong ExtraTrees, 2M | 501602 | 53 | −20,302 | −21 a −17 |
| Pong ExtraTrees, 2M | 501603 | 57 | −20,544 | −21 a −18 |
| Pong BART, ~562k | 501601 | 55 | −20,473 | −21 a −17 |
| Breakout ExtraTrees, 2M | 710601 | 201 | 3,945 | 0 a 31 |
| Breakout ExtraTrees, 2M | 710602 | 190 | 5,084 | 0 a 50 |
| Breakout ExtraTrees, 2M | 710603 | 188 | 4,707 | 0 a 33 |

Ninguna de estas partidas Pong terminó con victoria: todas perdieron 21 puntos y marcaron menos de uno por partida en promedio. Es evidencia de rendimiento pobre durante el entrenamiento tardío; no demuestra por sí sola ausencia absoluta de aprendizaje, sobreajuste, causalidad de la representación ni equivalencia entre métodos. BART además tiene menor presupuesto de interacciones. Las partidas sucesivas no son semillas de entrenamiento independientes: no se aplicaron tests de significancia a esta muestra.

Breakout ExtraTrees sí obtiene puntos, pero las medias recientes son modestas y los máximos aislados no prueban una política estable ni mejora frente al baseline. Faltan evaluaciones comunes de los checkpoints y curvas comparables.

## Coste observado

Sumando las fases instrumentadas de las semillas terminadas: CART Pong ~5,00–5,24h por 2M; ExtraTrees Pong ~6,70–7,03h; ExtraTrees Breakout ~5,64–5,92h. No son el tiempo Slurm completo: excluyen parte del arranque, restauraciones y limpieza. La escritura de checkpoints/logs continúa siendo una fracción dominante de los métodos baratos.

BART Pong acumula ~14,72h de ajuste y ~2,21h de predicción para 561.663 decisiones; el ajuste representa aproximadamente el 79% de sus fases registradas. Es el cuello de botella observado. No extrapolar el coste de las calibraciones de 20k como garantía para 2M.

## Almacenamiento y pendientes

La migración terminó: el log confirma comparación por checksum y retirada de los archivos de origen verificados. `~/projects/rrl_bayesian/results` es ahora un enlace a `~/archive/rrl_bayesian_results`. Esto resuelve la ubicación anterior, pero no implica espacio ilimitado en archive; la capacidad agregada de `df` no sustituye la cuota personal.

Cuando el usuario vuelva a dar prioridad a RRL: evaluar primero los checkpoints disponibles con el protocolo de desarrollo congelado, comparar recompensas sin exploración y luego decidir la ampliación. Los jobs de evaluación cancelados no producirán esos resultados automáticamente. El script 36 está preparado pero no se ejecutó. No se corrió test reservado.
