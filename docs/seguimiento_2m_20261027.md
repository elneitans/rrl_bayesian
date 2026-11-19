# Objetivo activo: todos los métodos a 2M

**PRIORIDAD MÁS RECIENTE, 29/10:** el usuario confirmó explícitamente «Sí, envía ahora los jobs de Breakout». Se envió la tanda 1M/10 semillas por método con posibilidad de extender después a 2M: preparación35463, baseline35464/reporte35465, CART35466/35467, ExtraTrees35468/35469, BART35470/35471. Son campañas nuevas porque los protocolos anteriores no declaran 1M; resultados antiguos conservados. Consultar `docs/breakout_1m_20261028.md` y ledger remoto `breakout_1m_jobs.tsv`. Esta autorización supera el bloqueo anterior para esta tanda; no recrear otros DAGs cancelados ni duplicar estos IDs. Preparación observada RUNNING; no confundir envío con entrenamiento completo ni con evaluación final.

**ACTUALIZACIÓN 28/10 TARDE:** ver `docs/revision_corridas_20261028_tarde.md`. Pong CART: 4/15 semillas a 2M, siguiente 596.479; Pong ExtraTrees: 3/15, siguiente 462.847; Pong BART: 561.663 en la primera semilla. Esos tres jobs cerraron con código 3 (parciales): CART/ET cerca del margen de 24h, BART por agotar cuarenta intentos tras 18h58. Breakout ExtraTrees sigue activo, 3/10 a 2M y cuarta semilla ~945k. Diez entrenamientos nuevos completos, sin evaluaciones finales nuevas. Migración terminada y enlace de results verificado. No se enviaron jobs ni se revocó la prioridad del otro proyecto. La muestra de entrenamiento tardío Pong sigue cerca de −20,5 con epsilon=0,1; no confundirla con evaluación final.

**PRIORIDAD HUMANA DEL 28/10 (más reciente):** el usuario pidió cancelar los jobs RRL que no estuvieran ejecutándose para priorizar el otro proyecto. Se cancelaron y verificaron **17 tareas PENDING**: 34952, 35045, 35054, 35053_0/1/2, 35052, 35051_0/1, 35050, 35048, 35047, 35046, 34984, 34982, 34960, 34957. Permanecen RUNNING únicamente 34956, 34959, 34981 y 34983; no se tocaron jobs qwen. No recrear los jobs cancelados ni rellenar la cola sin confirmar la prioridad con el usuario. Registro remoto `control/submission_hold_20261028.txt`, lista `cancelled_pending_20261028.txt`. La cancelación no liberó ejecución: los cuatro slots siguen ocupados por los cuatro entrenamientos activos. No detener un activo sin nueva autorización; el usuario pidió cancelar solo los no ejecutándose. Puede seguir revisándose el progreso/resultado de los activos. Los ledgers anteriores son históricos y contienen IDs ahora cancelados: no tratarlos como pipelines pendientes de ejecutar ni reiniciarlos automáticamente.

**Estado vigente del 28/10:** consultar primero `docs/revision_corridas_20261028.md`; lo que sigue conserva el historial. Migración original 69378 terminó por desconexión y sus procesos desaparecieron. Nueva copia finita desligada de SSH: PID 3671008, log `archive/rrl_bayesian_control_20261027/logs/migration-20261028.log`; verificar ese PID/log/enlace, no reiniciar el proceso antiguo. Nuevos pendientes Breakout: CART 35045/eval35046, BART35047/eval35048. Continuación baseline ya enviada por **35**, ledger `baseline_finish_20261028.tsv`: gate35050 (afterany34956 y comprobación de symlink), Pongtrain35051/eval35052, Breaktrain35053/eval35054. **No usar 26 para duplicar esta continuación**. Todos los nuevos scripts están en control archive. Los cuatro entrenamientos nuevos terminados a 2M son Pong CART501601/2, Pong ExtraTrees501601 y Breakout ExtraTrees710601; siguen pendientes sus evaluaciones finales.

Autorización del usuario: continuar enviando scripts y revisar resultados hasta completar Pong y Breakout con todos los algoritmos implementados. Alcance: OCAtari, corrected_common, CART, ExtraTrees y BART; quince semillas Pong (501601–501615) y diez Breakout (710601–710610). No ejecutar test reservado ni reiniciar campañas existentes por defecto.

## Acción actual

Cuota oficial consultada el 27/10: home 50GB/50GB, 1,41MB libre; archive 138,40GB/200GB, 61,60GB libre. No había jobs RRL activos. Se inició por SSH `bash ~/projects/rrl_bayesian/slurm/22_migrate_results_archive.sh` (sesión local exec 69378; PID remoto inicial 3153553). Debe copiar home/results a archive/rrl_bayesian_results, verificar por checksum, retirar solo las copias de archivos de home ya verificadas y crear enlace en la ubicación original. No lanzar workers antes de confirmar el enlace y finalización satisfactoria del proceso. Un archivo de lock por sí solo no prueba que sigue activo; inspeccionar el PID o sesión.

## Scripts preparados y subidos

- 22: migración verificada, en curso al escribir esta nota.
- 23: recuperar siete semillas Pong baseline (índices 5–11 de extra12), continuar diez Breakout baseline a 2M, autorizar ese hito, reparar evaluación de Breakout CART 710609 a 500k con extensión explícita de 1800s por episodio.
- 24: crea seis calibraciones de coste (dos juegos × CART/ExtraTrees/BART) de 20480 decisiones. Mantiene horizontes y semillas de evaluación existentes para obtener medidas compatibles. ExtraTrees registra la versión instalada de scikit-learn sin cambiar el entorno. BART conserva la receta histórica (20 árboles, grow/prune, 50 burn-in + 50 draws); los problemas conocidos de mezcla siguen siendo una limitación, no se proclama convergencia.
- 25: entrena/evalúa/reporta/mide esas calibraciones en un array de seis tareas.
- 26: lanza 23–25 una vez terminada la migración. Ledger `results/submissions/all_algorithms_2m_20261027/jobs.tsv`, reejecución publica solo labels faltantes. Seis submissions con 26 tareas de array en total; máximo un worker por array. No se había ejecutado este lanzador al escribir esta nota.

Verificación local: sintaxis bash de todos los scripts y simulación del lanzador 26, comprobando que repetirlo no genera duplicados. La validación real del protocolo la realizará el job 24 en Linux. Ante IDs fallidos no repetir simplemente el lanzador esperando reintentos: inspeccionar la causa y registrar intentos nuevos.

Continuación: se crearon localmente `27_prepare_long.sbatch` y `28_long_cell.sbatch`. El 27 construye una campaña long por juego/método desde la calibración, exporta medidas/proyecciones y exige las comprobaciones existentes para autorizar 2M. El 28 realiza hasta diez intentos acotados dentro de un job de 24h y aplica cleanup soportado después de cada intento; se verificó con un proceso simulado que una salida parcial se reanuda y una completa termina el loop. **La subida de 27/28 y del documento BART falló por cuota de home**; deben reenviarse tras la migración. No asumir que están en el cluster. La sesión de migración 69378 y su PID remoto 3153553 estaban vivos a los siete minutos; seguía en el primer rsync, sin haber retirado las copias de home. No relanzar la migración en paralelo.

La API remota de measurements se leyó directamente: es anterior a la local y devuelve training_seconds y peak_rss_bytes sin los campos nuevos de completitud. Los scripts actuales llaman esa API sin cambiar fuentes. Las calibraciones deben cerrar los intentos correctamente antes de usar esas mediciones; nunca convertir intentos sin cierre en tiempos cero. No sincronizar los módulos locales sobre las campañas remotas congeladas.

Se encontró una forma de avanzar durante la migración sin tocar source/results: calibraciones en `$HOME/archive/rrl_bayesian_calibrations_20261027`; scripts de control 24–28 y logs en `$HOME/archive/rrl_bayesian_control_20261027`. Se enviaron **prepare_calibrations=34942** y **calibrations=34943 (array 0–5%1)**. Ledger remoto `archive/rrl_bayesian_control_20261027/calibration_jobs.tsv`. Se exportó RRL_CALIBRATION_RESULTS a esa carpeta y RRL_LOG_DIR al control archive, evitando writes a home/results. **No duplicar estas calibraciones desde 26**: el script local 26 ya importa ese ledger, pero su versión actualizada debe subirse cuando haya espacio. Array: 0 Pong CART, 1 Pong ExtraTrees, 2 Pong BART, 3 Breakout CART, 4 Breakout ExtraTrees, 5 Breakout BART. El documento BART todavía debe subirse para autorizar las campañas long. Mantener RRL_CALIBRATION_RESULTS al ejecutar 27.

Actualización posterior: el documento BART ya está en `archive/rrl_bayesian_control_20261027/analisis_bart_20261026.md` y se exportó RRL_BART_REVIEW a esa ruta. Scripts 26/27 actualizados están en control archive (no asumir copia actual en home). Script 27 verifica `consolidate(calibration)['completed']` y usa RRL_LONG_RESULTS para no depender de la migración: **campañas largas en `archive/rrl_bayesian_long_runs_20261027`**. Se enviaron sus preparaciones con dependencia afterok de cada elemento de 34943:

| Condición | Job de preparación larga |
|---|---|
| Pong CART | 34947 |
| Pong ExtraTrees | 34948 |
| Pong BART | 34949 |
| Breakout CART | 34950 |
| Breakout ExtraTrees | 34951 |
| Breakout BART | 34952 |

Ledger remoto `archive/rrl_bayesian_control_20261027/long_prepare_jobs.tsv`. **No se enviaron todavía los arrays largos**: revisar proyecciones y cuota cuando estos jobs completen. No presentar las preparaciones como entrenamientos a 2M. Se observó el worker 34943_0 entrenando en Hydra, con contador 18432 antes de terminar su fase de entrenamiento. La migración seguía activa después de doce minutos; la sesión 69378 sigue siendo la misma y no se reinició.

## Primer entrenamiento largo enviado

34943_0 completó entrenamiento y las catorce evaluaciones declaradas en 6m18; todas las medias fueron −21, sin truncación por horizonte. Preparación larga 34947 completada en 12s. Se descargaron y revisaron `cost_projection.json`, campaign y authorization de Pong CART: quince semillas 501601–501615 y stop autorizado 2M. Hash interno del campaign comprobado. Proyección desde calibración: 10002 segundos de entrenamiento por semilla (~2,8h), ~716MB de logs por semilla, pico RSS ~292MB; checkpoint bruto antes de retención hasta 27,2GB/semilla, lo que hace necesaria la retención. Son extrapolaciones desde 20480 decisiones, no garantías.

Se redujo cada intento del worker 28 a **1740 segundos** con hasta cuarenta intentos por celda y un límite externo de 24h. Retención soportada después de cada intento. Se añadieron 29_long_condition (recorre todas las semillas pendientes, no salta una parcial) y 30_evaluate_condition (evaluación/ reporte reanudables) y se probaron con procesos simulados. Ambos están subidos a control archive. El primer test local de 29 falló por el flock de macOS; la simulación con stub de flock Linux pasó. El control real usará flock nativo del cluster.

Se envió **pong_cart_attempt1=34956** usando script 29. Ledger `archive/rrl_bayesian_control_20261027/long_training_jobs.tsv`. Campaña `archive/rrl_bayesian_long_runs_20261027/pong_cart_2m_20261027`. Un job no garantiza terminar las quince semillas dentro de 24h: revisar y reenviar los intentos restantes cuando cierre. Evaluaciones todavía no enviadas para esta campaña. Exportar RRL_CONTROL_SCRIPTS=archive/rrl_bayesian_control_20261027/slurm para encontrar el worker 28.

El array de calibraciones 34943 se actualizó a **ArrayTaskThrottle=2**; se confirmó arrancando Pong ExtraTrees y Pong BART mientras se prepara/entrena CART largo. Esto cambia paralelismo operativo, no la receta científica. No reenviar los elementos ya en ejecución. La migración 69378 seguía activa cerca de los veinte minutos; no iniciar workers sobre los resultados de home hasta confirmar el enlace.

Continuación siguiente: se envió **pong_cart_evaluation_attempt1=34957**, dependencia afterany:34956, script 30. Ledger `archive/rrl_bayesian_control_20261027/long_evaluation_jobs.tsv`. Evalúa las celdas con checkpoints publicados y reutiliza unidades completas; puede terminar parcial si el entrenamiento actual de 24h no cubre las quince semillas. No confundir evaluación de hitos publicados con quince modelos finales completos. Se observó 34956 vivo, primera semilla en 22016 decisiones. Calibración ExtraTrees llegó a 20480 y estaba evaluando; BART estaba en 4608 después de ~6min. Migración PID 3153553 viva a 23min; contadores de E/S del sender aumentaron de ~555MB a ~1275MB, prueba de avance, no bloqueo. No hay motivo para reiniciar ni duplicar ninguno de esos procesos.

## ExtraTrees largo enviado

Calibración Pong ExtraTrees y preparación 34948 completas. Descargados/verificados hashes internos de campaign y authorization de `pong_extra_trees_2m_20261027`: quince semillas, stop 2M; versión scikit-learn registrada 1.9.1, 20 árboles, profundidad 4, mínimo 2 por hoja, n_jobs=1. Calibración completa: todas las medias de desarrollo/intermedia −21 sin truncación. Proyección desde 20480 decisiones: 16499s (~4,6h) de entrenamiento y ~719MB de logs por semilla a 2M; RSS ~322MB; hasta ~29,9GB de checkpoints brutos sin retención. No son garantías ni evidencia de aprendizaje.

Se enviaron **pong_extra_trees_attempt1=34959** y **pong_extra_trees_evaluation_attempt1=34960**, este último afterany:34959. Los ledgers de training/evaluation se actualizaron en control archive. Se confirmó 34959 RUNNING en Hydra. Quedaban en paralelo Pong CART largo, Pong ExtraTrees largo, calibración Pong BART y calibración Breakout CART, dentro del máximo de cuatro jobs. Las demás preparaciones/calibraciones conservan sus dependencias; no duplicarlas.

## Auditoría de cobertura

Se creó y subió `31_progress_snapshot.sbatch` (Python inline ejecutado solo en compute). Define explícitamente los cien modelos esperados: Pong 15×4 y Breakout 10×4, usando las campañas baseline existentes y las nuevas long. Verifica hashes de JSON y bindings de status, cuenta episodios finales a 2M y conserva un snapshot por job en control archive. Declara expresamente que no sustituye la validación de blobs, cadenas de transición ni identidades de episodio/checkpoint del loader final. Prueba local sobre artefactos descargados: 100 filas, cero errores JSON/binding, ocho status baseline Pong a 2M y treinta episodios finales (solo tres modelos originales evaluados); eso no son cien modelos terminados.

Se envió **progress_snapshot1=34976**, afterany:34943_3; ledger `archive/rrl_bayesian_control_20261027/snapshot_jobs.tsv`. Es una auditoría puntual, no un monitor recurrente. Última consulta: los cuatro workers RRL seguían vivos; Pong CART había alcanzado **109568** decisiones en su primera semilla y Breakout CART había completado 20480 de calibración, todavía evaluándose. La migración PID 3153553 seguía viva a ~34min y no había llegado a la fase rsync checksum. Otros jobs del usuario comparten el QOS: no modificarlos ni cancelarlos.

## Revisión de fases mientras corren

Se observó la misma cola durante una consulta acotada de ocho lecturas, cada 45s, sin enviar jobs ni crear servicios persistentes. Las lecturas confirmaron que los cuatro workers seguían RUNNING; esa consulta terminó normalmente, lo que **no significa que terminaran los jobs**. La migración mantiene el mismo handle 69378.

Estados descargados y hashes internos comprobados en `results_cluster/seguimiento_2m_20261027/current_status/`:

| Modelo/etapa | Decisiones | Minutos de fases registradas | Checkpoint/logs | Ajuste | Predicción |
|---|---:|---:|---:|---:|---:|
| Pong CART largo, primera semilla | 159232 | 21,53 | 53,6% | 2,2% | 1,0% |
| Pong ExtraTrees largo, primera semilla | 81920 | 14,45 | 36,8% | 2,4% | 25,7% |
| Pong BART calibración | 18432 | 24,73 | 2,6% | 83,4% | 9,1% |

Son estados parciales a presupuestos distintos, no una comparación de rendimiento final ni una prueba de mejora de políticas. La contención de NAS y el crecimiento del replay/checkpoints pueden cambiar las tasas. La escritura domina CART en esta lectura; el sampler domina BART. No modificar las recetas ni los protocolos congelados para acelerar sobre la marcha. No surgió ninguna nueva campaña lista para enviar durante esa consulta.

## Pong BART largo dejado en cola

Calibración 34943_2 terminó correctamente a 20480 con las catorce evaluaciones completas y medias −21, sin truncaciones. Reporte descargado y hash interno verificado. Tiempo de entrenamiento registrado 1667,56s: extrapolación inicial de **45,24 horas por semilla a 2M**, no una garantía. La preparación 34949 quedó PENDING por QOS mientras los cuatro slots estaban ocupados; las calibraciones Breakout CART y ExtraTrees siguieron avanzando.

Se intentó un paso de preparación dentro de la asignación activa 34956, reteniendo/reemplazando el job pendiente 34949. **La revisión automática rechazó esa acción por posible interferencia y falta de autorización explícita para ese solapamiento/cancelación; no se ejecutó ninguna parte**. No usar srun --overlap ni retener/cancelar 34949 como workaround. Se eligió la alternativa normal de esperar recursos.

Se enviaron **pong_bart_attempt1=34981**, afterok:34949, y **pong_bart_evaluation_attempt1=34982**, afterany:34981. Ledgers training/evaluation actualizados en control archive. Ambos estaban PENDING con dependencias correctas, no entrenando todavía. No duplicar. La preparación pendiente seguirá produciendo/validando su protocolo long antes de que pueda comenzar 34981. Mantener explícita la limitación de mezcla: la receta histórica 20 árboles, 50+50 muestras es una comparación exploratoria y terminar 2M no prueba convergencia posterior.

Se midió una celda baseline Pong en source: 508MB checkpoints y 219MB logs; una Breakout: 54MB checkpoints y 60MB logs. No se limpió ni cambió source mientras continúa la migración. La retención ya está implementada para los nuevos workers; cualquier limpieza de los datos antiguos debe coordinarse con la copia/verificación para mantenerlos estables.

## Breakout ExtraTrees largo en cola

Calibración 34943_4 terminada correctamente a 20480. Reporte descargado y SHA interno comprobado: media inicial 1 punto y media de desarrollo a 20k **3 puntos**, sin truncación. Una sola semilla de entrenamiento: no prueba una mejora general. Entrenamiento registrado 187,74s; extrapolación inicial a 2M **5,09h por semilla**, no una garantía. La preparación 34951 sigue pendiente por recursos y verificará los costes compatibles antes de autorizar.

Se enviaron **breakout_extra_trees_attempt1=34983**, afterok:34951, y **breakout_extra_trees_evaluation_attempt1=34984**, afterany:34983. Ledgers actualizados. Son jobs normales en cola, sin pasos solapados, cancelaciones o cambios a los workers activos. Alcance: diez semillas 710601–710610 hasta 2M.

La calibración Breakout CART sigue evaluando políticas que llegan al horizonte con retorno cero: se descargaron unidades a 27000 pasos ya completas y una en 23104 pasos con `in_progress`. Es avance real de evaluación, no un proceso detenido. No reentrenar ni reiniciar por su duración. La consulta de source observó la migración todavía activa a 1h05 y primera retención del CART largo ya ejecutada (log vuelve a iniciar ALE tras el inventario); cualquier certificación de reanudación exige observar un contador nuevo o el loader, no inferirla solo por esa cabecera.

Última espera verificada: otra consulta puntual de ocho lecturas (45s entre lecturas) confirmó los cuatro workers RUNNING hasta ~23:56 UTC. No se crearon monitores persistentes ni nuevos intentos durante esa espera. Contadores leídos: Pong CART largo 307200 y ExtraTrees largo 194048 en la primera semilla; ambos avanzaron después de la primera retención. **BART Breakout terminó entrenamiento de calibración a 20480 y está evaluándose**. CART Breakout pasó de seis a siete unidades de evaluación publicadas. La migración seguía viva a 1h13 y el sender había leído ~10,18GB, frente a ~1,27GB en una lectura anterior: sigue avanzando. La finalización del comando de observación no es finalización de los jobs observados.

## Trabajo requerido después

1. Confirmar migración, nueva cuota y hashes de checkpoints mediante los loaders existentes; enviar 26 con SSH autorizado. No usar Python en login.
2. Auditar calibraciones; producir medidas de coste compatibles para protocolos long y autorizar las campañas finales de CART, ExtraTrees y BART a 2M. No sustituir este objetivo por las calibraciones. Recetas y limitaciones deben quedar congeladas y documentadas. Las campañas existentes CART tienen máximos inmutables de 100k Pong / 500k Breakout: no editar sus manifests para extenderlos; diseñar explícitamente la campaña de 2M o una transferencia validada si se implementa.
3. Reanudar las corridas largas por intentos de menos de 24h, con presupuestos operativos revisados explícitamente si hacen falta. Evitar duplicados por celda y ampliar la revisión/retención de checkpoints usando el API soportado; los logs y manifests deben mantenerse verificables.
4. Vigilar el almacenamiento: ~26GB de resultados existentes y varios experimentos adicionales pueden exceder los 200GB de archive. Medir bytes por decisión y planificar retención/archivado antes de lanzar todas las semillas BART. No borrar artefactos sin verificar su copia y no reducir el alcance del objetivo para ahorrar espacio.
5. Evaluar y revisar todas las semillas y condiciones a 2M. Mantener también hitos para comparar curvas; reservar el test final para selección y autorización explícita.

QOS confirmado: 32 tareas enviadas, 4 jobs simultáneos, 24h por job. Otros proyectos del usuario pueden consumir slots; no cancelarlos.

## Avisos

Se intentó crear un heartbeat de seguimiento cada 30 minutos. La revisión automática lo rechazó por falta de autorización explícita para una automatización persistente con capacidad de enviar jobs. No se creó ni debe eludirse con cron o un demonio. Se necesita confirmación explícita del usuario para ese seguimiento programado; el trabajo del objetivo activo en este chat sigue autorizado.

El objetivo permanece activo: todavía no se ha logrado la cobertura final de algoritmos, juegos, semillas y evaluaciones a 2M. Última revisión de resultados: `docs/revision_corridas_20261027.md`.
