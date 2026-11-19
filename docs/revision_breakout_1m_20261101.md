# Resultados Breakout a 1M — 1 de noviembre de 2026

Se consultó Slurm y se actualizaron estados/evaluaciones en `results_cluster/breakout_1m_latest/`. El resumen reproducible validó hashes, identidades y contratos compartidos de la descarga. No se descargaron checkpoints de modelos ni cadenas completas de entrenamiento para repetir localmente su validación.

## Evaluación final de desarrollo

| Método | Semillas evaluadas / solicitadas | Media entre agentes | Mediana | Episodios truncados |
|---|---:|---:|---:|---:|
| Baseline | 10/10 | 27,9 | 25 | 10/100 |
| CART | 10/10 | 9,1 | 2 | 0/100 |
| ExtraTrees | 10/10 | 5,3 | 3 | 0/100 |
| BART | 8/10 | 7,625 provisional | 3 provisional | 20/80 |

Total: 38/40 modelos entrenados a 1M y con sus diez episodios finales completos. No equivale todavía a cerrar cuarenta calendarios. Baseline, CART y ExtraTrees tienen sus archivos `calendar_progress_1m.json` completos y sus jobs de cierre terminaron correctamente; BART permanece abierto. Las cifras BART usan solo agentes ya evaluados; no imputan los pendientes como cero. No comparar formalmente esa media parcial con las cohortes completas.

Retornos medios BART por semilla: 710601=4, 710602=40, 710603=3, 710604=9, 710605=2, 710606=3, 710607=0, 710609=0. Las diez evaluaciones de cada agente repiten su retorno; la unidad independiente para comparar métodos sigue siendo el agente/semilla de entrenamiento, no cada episodio.

## Hallazgo: agentes que no ejecutan FIRE

Los agentes BART710607 y710609 completaron sus diez evaluaciones por horizonte: cada episodio llega a27.000 decisiones, retorno0 y cuatro estados codificados distintos. Sus conteos agregados son:

- 710607: LEFT269.990 y RIGHT10; FIRE0.
- 710609: RIGHT269.900, LEFT10 y NOOP90; FIRE0.

En ambos casos son políticas degeneradas de movimientos laterales sin FIRE durante la evaluación. Esto es coherente con no lanzar/reiniciar la pelota; para establecer el momento exacto falta inspeccionar una traza de objetos/Q o vídeo. No es evidencia de sobreajuste por sí sola ni un fallo de Slurm. Tampoco es un colapso universal a NOOP: uno repite LEFT y otro RIGHT.

710608 ya llegó a1M y tiene tres de diez evaluaciones completas, también truncadas con retorno0. En esos tres episodios alterna LEFT/RIGHT (40.497/40.503 acciones agregadas), FIRE0 y cuatro estados codificados por episodio. Se mantiene pendiente en la tabla hasta completar diez episodios. 710610 sigue entrenándose: snapshot en802.304 decisiones, sin evaluación final todavía.

La semilla BART710602 sí obtuvo40 puntos en diez partidas completas naturales. Ejemplo de un episodio:1.095 decisiones,205 FIRE,235 LEFT,445 NOOP,210 RIGHT y72 estados codificados. Esto demuestra capacidad de obtener puntos en un agente concreto, no estabilidad del método ni superioridad sobre CART/baseline.

## Estado operativo

Los cinco grupos originales BART35470_0–4 cerraron parciales con código3 cerca de23h50, preservando checkpoints. Continuaciones corregidas35782_0,35783_1 y35784_2 terminaron correctamente en2h18,2h59 y2h33, respectivamente. Confirma que la reparación del acceso a `sacct` permitió continuar sin reiniciar desde cero.

Siguen activos35785_3 (~6h27) y35786_4 (~4h09). No hay un nuevo fallo en estos workers ni motivo para duplicarlos. Reporte35471 y cierre35695 siguen pendientes de esas continuaciones.

Se envió además **35965**, evaluación anticipada del calendario BART, usando un slot libre y el script42 existente. Evalúa los agentes ya en1M y respeta el mismo lock de grupo que los entrenamientos. Si encuentra un grupo todavía ocupado o incompleto, guarda los episodios hechos y devuelve parcial. No entrena ni modifica la receta. Ledger `control/breakout_bart_early_calendar_jobs.tsv`.

El cierre definitivo35695 se actualizó para esperar también35965, evitando dos writers del calendario a la vez. En la consulta inmediata35965 estaba PENDING; enviar no demuestra que ya haya ejecutado evaluaciones. Si el cierre final termina parcial por tiempo, habrá que reanudar las unidades pendientes y volver a validar el reporte.

## Descarga y próximos diagnósticos

La primera descarga devolvió rsync24 porque desaparecieron snapshots temporales bajo `evaluations/.../objects` mientras el evaluador avanzaba. No fue corrupción de resultados confirmados. Se repitió con exclusión de `objects/`, `*.blob` y `*.tmp`, terminó con código0 y se regeneró el resumen desde los metadatos estables. Usar estos filtros en próximas descargas ligeras; el estado de evaluación sigue validándose contra su identidad y hash.

Falta completar las dos evaluaciones finales BART, terminar su calendario y validar el cierre de diez agentes. Para explicar los ceros, el siguiente análisis debería comparar Q(FIRE) con Q(LEFT/RIGHT/NOOP) y las relaciones del estado inicial y tras perder una vida. No cambiar ahora la política ni introducir FIRE automático dentro de esta cohorte congelada, pues alteraría la comparación.

El objetivo de cuarenta modelos a1M con su calendario permanece activo e incompleto. No se autorizó2M en estas campañas ni se utilizó el test reservado.
