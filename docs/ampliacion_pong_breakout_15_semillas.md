# Ampliación: Pong y Breakout, 15 semillas por condición

Fecha de preparación: 2026-10-26. Este documento y las configuraciones no envían jobs.

**Cambio de alcance solicitado posteriormente:** continuar sólo OCAtari. El objetivo activo pasa a Pong OCAtari y Breakout OCAtari, baseline/BART, 15 semillas por condición (60 entrenamientos). La tabla de 90 más abajo describe el alcance anterior, no una autorización para seguir enviando visual. Conservar los resultados visuales existentes como exploratorios.

Cancelar los jobs visuales ya aceptados por sus IDs verificados (el array `34400` en la consulta de transición). No cancelar arrays compartidos enteros si contienen tareas OCAtari. Para los nuevos envíos usar `bash slurm/17_submit_pong_breakout.sh --resume --ocatari-only`. La opción no cancela jobs por sí sola: omite los workers, evaluaciones y reportes visuales nuevos; en los nuevos diagnósticos sólo solicita el índice 0 (OCAtari). No borrar el registro de envíos ni los resultados parciales.

## Alcance

Objetivo de desarrollo: 2.000.000 decisiones por entrenamiento y 15 semillas de entrenamiento por condición. No son 15 episodios de evaluación.

| Juego y representación | Baseline RRTL corregido | BART |
| --- | --- | --- |
| Pong visual | 15 | 15 |
| Pong OCAtari | 15 | 15 |
| Breakout OCAtari | 15 | 15 |

Son 90 entrenamientos en total. No se añade Demon Attack, CART ni Breakout visual a este alcance. Los tres baselines visuales ya completos cuentan dentro de los 15, no como réplicas adicionales. Un checkpoint de 500k y otro de 2M de la misma semilla no son dos entrenamientos independientes.

## Qué se puede reutilizar

- Pong visual: semillas 501601–501603 completas a 2M, con evaluación de desarrollo completa. Conservarlas en su campaña original.
- Pong OCAtari: semillas 501601–501603 completas hasta 500k. Continuar sus checkpoints originales, sin volver a entrenarlas desde cero.
- Pong visual adicional: 501604–501615, campaña `results/pong_visual_2m_extra12_20261025_01` ya congelada y autorizada. Sus workers aún no se habían enviado al preparar este documento.
- Los controles de Pong y los datasets comunes están en `results/pong_analysis_v2_linux_20261025_01`, vinculados a `historical/pong_pilot_linux_01_code`. Operarlos desde esa copia; no sustituir su código ni desactivar verificaciones.
- Breakout ya pasó el smoke de integración Linux, pero 64 decisiones no son una medición de coste ni de aprendizaje a escala larga.

Los reportes de las campañas adicionales son separados. Para el análisis conjunto, agrupar por juego, representación, algoritmo y semilla de entrenamiento, comprobando los parámetros científicos; no promediar reportes como si fueran nuevas réplicas ni cambiar sus manifiestos para reunirlos.

## Configuraciones baseline

1. `configs/pong_online_baseline_visual_2m_extra12.json`: 12 nuevas semillas visuales, 501604–501615.
2. `configs/pong_online_baseline_ocatari_2m_extra12.json`: las mismas 12 semillas, pero OCAtari. Mantiene aprendizaje, exploración, evaluación y contrato de Pong del baseline existente. Presupuesto operativo de seis horas por celda; no es una promesa de duración.
3. `configs/breakout_online_baseline_15seeds.json`: 15 celdas explícitas de Breakout OCAtari, 710601–710615, en el runner común v3. Usa las acciones y relaciones auditadas de Breakout. Máximo 2M; hitos 100k/500k/2M; evaluaciones en 0/10k/100k/500k/2M y diez semillas de desarrollo por hito. Test reservado sin calendario.

La receta de Breakout utiliza el baseline común: profundidad 4, soporte 128, gamma 0,99 y epsilon 1→0,1 en 50k decisiones. No es el archivo legacy `configs/breakout_baseline.json` ni una configuración ya optimizada para Breakout. Se conserva esta receta entre baseline y sus futuras comparaciones, salvo que se declare otra variante antes de congelar.

Las dos configuraciones nuevas pasaron `bayesian_rrtl.online.protocol.resolve`: 12 y 15 celdas respectivamente, sin campos pendientes. Eso valida configuración, no competencia ni coste en el emulador.

## Orden de ejecución

### Lanzador preparado

`bash slurm/17_submit_pong_breakout.sh` se ejecuta en el login node y sólo envía jobs; no ejecuta Python allí. Exige las campañas y la copia histórica descritas antes. No combinarlo con envíos manuales de esas mismas celdas.

Envía preparación, las doce semillas visuales, continuación de tres OCAtari y doce OCAtari nuevas, y una sola semilla de Breakout a 100k, con evaluación y reportes dependientes. Dos ramas limitan a dos entrenamientos simultáneos del lanzador; evaluaciones y diagnósticos compiten además por los límites del scheduler.

También envía colección asistida de ambos pipelines, exportación histórica de X/y y auditoría BART base: dos pipelines × tres acciones seleccionadas × dos inicializaciones, cuatro cadenas por problema. Agrupa los seis problemas de cada pipeline en una tarea, sin cambiar semillas ni parámetros del muestreo. No abre sensibilidades, no certifica calidad estadística por el estado de Slurm y no lanza BART largo. Una unidad que agote su presupuesto requiere revisión y continuación explícita.

El registro `results/submissions/pong_breakout_20261026_01/jobs.tsv` conserva los IDs. El lanzador rechaza una segunda ejecución normal si existe esa carpeta. Para continuar un envío parcial usar **`bash slurm/17_submit_pong_breakout.sh --resume`**: reutiliza cada ID registrado y sólo envía etiquetas faltantes. No borrar el registro para forzar reenvíos. Un job ya registrado que haya fallado requiere inspección específica, no se vuelve a enviar automáticamente.

El QOS regular del cluster declara MaxSubmitJobsPU=32 y MaxJobsPU=4. Cada elemento de un array pendiente cuenta para el límite de envíos; `%1` sólo restringe ejecución simultánea. El lanzador cuenta tareas antes de enviar y pausa con salida 75 si no hay cupo suficiente. También conserva el registro si `sbatch` rechaza un envío por una carrera o política adicional. Repetir `--resume` cuando terminen tareas; no aumentar artificialmente `RRL_MAX_SUBMIT_JOBS` para eludir la política.

Para ahorrar tareas pendientes, `slurm/18_evaluate_cells.sbatch` evalúa las doce semillas adicionales secuencialmente en un único job por campaña. Conserva unidades completas, continúa las parciales, propaga errores reales y reenvía la señal de parada al evaluador activo. Esto no modifica la política ni el calendario científico.

Se probaron sintaxis, reanudación del registro parcial realista, rechazo de duplicados, pausa por cuota, rechazo de sbatch sin publicar IDs vacíos y dependencias completadas consultadas en accounting. Las pruebas del evaluador secuencial comprueban estados parciales, errores fatales y señal USR1. Se usaron procesos simulados: no equivale a ejecución real en Slurm ni a pruebas de los algoritmos.

### A. Completar baselines de Pong

Lanzar las doce visuales ya preparadas. Continuar las tres OCAtari existentes y crear la campaña de doce OCAtari adicionales. El manifiesto antiguo ya autoriza 2M, pero sus workers deben seleccionar exclusivamente los índices 3–5 para OCAtari.

Ejemplo de continuación de las tres OCAtari, desde Kraken (Python sólo dentro de Slurm):

```bash
cd ~/projects/rrl_bayesian
export RRL_ROOT="$PWD"
export RRL_CONDA_ENV="/home/natan.brugueras/archive/conda_envs/rrl-pong-analysis"
export RRL_BASELINE_CAMPAIGN="$PWD/results/pong_baseline_progressive_linux_20261014_01"

RRL_JOB_OC_RESUME=$(sbatch --parsable \
  --array=3-5%2 --time=04:00:00 \
  --output=rrl-ocatari-2m-train-%A_%a.out \
  slurm/11_online_resume.sbatch \
  --campaign "$RRL_BASELINE_CAMPAIGN" \
  --attempt-seconds 14280 \
  --extend-seconds 14400 --extension-id ocatari-2m-01 \
  --reason "Continuar exclusivamente baseline OCAtari desde 500k a 2M")

RRL_JOB_OC_EVAL=$(sbatch --parsable \
  --dependency="afterany:${RRL_JOB_OC_RESUME%%;*}" \
  --array=3-5%2 --time=04:00:00 \
  --output=rrl-ocatari-2m-eval-%A_%a.out \
  slurm/10_online_evaluate.sbatch --campaign "$RRL_BASELINE_CAMPAIGN")

sbatch --dependency="afterany:${RRL_JOB_OC_EVAL%%;*}" \
  slurm/12_online_report.sbatch --campaign "$RRL_BASELINE_CAMPAIGN"
```

La extensión es por celda, idempotente por ID y no modifica decisiones, modelos ni epsilon. No repetir el envío de workers concurrentes sobre la misma celda. Si se agota tiempo antes de 2M, revisar estado y reanudar: no reiniciar.

### B. Primera prueba de Breakout con la configuración definitiva del baseline

Subir únicamente las configuraciones nuevas; no sincronizar módulos de Python mientras haya campañas activas. Congelar una campaña independiente de Breakout. Ejecutar primero una sola semilla hasta 100k y sus evaluaciones para revisar funcionamiento, señal y coste; no lanzar las 15 directamente hasta 2M basándose en tiempos de Pong.

En un batch que cargue `slurm/common.sh`, el orden es:

```bash
python -m experiments.compare_pong_progressive --stage freeze \
  --protocol configs/breakout_online_baseline_15seeds.json \
  --campaign "$RRL_BREAKOUT_CAMPAIGN"

python -m experiments.compare_pong_progressive --stage authorize \
  --campaign "$RRL_BREAKOUT_CAMPAIGN" --game Breakout \
  --stop-at 100000 --reason "Validacion inicial de baseline Breakout a 100k"
```

Luego `slurm/09_online_train_cell.sbatch --campaign "$RRL_BREAKOUT_CAMPAIGN" --cell 0`, evaluación con `slurm/10_online_evaluate.sbatch --cell 0` después de terminar y reporte. Los otros 14 workers siguen sin enviar. El código 3 por hito intermedio es esperado. Revisar resultados y costes antes de autorizar otros hitos; la autorización v3 exige `--game Breakout`.

### C. Preparar BART antes de distribuir sus 45 entrenamientos

La infraestructura de BART existe, pero faltan evidencia y parámetros finales; `configs/pong_online_long_pending.json` y `configs/atari_comparison_final_pending.json` siguen pendientes. No sustituir sus `null` por parámetros arbitrarios ni cambiar `level=long` para eludir requisitos.

1. Ejecutar colección asistida de Pong: el gate común está habilitado, pero la colección aleatoria contiene sólo un punto positivo en ocho partidas. Mantener aleatoria y asistida como datasets diferentes; los datos asistidos son un diagnóstico offline, no una ventaja silenciosa exclusiva de BART online.
2. Exportar el problema fijo histórico con su código preservado y ejecutar A6: cadenas, sigma², ESS/Rhat, inicializaciones y sensibilidades revisadas. Los diagnósticos históricos no certifican por sí solos todos los ajustes online ni los datos de Breakout.
3. Elegir y registrar la receta BART: árboles, profundidad/soporte, prior, burn-in, muestras, frecuencia de ajustes, escala y calibración. Si se cambia el calendario de aprendizaje respecto del baseline, declararlo como una variante, no como una comparación idéntica.
4. Medir una ejecución corta de coste en Linux por condición, con esa receta y el horizonte definitivo, incluyendo evaluación. No extrapolar BART desde los tiempos del baseline o el smoke. Mantener intactas las semillas reservadas de test.
5. Crear configuraciones largas nuevas resueltas, con 15 semillas por condición y referencias verificadas a diagnósticos/costes. Para Pong usar 501601–501615; para Breakout 710601–710615. Mismos IDs permiten parear por semilla entre algoritmos, pero no fuerzan trayectorias idénticas.
6. Congelar, revisar el dry-run, autorizar y enviar celdas por Slurm. Extender por hitos y revisar diagnósticos online; no equiparar `completed` con calidad estadística.

## Interpretación y coste

- 2M es el presupuesto de decisiones elegido para cada juego, no una garantía de alcanzar competencia ni la misma cantidad de partidas o dificultad entre juegos.
- Comparar aprendizaje por decisiones y por tiempo activo. No promediar retornos brutos de Pong y Breakout.
- Quince semillas de entrenamiento permiten medir variación entre modelos; repetir una trayectoria determinista con diez semillas de evaluación no añade diez condiciones diversas.
- En Pong visual, dos de tres modelos siguen débiles a 2M. El resultado de las 15 debe incluir todas las semillas, no sólo la mejor.
- El coste de las 45 corridas BART y de Breakout todavía no está establecido. No hay una estimación válida del total de la ampliación.
- El test reservado se ejecuta sólo tras cerrar decisiones de desarrollo; ningún comando de esta guía lo abre.
