# Bloque B: comparación online progresiva de Pong

Actualización del bloque C (14/10/2026): la extensión multijuego y la ruta explícita de test sellado están descritas en [atari_comparison_guide.md](atari_comparison_guide.md). Las instrucciones siguientes documentan el alcance original de B; su campaña larga y su test no se ejecutaron.

Entregas independientes: [B8](pong_online/B8.md), [B9 opcional](pong_online/B9.md), [B10](pong_online/B10.md), [B11](pong_online/B11.md), [B12](pong_online/B12.md). El código integra RRTL corregido, CART por lotes, BART y ExtraTrees opcional. No cambia TD/F-test, recompensa o hechos del contrato común A3. No se ejecutó una campaña larga ni test reservado ni se enviaron jobs.

## Historia preservada

Antes de modificar los cuatro puntos de integración protegidos se creó `historical/pong_pilot_linux_01_code/` mediante `preserve_pong_history`. `historical-preservation.json` conserva el hash del protocolo y sus 40 fuentes verificadas. Esta copia ejecutable incluye los exportadores y diagnósticos A originales. Los resultados y checkpoints históricos permanecen en `results_cluster/pong_pilot_linux_01`; no se copian ni migran.

El directorio histórico está ignorado por Git, como artefacto local. Transferirlo aparte junto con los resultados si se necesita operar el piloto en otra máquina; conservar el entorno Linux original y ROM. Para trabajar en A6/A7 desde esa copia, usar una configuración de análisis nueva con `source_campaign`/`source_trace` absolutos, tal como indica `docs/pong_analysis_v2.md`. Ejecutar sus comandos desde la copia con `PYTHONPATH` apuntando allí. No ejecutar sobre el piloto los loaders del checkout modificado: su rechazo de hashes es correcto y se mantiene. No se rebajan comprobaciones de código o runtime.

## Preparar el entorno Linux

Usar el entorno ya instalado del proyecto. `requirements-ocatari.txt` fija el runtime común; para ExtraTrees, usar también `requirements-pong-analysis.txt` (scikit-learn 1.9.1). La instalación se hace una vez, fuera de jobs. `slurm/common.sh` verifica dependencias y registra el entorno; no las instala. Ejecutar los comandos desde la raíz del repo, con el mismo entorno y variables numéricas de freeze/train/evaluate. Un cambio de código, arquitectura, versiones o configuración de threads numéricos invalida la continuidad.

```bash
source slurm/common.sh
python -m pytest tests/test_pong_online.py tests/test_cluster_launch.py -q
python -c 'from bayesian_rrtl.ocatari_backend import verified_versions; print(verified_versions())'
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/pong_online_smoke.json
```

`source slurm/common.sh` requiere `SLURM_SUBMIT_DIR` o `RRL_ROOT`; fuera de Slurm usar `RRL_ROOT="$PWD" source slurm/common.sh` o activar el entorno y fijar los threads a 1 como el batch. No mezclar un freeze manual con otra configuración de threads al ejecutar Slurm.

## Smoke y recuperación

Prueba simulada completa, acotada y descartable:

```bash
python -m experiments.validate_pong_online_recovery --output results/pong_online_recovery_sim_01
```

Publica checkpoints, interrumpe mediante USR1 real, reanuda en proceso distinto y compara decisiones, estructura/parámetros y RNG con continuo. Verifica segundo writer, evaluación repetida y reconstrucción. Produce `recovery_report.json`, reportes por campaña, intentos y microbenchmark de caché.

Validación pequeña Atari en Linux, **comandos preparados, no enviados**:

```bash
python -m experiments.validate_pong_online_recovery --atari --cell 1 --output results/pong_online_recovery_atari_cart_01
# Las seis celdas de smoke con los tres learners y ambos extractores:
sbatch slurm/08_online_smoke.sbatch --output results/pong_online_recovery_atari_all_01
```

`--cell 1` valida CART visual; índices de smoke: 0 baseline visual, 1 CART visual, 2 BART visual, 3 baseline OCAtari, 4 CART OCAtari, 5 BART OCAtari. Los resultados simulados no demuestran desempeño ni compatibilidad exacta del emulador Linux. Los parámetros reducidos de smoke no satisfacen los diagnósticos científicos A6/A7.

## Congelar antes de distribuir

```bash
python -m experiments.compare_pong_progressive --stage freeze --protocol configs/pong_online_smoke.json --campaign results/pong_online_smoke_linux_01
# Entrada compatible desde el runner anterior, con selección explícita del protocolo nuevo:
python -m experiments.compare_pong --progressive --stage dry-run --protocol configs/pong_online_baseline.json
```

El manifiesto congela celdas, parámetros, calendario, runtime y hashes. Repetir freeze solo acepta identidad idéntica. No usar una configuración mutable para resume/evaluate: únicamente `--campaign`.

Cada job entrena una celda. Alternativa array (el batch toma el índice de `SLURM_ARRAY_TASK_ID`):

```bash
sbatch slurm/06_online_preflight.sbatch
sbatch slurm/07_online_freeze.sbatch --protocol configs/pong_online_smoke.json --campaign results/pong_online_smoke_linux_02
# Esperar que freeze termine antes de enviar workers:
sbatch --array=0-5 slurm/09_online_train_cell.sbatch --campaign results/pong_online_smoke_linux_02
# Evaluación después de publicar checkpoints; no escribe entrenamiento:
sbatch --array=0-5 slurm/10_online_evaluate.sbatch --campaign results/pong_online_smoke_linux_02
sbatch slurm/12_online_report.sbatch --campaign results/pong_online_smoke_linux_02
```

También se puede enviar un job individual con `--cell visual/cart/501501`. Un índice inválido no crea la carpeta de celda. La evaluación usa únicamente checkpoints confirmados; los episodios ya completos se reutilizan y los parciales continúan. El reporte puede ejecutarse con trabajos pendientes y termina con código 3; no simula completitud. Los reportes no abren entornos ni consumen semillas.

## Desarrollo progresivo del baseline

El máximo queda congelado en 2.000.000 decisiones y contiene hitos 100.000, 500.000 y 2.000.000. Autorizar uno no cambia epsilon, TD, F-test, horizonte, semillas o parámetros. Las evaluaciones intermedias y por hito ya están declaradas, y la principal final corresponde al máximo congelado.

```bash
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/pong_online_baseline.json
python -m experiments.compare_pong_progressive --stage freeze --protocol configs/pong_online_baseline.json --campaign results/pong_baseline_progressive_linux_01
python -m experiments.compare_pong_progressive --stage authorize --campaign results/pong_baseline_progressive_linux_01 --stop-at 100000 --reason "Desarrollo inicial aprobado a 100k"
sbatch --array=0-5 slurm/09_online_train_cell.sbatch --campaign results/pong_baseline_progressive_linux_01
sbatch --array=0-5 slurm/10_online_evaluate.sbatch --campaign results/pong_baseline_progressive_linux_01
python -m experiments.compare_pong_progressive --stage report --campaign results/pong_baseline_progressive_linux_01
```

Los workers al llegar a un hito intermedio quedan parciales con `authorized_milestone_reached` (código 3). Para continuar, registrar el hito siguiente y reanudar las mismas celdas:

```bash
python -m experiments.compare_pong_progressive --stage authorize --campaign results/pong_baseline_progressive_linux_01 --stop-at 500000 --reason "Continuación aprobada tras revisar desarrollo y costo"
sbatch slurm/11_online_resume.sbatch --campaign results/pong_baseline_progressive_linux_01 --cell visual/corrected_common/501601 --extend-seconds 3600 --extension-id desarrollo-500k-01 --reason "Ampliación operativa aprobada para esta celda"
```

Extensiones suman segundos al límite acumulado, mantienen consumo y contadores, y son idempotentes por ID. Repetir ID con otro monto/motivo se rechaza. Para un array, el incremento se aplica **a cada celda**; contabilizar la suma de recursos, no confundirla con wall time paralelo. La evaluación tiene extensión por episodio: `--extend-seconds 300 --extension-id eval-01 --reason "..."`, aplicada a cada unidad pendiente seleccionada. Se puede limitar con `--group intermediate --point 100000`.

## Costos y retención

```bash
python -m experiments.compare_pong_progressive --stage measurements --campaign results/pong_baseline_progressive_linux_01 > results/baseline_cost_linux.json
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/pong_online_baseline.json --cost-reference results/pong_baseline_progressive_linux_01
python -m experiments.compare_pong_progressive --stage inventory --campaign results/pong_baseline_progressive_linux_01 --cell 0
# Solo después de revisar el inventario; no hay limpieza automática:
python -m experiments.compare_pong_progressive --stage cleanup --campaign results/pong_baseline_progressive_linux_01 --cell 0
```

Retención protege todo el calendario, final/último, respaldo y bindings evaluativos. Los logs confirmados y metadatos no se retiran. La corrida no duplica checkpoints al cambiar solo estado administrativo en el mismo contador. Los segmentos no confirmados se listan aparte y no entran al reporte. Las mediciones simuladas y las de smoke con otra configuración/horizonte se rechazan como referencia de Atari larga.

Se reportan tasas y rangos observados, proyecciones de logs/checkpoints y RSS observado. No son cotas máximas: el llenado de replay, estructura aprendida, duración de partidas, carga del filesystem y hardware cambian los costos. El inventario no lee todos los grandes blobs para sumar espacio; cleanup verifica los retenidos antes de eliminar los demás.

Cada intento tiene recursos/nodo/entorno/versiones/duración. Los batches train/resume usan `--attempt-seconds 3480` para dejar dos minutos de margen en su solicitud de una hora; si se cambia `sbatch --time`, ajustar también este argumento. El límite es operativo, queda registrado y se combina con el presupuesto acumulado antes de cada fit; no cambia decisiones ni epsilon. Un SIGKILL puede dejar un intento sin final y hacer repetir decisiones hasta el último checkpoint (512 en baseline; 16 en smoke). Un modelo parcialmente ajustado nunca se guarda. La integridad se verifica antes de deserializar pickles locales confiables; corrupción o código incompatible se rechazan.

## Preparar la larga, sin activarla

```bash
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/pong_online_long_pending.json
sbatch slurm/13_online_dry_run.sbatch --protocol configs/pong_online_long_pending.json
```

Los dos comandos solo preparan. Para ejecutar posteriormente:

1. Completar A6/A7 y una revisión con conclusiones y hashes de artefactos; elegir parámetros BART desde esa evidencia. No rellenar `learners.bart` con ajustes inventados.
2. Medir en Linux una variante de costo con el mismo contrato, horizonte, aprendizaje, caché y estimadores definitivos de los tres métodos, con evaluaciones de desarrollo. El máximo corto de esa variante puede diferir; sus datos no consumen test.
3. Crear una **nueva** configuración resuelta a partir del template pendiente: BART/revisión, máximo de decisiones, hitos, calendario y presupuestos. Congelar una campaña nueva. No editar el manifiesto ya congelado.
4. Revisar `dry-run --cost-reference` y autorizar explícitamente el hito con `--cost-reference /ruta/campaña_de_costo_linux --reason "..."`. La autorización larga exige Linux y medidas compatibles de entrenamiento/evaluación para cada learner/pipeline.
5. Distribuir una celda por job y evaluar checkpoints independientemente. Reservar el test para otra etapa autorizada; este runner no tiene ruta para él.

No se garantiza competencia de ninguna política. Un resultado al piso se informa, junto con puntos y dispersión, sin declarar equivalencia entre learners.

El reporte final de esta entrega está en [pong_online_preparacion.md](pong_online_preparacion.md).
