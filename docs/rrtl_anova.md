# RRTL-ANOVA: variante experimental de crecimiento

Implementación del 01/11/2026. RRTL actual sigue siendo la opción predeterminada: `split_test="historical_f"`. Para activar la variante se declara `split_test="anova"` en `BaselineLearnerConfig` o `BaselineConfig`. El identificador del learner sigue siendo `corrected_common`; el criterio queda diferenciado en configuración, hash del contrato, checkpoint y métricas. No es un algoritmo TD distinto ni un nuevo pipeline.

`relational_regresion_tree.py` permanece intacto. La implementación nueva está en `bayesian_rrtl/baseline_anova.py`; `baseline.py` y el adaptador seleccionan el nodo correspondiente. El baseline predeterminado conserva su F-test, criterios alternativos y tratamiento de varianza cero.

## Qué calcula y qué conserva

Cada relación candidata acumula los mismos `n`, `mu` y `J` de antes. El orden sigue siendo: actualizar Q por TD, acumular estadísticas del **Q actualizado**, considerar crecimiento. No se sustituyen las observaciones por targets TD, recompensas ni ejemplos almacenados en replay.

Para los grupos observados no vacíos:

\[
n=\sum_c n_c,\quad K=\#\{c:n_c>0\},\quad
W=\sum_c J_c,\quad B=\sum_c n_c(\mu_c-\mu_{\rm total})^2,
\]
\[
F=\frac{B/(K-1)}{W/(n-K)},\qquad
p=\operatorname{scipy.stats.f.sf}(F,K-1,n-K).
\]

Se selecciona el menor p-value admisible y se divide si `p <= significance_level`. Los empates usan el primer candidato en el orden fijo del catálogo. En ANOVA, `best_literal_criteria` debe ser `"p-value"`; otras combinaciones se rechazan para evitar una configuración ambigua.

Se conservan:

- El umbral histórico estricto `n > min_sample_size`, profundidad máxima y acciones habilitadas para dividir.
- Catálogo, eliminación del literal usado en descendientes y condición histórica de presencia de alguna relación entre objetos.
- Ramas comparativas `more/same/less` y lógicas `True/False`, incluida la creación de ramas actualmente vacías.
- Herencia del Q del padre cuando está habilitada, inicialización en cero si no lo está, y estadísticas nuevas en cada hijo.
- Fallback al Q del nodo interno cuando falta su literal; no se crea una rama de valores ausentes.
- Actualizaciones TD, terminal/truncación, recompensa, exploración y presupuesto de interacciones.

Todos los hijos ANOVA son `ANOVANode`, también cuando siguen creciendo y después de restaurar. Los checkpoints validan el criterio, tipo de nodos y coherencia de las estadísticas; no pueden cambiarse de criterio para continuar el mismo experimento. La reanudación online contrasta la configuración del agente con el estimador congelado antes de restaurar el entorno.

## Degeneración y números inconsistentes

Con menos de dos grupos no vacíos, `n <= K` o dispersión total nula no se divide. Con `W=0` y `B>0`, teniendo grados de libertad válidos, se usa `p=0`. Con `B=0` y `W>0`, `F=0` y `p=1`. Los grupos vacíos no agregan grados de libertad.

La implementación comprueba conteos enteros no negativos, medias/dispersiones finitas, `J >= 0`, conteos por categoría que sumen el total, y coherencia del promedio y la descomposición total `J = W+B`. Las discrepancias grandes o un total nulo incompatible no se convierten silenciosamente en “no dividir”: producen error. Se permite redondeo pequeño entre acumulaciones independientes (tolerancia relativa `1e-8`; absoluta de medias `1e-10 * max(1, |mu_c|)` y de dispersión `1e-12`, manteniendo consistente el cero exacto). Se calcula la razón en logaritmos para evitar overflow/underflow intermedio; un p-value no finito de SciPy produce error explícito.

**ANOVA es un criterio experimental de crecimiento.** La dependencia temporal de Q, su distribución cambiante, la selección del menor p-value entre relaciones y las comprobaciones repetidas impiden asumir que estos p-values controlen la tasa de falsos positivos. `significance_level` es un umbral operativo nominal, no una garantía inferencial ni evidencia confirmatoria de que el split sea correcto. No se añadió una corrección de multiplicidad ni se afirma equivalencia entre criterios si el resultado es similar.

## Configuraciones pareadas preparadas

| Par | Archivo para RRTL actual | Archivo para RRTL-ANOVA | Trabajo por variante |
| --- | --- | --- | --- |
| Integración pequeña | `configs/rrtl_split_tests_smoke_historical_f.json` | `configs/rrtl_split_tests_smoke_anova.json` | 3 juegos × 1 semilla, 64 decisiones/celda; 12 episodios de desarrollo |
| Larga para revisión | `configs/rrtl_split_tests_long_review_historical_f.json` | `configs/rrtl_split_tests_long_review_anova.json` | 3 juegos × 5 semillas, 2.000.000 decisiones/celda; 330 episodios de desarrollo |

Dentro de cada par, los archivos difieren **únicamente** en nombre y `split_test`: mismas semillas, acciones, representación OCAtari, recompensa, hiperparámetros, exploración, descuento, horizontes, calendarios, recursos y límites operativos. Cada variante inicia un modelo independiente. Las políticas diferentes pueden producir trayectorias distintas; no se intenta igualarlas después de observarlas.

El smoke usa `min_sample_size=4` y horizonte 128 como prueba técnica. La larga conserva mínimo 128 y horizonte 27.000. Ambos usan `eta_q=.025`, `gamma=.99`, profundidad 4 y umbral `.0001`. Las configuraciones pequeñas son de nivel `baseline`, porque sólo incluyen RRTL, y requieren autorizar explícitamente su hito 64. No utilizar el wrapper de smoke C que exige los tres learners obligatorios.

La pareja larga representa **30 entrenamientos, 60 millones de decisiones y 660 episodios de desarrollo** en total. Los presupuestos operativos declarados son límites, no estimaciones de costo. No se autorizó ni ejecutó. `level="long"` exige autorización por juego y mediciones Linux compatibles con cada criterio, configuración y runtime; no se puede reutilizar una medición histórica de otro criterio ni el smoke reducido como si fuera la medición definitiva.

## Preflight y preparación

Desde la raíz del repo, activar el entorno fijado del proyecto y mantener la misma configuración de threads en freeze/train/evaluate. `slurm/common.sh` ya lo hace en el cluster. OCAtari debe seguir en las versiones fijadas; esta implementación no instala ni actualiza dependencias.

```bash
python -m pytest tests/test_baseline_anova.py tests/test_baseline_adapter.py tests/test_baseline_checkpoint.py -m 'not atari and not ocatari' -q
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/rrtl_split_tests_smoke_historical_f.json
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/rrtl_split_tests_smoke_anova.json
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/rrtl_split_tests_long_review_historical_f.json
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/rrtl_split_tests_long_review_anova.json
```

El dry-run sólo resuelve las celdas enumeradas y muestra trabajo, pendientes y costo desconocido cuando faltan mediciones. Se generaron los cuatro dry-runs localmente; están en `results/rrtl_anova_validation_20261101/`. No se ejecutó entrenamiento Atari real: OCAtari no está instalado en el entorno local de esta validación. Las pruebas del runner usan el driver RAM sintético de C.

## Campaña pequeña, comandos preparados

Ejecutar solamente cuando se quiera correr el smoke real en el entorno con OCAtari. No se enviaron estos jobs durante la implementación. Congelar desde el mismo checkout/runtime que ejecutará los workers:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.compare_pong_progressive --stage freeze --protocol configs/rrtl_split_tests_smoke_historical_f.json --campaign results/rrtl_split_smoke_historical_f_01
python -m experiments.compare_pong_progressive --stage freeze --protocol configs/rrtl_split_tests_smoke_anova.json --campaign results/rrtl_split_smoke_anova_01
```

Para **cada** campaña, revisar sus tres celdas y autorizar el hito en cada juego. Ejemplo para la variante ANOVA (repetir en la campaña histórica para compararlas):

```bash
python -m experiments.compare_pong_progressive --stage authorize --campaign results/rrtl_split_smoke_anova_01 --game Breakout --stop-at 64 --reason "Smoke pareado revisado"
python -m experiments.compare_pong_progressive --stage authorize --campaign results/rrtl_split_smoke_anova_01 --game DemonAttack --stop-at 64 --reason "Smoke pareado revisado"
python -m experiments.compare_pong_progressive --stage authorize --campaign results/rrtl_split_smoke_anova_01 --game Pong --stop-at 64 --reason "Smoke pareado revisado"
sbatch --array=0-2 slurm/09_online_train_cell.sbatch --campaign results/rrtl_split_smoke_anova_01
# Esperar entrenamiento antes de evaluar:
sbatch --array=0-2 slurm/10_online_evaluate.sbatch --campaign results/rrtl_split_smoke_anova_01
```

Para una sola celda se puede usar el mismo CLI con `--stage train-cell --cell 0`. Reanudación: `--stage resume --campaign ... --cell 0`; los presupuestos agotados requieren extensión explícita con ID y motivo según la [guía común](atari_comparison_guide.md). No cambia el criterio ni reinicia contadores. No hay un batch nuevo de entrenamiento ni ejecución automática de test.

`train_baseline.py --config ...` también acepta `split_test` en su JSON de `BaselineConfig`. El runner visual `train_atari.py --variant corrected` dispone de `--split-test anova`; por defecto sigue usando histórico. Para la comparación principal multijuego, usar los protocolos anteriores y el runner común.

## Reporte pareado y métricas

```bash
python -m experiments.report_rrtl_split_tests --historical results/rrtl_split_smoke_historical_f_01 --anova results/rrtl_split_smoke_anova_01 --output results/rrtl_split_smoke_pair_01.json
# Figuras regenerables de cada campaña con el exportador existente:
python -m experiments.report_atari_comparison --campaign results/rrtl_split_smoke_historical_f_01
python -m experiments.report_atari_comparison --campaign results/rrtl_split_smoke_anova_01
```

El reporte pareado rechaza cambios adicionales de semillas, representación, presupuesto, hiperparámetros, recursos, código o runtime. Sólo carga checkpoints verificados y resultados existentes; no ajusta ni evalúa políticas. Registra retorno bruto final de desarrollo, curva intermedia con alcance de episodios, divisiones, hojas totales/por acción, profundidad máxima/por acción, tiempo de entrenamiento y fases medidas. Cada punto se vincula al hash del checkpoint. Los costos incompletos por muerte abrupta quedan desconocidos, con el tiempo registrado como límite inferior aparte.

Se comparan retornos por semilla de entrenamiento, con diferencias `anova - historical_f` e intervalos bootstrap exploratorios por juego/pipeline. Faltantes y corridas parciales permanecen visibles; no se imputan ceros ni se promedian puntajes brutos entre juegos. Las curvas frente al tiempo usan fases activas medidas; no equivalen al tiempo calendario de trabajos paralelos. Una sola semilla del smoke no permite concluir eficacia ni equivalencia. El reporte parcial devuelve código 3; completo devuelve 0. La evaluación reservada mantiene las protecciones de C15 y no forma parte de estos comandos.

## Evidencia y compatibilidad

Validación local: **258 pruebas pasaron**, 20 casos Atari/OCAtari deseleccionados; 13 advertencias de dependencias. Incluye referencias `f_oneway`, degeneración, estadísticas inconsistentes, selección/umbral, ramas lógicas/comparativas, herencia, crecimiento de descendientes, Q actualizado frente a target, continuación exacta en proceso separado, rechazo de cambio de criterio y comparación predeterminada idéntica a `historical_f`. También incluye reanudación y reporte pareado mediante el runner común con drivers sintéticos.

Antes de modificar fuentes se verificó una copia de 94 archivos en `historical/pre_rrtl_anova_20261101_code/`. Los checkpoints/resultados anteriores se conservan con sus hashes y deben operarse desde su checkout original; no se saltaron verificaciones para cargar artefactos viejos como nuevos. El archivo histórico del árbol conserva SHA256 `5e6fb6015eddb875dc3a38a1981e0da1502fa2bcdeb1db59aba362635eab1a60`.

No se lanzó campaña larga, no se envió Slurm y no se consumió test real. Quedan la revisión de costo, preflight Linux y autorización explícita de cualquier ejecución larga. Ver también [validación estructurada](validation_rrtl_anova.json).
