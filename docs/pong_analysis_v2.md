# Diagnósticos de Pong: bloque A1–A6

El bloque prepara persistencia, controles, protocolo común, auditorías portables y comparaciones controladas. Terminar una ejecución no demuestra aprendizaje ni inferencia válida. No se ejecutan campañas ni se consumen semillas reservadas al instalar estos cambios. Las entregas se describen por separado más abajo.

## Preservar el piloto antes de cambiar código protegido

El piloto vincula sus checkpoints a todos los módulos de `bayesian_rrtl`, su registro de juegos, `preprocessing.py`, `relational_regresion_tree.py` y `experiments/compare_pong.py`. No añadir módulos a esas carpetas ni editar esos archivos antes de preservar el entorno histórico.

1. En Linux, con el entorno original, crear una copia independiente del checkout actual (excluyendo `.git`, resultados y entornos virtuales), usando `experiments.preserve_pong_history`. La herramienta verifica **cada hash del protocolo original antes de copiar**, conserva también las herramientas de exportación y vuelve a verificar la copia. Conservar además el entorno Conda original (`conda list --explicit > historical-conda-explicit.txt`) y las dependencias pip (`python -m pip freeze > historical-pip.txt`), la ROM legalmente instalada y su hash. La copia de código no reemplaza la conservación del entorno.
2. Ejecutar desde esa copia los validadores y la exportación, con el mismo entorno Linux del piloto. Usar la ubicación absoluta del piloto original como `source_campaign`/`source_trace` en una **configuración de análisis nueva**; jamás editar su `protocol.json`. `load_frozen`, `_finals` y `load_session` mantienen sus comprobaciones originales de código, checkpoint, runtime y ROM.
3. Crear una campaña v2 independiente y exportar `fixed_fit`/`fixed_transitions`. Conservar sus NPZ, metadatos y manifiesto. Tras la exportación, los análisis portables no necesitan deserializar checkpoints Linux ni abrir el emulador. Cada exportación tiene publicación atómica verificable: repetir el comando recupera bloques completos y regenera sus vistas NPZ/JSON desde el commit. Un bloque interrumpido antes de publicar puede recalcularse desde el checkpoint original.

Ejemplo, **sin enviar jobs**:

```bash
python -m experiments.preserve_pong_history --campaign results/pong_pilot_linux_01 --output /ruta/historical-pong-code
# Desde /ruta/historical-pong-code, con config v2 nueva que use las rutas absolutas originales:
python -m experiments.pong_analysis_common --config /ruta/analysis-v2.json --output /ruta/analysis-v2
python -m experiments.freeze_pong_analysis_data --config /ruta/analysis-v2.json --output /ruta/analysis-v2 --pipeline visual
```

Esta entrega solo modifica herramientas independientes de `experiments`, pruebas, documentación y el batch de análisis. No modifica código protegido, resultados, configuraciones congeladas ni checkpoints del piloto. No se desactivan verificaciones globales.

## Campañas y unidades v2

El manifiesto nuevo identifica configuración, protocolo de origen, runtime y código. Los manifiestos v1 se pueden leer, pero los runners no escriben en campañas v1 ni las migran. Las salidas ejecutables se separan en `smoke/<pipeline>/<etapa>` y `full/<pipeline>/<etapa>`. Los datasets portables verificados permanecen en `<pipeline>/fixed_*` como entradas compartidas, nunca como resultados smoke.

Cada unidad conserva `identity.json`, `state.json` y los objetos referenciados por SHA-256. La identidad incluye protocolo, pipeline, etapa, método, dataset, variante, modo, semillas, configuración efectiva, catálogo, datos, targets y código; evaluación añade modelo, horizonte y política; MCMC añade kernel, árboles, inicialización, prior, escala y referencias. Reutilizar una unidad exige identidad exacta, integridad y estado completo. Un archivo corrupto causa error: no se sobrescribe silenciosamente.

La publicación usa archivos temporales, `fsync` y reemplazo atómico; `state.json` es el punto de publicación. Un objeto huérfano por caída previa a ese punto nunca es un resultado válido. Cada unidad usa un bloqueo de sistema liberado al morir el proceso. Un segundo escritor recibe un conflicto inmediato. El filesystem del cluster debe soportar `flock` y reemplazo atómico; validar estas capacidades allí antes de la campaña.

Los episodios guardan entorno (incluido historial relacional), estado actual, controlador, RNG, acciones, acumuladores y decisiones. Publican una frontera cada 64 decisiones y siempre al alcanzar un límite administrativo. Un fallo vuelve a la última frontera publicada y recalcula su tramo pendiente con el mismo RNG; no junta fragmentos de partidas distintas. Los pickles son solo para artefactos locales confiables y se verifican antes de cargarlos.

FQI publica **cada** iteración y su procedencia, incluidos hashes de targets y modelo anterior. Una iteración incompleta se repite determinísticamente desde la anterior; la evaluación se realiza después de recuperar/ajustar los modelos. Los reportes se regeneran desde unidades válidas cada vez, con estados parciales explícitos.

## Validación y ejecución acotada

```bash
.venv/bin/python -m pytest tests/test_pong_analysis_runtime.py tests/test_pong_bart_analysis.py -q
python -m experiments.pong_analysis_common --config configs/pong_bart_analysis_v2.json --output results/pong_analysis_v2
python -m experiments.evaluate_pong_heuristics --config configs/pong_bart_analysis_v2.json --output results/pong_analysis_v2 --pipeline visual --smoke
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition base --smoke
python -m experiments.compare_fqi_trees --analysis results/pong_analysis_v2 --pipeline visual --stage fqi --smoke
```

Los últimos comandos requieren haber exportado los datos; la acción debe pertenecer a la selección predeclarada. Repetir el mismo comando recupera unidades compatibles y no repite las completas. Para un presupuesto agotado, usar `--extend-seconds 600 --extension-id continuacion-01`. La extensión es explícita, registrada y aplicada una sola vez por ID y unidad; repetir ese ID no vuelve a sumar tiempo. Un nuevo incremento necesita otro ID. No cambia horizonte, semillas, parámetros experimentales ni contadores. En Slurm usar `RRL_ANALYSIS_EXTEND_SECONDS` y `RRL_ANALYSIS_EXTENSION_ID`. No se envían jobs automáticamente.

Separar `status=completed` de `terminated` natural y `truncated` por horizonte; `partial` administrativo no es terminal del MDP. La calidad estadística se publica aparte y no se infiere del estado operativo. OCAtari debe verificarse en el entorno Linux original; las pruebas portables usan entornos sintéticos, sin ROM ni semillas del piloto.

## Entrega A2: controles y evidencia operativa

El controlador recibe una observación interpretada por `ObjectAdapter` o `FactAdapter`. Las coordenadas de pantalla y las relaciones RAM usan `less=arriba`; el visual histórico usa `more=arriba`. `FactAdapter` solo recibe hechos y no exige `present` para el visual. Diferencia ausencia observada, relación no disponible, estado vacío e historial insuficiente. Ambos controles comparten tolerancia de cuatro píxeles y reglas de recuperación FIRE; reinician memoria tras reset y puntos. El control por objetos visuales usa exclusivamente el extractor RGB existente.

Se comparan las cuatro políticas con semillas y horizontes comunes. Revisar `full/<pipeline>/heuristics/report.json` y las unidades por episodio: puntos positivos/negativos, retorno raw, decisiones, causas y motivos de acciones, terminal, horizonte y estado administrativo. Los retornos de partidas terminadas y los limitados por horizonte quedan etiquetados; no interpretar una pérdida al horizonte como resultado de partida completa. Una heurística fallida no demuestra incapacidad de la representación.

El gate asistido exige resultados completos requeridos, pares válidos frente a aleatoria, diferencia descriptiva positiva y puntos ganados. Es una condición operativa, no una prueba concluyente de superioridad. Se recalcula desde unidades verificadas; un reporte editado no puede habilitar la colección.

```bash
python -m experiments.evaluate_pong_heuristics --config configs/pong_bart_analysis_v2.json --output results/pong_analysis_v2 --pipeline visual --smoke
python -m experiments.evaluate_pong_heuristics --config configs/pong_bart_analysis_v2.json --output results/pong_analysis_v2 --pipeline visual --report-only --smoke
```

## Entrega A3: contrato común separado

`configs/pong_common_v1.json` declara el protocolo nuevo y el SHA256 de la ROM verificada localmente. La configuración de análisis v2 congela esa declaración. Una ROM distinta se rechaza; antes de cambiarla, crear una variante de configuración nueva y comprobarla. `CommonPongFactory` usa `ALE/Pong-v5`, frameskip cuatro una sola vez, sticky cero, las seis acciones en el orden original y reset sin acciones ocultas. La recompensa de aprendizaje es `sign(raw)` y el raw queda separado. El horizonte común es 27.000 decisiones.

Ambos extractores entregan `ObjectFrame` y pasan por **la misma instancia semántica de `PongRelationsV1`**, con su catálogo de veinte relaciones. Los IDs `pong_common_rgb_v1` y `pong_common_ram_v1` son distintos de los históricos. No se reetiquetan ni reconstruyen estados del piloto como comunes.

```bash
# Solo smoke; --common usa carpetas e identidades propias.
python -m experiments.evaluate_pong_heuristics --config configs/pong_bart_analysis_v2.json --output results/pong_analysis_v2 --pipeline visual --common --smoke
# Ejecutar únicamente en el entorno Linux original, cuando se disponga de OCAtari:
RRL_COMMON_EMULATOR_TEST=1 python -m pytest tests/test_pong_analysis_blocks.py -k real_ale -q
```

La prueba real compara RAM del emulador antes de considerar las diferencias de extracción. Es opt-in y queda omitida cuando OCAtari no está disponible; las pruebas sintéticas no reemplazan esa validación real.

## Entrega A4: auditoría portable

El reporte distingue replay y selección por IDs exactos, con conteos por acción/episodio, recompensas raw/aprendizaje, estados vacíos, fracción observada por relación, categorías/presencias, concentración de encodings y cobertura vertical/movimiento. Incluye rango, mediana, desviación, IQR y fracción de targets fuera de escala cuando existe `fixed_fit` junto al replay. La convención vertical es explícita.

Las colecciones nuevas incluyen geometría, reinicio de historial y causas por relación como información auxiliar; esas columnas **no se pasan al encoder ni a los learners**. Para el histórico, las causas quedan `unknown`. La auditoría `pong_representation_audit_v2` separa `geometry_collisions` de `same_action_evolution_differences`. La primera muestra geometría registrada distinta bajo el mismo encoding, sin demostrar que lo omitido importe para decidir. La segunda compara exclusivamente la misma acción en el mismo estado codificado: evolución diferente puede deberse a estocasticidad o historia no registrada y no certifica aliasing por sí sola. Evoluciones de acciones distintas no se presentan como evidencia de pérdida de información. La geometría JSON se compara por contenido, independientemente del orden de sus claves. Los reportes v1 históricos se conservan; generar un archivo nuevo para aplicar este diagnóstico corregido.

```bash
python -m experiments.audit_pong_representation --dataset results/pong_analysis_v2/visual/fixed_transitions --convention historical_visual --output results/pong_analysis_v2/visual/representation.json
```

Este comando lee NPZ/JSON y no carga un emulador. Si se analiza una selección en otra carpeta, conservar `fixed_fit` junto a ella o auditar también su problema fijo para obtener estadísticas de targets.

## Entrega A5: datasets y benchmarks controlados

El piloto exportado, los nuevos datos comunes y los asistidos tienen identidades y destinos distintos. La separación train/holdout se congela sobre episodios completos, con paso inicial cero, pasos contiguos y último terminal/truncación del entorno. Los episodios incompletos se excluyen y se reportan. Si hay menos de dos episodios completos, no se inventa un holdout; se publica la limitación y se rechaza una regresión independiente.

La selección uniforme por acción es principal. Las sensibilidades 128/512 son deterministas. La selección por estratos de recompensa tiene etiqueta `distribution_changed=true`; no duplica positivos y redistribuye cuotas cuando faltan muestras. No sustituye silenciosamente la uniforme.

La colección común aleatoria usa semillas nuevas distintas de la asistida. La asistida solo se habilita con el gate de controles **comunes y completos**, nunca smoke ni históricos. Actualiza la memoria del controlador aun cuando elige exploración; registra la política de cada acción y conserva el siguiente estado. RNG de exploración y selección son independientes. Las cuotas administrativas no añaden terminales ni truncaciones al MDP. Repetir recupera los episodios y evita duplicar transiciones.

```bash
# Smoke prepara persistencia; suele no tener episodios completos para fitting.
python -m experiments.collect_pong_dataset --analysis results/pong_analysis_v2 --pipeline visual --smoke
# Cuando se decida realizar la colección acotada nueva, sin --smoke:
python -m experiments.collect_pong_dataset --analysis results/pong_analysis_v2 --pipeline visual
# Tras gate común completo, colección asistida distinta:
python -m experiments.collect_pong_assisted --analysis results/pong_analysis_v2 --pipeline ocatari

# Sensibilidad: no cambia la partición congelada.
python -m experiments.pong_dataset_selection --dataset results/pong_analysis_v2/full/visual/collection_common/dataset --output results/pong_analysis_v2/full/visual/selection_512 --per-action 512 --seed 560702
# Añadir --rule stratified_reward solo en otro destino identificado.

# Nuevo problema fijo: el generador CART usa exclusivamente episodios train en TODOS sus ajustes.
python -m experiments.freeze_pong_regression --dataset results/pong_analysis_v2/full/visual/collection_common/dataset --output results/pong_analysis_v2/full/visual/common_fixed_fit --generator-epochs 1
python -m experiments.compare_fqi_trees --analysis results/pong_analysis_v2 --pipeline visual --stage regression --dataset common --dataset-folder results/pong_analysis_v2/full/visual/common_fixed_fit --smoke
python -m experiments.compare_fqi_trees --analysis results/pong_analysis_v2 --pipeline visual --stage fqi --dataset common --dataset-folder results/pong_analysis_v2/full/visual/collection_common/dataset --smoke
```

No ejecutar FQI con una selección sin cobertura de las seis acciones: se rechaza en lugar de completar acciones con datos artificiales. Regresión fija comparte exactamente X, y, IDs y máscaras entre BART/CART/ExtraTrees. FQI comparte transiciones y particiones, pero cada método genera targets desde su propio modelo anterior congelado. Los hashes de targets están en cada iteración; calibración usa solo train. ExtraTrees mantiene codificación nominal, incluidos faltantes explícitos, y la versión fijada de scikit-learn.

**Limitación histórica:** separar episodios del último lote no hace independientes sus targets; el generador pudo entrenar con ellos. El reporte histórico conserva esa advertencia. La regresión nueva excluye holdout de todos los ajustes y calibraciones del generador y congela su problema fijo antes de comparar estimadores. No confundir ambos benchmarks.

## Entrega A6: inferencia sobre el problema fijo

La base conserva cuatro cadenas, 1.000 burn-in y 2.000 retenidas sobre tres acciones elegidas por cobertura y recompensas. Cada identidad incluye X/y, IDs, referencias, catálogo, escala, árboles, kernel, inicialización y priors. Para explorar las seis acciones, añadir `--all-actions` y solicitar cada acción explícitamente; no se lanza una grilla automáticamente. Las tres acciones no certifican estabilidad de una decisión greedy de seis acciones.

Las trazas incluyen sigma², SSE, predicciones frecuentes/raras, hojas, profundidad, propuestas seleccionadas/imposibles/aceptadas/rechazadas y segundos de muestreo por barrido. El reporte separa grupos predictivos, ruido y estructura; publica ruido/SSE normalizados y originales con el factor `width²`, calibración y uso de floor. Incluye media, mediana, dispersión, intervalo posterior, MCSE y ESS por segundo. Los criterios Rhat ≤1,01 y ESS bulk/tail ≥400 siguen siendo operativos, no garantías.

Las constantes se clasifican: degeneración probada por soporte, constante no concluyente o cadenas atascadas en constantes diferentes. Nunca se rellenan valores indefinidos con un resultado favorable. `statistical_quality` es independiente de `status=completed`.

```bash
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --select-only
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition base --smoke
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition base --initialization dispersed --smoke
# Revisar primero ambas inicializaciones: no cambia el posterior objetivo.
# Tras revisar cadenas completas de la base, registrar una nota concreta:
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition base --review-base 'Nota de revisión de las trazas y diagnósticos'
# Después solicitar UN factor por vez, en su carpeta propia:
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition alpha_08
# Continuar una base agotada con extensiones explícitas de tiempo y longitud:
python -m experiments.audit_bart_fixed_data --analysis results/pong_analysis_v2 --pipeline visual --action 0 --condition base --extend-seconds 1800 --extend-draws 1000 --extension-id extension-01
```

Revisar `full/<pipeline>/audit/a<acción>/<condición>/<inicialización>/report.json`, `chain_*/trace.npz`, `metadata.json`, `length.json` y las gráficas. Las condiciones `alpha_08`, `beta_1`, `k_1`, `k_3`, `q_075`, `q_099` requieren revisión base registrada sobre los mismos hashes. No se reduce ningún umbral para hacer pasar BART. Extensiones por ID son idempotentes y conservan RNG y toda la traza previa; cadenas parciales no cuentan como completas.

## A7 y estado de validación

No se modifica el sampler ni la matemática de BART en este bloque. Las pruebas acotadas verifican señal sintética, unidades, inicializaciones válidas, casos atascados y continuación exacta; no identifican un nuevo defecto matemático que justifique A7. Esto no reemplaza revisar los diagnósticos reales ni demuestra superioridad frente al baseline. Si surge un problema concreto, entregar primero evidencia, reproducción, componente, propuesta, prueba y compatibilidad afectada.

Validación por entrega:

| Entrega | Pruebas principales | Comportamiento conservado / límite |
|---|---|---|
| A1 | `test_pong_analysis_runtime`: igualdad continuo/reanudado, unidades completas, identidad, corrupción, caída atómica, bloqueo multiproceso, extensiones, reportes frescos; `test_fqi_iteration_reuse...` y `test_mcmc_exact_resume...` | Piloto y checks originales intactos; el FS real del cluster debe verificar locks/atomicidad |
| A2 | Convenciones verticales, disponibilidad/ausencia, objetos vs Facts con historia/reset/puntos, gate positivo | Extractores históricos conservados; utilidad real requiere evaluación común completa |
| A3 | Dynamics/acciones/reset/reward/facts sintéticos; `test_real_ale...` opt-in | Contratos históricos intactos; validación OCAtari real pendiente donde esté instalado |
| A4 | Conteos reconciliados, IDs, causas conocidas/unknown, colisiones y escala | No añade información auxiliar al modelo; histórico sin imágenes no recupera geometría |
| A5 | Particiones completas, selección reproducible, no duplicados, generador sin holdout, mismos X/y, nominalidad, bootstrap | Comparaciones no entrenadas extensamente; cuotas pueden dejar insuficientes episodios/señal |
| A6 | RNG/trazas exactos, extensión idempotente, proposals, unidades, MCSE, señal, constantes/atasco e inicializaciones | Posterior objetivo y umbrales intactos; inferencia real aún sin certificar |

```bash
MPLCONFIGDIR=/tmp/rrtl-mpl python -m pytest tests/test_pong_analysis_runtime.py tests/test_pong_analysis_blocks.py tests/test_pong_bart_analysis.py -q
bash -n slurm/05_pong_analysis.sbatch
```

El batch de análisis conserva el comentario requerido por el cluster. `RRL_ANALYSIS_COMMON=1` selecciona controles comunes; `RRL_ANALYSIS_STAGE=collection` pide colección nueva; `RRL_ANALYSIS_DATASET_FOLDER` identifica datos independientes; `RRL_ANALYSIS_INITIALIZATION` y `RRL_ANALYSIS_EXTEND_DRAWS` se usan en la auditoría. Ejecutar/exportar cada etapa manualmente y revisar sus resultados antes de avanzar. No se enviaron jobs ni se ejecutó la campaña larga.

Resultado local del bloque: **36 pruebas aprobadas, una omitida** (evolución real RGB/RAM, requiere OCAtari y activación explícita). Pasaron también Ruff y la validación sintáctica del batch. Los cuarenta hashes protegidos del piloto se comprobaron sin diferencias. Para evitar demoras de lectura de dependencias del entorno local, la suite usó temporalmente SciPy y bibliotecas gráficas ya disponibles mediante `PYTHONPATH`; no se modificaron instalaciones, versiones fijadas de ExtraTrees, ni código del piloto. Repetir la validación normal y la prueba real en el entorno Linux antes de usar evidencia OCAtari para conclusiones científicas.
