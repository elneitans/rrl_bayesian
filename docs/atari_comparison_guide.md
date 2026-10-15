# Bloque C: reproducción de la comparación Atari

C13 y C14 añaden Breakout y Demon Attack al **runner de B**, `experiments.compare_pong_progressive`. [C13](atari_comparison/C13.md), [C14](atari_comparison/C14.md), [C15](atari_comparison/C15.md), [auditoría](ocatari_detector_audit_c.md) y [evidencia local](atari_comparison_preparacion.json). Este documento prepara comandos; no autoriza campaña larga ni test.

## Entorno y conservación histórica

Desde la raíz del repo, preparar un entorno separado; no modificar el entorno que opera resultados históricos. Python 3.11.7, OCAtari 2.2.1, Gymnasium 1.1.1 y ALE 0.10.2 siguen fijados. Las ROM deben estar disponibles legalmente mediante ALE, con los SHA de cada configuración. No hay descarga de ROM en los jobs.

```bash
conda env create -f environment-cluster.yml
conda activate rrl-pong
python -m pip check
python -c 'from bayesian_rrtl.ocatari_backend import verified_versions; print(verified_versions())'
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m pytest tests/test_atari_block_c.py tests/test_pong_online.py tests/test_cluster_launch.py -q
RRL_C_REAL_TEST=1 python -m pytest tests/test_atari_block_c_real.py -q
```

La instalación es previa a Slurm. `slurm/common.sh` activa `${RRL_CONDA_ENV:-rrl-pong}`, fija threads y registra dependencias; no instala paquetes. ExtraTrees requiere además el pin scikit-learn 1.9.1 de `requirements-pong-analysis.txt`, y sólo se incorpora mediante una celda explícita antes de congelar. No es necesario para RRTL/CART/BART.

Antes de C se verificaron 77 fuentes del B original en `historical/pong_online_B_20261014_code/historical-preservation.json`. También se conserva la copia anterior de 40 fuentes del piloto. Son artefactos locales ignorados por Git: transferirlos junto con resultados históricos si se requieren. No usar loaders actuales para migrar checkpoints viejos ni omitir verificaciones de hashes. La modificación de código/runtime durante una campaña provoca rechazo deliberado; continuar con su checkout y entorno originales.

Antes de corregir las mediciones tras SIGKILL y la auditoría A4, también se preservaron las 88 fuentes verificadas del smoke C en `historical/atari_common_C_20261014_code/`. Los resultados `c15_three_games_smoke_02` documentan la versión anterior a esas correcciones y se abren desde esa copia; no se cambiaron sus manifiestos ni checkpoints para hacerlos compatibles con el checkout posterior.

## Contratos de los juegos

| Juego/pipeline principal | Acciones en orden | Representación |
| --- | --- | --- |
| Pong/ocatari | NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE | contrato A3, catálogo Pong fijo; mantiene su regla de historia tras punto |
| Breakout/ocatari | NOOP, FIRE, RIGHT, LEFT | 2 objetos, 11 relaciones; pelota/paleta, sin ladrillos como entrada |
| DemonAttack/ocatari | NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE | 30 slots, 134 relaciones; identidad de slot, movimiento sólo del jugador |

Las etiquetas RIGHT/LEFT se conservan del catálogo real ALE; no se reinterpretan los nombres de Pong. La auditoría explica la semántica y pruebas empíricas. Ausencia certificada es presencia falsa; información imposible de calcular queda no observada. Las tolerancias de los nuevos juegos son espacial 4 y temporal 0 píxeles. Entrenamiento usa signo de recompensa, evaluación puntaje bruto; no hay shaping ni lanzamiento oculto. `terminated` corta bootstrap; truncación respeta `bootstrap_on_truncation=True`. Los métodos comparten condiciones dentro del juego y producen sus propias trayectorias online.

## C13 y C14 por separado

Las siguientes pruebas son pequeñas y usan sólo semillas de desarrollo. Las carpetas de continuidad deben ser nuevas:

```bash
python -m experiments.validate_atari_games --stage continuity --config configs/breakout_common_v1.json --output results/breakout_continuity_linux_01
python -m experiments.validate_atari_games --stage controls --config configs/breakout_common_v1.json --output results/breakout_controls_linux_01
python -m experiments.validate_multigame_smoke --protocol configs/breakout_integration_smoke.json --campaign results/breakout_smoke_linux_01
python -m experiments.validate_atari_games --stage continuity --config configs/demonattack_common_v1.json --output results/demonattack_continuity_linux_01
python -m experiments.validate_atari_games --stage controls --config configs/demonattack_common_v1.json --output results/demonattack_controls_linux_01
python -m experiments.validate_multigame_smoke --protocol configs/demonattack_integration_smoke.json --campaign results/demonattack_smoke_linux_01
```

La continuidad compara 64 decisiones posteriores a un snapshot en otro proceso. Los controles NOOP/aleatorio/Fact usan semillas 640201/640202 y horizonte 1024. Sus JSON conservan acciones, decisiones con información incompleta, puntaje, duración y alcance. Una heurística útil comprueba la interfaz; no prueba aprendizaje.

## Smoke reproducible con los tres juegos

```bash
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/atari_comparison_smoke.json > results/atari_smoke_dry_run.json
python -m experiments.validate_multigame_smoke --protocol configs/atari_comparison_smoke.json --campaign results/atari_smoke_linux_01
python -m experiments.report_atari_comparison --campaign results/atari_smoke_linux_01
```

Son **nueve celdas enumeradas**, 64 decisiones y un train seed por juego; no se crea una malla implícita. Los métodos por lotes tienen warmup/fit cada 16 decisiones y hacen cuatro ajustes, con parámetros BART reducidos sólo para comprobación técnica. El wrapper llama a los workers de B, pausa en 19, reanuda, evalúa checkpoints y comprueba reutilización de resultados completos. No realiza test.

Se publican `reports/report.json` (JSON con envoltura de integridad), `smoke_validation.json`, seis figuras, nueve exportaciones en `reports/models/` y `reports/inventory.json` con hashes/procedencia. `report_atari_comparison` sólo carga artefactos y regenera: no entrena ni abre entornos de evaluación. Sus referencias interpretativas son estado vacío y observación actual, identificadas en el JSON.

## Operación por celda y Slurm

Alternativa al wrapper anterior: congelar primero y después distribuir las celdas. No usar el wrapper sobre una campaña cuyos parámetros se editaron.

```bash
python -m experiments.compare_pong_progressive --stage freeze --protocol configs/atari_comparison_smoke.json --campaign results/atari_smoke_cells_01
python -m experiments.compare_pong_progressive --stage train-cell --campaign results/atari_smoke_cells_01 --cell Breakout/ocatari/cart/710501
python -m experiments.compare_pong_progressive --stage resume --campaign results/atari_smoke_cells_01 --cell Breakout/ocatari/cart/710501
python -m experiments.compare_pong_progressive --stage evaluate --campaign results/atari_smoke_cells_01 --cell Breakout/ocatari/cart/710501
python -m experiments.compare_pong_progressive --stage report --campaign results/atari_smoke_cells_01
```

El último reporte devolverá código 3 mientras falten otras celdas o evaluaciones; éstas permanecen pendientes. Para una evaluación concreta se admiten `--group intermediate --point 32`. La evaluación utiliza checkpoints confirmados, nunca el modelo mutable del worker. Entrenar una celda ya completa no vuelve a ajustarla. Los resultados de episodio se escriben atómicamente: repetir evaluate conserva completos y continúa parciales.

Los batches incluyen `--comment="Experimento rrtl. Supervisor: Guillermo Puebla"`. Enviar desde la raíz del repo, con Conda disponible:

```bash
sbatch slurm/14_atari_preflight.sbatch
# Ruta corta, después de revisar que preflight terminó correctamente:
sbatch slurm/15_atari_smoke.sbatch --campaign results/atari_smoke_slurm_01
sbatch slurm/16_atari_report.sbatch --campaign results/atari_smoke_slurm_01
```

Alternativa distribuida (esperar cada fase antes de enviar la siguiente):

```bash
sbatch slurm/07_online_freeze.sbatch --protocol configs/atari_comparison_smoke.json --campaign results/atari_cells_slurm_01
sbatch --array=0-8 slurm/09_online_train_cell.sbatch --campaign results/atari_cells_slurm_01
sbatch --array=0-8 slurm/10_online_evaluate.sbatch --campaign results/atari_cells_slurm_01
sbatch slurm/16_atari_report.sbatch --campaign results/atari_cells_slurm_01
```

El orden exacto del array está en `campaign.json`/dry-run: tres celdas de Breakout, tres de DemonAttack y tres de Pong. Para reanudar una celda parcial:

```bash
sbatch slurm/11_online_resume.sbatch --campaign results/atari_cells_slurm_01 --cell DemonAttack/ocatari/bart/720501 --extend-seconds 300 --extension-id c-smoke-01 --reason "Ampliación operativa revisada para completar el smoke"
```

La extensión agrega segundos al presupuesto acumulado; no reinicia consumo, decisiones, epsilon ni replay. El ID es idempotente y se rechaza si cambia su monto/motivo. Para evaluación se aplica por episodio pendiente. Train/resume reservan 120 segundos de margen mediante `--attempt-seconds 3480` en jobs de una hora; adaptar ambos valores conjuntamente si se cambia el wall time. No hay envío automático ni test en estos batches.

## Medir costo y preparar la campaña final

```bash
python -m experiments.compare_pong_progressive --stage measurements --campaign results/atari_smoke_linux_01 > results/atari_smoke_cost_linux.json
python -m experiments.compare_pong_progressive --stage dry-run --protocol configs/atari_comparison_final_pending.json > results/atari_final_pending_dry_run.json
python -m experiments.compare_pong_progressive --stage inventory --campaign results/atari_smoke_linux_01 --cell 0
```

El template final enumera 45 celdas, pero no puede entrenarse: faltan revisión A6/A7, BART, máximos, calendarios, AUC y presupuestos. Crear una configuración nueva y revisada para resolverlos; no editar una campaña congelada. Primero medir una variante de costo en **Linux** con las mismas condiciones definitivas de entorno, horizonte, aprendizaje, caché y estimador para cada método/juego. Puede reducirse su máximo de decisiones; no sustituir por una medición con otro contrato o por el smoke de 64 decisiones.

Después de revisar los recursos, `dry-run --cost-reference /ruta/campaña_de_costo_linux` o un export JSON de measurements muestra tasas/rangos compatibles. Mediciones antiguas de B con hashes o condiciones diferentes quedan como costo desconocido, no cero; no se extrapola silenciosamente desde ellas. La autorización larga exige costos Linux compatibles para todas las celdas. Congelar una configuración ya resuelta y autorizar cada juego con `--stage authorize --game Breakout --stop-at <hito_declarado> --cost-reference <mediciones> --reason <revisión>`, y análogamente los demás. Estos argumentos son una referencia de interfaz, no comandos que se hayan ejecutado.

El reporte conserva fases, memoria máxima, intentos, ajustes y complejidad por acción. RSS es una observación, no una cota garantizada; replay lleno y árboles más grandes pueden cambiar el costo. Cola queda desconocida si no se aportó medición externa. No interpretar la suma de segundos de celdas paralelas como duración calendario.

## Análisis y test final protegido

La métrica principal congelada es retorno bruto medio de desarrollo del checkpoint final. Se resumen episodios dentro de cada modelo y se comparan **semillas de entrenamiento**, con mediana/dispersión y diferencias pareadas. AUC requiere todo el intervalo y calendario declarados. Se identifica cada retorno como partida completa o limitada por horizonte; Pong además guarda puntos ganados/perdidos. No se promedian puntajes brutos entre juegos. Los smokes de una semilla no son una comparación científica concluyente.

**Comandos para una etapa futura, sólo tras autorización humana de test. No ejecutarlos sobre el smoke para investigar rendimiento.** Tras completar desarrollo y fijar las decisiones, preparar la selección (esto todavía no evalúa):

```bash
python -m experiments.compare_pong_progressive --stage prepare-test --campaign results/atari_final_autorizada --selection-record results/seleccion_final.json
```

Revisar ese archivo: completar únicamente `selection_decisions` con decisiones tomadas desde desarrollo y marcar `test_seeds_untouched_attestation=true` sólo si es cierto. No editar hashes, semillas o checkpoints. Después de la autorización humana:

```bash
python -m experiments.compare_pong_progressive --stage authorize-test --campaign results/atari_final_autorizada --selection-record results/seleccion_final.json --confirm-test --reason "Selección revisada únicamente con desarrollo; test autorizado"
# Ejemplo de interfaz; sustituir ID por una celda exacta del manifiesto final:
python -m experiments.compare_pong_progressive --stage test --campaign results/atari_final_autorizada --cell ID_EXACTO --confirm-test
python -m experiments.compare_pong_progressive --stage report-test --campaign results/atari_final_autorizada
```

Test nunca entrena, actualiza replay ni recalibra. Repetir `test` requiere confirmación, reutiliza episodios completos y reanuda pendientes; una extensión operativa exige ID y motivo. `report-test` no completa partidas faltantes. El sello y el registro de uso impiden presentar como reservado un conjunto observado y luego retocado. Los reportes de desarrollo nunca llaman estas etapas.

`report-test` también puede ejecutarse mientras un evaluador está activo: si el episodio tiene su bloqueo ocupado, lo muestra como pendiente con motivo `evaluation_running`, publica un reporte parcial y devuelve código 3. No lee el estado ni los blobs de progreso de ese episodio mientras está bloqueado. Después de finalizar la evaluación, repetir el reporte incorpora el resultado completo. Errores de integridad, identidad u otros fallos de lectura siguen propagándose.

## Estado observado y límites

El 14/10/2026 se completaron localmente continuidad real entre procesos, controles y el smoke de nueve celdas (`results/c15_three_games_smoke_02`). La ejecución local usó macOS y el código de la wheel fijada extraída del caché; se evitó un bloqueo de lectura de bytecode mediante un launcher temporal que compila las fuentes instaladas sin cambiarlas. Ese launcher no es una dependencia del repo ni una certificación Linux. Los comandos normales anteriores y el preflight Slurm quedan como reproducción en el entorno fijado.

No se envió Slurm, no se ejecutó campaña larga y no se consumió test reservado real. Las pruebas de sellado/test usan entornos y semillas sintéticos. Integración correcta, ejecución completa de smoke y aprendizaje útil se reportan por separado. Una política mala no demuestra por sí sola sobreajuste, mala mezcla o insuficiencia de representación.
