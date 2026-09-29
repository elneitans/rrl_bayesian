# Diagnóstico del piloto y comparación con el paper base

## Resultado de la auditoría de registros

Fuente: `results_cluster/pong_pilot_linux_01`. Once corridas completas y
`visual/bart/73` parcial en 5.120 decisiones. Los 11 resultados finales disponibles
son -21 (10 partidas de validación cada uno). No hay evidencia de aprendizaje útil
en este presupuesto, pero tampoco justificación para concluir que los algoritmos
sean incapaces de aprender Pong.

Las 12 corridas son 2 representaciones x 2 learners x 3 semillas. No son doce
partidas de entrenamiento. Cada agente completo recibió 20.480 decisiones.

| Condición | Episodios terminados, seeds 71/72/73 | Tiempo de entrenamiento por corrida completa |
|---|---|---|
| Visual baseline | 23 / 24 / 24 | 104–109 s |
| Visual BART | 23 / 22 / 5 (parcial) | 2.157–2.189 s |
| OCAtari baseline | 21 / 22 / 22 | 110–126 s |
| OCAtari BART | 22 / 22 / 22 | 1.789–2.446 s |

Cada corrida tiene además un episodio iniciado no terminado. La evaluación no
entrena. El tiempo CPU mayor de BART no significa más experiencia: observa el
mismo presupuesto de decisiones. Epsilon acaba cerca de 0,389; cada BART completo
hizo 40 ajustes. El cálculo reproducible queda en
`results/pong_diagnostic_logs_20260920/report.json`.

## Paper base

Fuente local: `capstone/paper_base.pdf`, *Learning Rules from Rewards*, Puebla y
Doumas. Sección 2.2 (p. 6), figura 4 y texto de resultados (p. 8), discusión
(pp. 10–11).

- Pong: 2.000.000 iteraciones por semilla, 100 semillas por versión.
- Epsilon: 1 a 0,1 durante 500.000 pasos.
- Profundidad máxima 10; mínimo de muestras 100.000; significancia 0,0001.
- Recompensa de Pong: sign(reward) + 0,1 por paso.
- Guardan el modelo cuando mejora el mejor retorno de entrenamiento; evalúan
  ese modelo con política greedy en 100 partidas por semilla. Nuestro primario
  es el checkpoint final, no el mejor observado.

Lectura visual aproximada de figura 4, segunda fila: las curvas que mejoran se
separan de la zona inicial alrededor de 1.200–1.600 episodios; muchas mejoras se
desarrollan entre 1.500 y 2.000. No es un tiempo de aparición exacto de política
útil: son medias móviles, con ventana máximo de episodios/12 y mínimo de
observaciones máximo/50. El texto no fija un umbral ni tiempo de pared para lograrlo.
La recompensa está moldeada para sobrevivir, y la curva no es nuestra evaluación
raw final. No se debe interpretar como porcentaje de partidas ganadas.

20.480 / 2.000.000 = 1,024%: nuestro presupuesto numérico es unas 98 veces menor.
La comparación no es una reproducción exacta: cambian catálogo, profundidad,
umbral de split, exploración, selección de checkpoint, versiones/pipeline y, para
OCAtari, recompensa. Tampoco se ha verificado identidad de frames por iteración.
El paper describe una política que sigue la pelota y advierte que la
representación no basta para ganar consistentemente; sugiere mayor granularidad
de trayectoria. "Relativamente útil" significa aquí mejor que aleatoria, no
dominar Pong.

La conclusión correcta es: este piloto valida ejecución y costos y no mostró
progreso a 20.480 decisiones; aún no evalúa un horizonte de aprendizaje comparable
al paper. Antes de ampliar BART 98 veces, diagnosticar comportamiento y costo de
inferencia; no extrapolar linealmente un presupuesto grande como garantía.

## Script de trazas

`experiments.diagnose_pong_policy`:

- Audita todos los logs y checksums; en modo traza usa solo corridas completed.
- Carga por las rutas estrictas existentes; no modifica código ni checkpoints.
- Abre un entorno nuevo y usa únicamente la primera seed de validación.
- Registra por decisión hechos antes de actuar, encoding/máscara, Q de las seis
  acciones, desempates, margen Q, acción, recompensa raw/transformada y flags.
- Para baseline registra profundidad y fallback interno por acción.
- Resume acciones, cambios, mayor racha de acción idéntica, estados únicos y puntos.
- No llama observe/fit; no es continuación de entrenamiento ni evaluación final.
- Salida incremental JSONL; límite por defecto 2.000 decisiones y 600 s por celda.
  Diferencia terminal, corte por horizonte y corte por tiempo.
- `--gif`: preview RGB cada cuatro decisiones, máximo 300 frames. La traza puede
  ser más larga que el GIF. No equivale necesariamente a una partida completa.

No compara todavía controles aleatorios o heurísticos: este script diagnostica
las políticas guardadas. Una acción dominante por sí sola no identifica la causa.
Correlacionar acciones/Q con posición de pelota/paleta y las imágenes antes de
concluir que hay un defecto en representación o actualización.

## Resultado local y cómo ejecutarlo en Linux

La auditoría JSON local se completó. Se intentó cargar las 11 corridas completas:
todas fueron rechazadas por validación estricta del historial (`History
return/policy mismatch` o `Comparison history mismatch`). Los logs revelan 32
discrepancias de último bit al recalcular epsilon localmente en cada corrida
completa: por ejemplo, paso 106, 0,9951304146652443 guardado frente a
0,9951304146652442 recalculado. Es coherente con diferencias numéricas entre
plataformas, no prueba corrupción o mal entrenamiento. No se relajaron controles
ni se migraron checkpoints. Evidencia del intento:
`results/pong_diagnostic_local_20260920/report.json`.

Subir los archivos nuevos `experiments/diagnose_pong_policy.py`,
`tests/test_diagnose_pong_policy.py` y `slurm/04_diagnose.sbatch` al repo Linux.
No sustituir las fuentes originales que forman parte de los hashes del piloto.

```bash
# En el clúster, desde rrl_bayesian. No exige terminar la corrida 12.
export RRL_OUTPUT="$PWD/results/pong_pilot_linux_01"
export RRL_DIAGNOSTIC_OUTPUT="$PWD/results/pong_policy_diagnostic_01"
sbatch slurm/04_diagnose.sbatch
```

Para enviar solo la traza `visual/bart/73` como batch Slurm, usar el mismo
archivo con `RRL_DIAGNOSTIC_CELL`. El archivo ya incluye el `#SBATCH --comment`
que exige el clúster; las directivas de otro `.sbatch` no se aplican a un envío
con `sbatch --wrap`.

```bash
export RRL_OUTPUT="$PWD/results/pong_pilot_linux_01"
export RRL_DIAGNOSTIC_OUTPUT="$PWD/results/pong_policy_diagnostic_visual_bart_73_final_01"
export RRL_DIAGNOSTIC_CELL=visual/bart/73
sbatch --job-name=rrl-trace-vb73 --time=01:00:00 \
  --output=rrl-trace-vb73-%j.out slurm/04_diagnose.sbatch
```

La ruta de salida debe ser nueva. `RRL_DIAGNOSTIC_CELL` sin definir mantiene el
diagnóstico de todas las celdas completas.

Para inspeccionar solamente una corrida:

```bash
conda activate rrl-pong
python -m experiments.diagnose_pong_policy \
  --campaign results/pong_pilot_linux_01 \
  --output results/pong_policy_one_new \
  --cell ocatari/bart/71 --horizon 2000 --seconds-per-cell 600 --gif
```

La salida debe ser una carpeta nueva. El reporte conserva fallos de carga y
devuelve código 1 si alguna celda falla, sin fingir trazas completas. Un corte
por tiempo aparece en `stop_reason`; no es una partida terminada.
Para una partida más larga cambiar explícitamente horizonte y presupuesto por
celda, manteniendo el límite Slurm suficiente. No usar seeds reservadas.

Pruebas: `python -m pytest tests/test_diagnose_pong_policy.py tests/test_cluster_launch.py -q`.
Incluyen trazas y GIF reales de los dos extractores y ambos learners, además de
terminales, cortes, no sobrescritura y ausencia de actualizaciones durante traza.
