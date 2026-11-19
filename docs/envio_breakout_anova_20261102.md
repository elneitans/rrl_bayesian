# Envío Breakout baseline + ANOVA — 2 de noviembre de 2026

Solicitud explícita: enviar el nuevo baseline con ANOVA a Breakout. Se conserva el alcance de la comparación reciente: OCAtari, diez semillas710601–710610, primera meta1M decisiones por agente y máximo futuro2M. No se ejecuta test reservado.

## Jobs enviados

| Trabajo | ID | Dependencia |
|---|---:|---|
| Pruebas Linux, smoke real, calibración y autorización | 36273 | — |
| Diez entrenamientos, cinco grupos de dos semillas, concurrencia2 | 36274_0–4 | afterok:36273 |
| Calendario de evaluación hasta1M y reporte | 36275 | afterany:36274 |

36273 se observó RUNNING en scylla. El entrenamiento largo no empieza hasta completar correctamente sus comprobaciones. Los jobs del otro proyecto no se modificaron.

## Código y condiciones

El checkout remoto anterior no incluía ANOVA. Se creó una copia aislada del commit **c2b4faa460904e67ec5ffd5203e54d9df51485bf**, en `~/archive/rrl_bayesian_code_anova_c2b4faa46_20261102`, con fuentes, configuraciones y pruebas necesarias. No se sobrescribió `~/projects/rrl_bayesian` ni se intentó cargar checkpoints históricos bajo el código nuevo. `SOURCE_REVISION` identifica el commit; la configuración nueva y pytest.ini se añadieron antes de congelar la campaña.

Configuración: `configs/breakout_anova_10seeds_1m_20261102.json`. Se verificó estructuralmente que difiere del protocolo anterior del baseline1M únicamente en nombre y `learners.corrected_common.split_test="anova"`: mismas semillas, entorno, relaciones, recompensas, parámetros TD, exploración, calendarios, presupuestos y recursos. El identificador del learner sigue siendo `corrected_common`; el contrato y hash distinguen el criterio. Como cambia la versión del código, no presentar una comparación con artefactos antiguos como una validación pareada del exportador estricto sin revisar esa diferencia de fuentes.

Antes del entrenamiento largo el job45 ejecuta pruebas ANOVA/adaptador/checkpoint, un smoke de64 decisiones con horizonte128 y mínimo4 para verificar integración/crecimiento, y una calibración de20.480 decisiones con mínimo128 y horizontes27000 como la campaña larga. La calibración preserva el criterio ANOVA y runtime nuevo; no reutiliza costes del F-test histórico. Tras evaluar el conjunto de desarrollo de calibración, calcula costes/proyección, congela el protocolo largo y autoriza solo el hito1M. Si una etapa queda parcial o falla, se revisa/reanuda la preparación; los entrenamientos quedan bloqueados por dependencia y no pasan por alto la validación.

## Rutas y operación

Control: `~/archive/rrl_bayesian_anova_20261102`.

- Ledger: `jobs.tsv`.
- Log preparación: `logs/prepare-36273.out`.
- Logs entrenamiento: `logs/train-36274_0.out` hasta `_4.out`.
- Log cierre: `logs/calendar-36275.out`.
- Campaña larga: `results/corrected_common/`.
- Agentes: `results/corrected_common/Breakout/ocatari/corrected_common/710601/` etc.
- Pruebas/calibración reales: `smoke/` y `cost/`.

Scripts nuevos45/46 están en el repo y en `control/slurm/`. Reutilizan los workers38 y cierre42/reporte39 copiados a este control independiente. El wrapper por grupos conserva checkpoints, permite reanudar y usa24h por allocation. La campaña no se considera completa solo por haber enviado el array.

Validaciones locales: sintaxis Bash/Python inline, igualdad de protocolos salvo nombre/criterio, y Slurm simulado que comprobó dependencias, cinco grupos/concurrencia2 e idempotencia del lanzador. La validación real Linux está delegada al job36273, aún no confirmada al escribir este envío.
