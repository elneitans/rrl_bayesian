# Pong: informe de D y E

La implementación y el smoke de cuatro celdas están completos. Suite: **377 pruebas pasaron**, sin fallos ni skips (13 advertencias de dependencias).

## Smoke medido

2.048 decisiones por celda, seed 71, validación en 0/2.048 con dos episodios. Los tiempos siguientes proceden de procesos secuenciales sin tests concurrentes. El smoke preliminar se conserva como integración y sus tiempos no se usan para estimar costos.

| Pipeline / learner | Train (s) | Eval (s) | Guardado (s) | Retorno final raw | Truncación |
|---|---:|---:|---:|---:|---:|
| visual/corrected_common/71 | 3.41 | 4.79 | 0.35 | -21.0 | 0% |
| visual/bart/71 | 11.98 | 7.30 | 0.34 | -21.0 | 0% |
| ocatari/corrected_common/71 | 3.04 | 3.36 | 0.32 | -21.0 | 0% |
| ocatari/bart/71 | 9.81 | 5.55 | 0.30 | -21.0 | 0% |

No hay evidencia de ventaja de BART: las cuatro políticas perdieron las partidas de esta validación corta. Una sola training seed no permite inferencia de eficacia. Las comparaciones visual/RAM no aíslan el efecto causal del extractor.

El baseline produjo cero splits en Pong; contó con 15 candidatos elegibles por acción en visual y 20 en OCAtari. No se redujeron umbrales para forzar divisiones. La prueba sintética con los defaults del piloto produjo un split en la transición 292.

## Mezcla posterior

Auditoría **completed**, 634.2 s de un máximo de 1.200 s. 24 de 24 combinaciones comprobadas; **0/24** cumplen todos los umbrales/observables. Cuatro cadenas por combinación, 500 burn-in y 1.000 draws, con IDs/X/targets/escala/prior idénticos entre kernels.

OCAtari reutiliza el checkpoint de desarrollo anterior compatible; visual usa su propio smoke. Referencias elegidas exclusivamente del replay por frecuencia/rareza. Los observables constantes no cuentan como convergencia.

| Pipeline / acción / kernel | Estado | Pasa todo | Observables que fallan o son constantes |
|---|---|---|---|
| visual/0/grow_prune | checked | no | total_leaves |
| visual/0/revision | checked | no | prediction_0, prediction_1, prediction_2, sigma2, total_leaves, max_depth |
| visual/1/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| visual/1/revision | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| visual/2/grow_prune | checked | no | sigma2, total_leaves, max_depth |
| visual/2/revision | checked | no | prediction_1, prediction_2, prediction_3, total_leaves, max_depth |
| visual/3/grow_prune | checked | no | prediction_2, sigma2, total_leaves, max_depth |
| visual/3/revision | checked | no | prediction_2, sigma2, total_leaves, max_depth |
| visual/4/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| visual/4/revision | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| visual/5/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| visual/5/revision | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, total_leaves, max_depth |
| ocatari/0/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| ocatari/0/revision | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| ocatari/1/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, total_leaves, max_depth |
| ocatari/1/revision | checked | no | total_leaves, max_depth |
| ocatari/2/grow_prune | checked | no | total_leaves |
| ocatari/2/revision | checked | no | total_leaves, max_depth |
| ocatari/3/grow_prune | checked | no | prediction_1, prediction_2, prediction_3, sigma2, total_leaves, max_depth |
| ocatari/3/revision | checked | no | sigma2, total_leaves, max_depth |
| ocatari/4/grow_prune | checked | no | sigma2, total_leaves, max_depth |
| ocatari/4/revision | checked | no | total_leaves, max_depth |
| ocatari/5/grow_prune | checked | no | prediction_0, prediction_1, prediction_2, prediction_3, total_leaves, max_depth |
| ocatari/5/revision | checked | no | total_leaves |

Los fallos de mezcla no se resuelven por añadir interacciones de entorno. El siguiente diagnóstico propuesto es alargar cadenas sobre estos mismos datasets y revisar geometría de estados raros y normalización/calibración, antes de proponer cambios de sampler o priors. No se ejecutó búsqueda de hiperparámetros ni se ajustó usando test. Estos diagnósticos de un fit fijo tampoco acreditan cada fit online.

## Piloto preparado, no ejecutado

12 runs × 20.480 decisiones, cinco evaluaciones × diez episodios por run: **600 episodios de validación**. Estimación central **3.3 horas**; escenario de episodios al horizonte **45.0 horas**.

Se incluyen entrenamiento, evaluación y guardado. La extrapolación escala entrenamiento por fits/árboles/sweeps/batch e inferencia por árboles×draws; la geometría de los árboles puede alterar bastante el costo. El escenario por horizonte es una cota de trabajo evaluativo, no una garantía de tiempo máximo.

Los límites operativos provisionales del piloto (2 h de entrenamiento por run y 4 h de evaluación conjunta) permitirían detener y conservar progreso, pero pueden dar un resultado parcial si los episodios duran más. El usuario debe decidir presupuesto y si acepta un piloto exploratorio con mezcla insuficiente. No debe interpretarse como campaña confirmatoria.

El piloto y el test reservado no se ejecutaron. H0/H5, F-test, TD, sampler y priors históricos permanecen intactos.

Uso en [pong_comparison.md](pong_comparison.md). Evidencia, hashes y comandos en [validation_pong_comparison.json](validation_pong_comparison.json).
