# Revisión independiente de hipótesis — BART Breakout 10 semillas

Datos locales: `rrl_bayesian/results_cluster/breakout_1m_latest/bart`, snapshot descargado 2026-11-01. No se verificaron blobs de modelos ausentes. Se inspeccionó implementación actual para interpretar campos, no se entrenaron modelos ni modificó el algoritmo.

## Hallazgo directo: objetivos fuera de los límites de retorno

El manifiesto declara recompensa de aprendizaje sign(raw), reward_min=-1, reward_max=1 y gamma=.99. Por tanto la cota conservadora del retorno descontado es [-100,100], y `FixedScale.from_rewards` usa ese intervalo. `BARTRegressor.fit` cuenta en `targets_outside_scale` los y originales fuera de esos límites; **no los recorta**. Un prior gaussiano tampoco impone una cota dura a Q.

Último ajuste publicado:

| Semilla | FIRE | LEFT | NOOP | RIGHT | Total /512 | Porcentaje |
|---|---:|---:|---:|---:|---:|---:|
|710601|126|127|128|128|509|99.414%|
|710602|0|0|0|0|0|0%|
|710603|125|127|126|128|506|98.828%|
|710604|0|0|0|0|0|0%|
|710605|0|0|0|0|0|0%|
|710606|0|0|0|0|0|0%|
|710607|0|2|2|1|5|0.977%|
|710608|0|0|0|0|0|0%|
|710609|0|0|0|0|0|0%|
|710610 (802816)|0|0|0|0|0|0%|

Los agentes601/603 presentan un problema de escala de objetivos, pero el resumen no contiene signo, magnitud ni origen temporal. Como y=r+.99 max Qanterior y |r|<=1, si |y|>100 hay necesariamente una predicción de bootstrap fuera de [-100,100] para esas transiciones. Es evidencia de valores incompatibles con la cota real de retorno. No basta para afirmar divergencia ilimitada. Los retornos de601/603 son4/3, pero ceros607–609 aparecen sin violación generalizada de cota: no hay una explicación única para todas las semillas.

Fuentes: `online/learners.py:LearningConfig.scale, fit_if_due`; `online/environments.py:CommonGameEnvironment.step`; `ensemble.py:BARTRegressor.fit`; `targets.py:frozen_q_targets`.

## Predicciones de Q y crecimiento estructural

La receta usa 20 árboles por acción, 4 acciones, 50 burn-in+50 draws, grow/prune, profundidad4 y min_child2. Se reinicia cada ajuste en árboles de una hoja (media y/m) y sigma²=lambda calibrado inicialmente por acción. `OnlineBatchAgent.fit_history` guarda sólo el último registro, pero los logs de sesión registran diagnósticos por ajuste.

La última muestra por acción tiene30–49 hojas sumadas entre20 árboles (1.5–2.45 hojas/árbol). El mejor agente602 tiene146 hojas en80 árboles; dos agentes cero607 y609 tienen173/154. Más estructura no equivale a mejor política.

Los movimientos aceptados sólo indican movilidad local de la cadena; no certifican mezcla ni precisión posterior. El status no incluye Rhat/ESS entre cadenas, así que **la mala mezcla de este experimento no puede afirmarse a partir de la auditoría anterior de Pong**. Es hipótesis plausible por la receta corta, no hecho medido para estos diez agentes.

Las etiquetas de árboles son intercambiables en el ensemble. Comparar árbol j con árbol j entre ajustes puede detectar el reinicio pero no mide consistencia de la función. Priorizar en un conjunto de estados fijo: RMSE/max salto Q por acción, cambio de argmax, margen entre primeras dos acciones, Q(FIRE) inicial/tras perder vida; complementar con histogramas de reglas/hojas/profundidad y ocupación.

## FIRE: entrenamiento y evaluación pueden diferir mucho

La configuración no fuerza FIRE en reset. Usa epsilon=.1 a partir50k decisiones, distribución exploratoria uniforme sobre4 acciones. Incluso si greedy nunca elige FIRE, exploración da probabilidad.1/4=.025 por decisión: espera geométrica media40 decisiones antes de FIRE, si se mantiene la condición. Eso puede permitir jugar en entrenamiento aunque evaluación greedy se quede sin lanzar pelota.

Estado final local:607/609 completan10 evaluaciones truncadas27k, retorno0, cuatro estados codificados, FIRE0;608 lleva3 completas idénticas. Esto es fuerte evidencia de una política atrapada en un ciclo de no-FIRE. Afirmar el estado físico exacto requiere Q/objetos/vídeo. Explica por qué curva de entrenamiento positiva no garantizaría retorno greedy positivo. En evaluación, duración27k no significa supervivencia hábil. Métricas útiles: puntos, proporción de episodios con puntuación>0, FIRE inicial, tiempo hasta primer punto/lanzamiento si objetos lo permiten, truncaciones.

Las evaluaciones completas por agente repiten exactamente retorno y duración; no tratar sus10 episodios como10 entrenamientos independientes. frameskip4 y repeat_action_probability0 pueden contribuir a poca diversidad; verificar realmente qué aleatoriedad introduce cada reset.

## Hipótesis teóricas a contrastar

1. Bootstrap y maximización pueden amplificar errores: y usa max sobre cuatro estimaciones con error y luego ajusta regresión. La contracción del operador Bellman exacto no garantiza estabilidad con aproximación de función y proyección/cambio de lote. Los601/603 motivan estudiar una curva de max|Q| y fracción objetivos fuera de cota. No declarar un bug específico sin reconstruir primer crecimiento anómalo.
2. Alias de representación: el catálogo de11 relaciones discretas puede juntar estados geométricamente diferentes. Esto crea objetivos heterogéneos que ningún crecimiento resuelve. Se debe comprobar con objetos o transiciones, no asumir por baja recompensa.
3. Replay FIFO50k y128 filas/acción: muestras distintas por bloque y cobertura desigual pueden cambiar estimaciones en estados raros/iniciales. Ajustes cada512 producen1953 modelos a1M; cambios argmax pueden amplificar por realimentación de política.
4. Reinicio y presupuesto MCMC corto: podría desperdiciar sweeps reconstruyendo estructura y producir error Monte Carlo. Comparar cold/warm start bajo idénticos lotes/objetivos y presupuesto; mantener prior idéntico. Los nuevos lotes imponen soporte min_child; validar árbol heredado.
5. Horizonte gamma=.99 es aproximadamente100 decisiones descontadas. Si FIRE tiene beneficio distante o poco representado, los márgenes entre FIRE/no-FIRE pueden ser pequeños. No es prueba de que gamma sea causa; medir margen y muestras de estados relevantes.
6. Bootstrap en truncación está activado: es correcto si horizonte es límite de colección de un proceso continuo, pero conserva estimaciones de ciclos sin premio, no aporta señal terminal negativa. No confundir truncaciones con muerte o éxito.

## Coste

Tiempo activo por semilla a1M:12.08–13.15h según sumas de fase, ajuste propiamente dicho70.42–71.73%. No son horas de pared de Slurm (colas/interrupciones/costos no contados). Semilla610 parcial10.87h. Mejorar ESS/segundo puede ser importante antes de multiplicar presupuesto; incrementar sweeps no garantiza solución.

## Prioridad de análisis

1. Curva de entrenamiento por episodio terminado y media móvil50, mostrando epsilon/horizonte; resumir entre semillas con media y dispersión, sólo sobre soporte común.
2. Curva de evaluación intermedia por agente si existe; no interpolar una evolución desde puntos finales.
3. Auditoría Q y acción greedy en estados fijos comunes entre checkpoints, y out-of-scale targets por ajuste.
4. Diagnósticos estructura y propuestas por ajuste; no equiparar aceptaciones con convergencia.
5. Pequeñas ablacaciones separadas de warm-start, revisión de kernel, objetivos acotados/Double-Q (estos cambian algoritmo y deben ser variantes nuevas), y diversidad de evaluación. No modificar la cohorte congelada.
