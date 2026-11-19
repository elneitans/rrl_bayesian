# Árboles y evolución temporal: baseline histórico frente a ANOVA

Análisis estático del 6 de noviembre de 2026, calculado desde los CSV producidos por `audit.py`: `historical_evaluations.csv`, `anova_evaluations.csv`, `*_structures.csv`, `*_nodes.csv`, `*_policy_probes.csv`, `*_policy_changes.csv` y `*_candidate_statistics.csv`. El auditor verificó las unidades individuales de evaluación y los checkpoints propios antes de deserializar. No se corrió entrenamiento ni evaluación nueva. Son diez agentes por criterio, OCAtari Breakout, 1M decisiones por agente; las evaluaciones observadas son desarrollo, no test reservado. Las versiones de fuentes difieren aunque los contratos científicos estén alineados. La comparación es descriptiva por semillas comunes, no una atribución causal estricta al criterio aislado.

## Aprendizaje observado y retrocesos

Cada celda muestra el retorno bruto greedy de los checkpoints 100k → 500k → 1M. Los puntos intermedios tienen dos episodios por agente; el final también tiene diez episodios de desarrollo. Por agente, sus retornos/duraciones se repiten bajo el entorno determinista actual. No hay cien entrenamientos independientes.

| Semilla | Histórico | ANOVA | Lectura ANOVA |
|---|---|---|---|
|710601|1 → 43 → 101|6 → 5 → 17|Mejora tardía moderada; queda bajo el histórico.|
|710602|2 → 1 → 3|34 → 83 → 160|Mejora en todos los puntos observados.|
|710603|7 → 59 → 25|15 → 32 → 3|Caída del 90,6% entre500k y1M.|
|710604|54 → 3 → 29|14 → 17 → 27|Mejora gradual en puntos observados.|
|710605|6 → 3 → 0|22 → 3 → 6|Caída temprana y recuperación parcial; termina bajo.|
|710606|92 → 2 → 25|30 → 49 → 139|Mejora sostenida entre puntos observados.|
|710607|3 → 3 → 55|56 → 330 → 87|Gran pico a500k y caída del73,6%, aunque supera el histórico final.|
|710608|3 → 3 → 1|46 → 3 → 42|Caída del93,5% y recuperación casi al nivel100k.|
|710609|37 → 31 → 37|15 → 65 → 95|Mejora sostenida entre puntos observados.|
|710610|1 → 4 → 3|18 → 2 → 8|Caída del88,9% y recuperación parcial; termina bajo.|

ANOVA tiene seis semillas con algún descenso observado y cinco que terminan por debajo de un punto anterior:603,605,607,608,610. En el histórico terminan por debajo de un punto previo603,604,605,606,608,610. No estimar una frecuencia de retroceso general con esta muestra pequeña y calendario escaso. Las líneas entre checkpoints no describen evaluaciones continuas ni permiten localizar el instante de caída.

| Decisiones | Histórico: media / mediana | ANOVA: media / mediana |
|---|---:|---:|
|0|1 / 1|1 / 1|
|100k|20,6 / 4,5|25,6 / 20|
|500k|15,2 / 3|58,9 / 24,5|
|1M|27,9 / 25|58,4 / 34,5|

La media ANOVA parece plana500k→1M, pero mezcla una caída243 puntos de607 con ganancias en otras semillas. Sin607, la media de los otros nueve sube28,78→55,22. La mediana global también sube24,5→34,5. Por tanto, esa meseta de la media no demuestra que todos hayan dejado de aprender. Tampoco el pico330 es rendimiento final ni un resultado de test independiente.

En las100 evaluaciones finalesANOVA todas las partidas terminan naturalmente, sin truncación. Los resultados bajos603/605/610 corresponden a212/317/435 decisiones y3/6/8 puntos; sí ejecutanFIRE (68/69/72 veces). Por tanto, estos fallos finales no son la espera hasta el horizonte por ausencia deFIRE del histórico608. Este último daba1 punto, unFIRE y27000 decisiones; ANOVA608 obtiene42 puntos en1125 decisiones,266FIRE y terminación natural. Las frecuencias no dicen por sí solas siFIRE se usa en la fase adecuada.

El archivo `comparison_stats.json` contiene bootstrap por semillas y sensibilidad: final58,4 frente27,9; diferencia+30,5, IC95% bootstrap pareado[-8,6;71,4]; siete victorias y tres derrotas; test de signos bilateralp=0,34375. La señal favorece ANOVA, sin estimación precisa de superioridad poblacional.

## La diferencia estructural es grande

| Etapa | Histórico: divisiones medias / hojas medias | ANOVA: divisiones medias / hojas medias |
|---|---:|---:|
|100k|8,4 / 19|68 / 137,6|
|500k|12,3 / 26,1|97,7 / 195,8|
|1M|14,4 / 30|108,5 / 217,3|

Son totales entre los cuatro árboles de cada agente, no cantidades por árbol. ANOVA produce7,53 veces más divisiones y7,24 veces más hojas finales en promedio. Todos sus agentes llegan a profundidad4 ya a100k. Sus modelos finales tienen101–123 divisiones y200–249 hojas; el histórico7–23 y15–47.

A1M,1835/2173 hojas ANOVA (84,4%) están en el límite de profundidad4. Solo9/300 hojas históricas (3%) lo están. Son proporciones de hojas estructurales, no proporciones de visitas del agente. El cuello de crecimiento histórico es principalmente el criterio/evidencia de separación; en ANOVA la restricción de profundidad aparece mucho más relevante. Eso no justifica aumentar profundidad automáticamente: mayor capacidad también puede fragmentar más las observaciones.

| Semilla | Divisiones ANOVA | Hojas ANOVA | Hojas a profundidad4 |
|---|---:|---:|---:|
|710601|114|229|198|
|710602|108|217|185|
|710603|109|218|180|
|710604|101|200|166|
|710605|101|205|173|
|710606|112|220|186|
|710607|123|249|218|
|710608|102|205|166|
|710609|111|225|195|
|710610|104|205|168|

La complejidad final de603 (109 divisiones,3 puntos) es parecida a602 (108 divisiones,160 puntos).601 tiene114 divisiones y17 puntos;606112 y139. Número de nodos y presencia de posiciones útiles no garantizan una buena política.

Todas las reglas antiguas permanecen iguales en su ruta en ambos criterios: solo se añaden descendientes. ANOVA cambia el test de crecimiento, pero no incorpora poda, reemplazo de reglas antiguas ni reversión de una partición temprana.

## Posición horizontal, raíces y política

En el histórico, LEFT comienza por `x(player_t,ball_t)` en10/10 agentes; RIGHT lo hace en0/10. ANOVA coloca esa posición en la raíz de RIGHT en5/10 y deLEFT en6/10; en el conjunto de40 raíces aparece21 veces frente10 históricas. También usa movimiento horizontal/vertical de pelota y movimiento previo de paleta. No hay una única raíz que explique el éxito:

- ANOVA602,160 puntos: RIGHT empieza por movimiento vertical de pelota y luego incluye posición horizontal siete veces en ramas distintas.
- ANOVA606,139 puntos: RIGHT empieza por posición horizontal; LEFT por movimiento previo de paleta, con posición horizontal en ramas posteriores.
- ANOVA601,17 puntos: ambosRIGHT yLEFT empiezan por posición horizontal, pero eso no basta para llegar al rendimiento histórico101.
- ANOVA610,8 puntos: RIGHT empieza por movimiento previo, pero ahora incorpora posición horizontal en cuatro ramas. Ha corregido parte de la ceguera del árbol histórico610, que no usaba esa posición enRIGHT, sin conseguir todavía una política robusta.

Las sondas del auditor cruzan posición horizontal relativa con movimiento previo de paleta, manteniendo fija pelota presente, descendente y hacia la derecha, entre otras relaciones. Son estados sintéticos de la función aprendida; no representan todas las geometrías ni prueban una acción óptima para cada trayectoria compatible. `position=less` significa paleta a la izquierda de pelota: seguir su posición sugeriríaRIGHT.

| Semilla | Histórico: acción cuando pelota está a derecha, paleta venía izquierda / quieta / derecha | ANOVA: mismas sondas |
|---|---|---|
|710601|RIGHT / RIGHT / RIGHT|LEFT / RIGHT / FIRE|
|710602|FIRE / RIGHT / LEFT|RIGHT / RIGHT / RIGHT|
|710603|RIGHT / RIGHT / RIGHT|RIGHT / NOOP / RIGHT|
|710605|RIGHT / RIGHT / RIGHT|NOOP / RIGHT / RIGHT|
|710607|RIGHT / RIGHT / RIGHT|RIGHT / RIGHT / RIGHT|
|710608|LEFT / RIGHT / LEFT|RIGHT / RIGHT / RIGHT|
|710610|FIRE / FIRE / FIRE|NOOP / RIGHT / FIRE|

La corrección de602/608 es consistente con su mejora final3→160 y1→42.601 muestra el caso inverso. Pero605/603 muestran que aprobar unas pocas sondas de seguimiento no garantiza buen resultado en el juego.607 mantieneRIGHT en las tres sondas a500k y1M pese a330→87: las nueve sondas son insuficientes para explicar esa caída.

En total, de30 sondas finales con pelota a derecha, RIGHT aparece23 veces en histórico y24 enANOVA; con pelota a izquierda, LEFT27 frente23. Este conteo no es una métrica de optimalidad y no respalda afirmar que ANOVA haya aprendido universalmente mejor seguimiento. La política completa comparaQ entre cuatro árboles en muchas más condiciones que las sondas.

## Criterio estadístico, capacidad y cobertura

El histórico evalúa `F.cdf(W/J,n-1,n-1)`; ANOVA usa diferencias de medias entre categorías, `F=(B/(K-1))/(W/(n-K))`. Ambos acumulan estadísticas delQ de la hoja después de cada TD, no de targetsTD individuales ni retornos futuros independientes. Los p son scores nominales de crecimiento bajo dependencia temporal y pruebas adaptativas repetidas.

En los modelos históricos finales,426 candidatas repartidas entre186 hojas cumplen mínimo de muestras, profundidad disponible y pANOVA≤1e-4. El histórico las conservó sin dividir. En los ANOVA finales no queda ninguna candidata con esas tres condiciones. En cambio,412 candidatas de289 hojas ANOVA tienen mínimo y p≤1e-4 pero están bloqueadas por profundidad4. Algunas separaciones tienen fracciónB/(B+W) menor de0,1%: significación nominal no equivale a separación grande o utilidad para el control.

Hay también fragmentación. Tomando por hoja el máximo contadorn entre sus candidatas restantes como indicador disponible de soporte estadístico,49/2173 hojas ANOVA tienen todos esos contadores en0 y805/2173 (37%) no superan128; en el histórico0/300 y8/300, respectivamente. No equiparar este máximo sin más con todas las visitas posibles: los contadores dependen de presencia de cada relación y se reinician al crear hijos. Las hojas nuevas heredanQ. La observación evidencia soporte local desigual y ramas escasas, no demuestra sobreajuste ni que cada rama sin muestras sea físicamente alcanzable.

## Q cambia aunque el árbol no crezca

El mismo panel fijo externo por semilla se usa ahora para ambos métodos; procede de estados de la campaña BART previa y no se pondera por visitas baseline/ANOVA. El auditor cuenta estados donde los conjuntosargmax de dos checkpoints son disjuntos. Desde el checkpoint cero todo es empate, de modo que cero cambios frente a cero no implica una política equivalente.

Entre999936 y1M hay64 transiciones y ninguna división nueva en ambos criterios:

- ANOVA601 cambiaargmax en4,55% del panel;6031,19%;6052,60%;6092,56%. Los demás0%.
- Histórico60514,29%,6083,90%,61034,21%; demás0%.

Estas cifras históricas difieren de la auditoría antigua (13,41/3,66/32,10%) porque aquella añadía los estados guardados del colector al panel; no mezclar denominadores. No son porcentajes de decisiones reales cambiadas. Demuestran que las actualizacionesQ alteran rankings sin cambios estructurales. No demuestran que esos64 pasos causen una caída de retorno: no se evaluógreedy999936.

Coneta=.025, cadaQ recibe una actualización exponencial con vida media de aproximadamente27,4 actualizaciones de esa hoja si se considera solo el peso de valores pasados; las estadísticas para crecimiento, en cambio, acumulan toda la historia desde la creación de la hoja. El contraste entre adaptación recienteQ y estadísticas acumulativas es un mecanismo plausible para estudiar, no una causa confirmada.

## Qué observar antes de proponer una mejora definitiva

1. Congelar y reevaluar100k/500k/1M bajo un conjunto de inicios realmente diversos, conservando una evaluación comparable al protocolo actual. Medir greedy yepsilon0,01/0,1 sin actualizar. Esto separa fragilidad de una trayectoria determinista de una política más general y del rescate por exploración.
2. Dar prioridad a603 (32→3),607 (330→87),608 (46→3→42),610 (18→2→8) y controles602/606. Registrar fases: bola ausente, relanzamientoFIRE, pérdidas de vida, tiempo sin puntos, geometría/velocidad y márgenesQ. No inferir esas fases solo por frecuencia global de acciones.
3. En continuaciones diagnósticas cortas, evaluación más frecuente y log explícito de eventos de división: ruta, candidata, n, B,W, efecto y predicciones antes/después. Los JSONL históricos guardanTD pero no estados/identidad de hoja ni eventos detallados de split; no se puede reconstruir causalmente cada descenso con la evidencia retenida.
4. Comparar desde el mismo checkpoint una continuación con crecimiento normal frente a estructura congelada, manteniendoTD y presupuesto. Separadamente, evaluar una política totalmente congelada. Así se distingue cambioQ, crecimiento y simple variabilidad de evaluación.
5. Medir en hojas la heterogeneidad de targets/errorTD por posición y acción con un panel de estados reales. Decidir entonces si hace falta cambiar la variable del testQ→target, revisar estructura o ajustar cobertura.
6. Una vez observados los mecanismos, hacer ablaciones separadas del mínimo128, profundidad4 y tasaTD. ANOVA está mucho más cerca del límite de capacidad y muestra más ramas de soporte pequeño; el histórico está lejos de saturar profundidad. Cambiar todo a la vez impediría atribuir una mejora.

No hay evidencia suficiente para afirmar que el fallo sea sobreajuste, que una raíz específica sea causal, que aumentarprofundidad solucione las caídas ni queANOVA haya resuelto la estabilidad. Sí hay evidencia de mayor sensibilidad de crecimiento, árboles mucho más grandes, mejor rendimiento agregado observado y retrocesos todavía importantes.
