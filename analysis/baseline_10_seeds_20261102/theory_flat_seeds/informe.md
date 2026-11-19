# Hipótesis para las semillas baseline 710602, 710608 y 710610

Auditoría estática del 2/11/2026. Se verificaron hashes de los checkpoints finales propios de la campaña antes de cargarlos. No se entrenó, restauró el emulador ni ejecutó evaluación nueva. Los árboles a 100 k/500 k/1M ya estaban exportados por la auditoría previa. No se editó la presentación.

## Qué falla, con evidencia directa

Estas semillas sí mejoran algo: medias primeras 100→últimas 100 partidas de 1,32→2,99;1,77→3,93;2,04→3,43. El problema es la meseta de bajo retorno, no ausencia absoluta de aprendizaje. Sus retornos greedy finales son 3,1,3.608 llega al horizonte tras puntuar 1;602/610 terminan naturalmente.

Las tres construyeron particiones insuficientes para una política robusta de seguimiento horizontal de la pelota. Los árboles RIGHT parten preguntando x(player_t,player_t-1), es decir, movimiento previo de la paleta. En 602/608, solo algunas ramas consultan x(player_t,ball_t). En 610, ninguna rama de RIGHT usa esa relación. RIGHT puede aprender un valor distinto según movimiento previo o presencia de pelota, pero no según si la pelota está a izquierda/derecha mientras esas variables permanecen iguales.

Fragmento real de RIGHT,610, a 1M:

```
movimiento previo de la paleta
├── more: se movió a derecha → Q=1,057838
├── same: no se movió        → Q=1,128528
└── less: se movió a izquierda
    └── pelota presente en instante anterior
        ├── True  → Q=1,182574
        └── False → Q=1,055370
```

No es una cuestión de poca profundidad permitida: profundidades finales 602=3,608=2,610=2 con límite 4. Y no están inicialmente vacíos: hay 11/11/7 divisiones.610 ya tenía exactamente esa estructura RIGHT a 100 k; siguió sin incluir posición horizontal a 1M.

## Sondas controladas de la política

Se construyeron 9 estados relacionales por agente cruzando x(player,ball)={less,same,more} y desplazamiento horizontal previo del jugador={less,same,more}. Se mantuvieron fijos: pelota/jugador presentes ahora y antes; pelota moviéndose abajo/derecha; jugador por debajo de pelota; contacto falso y desplazamiento vertical jugador same. Son sondas sintéticas de la función aprendida, no trayectorias del emulador ni muestras ponderadas por visitas. No se demuestra que cada acción sea óptima para toda geometría compatible.

Con **pelota a la derecha de la paleta**, la mejor acción resulta:

| Movimiento previo de paleta |602|608|610|607, referencia de 55 puntos|
|---|---|---|---|---|
| Izquierda |FIRE|LEFT|FIRE|RIGHT|
| Quieto |RIGHT|RIGHT|FIRE|RIGHT|
| Derecha |LEFT|LEFT|FIRE|RIGHT|

Ejemplo 608 con paleta que venía moviéndose a derecha: Q(NOOP)=1,218940, Q(FIRE)=1,192402, Q(RIGHT)=1,196127, Q(LEFT)=1,471520. Por tanto elige LEFT aunque la pelota esté a la derecha. Si luego se mueve a izquierda manteniéndose a la izquierda de la pelota, elige LEFT otra vez; quieta en esa relación elige RIGHT. El ranking permite oscilación o movimiento lateral poco orientado, pero **no se simuló ni demostró ese ciclo completo**.

602 comparte un patrón parecido, aunque usa FIRE cuando venía moviéndose a izquierda.610 no elige RIGHT en ninguna de las 9 sondas: con pelota a derecha domina FIRE, y alineada/aizquierda dominaLEFT. Esto no significa que jamás use RIGHT en otros estados incompletos o fases: sí hay acciones RIGHT en sus registros.

Comparación importante:601, la mejor evaluación final 101, también comienza RIGHT con movimiento previo. Pero se subdividió para distinguir posición relativa bajo las tres ramas principales de movimiento (en algunas, tras preguntar presencia). Por tanto **preguntar por movimiento primero no es suficiente para explicar el fracaso**. El problema observado es la combinación de particiones posteriores y valores, no el nombre de la raíz aislado.607 ofrece otra estructura que distingue posición y muestra seguimiento estable en estas sondas.

## Por qué no siguen subdividiéndose

En RIGHT de 610, rama same, la relación candidata x(player,ball) acumuló 16182 observaciones, con J_total=33,45617 y sumaJ_grupos=33,34021. El cociente histórico esR=0,996534 y su p implementado=0,412611. Con nivel.0001, no supera el criterio. En las ramas more de 602/608/610:

| Semilla | n de la candidata |R|p histórico|
|---|---:|---:|---:|
|602|2949|0,95403|0,10074|
|608|1360|0,99272|0,44644|
|610|674|0,99925|0,49614|

No basta reunir 128 observaciones: según el criterio almacenado actual, esos grupos no separan suficiente dispersión. La candidata sí existe y hay profundidad disponible. Estos valores son diagnósticos del criterio implementado, no p-values de un test F cuya calibración se haya validado para datos TD dependientes. Tampoco permiten reconstruir cada comprobación histórica.

Crucialmente, la estadística acumula el valor Q compartido de la hoja después de cada actualización, no targets TD individuales. En 610/RIGHT/same las medias históricas de ese Q bajo posiciones more/same/less son aproximadamente 1,09586/1,08637/1,08977: casi iguales. Un valor compartido y suavizado puede mostrar poca separación aun si los resultados futuros requieren distinguir posición. Es una hipótesis sobre la falta de sensibilidad del criterio; no se midió aquí la heterogeneidad de targets por estado.

## Hipótesis mecanística

Una partición inicial centrada en movimiento previo agrupa situaciones que requieren acciones distintas. Las hojas promedian mediante TD sobre esos grupos; la acción greedy resulta demasiado dependiente de historia de movimiento o demasiado dominada por FIRE/LEFT. Eso afecta la distribución de experiencias posterior. Los candidatos de posición tienen evidencia efectiva local muy inferior al millón global, y el criterio sobre Q suavizado no admite su división. La exploración mantiene algo de puntuación, pero el modelo no desarrolla un seguimiento robusto. Ningún mecanismo permite reemplazar directamente las particiones antiguas.

La distribución de acciones es compatible: en los últimos 100 k movimientos,602 usaRIGHT18,77% yLEFT44,94%;608 RIGHT16,56% yLEFT47,46%;610 RIGHT10,34%,LEFT36,42%,FIRE41,32%. La referencia 607 usaRIGHT33,18% yLEFT37,16%. Son frecuencias de trayectorias distintas, no una prueba de que más simetría cause mayor retorno. Tampoco se puede atribuir todo movimiento RIGHT a epsilon.

## Alcance y prueba que falta

Hay evidencia exacta de insensibilidad a posición en ramas y de rankings poco útiles en sondas. **No se demostró que una división temprana específica cause la meseta.** La política de exploración, historia, TD y estructura cambian conjuntamente, y la representación es aproximada.

Para probar el mecanismo: partir del mismo checkpoint y comparar una continuación control con una variante que refine explícitamente RIGHT por posición o permita revisar esa rama, manteniendo presupuesto y protocolo. Inicializar con el mismo Q permite que la separación aprenda antes de evaluar. Instrumentar visitas y objetivos/errorTD por posición permite distinguir poca cobertura de estadística de división insensible. Comparar ANOVA o targetsTD sería una ablación separada; cambiar todo a la vez no identifica causa.

Archivos: `policy_probe.csv` contiene todas las predicciones; `nodes.csv` el inventario de nodos de 10 agentes; `leaf_statistics.json` las estadísticas de candidatas en hojas; `analyze.py` reproduce esta auditoría desde la raíz capstone con las dependencias del repo.
