# Piloto de estabilidad de BART en Breakout OCAtari

El piloto compara cuatro variantes pareadas con tres semillas nuevas de entrenamiento y 100.000 decisiones por semilla y variante. Todas usan 20 árboles por acción, 50 sweeps de burn-in, 50 retenidos y el kernel `grow_prune`.

| Variante | Árboles al comenzar | Primera decisión de ajuste | Decisiones entre ajustes |
| --- | --- | ---: | ---: |
| Control | Árboles de una hoja | 512 | 512 |
| Warm start | Última muestra disponible, con validación de soporte | 512 | 512 |
| Intervalo 1.024 | Árboles de una hoja | 512 | 1.024 |
| Warm-up 4.096 | Árboles de una hoja | 4.096 | 512 |

Cada árbol previo que quede fuera del soporte del lote nuevo vuelve a una hoja. El prior de ruido y el valor inicial de la varianza se tratan igual en ambas variantes. Los detalles de cada ajuste cuentan árboles reutilizados y reiniciados. Los checkpoints conservan el estado necesario para retomar la corrida.

La etapa fija usa los checkpoints de 710602 y 710608 a 1M. Por semilla, el primer lote es el que se usó en el ajuste histórico 1.953 y los otros dos se muestrean del replay final con el modelo objetivo histórico congelado. Cuatro cadenas por tratamiento recorren la misma secuencia. En el primer lote, cada cadena con warm start parte de la última muestra del modelo objetivo histórico. En lotes sucesivos, parte de su propia última muestra. Los reportes registran los hashes de los lotes, el tiempo, las trazas de Q y el cambio de acción en un panel fijo ponderado por visitas.

Los doce entrenamientos usan el mismo conjunto de semillas por variante. Se evalúan dos episodios de desarrollo por checkpoint a 25.000 y 50.000 decisiones, y cinco por checkpoint final a 100.000. Cada episodio registra retorno bruto, truncación y la demora hasta la primera acción `FIRE` desde el inicio y desde cada pérdida de vida que permita seguir jugando. Las oportunidades que terminan sin `FIRE` quedan registradas como censuradas. La unidad del contraste de retorno es la semilla de entrenamiento. El test reservado queda fuera del calendario.

El código de la campaña se ejecuta desde una copia aislada bajo `~/projects/rrl_breakout_bart_ablation_20261110_01/code`. El preflight ejecuta pruebas unitarias, un smoke con ALE para los cuatro brazos y la auditoría de datos fijos. Después arranca el array de entrenamiento y evaluación. `jobs.tsv` registra las dependencias y `pilot/report.json` se genera tras completar el array.

Jobs enviados el 10 de noviembre de 2026. Preflight `38660`, array del piloto `38661` (dos tareas simultáneas) y reporte `38662`.

El preflight confirmó 13 pruebas unitarias y completó los cuatro brazos del smoke en Hydra. Con el presupuesto reducido del smoke, control y warm start ajustaron 12 veces a partir de la decisión 8. El intervalo mayor ajustó 6 veces desde esa misma decisión. El warm-up extendido ajustó 5 veces desde la decisión 64. El brazo warm start registró 86 reutilizaciones de árboles.

## Resultados del piloto

Los 12 entrenamientos llegaron a 100.000 decisiones y las 108 evaluaciones de desarrollo terminaron. Verifiqué cada evaluación contra su archivo individual. Los cinco episodios finales de cada combinación de semilla y variante dieron el mismo puntaje, así que los contrastes de retorno usan las tres semillas de entrenamiento como unidades.

| Variante | Retornos a 100k por semilla 731101, 731102, 731103 | Media | `FIRE` al inicio | `FIRE` tras perder una vida | Truncaciones | Minutos de entrenamiento por semilla |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control | 2, 0, 0 | 0,67 | 5/15 | 20/20 | 10/15 | 61,5 |
| Warm start | 3, 3, 0 | 2,00 | 15/15 | 60/60 | 0/15 | 85,5 |
| Intervalo 1.024 | 3, 7, 1 | 3,67 | 15/15 | 60/60 | 0/15 | 33,6 |
| Warm-up 4.096 | 2, 0, 1 | 1,00 | 10/15 | 35/40 | 5/15 | 57,1 |

Los minutos son la suma de las fases instrumentadas de entrenamiento en Hydra. Con el intervalo de 1.024 hubo 98 ajustes frente a 195 en el control, una reducción del 45 % del tiempo de entrenamiento. Las tres semillas mejoraron frente a su control y `FIRE` se ejecutó en las primeras 32 decisiones de las 60 oportunidades tras perder una vida. El warm start reutilizó el 91,9 % de los árboles después del primer ajuste. Lanzó en las 60 oportunidades tras perder una vida, 55 de ellas dentro de 32 decisiones. Un lanzamiento registrado corresponde a una acción `FIRE`.

| Variante | Retorno medio a 25k | A 50k | A 100k |
| --- | ---: | ---: | ---: |
| Control | 0,00 | 1,67 | 0,67 |
| Warm start | 0,00 | 3,67 | 2,00 |
| Intervalo 1.024 | 5,00 | 6,33 | 3,67 |
| Warm-up 4.096 | 3,67 | 1,33 | 1,00 |

Las medias intermedias y finales usan grupos de semillas de evaluación distintos. La variación entre checkpoints es considerable. Por ejemplo, en el brazo de 1.024 la semilla 731103 pasó de 16 puntos a 50k a 1 a 100k. El intervalo de 1.024 es el candidato prioritario para una corrida más larga por el contraste pareado de retorno y su menor costo. Tres semillas y 100k decisiones permiten seleccionar esa variante para seguimiento. La magnitud del efecto a mayor escala sigue abierta.

### Auditoría con datos fijos

| Semilla histórica | Cambio greedy ponderado, control → warm start | Cadenas que eligieron `FIRE`, control → warm start |
| --- | ---: | ---: |
| 710602 | 52,8 % → 46,4 % | 11/12 → 5/12 |
| 710608 | 74,1 % → 72,4 % | 10/12 → 4/12 |

La primera columna promedia cambios frente al modelo anterior en el panel de 17 estados. La segunda resume cuatro cadenas por cada uno de los tres lotes. Para 710608, en el primer lote las cuatro cadenas con warm start mantuvieron `LEFT`, que era la acción preferida por el modelo desde el que se inicializaron. Las cuatro cadenas de control eligieron `FIRE` con esos mismos datos y objetivos. En estas dos semillas, el warm start redujo algo el cambio de acción y a la vez conservó decisiones de saque peores. En el piloto en línea, el warm start mejoró el arranque respecto del control, con un costo medio de entrenamiento 39 % mayor. La auditoría fija y el piloto responden preguntas distintas porque en línea también cambian los datos y objetivos que produce cada política.

### Integridad y archivos

El reporte Slurm 38662 falló al leer los resúmenes porque el checksum se calculó con claves numéricas de checkpoints que JSON recupera como texto. Verifiqué los 12 checksums con las claves originales, cotejé las 108 evaluaciones y reconstruí el reporte local. Corregí la escritura para usar claves de texto desde el inicio.

[Reporte verificado](../results_cluster/rrl_breakout_bart_ablation_20261110_01/pilot/report_verified.json) · [Auditoría 710602](../results_cluster/rrl_breakout_bart_ablation_20261110_01/fixed/710602/report.json) · [Auditoría 710608](../results_cluster/rrl_breakout_bart_ablation_20261110_01/fixed/710608/report.json)

## Seguimiento pareado a 500.000 decisiones

El 12 de noviembre se enviaron los brazos de intervalo 512 y 1.024 con cinco semillas nuevas de entrenamiento (733101–733105). Ambos empiezan a ajustar en la decisión 512 y comparten los 50 + 50 sweeps, los 20 árboles por acción y la inicialización desde una hoja. Se evalúan los checkpoints de 100k, 200k, 300k y 400k con dos semillas de desarrollo compartidas, y el de 500k con diez semillas de desarrollo distintas. Se mantienen fuera del calendario las semillas reservadas.

La carpeta de Slurm es `~/projects/rrl_breakout_bart_cadence_500k_20261112_01`. La preparación `39140` ejecuta las pruebas y el smoke, y congela ambos protocolos. El array `39141` alterna control e intervalo 1.024 dentro de cada semilla y admite dos tareas simultáneas. El consolidado `39142` se ejecuta tras completar las diez tareas y escribe `pilot/report.json`. [Configuración a 500k](../configs/breakout_bart_cadence_500k_20261112.json)

### Resultados verificados a 500.000 decisiones

Los jobs 39140, 39141 y 39142 terminaron con código de salida cero. Se verificaron los manifiestos, estados y checksums de las diez celdas y las 180 evaluaciones de desarrollo. Cada resultado individual coincide con su resumen y el consolidado final. La única diferencia metodológica entre los manifiestos de ambas variantes es el intervalo de ajuste. Las 100 evaluaciones finales usaron diez semillas comunes de desarrollo por política.

| Semilla de entrenamiento | Media final con 512 | Media final con 1.024 | Diferencia |
| --- | ---: | ---: | ---: |
| 733101 | 0 | 0 | 0 |
| 733102 | 0 | 4 | +4 |
| 733103 | 0 | 3 | +3 |
| 733104 | 0 | 2 | +2 |
| 733105 | 0 | 0 | 0 |
| **Media** | **0,0** | **1,8** | **+1,8** |

| Métrica final | Intervalo 512 | Intervalo 1.024 |
| --- | ---: | ---: |
| Episodios con `FIRE` en las primeras 32 decisiones | 10/50 | 50/50 |
| Episodios truncados al alcanzar 27.000 decisiones | 40/50 | 0/50 |
| `FIRE` dentro de 32 decisiones tras perder una vida, cuando el episodio continúa | 40/40 | 180/200 |
| Ajustes BART por semilla | 976 | 488 |
| Horas medias de entrenamiento por semilla | 5,56 | 2,98 |
| Horas medias en ajustes BART por semilla | 4,47 | 2,18 |

Las cuatro políticas de control que llegaron al horizonte nunca ejecutaron `FIRE` en sus 40 episodios. El quinto control sí inició el juego y perdió sus vidas sin puntuar. Con 1.024 las cinco políticas iniciaron el juego y los 50 episodios terminaron naturalmente. Dos de las cinco políticas siguieron con retorno cero. Las cifras de tiempo provienen de las fases instrumentadas. Slurm usó Hydra, Scylla, Yodaxico y Ventress, con dos pares evaluados en Hydra y uno en Scylla en ambos brazos. La reducción media observada fue del 46,4 %, junto con una reducción exacta del 50 % en el número de ajustes.

| Decisiones | Retorno medio con 512 | Retorno medio con 1.024 |
| --- | ---: | ---: |
| 100.000 | 9,4 | 1,8 |
| 200.000 | 1,6 | 3,4 |
| 300.000 | 5,0 | 3,0 |
| 400.000 | 0,6 | 8,0 |
| 500.000 | 0,0 | 1,8 |

En el intervalo de 1.024, la semilla 733102 obtuvo 34 puntos a 400k y 4 a 500k. Los checkpoints intermedios comparten dos semillas de evaluación y el final usa otras diez. El puntaje final coincidió entre los diez episodios de cada política congelada. El contraste inferencial tiene cinco pares de entrenamiento, con tres diferencias positivas y dos empates. El contraste bilateral descriptivo por inversión de signos da **p = 0,25**. La evidencia favorece el intervalo de 1.024 para reducir el bloqueo de saque y el cómputo bajo esta receta. El puntaje sigue siendo bajo y variable entre checkpoints.

[Resultados y comprobaciones](../results_cluster/rrl_breakout_bart_cadence_500k_20261112_01/pilot/report_verified.json) · [Consolidado Slurm](../results_cluster/rrl_breakout_bart_cadence_500k_20261112_01/pilot/report.json)
