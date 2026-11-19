# Dos variantes de estabilidad de BART a un millón de decisiones

La campaña implementa por separado la conservación de estructuras con distancia acotada y la actualización menos frecuente del modelo objetivo. Usa cinco semillas pareadas por variante, 733101–733105, con diez entrenamientos desde cero hasta 1.000.000 de decisiones. El protocolo de referencia es la variante cada 1.024 de la campaña del 12 de noviembre. Cada decisión corresponde a cuatro frames de ALE.

| Variante | Inicialización de reajustes posteriores al primero | Distancia estructural máxima por árbol | Reajustes por modelo objetivo |
| --- | --- | ---: | ---: |
| `bounded_structure` | Última muestra del reajuste anterior | 2 | 1 |
| `slow_target` | Árboles de una hoja | Sin restricción adicional | 4 |

Ambas variantes usan 20 árboles por acción, 50 sweeps de burn-in, 50 muestras retenidas, kernel `grow_prune`, 128 transiciones por acción y replay de 50.000 decisiones. El primer ajuste ocurre a las 512 decisiones y los siguientes cada 1.024. Reutilizar las semillas históricas permite contrastar los resultados hasta 500.000 con la referencia guardada. El contraste entre estas dos variantes mide la diferencia entre sus dos intervenciones completas.

## Estructuras heredadas

La distancia cuenta reglas de partición añadidas, eliminadas o cambiadas en rutas correspondientes del árbol. Dividir una hoja o podar una pareja de hojas cuesta una unidad. Eliminar un subárbol cuesta una unidad por cada regla eliminada. Los valores de las hojas se actualizan mediante Gibbs y quedan fuera de esta distancia.

Cada reajuste hereda la última muestra de cada árbol del posterior anterior. Durante todos los sweeps, las propuestas que exceden la distancia dos respecto de ese árbol heredado se rechazan. Esto permite explorar una vecindad fija mediante propuestas locales reversibles. La distribución objetivo queda restringida a esa vecindad. Las pruebas verifican el balance detallado frente a una distribución posterior enumerada en un problema pequeño.

Cuando el nuevo lote deja sin soporte una partición, se poda el subárbol correspondiente y esas reglas eliminadas consumen el mismo presupuesto. Si la poda necesaria excede dos, se conserva el posterior anterior de esa acción para ese reajuste. El log registra el árbol que impidió la adaptación, la distancia de reparación y la acción conservada. También registra la distancia máxima alcanzada por cada árbol, las podas iniciales y los rechazos por exceder la vecindad. El primer ajuste construye los árboles con la receta de referencia.

Esta variante controla la estructura. El remuestreo de valores de hojas y el promedio posterior también pueden cambiar las predicciones y las decisiones greedy. Las acciones conservadas por soporte deben interpretarse junto con el puntaje para detectar si la restricción dificulta aprender.

## Modelo objetivo cada cuatro reajustes

El modelo publicado sigue actualizándose cada 1.024 decisiones. Los objetivos TD usan un modelo congelado durante cuatro ajustes consecutivos. Los ajustes 1–4 usan el modelo 0, los ajustes 5–8 usan el modelo 4 y así sucesivamente. Los identificadores de los modelos publicados avanzan en cada ajuste y los logs guardan el identificador del modelo que realmente generó sus objetivos.

La publicación del modelo, la calibración y los generadores aleatorios es atómica. El checkpoint conserva el modelo objetivo congelado y el contador que fija su calendario. Las pruebas comparan la continuación tras un checkpoint con la ejecución continua, incluyendo objetivos, muestras, modelos y decisiones. Esta actualización más lenta también reduce la frecuencia de propagación de nueva información de Bellman, cuyo efecto se medirá dentro del mismo presupuesto de interacciones.

## Calendario y verificación

Los checkpoints de desarrollo están en 100k, 200k, 300k, 400k, 500k, 600k, 700k, 800k, 900k, 925k, 950k, 975k y 1M. Los doce intermedios usan las semillas compartidas 734001–734002. El final usa 734101–734110. El test reservado conserva su calendario vacío. Los checkpoints adicionales al final permiten medir el rendimiento sostenido.

La preparación ejecuta las pruebas unitarias y una integración técnica de 96 decisiones por variante con ALE. El array alterna las variantes dentro de cada semilla y permite dos tareas simultáneas. El reporte depende de que terminen las diez tareas. Cada tarea verifica su frecuencia de ajuste, los identificadores de los objetivos y los límites estructurales. El reporte vuelve a contrastarlos con los logs y coteja los resultados de desarrollo con sus archivos individuales.

La carpeta aislada del clúster es `~/projects/rrl_breakout_bart_stability_1m_20261114_01`. Su `jobs.tsv` registra los envíos y `pilot/report.json` consolidará los resultados. Cada celda dispone de un presupuesto operativo de 24 horas de entrenamiento y el tope científico de un millón de decisiones.

La preparación Slurm `39800` terminó con código cero. Pasaron 78 pruebas en 139 segundos y las dos integraciones con ALE completaron seis ajustes y tres evaluaciones por variante. El array `39801` contiene las diez corridas y el reporte `39802` depende de su finalización. En la verificación de envío del 14 de noviembre, las tareas 0 y 1 estaban ejecutándose en Hydra y las ocho restantes esperaban su turno. La cola admite un máximo de 24 horas por tarea, límite que se aplicó al array.

[Verificación del código y los protocolos enviados](../results_cluster/rrl_breakout_bart_stability_1m_20261114_01/launch_verification.json) · [Salida de las pruebas](../results_cluster/rrl_breakout_bart_stability_1m_20261114_01/slurm/prepare-39800.out)

[Configuración](../configs/breakout_bart_stability_1m_20261114.json) · [Ejecución y comprobaciones](../experiments/breakout_bart_stability.py) · [Pruebas específicas](../tests/test_bart_stability.py)

## Propuesta para el ensayo conjunto

El candidato conjunto usaría `warm_start=true`, `max_structure_distance=2` y `target_update_interval=4`, con reajustes cada 1.024 decisiones. El árbol heredado acotaría la variación estructural y el objetivo compartido por cuatro reajustes reduciría la frecuencia con que cambian las etiquetas TD. Los valores de las hojas seguirían aprendiendo en cada reajuste.

La decisión de ensayar esa combinación se revisará con los resultados de los dos brazos individuales. Se compararán los puntajes de la ventana 900k–1M por semilla, los bloqueos de saque, las acciones conservadas por falta de soporte y los objetivos fuera de escala. La implementación permite expresar ambos parámetros juntos. Las diez corridas enviadas evalúan los dos mecanismos por separado y el ensayo conjunto requiere un presupuesto posterior.

## Resultados revisados el 17 de noviembre

Las diez corridas alcanzaron 1.000.000 de decisiones y 977 reajustes cada una. Slurm marcó las diez tareas del array `39801` y el reporte `39802` como completados con código cero. Se cotejaron las 340 evaluaciones de desarrollo con sus archivos de identidad y estado y con el consolidado. Dentro de cada política, las evaluaciones repitieron el puntaje, la duración, el histograma de acciones y el registro de saques. Las cinco semillas de entrenamiento son las réplicas del contraste pareado.

| Decisiones | Estructura acotada | Objetivo cada cuatro reajustes |
| ---: | ---: | ---: |
| 500.000 | 3,40 | 11,60 |
| 600.000 | 8,80 | 3,20 |
| 700.000 | 15,20 | 6,60 |
| 800.000 | 2,80 | 9,20 |
| 900.000 | 2,00 | 7,60 |
| 925.000 | 1,20 | 2,00 |
| 950.000 | 1,20 | 2,20 |
| 975.000 | 1,20 | 1,80 |
| 1.000.000 | 0,20 | 0,00 |

Cada valor es la media del puntaje bruto entre semillas de entrenamiento. Estructura acotada llegó a 67 puntos en 733102 a 700k. Los checkpoints de la misma corrida dieron cero puntos en 600k y tres en 800k. Objetivo cada cuatro reajustes llegó a 44 puntos en 733102 a 500k y tres en 600k. El puntaje medio en 900k–1M fue 1,16 con estructura acotada y 2,72 con objetivo cada cuatro reajustes. Al usar los cuatro checkpoints desde 925k, las medias fueron 0,95 y 1,50.

| Resultado por semilla a 1M | Estructura acotada | Objetivo cada cuatro reajustes |
| --- | --- | --- |
| 733101 | 1 punto, terminó | 0, sin saque inicial |
| 733102 | 0, sin saque inicial | 0, sin saque después de perder la primera vida |
| 733103 | 0, sin saque inicial | 0, sin saque inicial |
| 733104 | 0, sin saque inicial | 0, sin saque inicial |
| 733105 | 0, sin saque inicial | 0, sacó de inmediato en las cinco vidas |

En 900k–1M, 15 de las 25 políticas evaluadas con estructura acotada truncaron la partida a 27.000 decisiones, frente a 9 de 25 con objetivo cada cuatro reajustes. A 1M truncaron cuatro de cinco en cada variante. En el período comparable de 100k–400k, con las mismas semillas de evaluación, la referencia histórica cada 1.024 obtuvo 4,05 puntos de media, frente a 3,05 y 2,00 en los dos brazos nuevos. A 500k la referencia histórica obtuvo 1,80 con el grupo de evaluación final y las variantes nuevas usaron el grupo intermedio. La referencia pareada cada 1.024 llega hasta 500k.

[Reporte consolidado de Slurm](../results_cluster/rrl_breakout_bart_stability_1m_20261114_01/pilot/report.json) · [Referencia histórica a 500k](../results_cluster/rrl_breakout_bart_cadence_500k_20261112_01/pilot/report_verified.json)

La auditoría de los logs completos y de los diez checkpoints finales mostró que los dos mecanismos actuaron de forma distinta.

| Diagnóstico | Estructura acotada | Objetivo cada cuatro reajustes |
| --- | ---: | ---: |
| Objetivos TD ajustados fuera de [−100, 100] durante toda la corrida | 81,76 % | 0 de 2.501.038 |
| Objetivos TD ajustados fuera de la cota en las últimas 100k decisiones | 99,58 % | 0 de 250.880 |
| Máximo Q final sobre el replay, rango entre semillas | 163,98–185,52 | 15,90–60,18 |
| Cambio de acción greedy en el último reajuste, mediana entre semillas | 68,19 % | 67,20 % |
| Tiempo medio de entrenamiento instrumentado | 7,23 h | 5,86 h |

El cambio de acción se calculó entre los ajustes 976 y 977 para ambas variantes, sobre los estados con pelota visible de cada replay final y ponderado por frecuencia de visita. El límite estructural se respetó en cada árbol ajustado. Por incompatibilidad de soporte se conservó la política anterior en 627 de 19.520 actualizaciones de acción posteriores al primer ajuste (3,21 %). Esas actualizaciones se excluyen del denominador de objetivos TD ajustados. La inflación apareció en las cinco semillas con estructura acotada entre 62.976 y 163.328 decisiones, y al final todos los Q predichos sobre sus estados visitados superaban 100. El modelo objetivo actualizado cada cuatro reajustes mantuvo los objetivos ajustados dentro de la cota en las cinco semillas. Sus bloqueos finales muestran que controlar esa escala deja abierta la estabilidad de la decisión greedy.

Con estos resultados, el objetivo cada cuatro reajustes es el candidato para seguir investigando. Dejaría la combinación con el límite de dos cambios para después de una prueba de cobertura del replay que preserve ejemplos de saque, reinicio de vida y pelota en juego, con semillas pareadas. Evaluaría el puntaje sostenido y la frecuencia de bloqueos en toda la ventana final.

[Auditoría de los logs y checkpoints finales](../results_cluster/rrl_breakout_bart_stability_1m_20261114_01/pilot/fit_audit_20261117.jsonl)
