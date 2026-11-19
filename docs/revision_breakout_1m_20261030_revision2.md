# Breakout 1M: segunda revisión del 30 de octubre

Se recuperó la conexión SSH. No se asumió fallo de los jobs por los timeouts previos y no se duplicaron envíos. Descarga separada en `results_cluster/breakout_1m_20261030_revision2/`.

## Avance verificado

| Método | Modelos a 1M con 10 evaluaciones de desarrollo | Modelo entrenándose |
|---|---:|---|
| Baseline | 9/10 | 710610: 139.264 decisiones |
| CART | 7/10 | 710608: 62.976 |
| ExtraTrees | 5/10 | 710606: 891.392 |
| BART | 1/10 | 710602: 488.448 |

Total: 22 de 40 modelos evaluados; faltan 18. Los cuatro workers continúan vivos: 35464_4, 35466_3, 35468_2 y 35470_0. Los grupos posteriores siguen en cola; ningún grupo observado terminó fallido/parcial ni requiere reenvío en esta consulta. BART35470_0 llevaba 18h54 y puede necesitar continuación si no completa la segunda semilla y sus evaluaciones antes del límite de 24h. Comprobar el resultado terminal antes de decidir ese reenvío.

## Retornos medios de desarrollo a 1M

Cada entrada usa las diez semillas de evaluación comunes. No es una réplica independiente de entrenamiento por episodio.

| Semilla entrenamiento | Baseline | CART | ExtraTrees | BART |
|---|---:|---:|---:|---:|
| 710601 | 101 | 0 | 0 | 4 |
| 710602 | 3 | 11 | 19 | pendiente |
| 710603 | 25 | 1 | 3 | pendiente |
| 710604 | 29 | 0 | 0 | pendiente |
| 710605 | 0 | 0 | 7 | pendiente |
| 710606 | 25 | 11 | pendiente | pendiente |
| 710607 | 55 | 12 | pendiente | pendiente |
| 710608 | 1 (*) | pendiente | pendiente | pendiente |
| 710609 | 37 | pendiente | pendiente | pendiente |
| 710610 | pendiente | pendiente | pendiente | pendiente |

(*) Baseline710608 terminó sus diez evaluaciones por truncación del horizonte, no por final natural del juego. El retorno observado sigue siendo válido para el presupuesto declarado de 27.000 decisiones, pero no debe describirse como puntuación de una partida completa natural. Los otros 210 episodios disponibles no están truncados. La semilla BART710601 tiene diez episodios completos con retorno medio 4; aún no permite estimar estabilidad entre semillas BART ni comparaciones concluyentes.

Los datos sugieren gran variación entre agentes baseline y resultados modestos en los reemplazos por árboles para las semillas disponibles. No comparar medias agregadas de subconjuntos distintos como si ya fueran el contraste de diez semillas pareadas. Tampoco usar diez episodios idénticos por agente como diez muestras independientes para significancia estadística. El análisis final debe esperar las semillas faltantes y conservar la frecuencia de truncación.

## Validaciones y trabajo pendiente

Se comprobaron los hashes internos de estados de entrenamiento, estados de evaluación e identidades descargados, y la vinculación del estado a su identidad. No se descargaron blobs ni logs completos para una nueva auditoría local de las cadenas. Los jobs de reporte en compute siguen pendientes de sus arrays y deben verificar las fronteras completas. No se ejecutó test reservado ni se autorizaron 2M en la nueva tanda.

La preparación y los envíos son correctos operativamente hasta lo observado. Falta terminar los 18 modelos restantes, cerrar los cuatro reportes de 1M y comprobar que sus cuarenta identidades corresponden exactamente a los diez seeds solicitados por algoritmo. Mantener el objetivo activo; 22/40 no es el experimento completo.
