# Revisión de las corridas descargadas el 27 de octubre de 2026

Fuente: estados JSON, evaluaciones por episodio, reportes y logs Slurm descargados en `results_cluster/revision_20261027/`. Consulta remota de `squeue` vacía. No se enviaron nuevos jobs ni se modificaron campañas durante esta revisión.

## Estado operativo

| Campaña | Entrenamiento | Evaluación de desarrollo |
|---|---|---|
| Breakout OCAtari baseline | 10/10 semillas 710601–710610 alcanzaron 500k | 100/100 episodios completos a 500k |
| Breakout OCAtari CART | 10/10 semillas 710601–710610 alcanzaron 500k | 90/100 episodios completos; faltan los diez del modelo 710609 |
| Pong OCAtari CART piloto | 1/1 semilla 501601 alcanzó 100k | 10/10 episodios completos a 100k |
| Pong OCAtari baseline, 12 adicionales | 5/12 alcanzaron 2M; una parcial; seis sin estado publicado | El job de evaluación no pudo arrancar |

Breakout baseline devuelve código 3 porque llega al hito autorizado de 500k en una campaña cuyo máximo sigue siendo 2M y declara quince semillas. Los tres grupos 34628 terminaron en ese hito; no agotaron el walltime. Los reportes globales parciales también incluyen evaluaciones no programadas por esta tanda. No son una medida suficiente del cumplimiento del subconjunto solicitado.

Breakout CART terminó sus entrenamientos 34631 sin error. El grupo de evaluación 34632_2 devolvió 3 porque los diez episodios del modelo 710609 agotaron el presupuesto de 600 segundos por episodio. Sus estados son `administrative_partial`, con ~13–14k decisiones en los ejemplos inspeccionados; no alcanzaron el horizonte de 27000 ni el final natural del juego. Su retorno parcial de 1 no puede usarse como resultado final ni sustituirse por cero.

Pong baseline, array 34619:

| Semilla | Último contador publicado | Estado |
|---|---:|---|
| 501604 | 2.000.000 | Completado |
| 501605 | 2.000.000 | Completado |
| 501606 | 2.000.000 | Completado |
| 501607 | 2.000.000 | Completado |
| 501608 | 2.000.000 | Completado |
| 501609 | 672.768 | Parcial; job terminó con código 1 |
| 501610–501615 | Sin estado publicado | Jobs fallaron al arrancar con señal 53 |

Los jobs de evaluación y reporte 34620/34621 también fallaron al inicio con señal 53. No es un timeout: los cinco primeros entrenamientos finalizaron en 2h17–4h21 y el sexto falló a 1h23. El log del sexto contiene solo la cabecera y ALE, sin traceback. El patrón es compatible con el problema previo de cuota, pero esta revisión no obtuvo una nueva lectura de la cuota individual: no se afirma como causa demostrada. `du -sh results` remoto devolvió 26G, frente a los 6,9G observados anteriormente. La RAM registrada fue de unos 190–196 MiB para estos entrenamientos, lejos de los 20 GiB solicitados.

## Breakout: comparación a 500k

Cada valor es la media del retorno bruto en diez semillas de evaluación de desarrollo, no recompensa de entrenamiento.

| Semilla de entrenamiento | Baseline | CART |
|---|---:|---:|
| 710601 | 43 | 23 |
| 710602 | 1 | 7 |
| 710603 | 59 | 2 |
| 710604 | 3 | 30,4 |
| 710605 | 3 | 0 |
| 710606 | 2 | 3 |
| 710607 | 3 | 24 |
| 710608 | 3 | 11 |
| 710609 | 31 | Pendiente |
| 710610 | 4 | 9 |

Baseline: media entre diez modelos 15,2, mediana 3, desviación estándar muestral 21,18. CART: media provisional entre nueve modelos completos 12,16, mediana 9, desviación estándar muestral 10,97. No comparar estas medias como resultado definitivo: falta un modelo CART y los subconjuntos difieren.

En los nueve pares completos, CART supera al baseline en seis, pero la diferencia media CART−baseline es −1,29 puntos, debido a las grandes diferencias de algunas semillas. Esta descripción no demuestra superioridad de ninguno. No se hicieron pruebas de significancia con el modelo faltante. La unidad principal de replicación es la semilla de entrenamiento; los episodios de un mismo modelo no son diez entrenamientos independientes.

Los cien episodios baseline y los noventa CART completados terminaron naturalmente, sin truncación por horizonte. En baseline los diez retornos de evaluación fueron idénticos para cada modelo. En CART siete de los nueve modelos completos también tuvieron retornos idénticos; los modelos 710604 y 710610 variaron. Retornos repetidos no prueban por sí solos un fallo de semillas, pero hay poca diversidad observable y conviene auditar el reinicio y aleatoriedad del entorno antes de presentar intervalos como si cada episodio aportara una réplica independiente.

Sí hay políticas que consiguen decenas de puntos en Breakout; eso contrasta con los resultados débiles anteriores. No basta para demostrar una curva de mejora general: esta tanda evaluó el hito de 500k, y faltan controles y una comparación homogénea entre hitos para las diez semillas.

Coste aproximado de entrenamiento, sumando los segundos registrados en intentos por modelo: baseline 57,9 minutos de media por semilla y CART 80,9 minutos. CART costó ~1,40 veces en esta muestra. Baseline incluye el intento previo a 100k de la primera semilla. Los grupos Slurm duraron 2h03–4h14 para baseline y 2h34–5h49 para CART; cada grupo contiene varias semillas y no debe confundirse con el tiempo por modelo.

## Pong CART

El piloto 34624 terminó a 100k en 15m37; la evaluación 34625 tardó 11m12. Las diez evaluaciones finales dieron −21, terminaron naturalmente y duraron 764 decisiones cada una. No hay señal de una política útil a ese presupuesto. No demuestra que CART nunca aprenda; tampoco justifica extrapolar el resultado positivo de algunas semillas de Breakout a Pong. No se ejecutó una nueva corrida larga de BART en esta tanda.

## Qué falta y orden sugerido

1. Revisar cuota individual y resolver almacenamiento antes de reenviar Pong. El crecimiento de `results` hace preferible trasladar resultados a archive con verificación y conservar rutas mediante enlace, o aplicar la limpieza soportada por el pipeline después de inventariar artefactos. No borrar checkpoints manualmente ni alterar manifests.
2. Reanudar solo Pong baseline 501609 desde su último checkpoint validado e iniciar 501610–501615. Las cinco semillas adicionales completas se conservan. Después evaluar las doce adicionales y regenerar el reporte. Sumadas a las tres originales completas, hay ocho entrenamientos a 2M, una corrida adicional parcial y seis pendientes; no quince completados.
3. Completar la evaluación de Breakout CART 710609 a 500k mediante una extensión operativa explícita, registrada, y reanudar los estados por episodio; no reentrenar ni modificar el protocolo congelado para cambiar el presupuesto. Después cerrar la tabla con diez pares.
4. Exportar un resumen del subconjunto de diez semillas a 500k: el reporte global del baseline seguirá incluyendo otras cinco semillas y el máximo de 2M. Completar las evaluaciones de hitos anteriores si se desea una curva comparable, porque esta tanda programó específicamente la evaluación final de desarrollo a 500k.
5. Revisar políticas y trazas de semillas buenas/malas de Breakout y decidir la ampliación de Pong CART con evidencia de coste y controles. Los resultados no son una comparación BART contra baseline.

Se verificaron sin discrepancias los hashes internos de 20.552 JSON envueltos disponibles en las dos campañas Breakout, además de los estados/evaluaciones Pong descargados prioritariamente. Esto valida integridad interna de esos JSON, no los blobs de checkpoint ni las cadenas JSONL de transiciones, que se excluyeron de esta descarga ligera. La próxima reanudación debe mantener las verificaciones del cargador. No se ejecutó el test reservado.
