# Breakout 1M: avance observado el 30 de octubre

**Consulta posterior:** dos intentos de SSH al mismo host/puerto terminaron con `Operation timed out`. No se obtuvo un estado nuevo de Slurm. Los contadores siguientes corresponden a la última descarga válida, no a esta consulta fallida. No se infirió finalización ni caída de jobs a partir del timeout, no se reenviaron workers y no se cambiaron sus dependencias. Reconsultar los IDs existentes al recuperar la conexión.

La preparación 35463 terminó correctamente (0:0, 16m40) y autorizó los cuatro protocolos exactamente hasta 1M. Las campañas conservan 2M como máximo futuro. No se ejecutó test reservado ni se cambiaron los módulos Python congelados.

## Estado de la tanda

| Método | Semillas entrenadas y evaluadas a 1M | Semilla activa en snapshot |
|---|---:|---|
| Baseline corrected_common | 3/10 | 710604: 356.864 |
| CART | 2/10 | 710603: 615.424 |
| ExtraTrees | 2/10 | 710603: 64.000 |
| BART | 0/10 | 710601: 508.928 |

Workers activos: 35464_1 (baseline), 35466_1 (CART), 35468_1 (ExtraTrees), 35470_0 (BART). Sus grupos posteriores permanecen pendientes por el límite de un elemento activo por array. Los grupos iniciales baseline35464_0, CART35466_0 y ExtraTrees35468_0 completaron con código 0 en 4h11m43, 5h11m18 y 6h40m20 respectivamente. Cada grupo incluye dos entrenamientos y sus evaluaciones, así que esos tiempos no son tiempos de entrenamiento de una sola semilla.

Los status a 1M se llaman `partial` con razón `authorized_milestone_reached`: es correcto porque el máximo congelado es 2M. Se confirmó que el worker reconoce el contador y sigue con evaluación/otra semilla sin entrenar automáticamente hasta 2M.

## Primeros resultados de desarrollo

Evaluación greedy del checkpoint de 1M, diez semillas de desarrollo por modelo, horizonte 27.000 decisiones:

| Semilla de entrenamiento | Baseline | CART | ExtraTrees |
|---|---:|---:|---:|
| 710601 | 101 | 0 | 0 |
| 710602 | 3 | 11 | 19 |
| 710603 | 25 | pendiente | pendiente |

Cada entrada resume diez episodios completados y terminados naturalmente; cero truncaciones en los setenta episodios disponibles. Dentro de cada modelo el retorno es idéntico en las diez semillas de evaluación. Por tanto, no tratar esos episodios como diez réplicas independientes de aprendizaje ni interpretar su dispersión nula como certeza sobre rendimiento general. El contraste principal debe hacerse entre semillas de entrenamiento y esperar los diez modelos de cada método.

La semilla baseline710601 muestra una política capaz de obtener 101 puntos en esta evaluación. Eso es evidencia positiva sobre ese agente concreto, no una demostración de aprendizaje consistente, superioridad global ni mejora temporal sin comparar sus checkpoints anteriores. Baseline710602 obtiene solo tres puntos. CART y ExtraTrees también muestran variación fuerte. BART aún no tiene evaluación a 1M; compararlo ahora con los otros a ese presupuesto sería prematuro.

## Evidencia y límites

Se descargaron estados, manifests y evaluaciones en `results_cluster/breakout_1m_20261030/`. Se comprobaron hashes internos de los once status de entrenamiento y de los setenta estados e identidades de evaluación, además de la referencia del estado a su identidad. No se descargaron blobs de checkpoint ni logs completos: la revisión local no constituye una nueva validación integral de sus cadenas. Los workers reales y el reporte `consolidate` realizan esa validación en compute; los tres grupos completados son evidencia operativa adicional.

El objetivo permanece incompleto: faltan 33 de los 40 modelos a 1M evaluados. No se enviaron duplicados mientras los cuatro workers siguen vivos. Si un grupo acaba parcial por las 24h, habrá que reanudar ese grupo con sus mismos checkpoints; no confundir el cierre de un job con el cierre del experimento.
