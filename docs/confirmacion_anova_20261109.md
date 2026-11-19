# Confirmación prospectiva: baseline, TD lento y reestimación ANOVA

**Campaña cerrada el 10 de noviembre de 2026.** Terminaron las cinco semillas y las quince continuaciones, con evaluaciones y auditorías completas. Resultados y límites: [informe final](../analysis/confirmation_20261109/informe.md). Las secciones de diseño y envío conservan lo fijado antes de observar los resultados.

La evaluación anterior terminó: ninguna variante mejora de manera consistente ambos criterios. TD lento muestra una señal favorable sencilla; reestimación obtiene la mayor media ANOVA, con fuerte influencia de una semilla. Esta nueva campaña busca comprobar esas hipótesis con agentes nuevos y seguimiento más largo, sin reutilizar como confirmación los resultados que motivaron el diseño.

## Diseño fijado antes de ejecutar

- Solo Breakout/OCAtari y crecimiento ANOVA. Misma representación, recompensa sign, descuento .99, exploración y criterio de crecimiento.
- Cinco semillas de entrenamiento nuevas: **721101–721105**. No son los agentes originales ni las semillas de test reservado.
- Entrenar cinco baselines desde cero hasta 1M, conservando el checkpoint de 500k y la historia completa. La mitad posterior sirve además de control reproducible.
- Desde cada padre de 500k, crear tres continuaciones independientes hasta 1M: baseline, TD lento (.005) y reestimación sin poda (.025). No se combinan Double ni poda.
- Quince continuaciones de 500k: **7.5M decisiones nuevas**, más 5M de entrenamiento de padres. La repetición de baseline es un control de instrumentación: debe reproducir exactamente las 2.5M transiciones originales correspondientes.
- Evaluación greedy sin aprendizaje cada 25k adicionales, incluidos inicio y final: 21 puntos, 12 snapshots compartidos y horizonte 3000; **3.780 rollouts**. El endpoint principal es el final fijado, no el máximo retrospectivo.
- Mantener el refit cada **25k**, el buffer 8192, soportes y márgenes anteriores. Así se evita confundir seguimiento largo con un cambio de frecuencia del mantenimiento.
- Guardar cada 25k, frente a cada 5k antes. Solo cambia el costo de almacenamiento y el máximo trabajo perdido al interrumpir, no las actualizaciones ni evaluaciones.
- Medir curvas, diferencias pareadas por semilla/arranque, medianas/rangos, caídas desde el padre y desde picos observados, y entrenamiento por ventanas. Informar las cinco semillas; no eliminar retrospectivamente extremos.

La evaluación desde snapshots conserva comparabilidad con el diagnóstico anterior y su limitación: situaciones tempranas relacionadas, no un benchmark completo. Este diseño confirma con nuevas semillas de entrenamiento; no convierte esos arranques en muestras independientes ni consume test reservado. Una evaluación adicional de generalización física requerirá otro protocolo explícito.

## Implementación y compatibilidad

`experiments/confirm_rrtl_anova.py` usa un esquema nuevo `rrtl_anova_confirmation_v1`, valida las tres variantes y las semillas antes de operar. Reutiliza TD, refit, snapshots, trazas y restauración existentes. Los runners anteriores mantienen por defecto su diseño de siete modalidades y ambos criterios; no aceptan silenciosamente la nueva configuración.

Los hooks explícitos de carga permiten usar el mismo reporte y análisis con el esquema nuevo. El reporte debe verificar 15 unidades completas, 7.5M transiciones, 3.780 evaluaciones y 2.5M controles idénticos a sus historias originales. La auditoría posterior debe reconstruir todas las reestimaciones aceptadas, conservando particiones, identidades y estadísticas durante cada mantenimiento; el crecimiento online continúa habilitado.

Configuraciones: `configs/breakout_confirmation_parents_20261109.json` y `configs/breakout_anova_confirmation_20261109.json`. Pruebas: `tests/test_confirmation_design.py`, además de las de reanudación, baseline y revisión existentes.

## Secuencia operativa y criterios para avanzar

1. Congelar las fuentes en una carpeta nueva. Las campañas anteriores se siguen reproduciendo con sus snapshots originales; no se migran ni se sobrescriben.
2. Ejecutar `slurm/62_confirmation_preflight.sbatch`: pruebas Linux y una medición real de 20.480 decisiones, entorno real y evaluación completa corta. La etiqueta baseline se aplica solo a esa medición limitada; la campaña de 1M conserva nivel long y su autorización con costos compatibles.
3. Revisar tiempos y almacenamiento proyectados, comprobar cuota efectiva en home y reservar margen antes del envío largo. `archive` estaba sin cuota; el espacio global del NAS no basta como comprobación. No borrar datos de otros proyectos.
4. Autorizar y ejecutar padres hasta 1M con concurrencia máxima inicial de dos, checkpoints y reanudación. Completar sus cinco historias antes de preparar los forks.
5. Preparar los bancos y ejecutar integración real de las tres variantes con restauración de emulador y equivalencia del control; después congelar el manifiesto principal.
6. Ejecutar quince continuaciones, reporte, análisis y auditoría independiente; descargar y comprobar los artefactos completos.

**Enviar la medición de costo no completa esta tarea.** El cierre requiere los resultados de las cinco semillas nuevas, todas las evaluaciones y auditorías, y un informe que compare conservación y progreso posterior. Si fallan recursos, se retoma el mismo checkpoint sin reducir el diseño ni retocar los parámetros después de ver retornos.

## Registro del envío del 9 de noviembre

El preflight38115 terminó COMPLETED0:0 en6m46s:26 pruebas Linux y20.480 decisiones reales con nueve evaluaciones. Las fuentes están congeladas en `~/projects/rrl_confirmation_code_20261109_v2`; control/datos en `~/projects/rrl_confirmation_20261109_01`. SHA del archivo fuente:`ccd1a98eb65c40d085d9d0fc7cdc31e7ca0534819296dbe0ce2e7f4bdc602f8a`.

DAG enviado: padres38120(array0–4%2), integración38121, preparación38122, continuaciones38123(array0–4%2), reporte38124, análisis38127 y auditoría38128. Los batches operativos63–65 se copiaron al directorio de control para serializar únicamente la creación del protocolo y agregar la integración; sus hashes se registran en `OPERATIONS_SHA256`, separados del snapshot científico.

La proyección observada es4017s de entrenamiento y419MB de logs por padre a1M; no es una cota superior. `du` registró22.86GB usados en home, frente a la cuota de50GB comunicada antes por el usuario; esa cuota no pudo consultarse de forma independiente. Las escrituras y el preflight real pasaron. Se reserva un margen de planificación de12GB adicionales y se vigilará crecimiento; `archive` no se utiliza para resultados nuevos.

Estado completo y evidencia se mantienen en `analysis/confirmation_20261109/progress.json`. En el momento del envío la corrida larga seguía pendiente; su cierre posterior exige las verificaciones descritas abajo y se documenta en el informe final.

## Verificación preparada para el cierre

Los verificadores adicionales están fuera de las fuentes científicas congeladas y no alteran los entrenamientos. Su mera existencia no certifica que la campaña esté terminada:

- `analysis/confirmation_20261109/verify_parents.py`: ejecutar en cómputo con los blobs disponibles. Comprueba configuración y fuentes, las cinco historias completas de 1M, bytes de checkpoints en los hitos y las 55 evaluaciones de desarrollo previstas. No deserializa modelos ni vuelve a jugar.
- `analysis/confirmation_20261109/verify_download.py`: comprueba los estados, identidades y bancos descargados de las quince continuaciones y reconstruye las 1.500 ventanas de entrenamiento a partir de las 7.5M transiciones. Requiere las fuentes congeladas en `PYTHONPATH`.
- `analysis/confirmation_20261109/verify_results.py`: contrasta manifiesto, reporte, análisis y auditoría independiente; exige 3.780 evaluaciones y 2.5M transiciones de control idénticas antes de generar el resumen por método y semilla.

El cierre debe conservar también los hashes de los archivos descargados y los resultados de estos verificadores, junto con el informe científico de las cinco semillas completas.

La auditoría de padres se envió como job **38132**, dependiente de la finalización correcta del array38120. Usa `slurm/66_confirmation_verify_parents.sbatch` y copias de los verificadores en el directorio operativo, con hashes en `PARENT_AUDIT_SHA256`. El resto del DAG permanece igual.

## Cierre observado

Todos los jobs científicos terminaron `COMPLETED0:0`. Se verificaron 5M transiciones de padres y55 evaluaciones de desarrollo; 7.5M transiciones de continuaciones,3.780 rollouts y2.5M transiciones exactas de control; y93 reestimaciones aceptadas reconstruidas independientemente. La descarga completa de trazas y metadatos se comprobó localmente y las figuras finales se revisaron visualmente.

Las medias finales desde snapshots fueron16.65 para baseline,32.23 para TD lento y27.10 para reestimación. Ambas intervenciones ganan tres semillas y pierden dos; ninguna establece estabilidad sostenida ni superioridad general. La evaluación reservada no se ejecutó. Los detalles y el certificado están en `analysis/confirmation_20261109/`.
