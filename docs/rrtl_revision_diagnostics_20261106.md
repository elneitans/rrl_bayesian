# Poda, recrecimiento y reemplazo local: campaña diagnóstica

Solicitud: implementar las tres operaciones, evaluar políticas congeladas desde situaciones realmente diferentes, observar continuaciones cortas y auditar la señal del test; revisar los resultados. Esta campaña es un experimento nuevo con padres históricos verificados. No modifica sus protocolos, checkpoints, evaluaciones ni resultados.

## Estado final, revisado el 8 de noviembre de 2026

La campaña está completa: 30 continuaciones de 100k decisiones, 3.000.000 de transiciones, 324 rollouts congelados y 900 evaluaciones de continuación. Los seis grupos de entrenamiento/evaluación y los jobs 36905/36919/36931 terminaron COMPLETED con código cero. La preparación pasó 107 pruebas Linux. Los seis controles reproducen exactamente 600k transiciones originales.

Se verificaron 32 podas, 16 reemplazos y 78 crecimientos. La auditoría reproduce las pérdidas de las 48 revisiones aceptadas y confirma prefijos de aprendizaje idénticos hasta sus 14 primeras intervenciones. Hay seis crecimientos en las propias hojas creadas por podas, cinco con la relación que se había eliminado. Los tres archivos principales de la implementación y su configuración locales coinciden con la campaña ejecutada.

Resultados e interpretación: [informe final](../analysis/revision_20261106/informe.md). Tablas, figuras y pruebas de integridad: `analysis/revision_20261106/final/`. Los modelos y holdouts completos permanecen en `~/archive/rrl_revision_20261106_02/campaign/`; sus árboles JSON y trazas están descargados.

El criterio de pérdida no garantiza conservar la política. Hay beneficios en casos concretos y deterioros importantes; no se selecciona una modalidad ganadora general con estos casos diagnósticos. Las anotaciones parciales de la bitácora siguiente corresponden al desarrollo y no describen el estado actual.

## Implementación y criterio

`bayesian_rrtl/tree_revision.py` añade `RevisionAgent` como fork explícito del baseline. Las modalidades son:

- `normal`: crecimiento y TD originales; control de equivalencia comprobado para histórico y ANOVA.
- `frozen`: estructura fija, TD activo.
- `prune`: TD y poda; ningún crecimiento posterior.
- `prune_regrow`: TD, poda y posterior crecimiento con las candidatas restauradas; espera mínima tras cambiar una rama.
- `replace`: TD y crecimiento, más reconstrucción local alternativa con una relación raíz distinta. No es un ajuste de los cuatro árboles completos.

La poda compara el subárbol con una hoja ajustada a targets Bellman recientes. El reemplazo propone un subárbol de profundidad local máxima2, ajustado por reducción de SSE sobre los mismos targets; no instala estadísticas de targets como si fueran estadísticas históricas de Q. Los nodos nuevos comienzan con estadísticas vacías de su criterio original. Las ramas sin datos heredan Q; la nueva hoja tiene Q reciente, nunca el Q antiguo almacenado del padre.

Las propuestas se ajustan exclusivamente con un buffer reciente del entrenamiento. Se revisan hasta4 candidatas, seleccionadas por criterios de entrenamiento, y se acepta como máximo un cambio cada25k. Los targets de ajuste y validación utilizan la misma referencia Q congelada en ese mantenimiento, con los mismos terminales/truncaciones del baseline. El conjunto de validación proviene de8 sondas nuevas de128 decisiones en otro emulador, con comportamiento mixto independiente del modelo y streams nuevos por revisión. Se utiliza un banco adicional de6 arranques que no comparte hashes RAM con el banco de evaluación. Nunca entra al TD ni al buffer del agente. Se guardan tanto esos datos como los pesos de referencia para reproducir el criterio.

Se calcula diferencia de MSE candidata−actual por bloque de sonda. La poda exige cota bootstrap unilateral95% menor de.0025; el reemplazo exige menor de−.001. Mínimos:128 ejemplos de ajuste,64 de validación y4 bloques con datos. Los valores son umbrales operativos de un piloto, no calibración confirmatoria. Dependencia, arranques compartidos y selección repetida impiden prometer un control nominal de falsos positivos. No se poda por pANOVA>.05. Una mejor MSE tampoco garantiza mejor política: el retorno se mide separadamente y no se usa test reservado.

El contador `n_splits` conserva el significado de divisiones actualmente presentes para validar los árboles originales. La variante guarda además crecimientos, podas y reemplazos acumulados, IDs persistentes de nodos y versiones de estructura. Los padres, profundidad, catálogo disponible, ramas lógicas/comparativas y fallback por ausencia se conservan. Las candidatas se recuperan respecto de los ancestros; ausencia nunca se convierte en False.

## Arranques y evaluación congelada

`bayesian_rrtl/revision_diagnostics.py` construye6 snapshots completos alcanzables mediante prefijos explícitos de acciones. Exige RAM realmente distinta y al menos4 codificaciones relacionales diferentes; guarda RAM hash, estado físico/relacional, objetos, historial del detector y prefijos. No basta cambiar el seed. Se verifica restauración y repetición exacta de la transición siguiente antes de correr.

Se evalúan los pesos existentes100k/500k/1M de semillas710603,710607,710610, con histórico y ANOVA, epsilon0/.01/.1 y los mismos6 arranques: **324 rollouts**. La política, Q, estadísticas, encoder y RNG del agente permanecen inmutables. El emulador y el RNG de evaluación son independientes del entrenamiento.

Se mide retorno bruto incremental desde el snapshot, duración, vidas, FIRE, decisiones sin pelota, truncación, estados codificados y hash de la trayectoria. Horizonte diagnóstico3000 decisiones. Estos retornos son continuaciones desde snapshots con distintas vidas/ladrillos/puntaje previo: **no equivalen a partidas completas ni sustituyen los resultados oficiales de1M**. Las diferencias entre métodos se comparan usando los mismos arranques y presupuesto.

## Continuaciones e instrumentación

Desde cada padre500k, las cinco modalidades reciben100k decisiones adicionales: **30 continuaciones y3M transiciones**. Son casos elegidos para estudiar una caída, una política buena y una meseta; no una nueva estimación poblacional de superioridad. Se conserva el estado completo del emulador, Q y los streams originales de exploración/episodios. La instrumentación y las evaluaciones utilizan streams separados.

Evaluación greedy cada25k, incluido el punto inicial, con los6 snapshots: **900 rollouts** adicionales. Se guardan pesos cada5k y trazas comprimidas de todas las transiciones, con Q antes/después de las cuatro acciones, margen greedy, hoja/ruta/versión, relaciones codificadas, TD y objetos posteriores. Se registran todas las divisiones y revisiones, incluidas propuestas rechazadas, soporte, efectos y ambos criterios.

Para cada candidata y hoja se acumulan separadamente momentos por categoría de Q actualizado, target TD y error TD. Las auditorías por ventanas5k muestran ambos p nominales, efecto entre categorías, soporte y ausencias. No se confunden con la estadística histórica acumulada desde que nació la hoja. Permite observar desacuerdos entre señales sin cambiar simultáneamente la señal real usada por el baseline.

## Archivos, integridad y operación

- Runner: `experiments/diagnose_rrtl_revision.py`, etapas `prepare`, `frozen`, `continue`, `report`.
- Diseño: `configs/breakout_revision_diagnostics_20261106.json`.
- Pruebas: `tests/test_tree_revision.py` y `tests/test_revision_runner.py`, más adaptador/ANOVA/checkpoint existentes.
- Slurm47: pruebas Linux, smoke real de ambas reglas y cinco modalidades, luego preparación completa y verificación de historias de padres.
- Slurm48: worker por grupo criterio/semilla, o reporte.
- Lanzador49: DAG finito e idempotente con ledger; `--prepare-only` permite revisar la integración antes del envío principal. No consulta sacct desde nodos de cómputo.

Se verifica el binding y SHA de padres antes de cargar modelos propios, runtime numérico fijado y manifiesto exacto del entorno antes de restaurar. Es un fork diagnóstico con procedencia nueva, no una migración que eluda el protocolo original. El manifiesto nuevo fija todas las fuentes y el runtime de este experimento. Las identidades de evaluación incluyen checkpoint, banco de arranques, epsilon y horizonte. Checkpoints, trazas, auditorías y fronteras permiten reanudar sin duplicar transiciones. Unidades completadas se reutilizan por identidad; faltantes no se imputan.

El reporte comprueba modelos finales, continuidad de trazas, operaciones acumuladas, holdouts, identidades y todas las evaluaciones esperadas antes de declarar completitud. Si se agota tiempo o llega una señal, se publica la última frontera y el trabajo queda parcial; los pesos y logs permanecen reanudables. Las otras campañas y los trabajos de otros proyectos no se modifican.

## Bitácora de implementación y estados históricos

Validación local: 105 pruebas pasan, 2 casos Atari/OCAtari excluidos porque el runtime local no dispone de OCAtari; se ejecutan en Linux. Incluye equivalencia incremental, operaciones de estructura reales, independencia del holdout, Q reciente al podar, continuidad de trazas, reanudación y evaluación congelada sin mutar el encoder ni el agente.

Primer job de integración 36895 (`rrl_revision_code_20261106_v1`, control `_01`):106 pruebas Linux pasaron; falló antes de entrenar al comprobar RAM inmediatamente tras restaurar un snapshot. El job36898 aisló la diferencia: `getRAM()` conservaba la caché del reset aunque las relaciones ya estaban restauradas. La fuente oficial ALE0.10.2 confirma que `restoreState` carga Stella sin llamar a `processRAM`, mientras `getRAM` devuelve `m_ram`; esta caché se refresca al emular. Véanse [restoreState](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.10.2/src/ale/environment/stella_environment.cpp) y [getRAM](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.10.2/src/ale/environment/stella_environment.hpp).

La segunda versión compara el estado serializado completo de ALE tras restaurar, la memoria relacional y los objetos del detector. Además reproduce los prefijos registrados desde reset, comprueba su RAM/codificación original y exige igualdad de la siguiente transición y RAM con la continuación restaurada. No se avanza un paso oculto para “arreglar” el snapshot. No se modifica el backend ni su contrato de las campañas originales. También separa los bancos de arranques de validación/evaluación, y conserva por separado las semillas de entrenamiento y de evaluación en el reporte.

Archivo fuente v2 SHA256 `b525f27011bc2211da328eb0fdf1ccd436bd262c0038a5a772ce4f4fa644ed6e`; carpeta aislada `~/archive/rrl_revision_code_20261106_v2`; control `~/archive/rrl_revision_20261106_02`. Ninguna conclusión de eficacia corresponde todavía a esta campaña. Los jobs y hallazgos reales se registrarán aquí después de comprobar su estado y artefactos.

Preparación v2 enviada como **36899**. El job original36895 quedó FAILED antes del entrenamiento;36898 terminó el diagnóstico. No se enviaron dependientes del intento fallido. Los resultados de ambas integraciones se mantienen separados.

Envío principal, con dependencias de preparación exitosa:

| Etapa | Job | Dependencia |
|---|---:|---|
| Integración real y preparación |36899|—|
| Evaluación congelada,6 grupos,concurrencia1 |36903|afterok:36899|
| Continuaciones,6 grupos,concurrencia2 |36904|afterok:36899|
| Auditoría completa y reporte |36905|afterany:36903:36904|

Ledger remoto: `~/archive/rrl_revision_20261106_02/jobs.tsv`. Resultados principales: `campaign/`; logs: `logs/`. Se verificó que el lanzador reutilizó36899 y no duplicó su preparación. El envío no prueba que la campaña esté completa; el cierre debe comprobar todos los artefactos y faltantes.

Análisis preparado: `analysis/revision_20261106/build_report.py` verifica hashes/continuidad de las trazas y produce tablas, figuras y `resultados.md` a partir del reporte auditado. Excluye de promedios por episodio la partida ya en curso en el checkpoint padre; sí contabiliza todas las recompensas nuevas en la tasa por decisiones. El pipeline de análisis se ejecutó con un caso sintético de continuidad y render para comprobarlo, sin presentarlo como resultado Atari.

## Diseño de comparación antes de ver resultados

La comparación primaria diagnóstica es cada modalidad frente a `normal`, pareada por criterio, semilla y snapshot, al finalizar100k decisiones adicionales. `frozen` ayuda a separar derivaQ de crecimiento, `prune` elimina crecimiento nuevo y `prune_regrow` lo permite; no se interpretan como intervenciones idénticas. Se reportan curvas completas, no solo el mejor punto. Los tres casos seleccionados no justifican p-values poblacionales ni elegir retrospectivamente una variante como ganadora general.

El reporte distingue fracaso operativo, cambio rechazado por falta de soporte y ausencia de beneficio observado. Una modalidad que no acepta ninguna poda/revisión no se describe como una política efectivamente podada. También se informa cuántas operaciones ocurrieron y su efecto inmediato sobre el rankingQ; una menor pérdida sobre targets no se convierte automáticamente en mejora de retorno. Las primeras conclusiones requieren el reporte auditado completo, no un log que solo indique que terminó un worker.

Comprobación de dependencias vía `scontrol`:36903 y36904 esperan `afterok:36899`;36905 espera `afterany` de ambos arrays. Al revisar el piloto v2 se observaron cuatro podas y dos reemplazos en el histórico; en ANOVA también se observaron podas. Son operaciones de integración con umbrales reducidos, no conclusiones sobre eficacia ni resultados de la configuración principal.

Verificación adicional enviada como **36906** mediante `slurm/50_verify_revision_control.sbatch`: contrasta las1000 transiciones del control normal de cada criterio contra los logs originales500001–501000, incluidos acción, RNG observable, epsilon, TD, terminales, retornos y contadores. No entrena ni modifica datos originales. Las dependencias de36903/36904 se amplían a `afterok:36899:36906`; el reporte conserva su dependencia de ambos arrays. Esta comprobación identifica cambios de comportamiento introducidos accidentalmente por la instrumentación, además de las pruebas sintéticas.

Piloto real v2 cerrado:107 pruebas Linux,10 continuaciones de1000 decisiones (10.000 transiciones),4 evaluaciones congeladas, sin unidades faltantes. La copia local está en `results_cluster/rrtl_revision_20261106_02/smoke/`; su análisis verificado en `analysis/revision_20261106/smoke/`. Las recompensas de horizonte128 de este smoke no sirven para concluir eficacia. La campaña principal conserva100k adicionales por agente, horizonte3000 y mínimos originales declarados.

Estado observado tras las validaciones:36899 COMPLETED0 en13m47;36906 COMPLETED0 en32s. El control normal coincide exactamente con las1000 transiciones originales de cada criterio, incluidos acciones, TD y recompensas. Arrancaron36903_0,36904_0 y36904_1; faltan las corridas principales y su análisis. El objetivo global permanece en curso.

Seguimiento parcial: la concurrencia máxima del array36904 se amplió de2 a4 para utilizar los cupos que permita Slurm, sin ampliar semillas, decisiones ni recursos por tarea. El límite global de jobs y los trabajos de otros proyectos permanecen bajo el control del scheduler. Se descargaron/verificaron las primeras162 unidades congeladas; cada política produce seis trayectorias distintas en esos arranques. Los primeros controles históricos muestran ventanas donde la puntuación nominal sobre targets y la deQ actualizado discrepan; las ventanas no son muestras independientes y no prueban utilidad causal de la división.

Análisis final encadenado como **36919**, dependencia `afterok:36905`, mediante `slurm/51_analyze_rrtl_revision.sbatch`. Produce tablas/figuras en el runtime Linux fijado y verifica adicionalmente las100k transiciones de los seis controles contra sus continuaciones originales (600k en total). Sus fuentes están en `control/analysis_code`, fuera del código congelado; `analysis/completion.json` conservará sus hashes y los del reporte. Las figuras no son necesarias para validar las trazas: los analizadores admiten `--no-plots` para separar análisis numérico de fallos del entorno gráfico local.

Descarga durante ejecución: `analysis/revision_20261106/download.sh` fija primero `status.json` y luego copia sus dependencias inmutables, sin sobreescribir esas fronteras en la segunda pasada. Evita analizar un estado más nuevo que las trazas descargadas. `partial_continuations.py` y `partial_frozen.py` distinguen observaciones parciales del cierre completo; `audit_signals.py` compara señales pareadas en la misma ventana/candidata.

Primer contraste denso observado: en histórico710603, el control normal obtiene65,33→56,5→2,17→41,33→51,5 desde los seis snapshots comunes; con estructura fija15 divisiones obtiene65,33→56,5→1→85,83→1,33. Son puntos0/25k/50k/75k/100k adicionales. Por tanto, el crecimiento no es condición necesaria para estos retrocesos: Q sigue cambiando aun sin divisiones. Parte de los retornos del control normal son limitados por horizonte y muestran largas secuencias sin pelota/FIRE escaso; el bajo retorno final de estructura fija termina naturalmente y sí usaFIRE. No atribuir ambos fallos al mismo mecanismo ni declarar eficacia global de la poda con esta observación parcial. Los tres controles históricos completos coinciden exactamente con las continuaciones originales100k cada uno; los controles ANOVA y la comparación completa siguen pendientes.

Observación parcial posterior:915.000 transiciones descargadas/verificadas,8 fronteras de continuación reportadas completas de30 y162 evaluaciones congeladas de324. Se añadieron las primeras podas de la configuración principal; el análisis final36919 espera el cierre36905. Estas cifras son una captura de progreso, no el estado vivo ni una declaración de completitud.

Primer resultado adverso de poda pura: histórico710607 termina la continuación con media63,33 en `normal` y63,33 en estructura fija, frente2,5 con poda, sobre los mismos seis snapshots. El control no añadió divisiones. Esto demuestra en ese caso que aceptar la poda con el criterio de pérdida no garantizó preservar retorno; no se cambia el umbral después de verlo. La atribución a una rama, falta de cobertura o cambio de ranking requiere el análisis de eventos y árboles; aún no se declara un mecanismo causal específico ni una conclusión para todos los casos.

Las324 unidades congeladas ya fueron descargadas y sus identidades JSON verificadas; el cierre completo aún requiere el reporte de modelos y las30 continuaciones. En histórico710603 se observó recrecimiento real: tres crecimientos y cuatro podas al completar100k adicionales. El hecho de que una modalidad permita recrecimiento no implica que ocurra en todas las semillas.

Auditoría adicional de solo lectura: `analysis/revision_20261106/audit_revision_effects.py`, encadenada como36931 mediante `slurm/52_audit_revision_policy.sbatch`, después del análisis36919. Para cada operación aceptada compara las raíces guardadas inmediatamente antes del mantenimiento con el checkpoint inmediatamente posterior, sin emular ni actualizarQ. Reproduce las pérdidas de validación y la diferencia media por bloques del gate; mide cambios del conjunto greedy, distribución uniforme de empates, margenQ y acciónFIRE. Conserva los árboles antes/después. Esta auditoría utiliza las sondas del propio gate: no es una segunda validación independiente ni una estimación de regret real. No añade decisiones de entrenamiento ni selecciona otra política.
