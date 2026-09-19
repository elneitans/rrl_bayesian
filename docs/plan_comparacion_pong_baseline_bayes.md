# Plan para developer: Pong bayesiano vs baseline, visual y OCAtari

## 0. Encargo y alcance de esta entrega

Trabajar en `capstone/rrl_bayesian/`. Las rutas de este documento son relativas
a esa raíz. Implementar y validar la infraestructura necesaria para comparar
BART con el baseline corregido en dos pipelines de Pong: visual legacy y
OCAtari RAM. Dejar preparado un piloto pareado de 12 ejecuciones, pero NO
lanzarlo automáticamente. No ejecutar test reservado ni campaña larga.

Entrega autorizada por este plan: código, pruebas, smoke pareado acotado,
auditoría diagnóstica offline acotada, protocolo preparado e informe de readiness.
Al terminar, presentar costo estimado y limitaciones para que el usuario autorice
el piloto. Si aparecen cambios necesarios al sampler, priors o F-test, proponerlos
por separado; no introducirlos para obtener resultados favorables.

No editar los módulos históricos `relational_regresion_tree.py`,
`preprocessing.py`, `train_and_test.py` ni resultados previos. Preservar H0–H7,
la CLI legacy y checkpoints existentes. No añadir nuevos juegos en esta entrega.

## 1. Punto de partida verificado

- BART + OCAtari Pong funciona: 2.048 interacciones, 8 fits, 217 estados distintos,
  cobertura de seis acciones y reanudación exacta entre procesos.
- La suite revisada completó 265 pruebas. Repetirla al inicio para registrar la
  línea base actual, no asumir que el worktree no cambió.
- De seis acciones, solo LEFTFIRE cumplió todos los umbrales MCMC comprobados.
  La integración está validada; la calidad posterior global no.
- `CorrectedRRLAgent` en `baseline.py` conserva TD incremental y F-test, pero su
  entrenamiento/evaluación construyen un entorno Gym visual y un preprocesador.
- `train_atari.run_corrected` rechaza OCAtari. Ese rechazo evita un fallback
  engañoso y debe permanecer hasta disponer de una ruta común real y probada.
- `campaign_h7.settings` fija explícitamente backend visual. No basta con cambiar
  el nombre del juego para obtener una campaña OCAtari.
- `Node` utiliza valores lógicos string (`'True'/'False'`), mientras
  `PongRelationsV1` emite booleanos. El encoder bayesiano sí admite ambos.
- El catálogo histórico Pong de `RRLAgent.get_all_relations_pong` omite la
  trayectoria Y del jugador y no incluye las nuevas presencias. No reutilizarlo
  sin adaptación para el esquema OCAtari de 20 claves.

## 2. Preguntas experimentales y cuatro condiciones

### Comparaciones primarias de esta entrega

| ID | Pipeline de entrada | Aprendizaje |
|---|---|---|
| `visual/corrected_common` | Visual legacy + relaciones legacy | TD/F-test corregido adaptado al entorno común |
| `visual/bart` | Exactamente el mismo pipeline visual | BART |
| `ocatari/corrected_common` | OCAtari RAM + PongRelationsV1 | TD/F-test corregido adaptado al entorno común |
| `ocatari/bart` | Exactamente el mismo pipeline OCAtari | BART |

Los contrastes primarios son `bart - corrected_common` DENTRO de cada pipeline.
No promediar ambos pipelines ni anunciar que la diferencia entre ellos mide
aisladamente el efecto de RAM. Además de extracción, actualmente cambian las
relaciones, el manejo de estados incompletos, el reset y la recompensa.

`corrected_common` debe identificarse en manifiestos como
`rrtl_corrected_common_v1`: conserva el algoritmo de actualización/split,
pero hace explícitos catálogo, adaptación de hechos, entorno y desempate. No
presentarlo como reproducción bit a bit del runner H0. `rrtl_corrected` histórico
sigue existiendo y sirve como referencia de regresión, no como sustituto
silencioso de esta condición.

### Condiciones ambientales cerradas

| Parámetro | visual | ocatari |
|---|---|---|
| backend | `legacy_visual_pong_v1` | `ocatari_ram_v1` |
| env_id | `PongNoFrameskip-v4` | `ALE/Pong-v5` |
| esquema | `legacy_pong_v1` | `pong_relational_v1` |
| representación | comparative | comparative |
| incompletos | false, conserva configuración original | true, contrato OCAtari |
| reset | legacy_fire | none |
| recompensa entrenamiento | sign(raw)+0.1 | sign(raw) |
| cotas recompensa | [-0.9,1.1] | [-1,1] |
| retorno evaluación | suma raw, sin bonus | suma raw |
| frameskip / sticky | 4 / 0 | 4 / 0 |
| acciones mínimas | NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE | mismas |

Registrar estas diferencias, no ocultarlas normalizando los gráficos. Igualar
algoritmos dentro de cada pareja no requiere cambiar las rutas históricas.

**Estudio posterior opcional, fuera de esta entrega:** visual y RAM alimentan
el MISMO `PongRelationsV1`, mismo ID/parámetros ALE, recompensa y reset. Eso
permitiría estudiar extracción. Exigiría una condición nueva versionada; no
reinterpretar `visual` ni mezclarla con estos resultados.

## 3. Arquitectura y archivos

Reutilizar `AtariRelationalEnvironment`, registro, `ObjectFrame`, relaciones,
snapshot v1/v2, encoder y `QTrainingSession` bayesiano. El baseline debe consumir
hechos directamente, sin volver a extraerlos ni crear un segundo emulador.

| Archivo propuesto | Entregable |
|---|---|
| `bayesian_rrtl/baseline_adapter.py` | Adaptación de hechos, catálogo y learner TD/F-test sin emulación |
| `bayesian_rrtl/baseline_training.py` | Sesión relacional incremental, logs y continuidad de episodio |
| `bayesian_rrtl/baseline_checkpoint.py` | Formato nuevo versionado para baseline común y recolector |
| `bayesian_rrtl/comparison.py` | Validación de protocolo y factories de entorno/learner/sesión |
| `experiments/compare_pong.py` | CLI de campaña pareada, train/resume/report/test separado |
| `experiments/audit_pong_readiness.py` | Costos, capacidad de split y diagnósticos offline |
| `configs/comparison_pong_smoke.json` | Cuatro celdas, una semilla, presupuesto pequeño |
| `configs/comparison_pong_pilot.json` | Cuatro celdas, tres semillas, 20.480 interacciones |
| `tests/test_baseline_adapter.py` | Equivalencia del learner, hechos, literales, terminales |
| `tests/test_comparison_protocol.py` | Validación, congelado, semillas, lifecycle y reportes |
| `tests/test_pong_comparison_pipeline.py` | ALE real, ambas entradas, CLI, resume y smoke |
| `docs/pong_comparison.md` | Uso, decisiones, resultados acotados y límites |
| `docs/validation_pong_comparison.json` | Evidencia nueva medida, por criterio |

Nombres equivalentes son aceptables si mantienen separación. No forzar al baseline
a heredar `BayesianQAgent`, simular posteriors ni llenar campos ficticios de BART.
Una interfaz de sesión común pequeña basta: `run(n)`, `interactions`, `evaluate`,
`save/load`, `metrics`. Mantener el loop H5 bayesiano intacto; probar ambos loops
con la misma traza de transiciones.

## 4. Implementación en orden

### A. Baseline relacional independiente del entorno

1. Crear un adaptador que reutilice `CorrectedRRLAgent.update_transition` y la
   lógica existente de Node/CorrectedNode; preferir composición/delegación y
   factories controladas a copiar y mantener una segunda implementación del F-test.
2. Construir acciones desde la especificación real del entorno. Construir
   `Literal(name,type,obj1,obj2,obj3)` desde el catálogo estable del encoder,
   conservando orden determinista. Guardar hash y lista exacta en checkpoint.
   No inferir literales de test ni crearlos dinámicamente según lo observado.
3. Para la equivalencia histórica, permitir pasar explícitamente el catálogo
   histórico y su orden. Para las cuatro condiciones principales usar el catálogo
   completo publicado por su entorno; registrar el cambio de soporte respecto
   del catálogo restringido H0. No confundir igualdad de hechos con igualdad de
   hipótesis: BART usa splits binarios; baseline usa sus ramas históricas.
4. Normalizar hechos SOLO en la frontera del baseline: bool/np.bool_ a strings
   `True/False`; conservar comparativos, nombres y objetos. No mutar el estado
   original ni convertir un hecho ausente a False. Comparar igualdad semántica
   canónica, no identidad de objetos Python entre los dos learners.
5. Preservar orden TD→estadísticas→split, herencia de Q, criterio y guardia de
   varianza nula. No sustituir F-test por CART/SSE ni modificar su fórmula.
6. Preservar el fallback histórico ante literal ausente, documentarlo y medir
   cuántas consultas terminan en nodo interno. No inventar una rama missing
   dentro de este control. Probar estados vacíos y presencias sin comparativos.
7. El baseline no debe aplicar shaping, reset, extracción, frame skip ni otra
   actualización TD adicional. Se le entrega reward ya transformada una vez.
8. Interfaz mínima: predicción Q ordenada por acción, selección, observe y estado
   serializable. Rechazar explícitamente requests de posterior/draws.

**Puerta A:** sobre una secuencia fija de hechos/transiciones con catálogo
histórico, el update adaptado coincide con H0 corregido en Q, target, contador,
estadísticas y splits. Incluir una señal sintética que efectivamente produzca
splits; no verificar equivalencia solo mientras todos los árboles son raíces.
Con catálogo OCAtari, pruebas de True/False/ausente, candidatos de presencia y
trayectoria del jugador. Si se detecta un defecto del algoritmo histórico que
impide estas pruebas, reportarlo y detener esa puerta, no cambiar el baseline
silenciosamente. Las diferencias ya declaradas de catálogo/política no son bugs.

### B. Política, recolección y evaluación comunes en semántica

1. Usar una fábrica de entornos desde la configuración completa. Baseline y BART
   no comparten una instancia simultáneamente, pero sí el mismo contrato.
2. Igualar epsilon como función del número de decisiones: inicial 1, mínimo 0.1,
   decay_steps=50.000. Una actualización por decisión, no por sweep ni evaluación.
   Usar la fórmula cerrada actual de BART para evitar deriva multiplicativa.
3. Desempate greedy uniforme entre máximos con RNG explícito. Reproducir en el
   adaptador la política epsilon-greedy usada por BART; no exigir trayectorias
   iguales después de que los Q diverjan. Declarar esta diferencia respecto del
   desempate de primera acción del runner H0. Remuestreo de repetidas=false.
4. Separar RNG de entorno y política. Usar el mismo esquema de child seeds H5
   para esos dos streams; el baseline no necesita consumir los RNG MCMC/replay.
   Excluir validación/test del generador de semillas de episodios.
5. Un punto no es terminal. Terminal real no consulta next Q. Truncación hace
   bootstrap con observación final si está configurado. Corte de presupuesto
   preserva episodio y no sintetiza terminal ni fuerza fit bayesiano.
6. Evaluar copias/modelos congelados, epsilon=0, entornos nuevos, recompensa raw.
   No modificar árboles, estadísticas, replay, contador o RNG de entrenamiento.
7. Emitir logs comunes por transición y episodio: IDs, seed episodio, acción,
   reward/raw, flags, epsilon y retornos parciales/completos diferenciados.
   Guardar métricas específicas (TD/splits o fits/MCMC) sin equipararlas.
8. No agregar el retorno parcial de entrenamiento a la media de episodios
   terminados. En evaluación, reportar retornos limitados por horizonte junto a
   fracción de truncación; no etiquetarlos como partidas completas si no lo son.

**Puerta B:** reproducir una traza ALE con acciones prefijadas en dos entornos
del mismo backend; ambos learners reciben estados semánticamente equivalentes,
reward/raw y flags iguales. Probar visual y OCAtari, con punto/truncación/reset.
Capturar antes de actualizar; no exigir igualdad online bajo políticas distintas.

### C. Checkpoint del baseline común y reanudación

1. No reutilizar el checkpoint H0 de evaluación como si tuviera estado del entorno.
   Crear formato identificado `rrtl_corrected_common_v1`, versionado, atómico,
   con checksum y uso solo de pickle local confiable si se necesita.
2. Guardar configuración resuelta/hash, catálogo/literales ordenados, raíces,
   estadísticas, contador de splits, RNG, epsilon derivable, semillas usadas,
   buffer de política si existe, historial/recolector y snapshot ambiental v1/v2.
3. La carga valida learner, versiones, acciones, catálogo, semántica de relaciones,
   reward/reset y coherencia de contadores. No reconstruir estadísticas vacías
   ni escoger otro extractor porque falte OCAtari. Validar antes de continuar.
4. Guardar entre actualizaciones completas. Mantener separadas las rutas de carga
   históricas H0/H5 y el formato nuevo. No cambiar hashes de archivos antiguos.
5. Evaluación final debe cargar el artefacto guardado, no solo usar el modelo
   en memoria que lo produjo.

**Puerta C:** pausa a mitad de episodio y continuación de al menos 128 decisiones
idénticas local/otro proceso, para visual y OCAtari. Igualdad de acciones,
transiciones, Q, estadísticas, splits y RNG; tiempos excluidos. Incluir prueba
sintética con split antes de guardar y otro después, para cubrir estadísticas
acumuladas. La evaluación deja intacto el SHA-256 del checkpoint.

### D. Runner de comparación y protocolo inmutable

Implementar nueva CLI `experiments.compare_pong` para no alterar el protocolo H7
ya consumido. Reutilizar utilidades de estadísticas y auditoría, no hardcodes de
`campaign_h7.settings`. Protocolo identificado por schema/version y hash:

```text
purpose: development
environments: {visual: configuración completa, ocatari: configuración completa}
learners: {corrected_common: parámetros específicos, bart: parámetros específicos}
training_seeds: lista auditada
validation_seeds: lista auditada
test_seeds: lista reservada, NO consumida por train/pilot
interactions, curve_checkpoints, evaluation_episodes, evaluation_horizon
contrasts: [visual/bart - visual/corrected_common,
            ocatari/bart - ocatari/corrected_common]
execution_order, budgets, primary_checkpoint: final
```

- Expandir y validar las cuatro configs ANTES de crear runs. El hash por celda
  incluye entorno, semántica, learner y seed; no heredar defaults sin registrarlos.
- Todas las celdas usan gamma=.99, mismas seis acciones y horizonte/budget dentro
  de su pareja. Mantener cotas y escala correctas para cada reward transform.
- Copiar/fijar protocolo, orden y hashes de fuentes. Rechazar resume sobre código
  de entrenamiento incompatible; registrar cambios permitidos de reporting.
- Ejecutar un run por proceso, secuencialmente por defecto; fijar orden
  pseudoaleatorio por bloques de seed para no ejecutar siempre un método primero.
  No mezclar mediciones de costo concurrentes con secuenciales.
- Cada run guarda manifiesto/config/encoder, curvas de validación, logs,
  complejidad, checkpoint de progreso y final, resumen y estado completed/partial/
  failed. Checkpoint cada punto de curva. No publicar partial como completed.
- `resume` conserva runs completados y continúa solo incompletos desde el último
  checkpoint, sin duplicar filas ni evaluaciones ya guardadas. Validar IDs y hash.
- Separar etapas train, resume, report-validation y test/report-test. No evaluar
  test desde train; stage test requiere confirmación explícita por CLI, protocolo
  congelado y todas las celdas completas. En esta entrega probarlo únicamente con
  dobles o seeds sintéticas, nunca consumir las reservadas de Pong.
- Resultados de test por run: escritura atómica; nunca sobrescribir; al reanudar,
  validar y omitir resultados completos compatibles, completar solo pendientes.
  Vincularlos al hash de checkpoint/protocolo/código de evaluación. Reutilizar
  la solución H7 existente si corresponde, incluyendo sus pruebas de interrupción.
- Mantener por ahora el rechazo OCAtari en el runner H0 existente; la nueva CLI
  usa `corrected_common`. Documentar el comando correcto, sin aliases ambiguos.

**Puerta D:** smoke de las cuatro celdas, tests de resume parcial, protocolo
alterado, checkpoint incompatible y evaluación no mutante. Un error de setup
no consume semillas de evaluación ni deja un run etiquetado como terminado.

### E. Comprobar capacidad de aprendizaje y mezcla antes del piloto

No ajustar hiperparámetros sobre test. Implementar un informe de readiness que
separe problemas de wiring, costo, capacidad de split y calidad de inferencia.

**Baseline:** parámetros iniciales explícitos para el piloto de desarrollo:
eta=.025, gamma=.99, max_depth=4, min_sample_size=128,
significance_level=.0001, best_literal_criteria=p-value, inherit_q_values=true.
Esto NO reproduce el min_sample_size=100000 histórico; declararlo como
configuración del control adaptada al presupuesto. Probar que el criterio puede
dividir en sintéticos, registrar candidatos con suficientes muestras/splits por
acción en Pong. Cero splits con datos reales puede ser un resultado legítimo;
no forzar divisiones ni reducir el umbral hasta que gane. Cualquier cambio
posterior define otra configuración de desarrollo y queda registrado.

**BART:** no declarar resuelta la mezcla por aumentar pasos de entorno. Sobre
un checkpoint de desarrollo por pipeline y un fit fijo, comparar grow_prune y
revision con los mismos IDs, X, targets, escala y prior de ruido. Si el checkpoint
OCAtari existente es compatible, reutilizarlo. Para visual se puede recolectar
con el smoke. No fabricar un batch visual a partir de estados RAM.

Diagnóstico inicial por kernel y acción: cuatro cadenas, 500 burn-in y 1.000
draws, RNG reproducibles. Guardar predicciones en estados frecuentes y raros
elegidos con regla declarada solo desde replay, sigma², hojas y profundidad;
R-hat rank-normalized, ESS bulk/tail y ESS/segundo. Guardar qué observables son
constantes; no tratarlos como convergencia garantizada. Umbrales orientativos:
R-hat≤1.01, ESS bulk/tail≥400 para observables no constantes.

Presupuesto conjunto de auditoría offline: máximo 20 minutos. Si no caben todas
las combinaciones, marcar not_run/partial y conservar artefactos; no bajar
longitudes silenciosamente. Esta auditoría valida el instrumento sobre datos
fijos, NO acredita mezcla de cada fit online ni decide automáticamente un ganador.

Si falla mezcla: informar por acción/observable y proponer siguiente diagnóstico
(más longitud, geometría de datos, normalización/calibración). No ejecutar una
búsqueda masiva de m/lotes/priors ni ajustar el ruido para mejorar retorno. El
piloto puede autorizarse como exploratorio aproximado con estas reservas; la
campaña confirmatoria requiere una decisión explícita sobre el método evaluado.

## 5. Presupuestos y configuraciones preparadas

### Smoke que sí debe ejecutar el developer

Cuatro celdas, una seed de desarrollo auditada (candidata 51), 2.048 decisiones
por celda, horizonte 1.000. Evaluación de validación de dos episodios en 0 y
2.048; no test. BART usa m=5, burn-in/draws=20/20, batch=64, replay=5000,
warmup=fit_interval=256, depth=4, min_child=2 y kernel=grow_prune.
Baseline usa los parámetros del apartado E.

Tope operativo inicial: 5 minutos de entrenamiento por celda y 20 minutos
totales para evaluaciones smoke; guardar progreso y devolver evidencia parcial
si se excede. Estos topes no permiten reducir datos o eliminar runs lentos.
Medir entrenamiento, evaluación y guardado por separado.

### Piloto preparado, pendiente de autorización

- 3 training seeds auditadas, candidatas 51/52/53: 12 ejecuciones.
- 20.480 decisiones por celda; checkpoint final como primario.
- Horizonte train/eval=27.000 decisiones por episodio en ambos pipelines.
- Validación en 0, 5.120, 10.240, 15.360 y 20.480 decisiones; 10 episodios
  de validación por punto, mismas seeds para todos los modelos.
- BART inicial: m=20, burn-in/draws=50/50, replay=50.000, batch=128,
  warmup=fit_interval=512, depth=4, min_child=2, k=2, kernel=grow_prune y
  demás priors/calibración actuales explícitos. Etiquetar inferencia aproximada;
  no llamar convergidos a fits de 50 draws. Si la auditoría aconseja otra
  configuración, presentar propuesta y costo, no cambiar el JSON congelado.
- Baseline: mismos parámetros del apartado E. Epsilon y controles comunes según B.
- No lanzar estas 12 ejecuciones como efecto secundario de validar el protocolo.

Seeds candidatas adicionales: validación 320101–320110 y test reservado
420101–420120. Auditar configuraciones, logs y evidencia existentes ANTES de
fijarlas; si fueron consumidas, elegir/documentar un conjunto nuevo. Todas las
listas son disjuntas. Validación se puede reutilizar durante desarrollo; test no
se usa para tuning. El piloto no consume test y ese tamaño no fija el test final.

La evaluación puede costar más que entrenar 20.480 pasos. Calcular costo de las
12 ejecuciones MÁS 5×10 episodios por run a partir de tiempos observados; ofrecer
estimación central y cota por horizonte. No presupuestar solo sweeps BART.

## 6. Métricas y reglas de interpretación

Por run/pipeline/learner/seed:

- Retorno raw medio/mediano y dispersión por punto de evaluación; episodios
  terminados/truncados, duración y conteo de puntos ganados/perdidos.
- Curvas retorno vs decisiones y tiempo de entrenamiento; AUC normalizada por
  decisiones en la MISMA malla. No comparar AUC de horizontes diferentes.
- Tiempo de entrenamiento, evaluación, emulación/detección, actualización/fits,
  inferencia; RSS pico por proceso y tamaño de checkpoints.
- Complejidad por acción: nodos/hojas/profundidad y splits; para BART suma de
  árboles y variación posterior. No comparar número de ensembles con número de
  hojas como si fueran la misma medida.
- Cobertura por acción, estados incompletos/vacíos, fallbacks baseline, Q máximo,
  targets fuera de escala y calidad MCMC cuando se haya medido.

Promediar episodios DENTRO de cada training seed. Para las diferencias pareadas,
la unidad de remuestreo es la training seed, no cada episodio. Reutilizar
`evaluation.paired_bootstrap`; con tres seeds los intervalos son descriptivos e
inestables. No reportar superioridad confirmatoria ni efecto causal del extractor.
Las políticas generan trayectorias distintas incluso con semillas pareadas.

La campaña final se diseña después: presupuesto suficiente para observar curva,
variabilidad entre seeds, efecto mínimo relevante, cantidad de seeds/episodios,
regla de selección y test no observado. No fijar automáticamente 30 seeds o
millones de interacciones sin estimar costo y justificarlo.

## 7. Pruebas mínimas adicionales

1. Equivalencia TD/F-test con H0 en trazas que incluyan splits y herencia.
2. bool/string/False/ausente correctos sin mutar estados originales.
3. Literales del catálogo completos, orden estable y hash persistente.
4. Política: epsilon por decisión, desempate uniforme, evaluación no mutante.
5. Terminal/truncación/punto/corte de presupuesto para baseline común.
6. Igualdad semántica de entradas a ambos learners con acciones fijadas, visual
   y OCAtari; una llamada al entorno y una transformación de recompensa.
7. Reanudación del baseline común entre procesos para ambos backends, con
   estadísticas de split y memoria temporal del entorno.
8. Cuatro celdas reales por CLI y persistencia del modelo final evaluado.
9. Protocolo/hash/código alterado rechazados; run parcial reanudable sin duplicación.
10. Test interrumpido simulado: completar pendientes sin reevaluar ni sobrescribir.
11. Mantener suite H0–H7/OCAtari existente; reportar regresiones y skips por separado.

## 8. Comandos objetivo y definición de terminado

La implementación debe dejar operativos estos comandos (son contratos futuros):

```bash
.venv/bin/python -m pytest tests/test_baseline_adapter.py tests/test_comparison_protocol.py tests/test_pong_comparison_pipeline.py -q
.venv/bin/python -m experiments.compare_pong --protocol configs/comparison_pong_smoke.json --output results/pong_comparison_smoke_new --stage train
.venv/bin/python -m experiments.compare_pong --output results/pong_comparison_smoke_new --stage resume
.venv/bin/python -m experiments.compare_pong --output results/pong_comparison_smoke_new --stage report-validation
.venv/bin/python -m experiments.audit_pong_readiness --campaign results/pong_comparison_smoke_new --output results/pong_readiness_new --budget-seconds 1200
.venv/bin/python -m experiments.compare_pong --protocol configs/comparison_pong_pilot.json --dry-run
```

Usar el intérprete aislado realmente instalado si `.venv` no es el validado;
registrarlo. `dry-run` valida/expande y estima costos; no crea emuladores de test
ni entrena. El modo resume puede informar que todo está completo sin mutaciones.

Entrega completa de esta fase requiere:

- [ ] Baseline común con contratos A–C verificados, sin cambios al F-test/TD.
- [ ] Las cuatro condiciones consumen las entradas declaradas; sin fallback visual.
- [ ] Runner/protocolo versionados, semillas auditadas y test protegido.
- [ ] Smoke de cuatro celdas completo y regresiones ejecutadas.
- [ ] Diagnóstico offline/costo con resultados o límites presupuestarios explícitos.
- [ ] Piloto configurado, NO ejecutado; propuesta de presupuesto al usuario.
- [ ] Informe con comandos, pruebas, artefactos, semántica del baseline y reservas.

No declarar que Pong está aprendido ni que BART supera al baseline al cerrar
esta fase. El resultado es una comparación bien conectada, reproducible y lista
para que el usuario autorice el siguiente experimento.
