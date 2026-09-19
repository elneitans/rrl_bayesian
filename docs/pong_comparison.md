# Comparación Pong: pasos A–E

Infraestructura de los pasos A–E de [plan_comparacion_pong_baseline_bayes.md](plan_comparacion_pong_baseline_bayes.md).
El learner `RelationalBaselineAgent`, en `bayesian_rrtl/baseline_adapter.py`, se
identifica como `rrtl_corrected_common_v1`. Aprende de hechos y recompensas ya
transformadas, sin crear emuladores ni extraer imágenes.

## Algoritmo y catálogo

Cada `observe` delega una vez en `CorrectedRRLAgent.update_transition` y conserva
TD → estadísticas sobre el Q actualizado → split. Reutiliza `Node` y
`CorrectedNode`, incluida la exclusión de candidatos de varianza cero, los tres
criterios de selección y la herencia opcional de Q. Los módulos históricos y el
baseline H0 siguen intactos.

`from_environment(environment)` toma las acciones efectivas, comprueba su
igualdad con `environment.spec.actions` y usa el catálogo completo del encoder
publicado. `predict_q` devuelve Q en ese mismo orden; `select_action` devuelve el
índice de la acción. No se crean candidatos a partir de observaciones.

| Catálogo comparativo | Candidatos | Diferencia respecto de H0 |
|---|---:|---|
| H0 Pong explícito | 14 | Orden histórico, sin presencia ni trayectoria Y del jugador |
| Visual común | 24 | Añade trayectoria Y, seis presencias y tres contactos temporales publicados por su encoder |
| OCAtari Pong | 20 | Añade trayectoria Y y seis presencias; solo contactos jugador–pelota y pelota–enemigo |

Usar el catálogo publicado no implica que todas las relaciones vayan a aparecer
en cada extractor. Las cuatro condiciones futuras deben usar el catálogo
completo de su propio entorno. Para la regresión con H0 se admite
`ordered_catalog=...`: subconjunto explícito del encoder, sin duplicados y
respetando exactamente el orden suministrado. El orden importa para desempates
del F-test. El estado serializable incluye catálogo, lista ordenada de literales,
hash SHA-256 de esa lista, configuración y hash del contrato.

Igualar hechos no iguala el espacio de hipótesis: el baseline conserva tres
ramas para comparativos y dos para lógicos; BART usa splits binarios. Este
adaptador tampoco reproduce bit a bit la política del runner H0: permite
epsilon explícito y desempata uniformemente con RNG. El calendario de epsilon,
los streams de semillas y la evaluación congelada están integrados en las
sesiones del paso B descritas más abajo.

Los defaults del learner son eta=.025, gamma=.99, profundidad=4,
min_sample_size=128, significancia=.0001, criterio=p-value y herencia activada.
Son los parámetros propuestos para el control de desarrollo del plan; el umbral
128 no reproduce el histórico 100000. La comparación sintética usa umbrales
menores declarados para probar splits, sin cambiar estos defaults.

## Frontera de hechos y ausencia

`normalize_facts` convierte bool/np.bool_ a `"True"`/`"False"`, conserva el resto
del hecho y devuelve una tupla canónica sin mutar la entrada. El encoder valida
claves y categorías; los duplicados equivalentes se colapsan y los contradictorios
se rechazan. Una ausencia sigue siendo ausencia. Se rechazan catálogos ambiguos
para la búsqueda histórica por nombre/obj1/obj2.

Cuando falta el literal de un nodo interno, se devuelve su Q, incluso en estados
vacíos. Ese nodo recibe la actualización TD y las estadísticas que correspondan;
no aparece una nueva rama missing. `query_counts` mide consultas y
`internal_fallbacks` por propósito: `prediction`, `update` y `bootstrap`.
Una predicción/bootstrap cuenta una consulta por acción y cada actualización
cuenta una consulta a la acción ejecutada. Se cuentan consultas semánticas,
sin inflar las cifras por recorridos redundantes internos de `Node`. Por ello,
consultar Q modifica solo esta telemetría; la evaluación de las sesiones comunes
usa una copia y conserva el estado de entrenamiento.

Se conserva otra condición histórica: `split_iteration` exige al menos un
hecho con segundo objeto. Un estado que contiene únicamente presencias puede
acumular estadísticas y actualizar Q, pero no dividir. Las presencias sí son
candidatos efectivos cuando hay también un hecho entre dos objetos; una prueba
produce un split de presencia y comprueba rutas True/False/ausente.

## Interfaz disponible

Ejemplo de una decisión con un `AtariRelationalEnvironment` ya creado y un
`state` obtenido del entorno:

```python
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig, RelationalBaselineAgent

agent = RelationalBaselineAgent.from_environment(environment, BaselineLearnerConfig())
action = agent.select_action(state, epsilon=0.1)
next_state, reward, raw_reward, terminated, truncated = environment.step(action)
td = agent.observe(state, action, reward, next_state, terminated, truncated)
saved = agent.state_dict()
restored = RelationalBaselineAgent.from_environment(environment, agent.config)
restored.load_state_dict(saved)
```

El caller administra el entorno, el presupuesto y los episodios. `reward` entra
ya transformada; el learner no añade bonus, repite acciones ni hace reset.
Un terminal real no inspecciona `next_state`; una truncación hace bootstrap con
la observación final según configuración y rechaza su ausencia si es necesaria.

`state_dict` devuelve una copia independiente serializable mediante pickle:
raíces con estadísticas y enlaces padre/hijos, interacciones, splits, RNG de
selección y telemetría. `load_state_dict` exige el mismo contrato. Solo debe
deserializarse pickle de origen local confiable. Esto es estado del learner;
el archivo atómico con checksum y el recolector/snapshot ambiental se guardan
mediante la sesión del paso C descrita abajo. `predict_posterior` y `predict_draws`
rechazan la operación explícitamente.

## Puerta A

La prueba `test_historical_gate_with_real_splits` compara después de cada una de
120 transiciones los TDUpdate (incluidos target y Q), contador, estadísticas,
estructura, orden de candidatos y predicciones con H0 corregido. Las seis
combinaciones de criterio y herencia producen **tres splits cada una**. Otra
traza cubre las seis acciones, bootstrap, terminales y truncaciones. La
serialización se prueba con un split antes de guardar y dos posteriores, además
de continuidad del RNG. Las pruebas nuevas no requieren emular Atari.

Comando de las pruebas del paso A, usando el intérprete aislado disponible:

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest tests/test_baseline_adapter.py -q
```

Resultados y comandos de regresión en
[validation_pong_comparison.json](validation_pong_comparison.json).
La suite inicial pasó 265 pruebas; la final pasó 295 (30 nuevas), sin fallos ni
skips. Ambas emitieron las mismas 13 advertencias de deprecación de
Matplotlib/Pyparsing. Ruff y `git diff --check` pasaron.

## Paso B: recolección, política y evaluación

`make_comparison_session` en `bayesian_rrtl/comparison.py` valida la configuración
antes de crear ALE. `PongEnvironmentFactory` resuelve todos los campos de
`AtariConfig` y conserva las condiciones declaradas: visual comparativo con
incompletos=false, reset legacy_fire y sign(raw)+0.1; OCAtari comparativo con
incompletos=true, reset none y sign(raw). Ambos usan frameskip=4 y sticky=0.
Cada sesión tiene su propia instancia del entorno. Se validan acciones,
catálogo, gamma=.99 y, para BART, cotas de recompensa y configuración de política.

`BaselineTrainingSession` ejecuta una interacción y un `observe` por decisión.
`BayesianComparisonSession` añade los campos comunes de reporting alrededor de
`QTrainingSession.step`, sin cambiar el loop H5, replay, targets ni sampler.
La prueba con fits reales sobre un entorno controlado comprueba que conserva
acciones, logs H5, targets, draws y RNG; los tiempos de ejecución se excluyen.

La política común deriva epsilon del número de interacciones:

```text
epsilon(t) = max(0.1, 1.0 * (0.1 / 1.0) ** (t / 50000))
```

`t` se incrementa una vez por decisión, nunca por sweep ni durante evaluación.
El desempate entre máximos es uniforme, con remuestreo de acciones repetidas
desactivado. El baseline usa los hijos 0 y 1 de `SeedSequence(seed)`, como H5:
entorno y exploración respectivamente. No consume streams de replay/MCMC.
Las semillas de validación/test se excluyen del generador de episodios. El RNG
propio del learner de A queda disponible para su API directa; durante entrenamiento
la sesión le pasa explícitamente su stream de exploración. Este estado adicional
del recolector se persiste en el checkpoint de C.

`run(n)` añade exactamente n decisiones. Si se agota el presupuesto a mitad de
episodio, conserva estado, seed y retornos para el siguiente `run`; no añade un
terminal ni fuerza un fit de BART. Un punto conserva los flags del entorno.
Terminales reales no consultan next Q; truncaciones usan la observación final
cuando bootstrap está habilitado.

Las dos sesiones exponen:

- `history`: filas por transición con IDs, seed del episodio, acción, recompensa
  transformada/raw, flags, epsilon y retornos acumulados.
- `episode_history`: filas de episodios finalizados; `episode_rows()` añade una
  copia del episodio parcial actual, si existe, sin duplicarlo en el historial.
- `return_scope`: `complete_game`, `horizon_limited` o `partial`.
- `metrics()`: medias de partidas completas separadas de episodios limitados por
  horizonte, junto al episodio parcial. Si no hay partidas completas, la media
  es `None`; el retorno parcial no se incluye.
- `learner_metrics`: TD y splits del baseline, o fit ID/diagnósticos de BART.
  Los diagnósticos por transición se adjuntan solo cuando hubo un fit.

Estas estructuras son logs serializables a JSON. La escritura de archivos y el
lifecycle de campaña corresponden a D.

`evaluate(seeds, horizon=...)` usa una copia del learner, epsilon=0, RNG privado
por seed y un entorno nuevo por episodio. Suma recompensa raw y reporta la
fracción truncada, horizonte, flags y alcance de cada retorno. No modifica
árboles, estadísticas, replay, RNG, contadores ni el entorno de entrenamiento.
Rechaza solapamiento con seeds de entrenamiento, reutilizar el entorno de
entrenamiento y cambios de acciones/catálogo/semántica. Puede cambiar el horizonte
de evaluación; un corte de evaluación queda etiquetado como retorno limitado,
sin declarar una partida completa.

Ejemplo de uso por API para el baseline OCAtari:

```python
from bayesian_rrtl.atari import AtariConfig
from bayesian_rrtl.baseline_adapter import BaselineLearnerConfig
from bayesian_rrtl.comparison import ComparisonPolicyConfig, make_comparison_session

environment = AtariConfig(
    game="Pong", extractor="ocatari_ram_v1",
    include_incomplete_states=True, max_episode_steps=512,
)
policy = ComparisonPolicyConfig()
session = make_comparison_session(
    "corrected_common", environment, BaselineLearnerConfig(seed=61),
    policy_config=policy,
)
try:
    session.run(128)
    session.run(128)  # Continúa el mismo episodio si no terminó.
    report = session.evaluate(policy.validation_seeds, horizon=512)
    metrics = session.metrics()
finally:
    session.environment.close()
```

Para BART, pasar `"bart"` y un `BayesianQConfig` con las acciones del entorno,
gamma=.99, cotas correspondientes, epsilon_decay_steps=50000 y las mismas listas
de semillas de `policy`. La fábrica rechaza discrepancias en vez de sobrescribir
la configuración silenciosamente. Las seeds predeterminadas son las de desarrollo
de H5; las candidatas del futuro piloto siguen pendientes de auditoría en D.

## Puerta B

La validación acotada `experiments.validate_pong_comparison_inputs` abre dos
emuladores independientes por backend. Usa seed de desarrollo 61, horizonte
512 y 1.030 decisiones con acción `i % 6`. Captura hechos antes de `observe` y
compara su codificación canónica, acciones, reward/raw, flags y logs. El baseline
aprende con sus defaults; BART tiene warmup explícito 1031, fuera de la traza,
porque esta puerta comprueba entradas/recolección y no capacidad de aprendizaje.

Pasó para visual y OCAtari: **26 puntos no terminales, dos truncaciones y tres
resets por backend**, igualdad de las 1.030 transiciones por learner y exactamente
una llamada al entorno por decisión. No hubo transformación adicional de
recompensa. Los hashes de las trazas, configuración, seeds efectivas y manifiesto
runtime quedan en la evidencia de B. Los tests unitarios cubren además terminales,
continuidad de presupuesto, fits H5 y evaluación congelada con modelos entrenados.
La suite previa a B pasó 295 pruebas y la final pasó **317** (22 nuevas), sin
fallos ni skips; se mantienen las 13 advertencias de Matplotlib/Pyparsing. Ruff
y `git diff --check` pasaron. La evidencia JSON conserva A en `step_A` y añade
las mediciones de esta entrega en `step_B`.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest tests/test_comparison_training.py tests/test_pong_comparison_inputs.py -q
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.validate_pong_comparison_inputs --output results/comparison_gate_b_new
```

## Paso C: checkpoint del baseline común

`bayesian_rrtl/baseline_checkpoint.py` define un formato independiente con cabecera
`RRTL-CORRECTED-COMMON-CHECKPOINT-1`, schema=1 y learner
`rrtl_corrected_common_v1`. El payload pickle es para archivos locales confiables:
el checksum detecta corrupción, no convierte pickle en un formato seguro para
archivos de terceros.

`BaselineTrainingSession.save(path)` guarda:

- Contrato/hash del learner, configuración de política y configuración ambiental
  resuelta; catálogo, lista ordenada de literales y sus hashes.
- Raíces completas con estadísticas y enlaces padre/hijos, interacciones, splits,
  telemetría y RNG propio del learner.
- RNG de entorno/exploración, semillas de episodios, estado relacional actual,
  contadores, retornos parciales e historiales. Epsilon se deriva de la política
  y el contador, sin una segunda variable susceptible de divergir.
- Snapshot ambiental v1 visual o v2 OCAtari, incluida memoria temporal, wrappers,
  detector y RNG según el contrato del backend.
- Versiones de runtime y hashes de fuentes que determinan el entrenamiento.

Se escribe en un temporal del mismo directorio, se vacía/fsync y se publica con
`os.replace`. Un fallo antes de publicar conserva el checkpoint anterior y
elimina el temporal. El guardado no avanza el entorno ni actualiza Q. Solo se
permite entre actualizaciones completas; si una actualización falla, la sesión
rechaza continuar/guardar y debe recuperarse desde el último checkpoint válido.

La carga verifica checksum, versiones, learner, contratos, acciones, catálogo,
estadísticas y estructura de árboles, contadores, seeds/RNG de episodios,
historiales, retornos, epsilon y coherencia del episodio con `done` ambiental.
Comprueba semántica de relaciones, reward/reset y el snapshot antes de restaurar.
Las estadísticas se recuperan del artefacto, sin reinicializarlas. Cambios de
fuentes/runtime requieren una migración explícita; no se aceptan silenciosamente.
Un checkpoint RAM crea su backend RAM y falla si falta OCAtari, sin fallback visual.

Uso con una sesión de baseline creada por la fábrica de B:

```python
from bayesian_rrtl.baseline_training import BaselineTrainingSession
from bayesian_rrtl.baseline_checkpoint import evaluate_baseline_checkpoint

session.run(100)
session.save("results/my_run/progress.rrtlc")
session.environment.close()

# Crea el backend indicado por el archivo y restaura el episodio en curso.
resumed = BaselineTrainingSession.load("results/my_run/progress.rrtlc")
try:
    resumed.run(128)
    resumed.save("results/my_run/final.rrtlc")
finally:
    resumed.environment.close()

# Carga el artefacto final; no evalúa la referencia en memoria del entrenamiento.
report = evaluate_baseline_checkpoint(
    "results/my_run/final.rrtlc", (100001,), horizon=512,
)
```

La carga admite `expected_config`, `expected_policy` y `expected_environment`
para rechazar discrepancias con una configuración esperada. También admite un
entorno explícito compatible; en ese caso el caller mantiene su propiedad y lo
cierra. Los entornos de prueba no Atari requieren un entorno/factory explícito.
La evaluación crea y cierra sus propios entornos, conserva el estado del modelo
y nunca escribe sobre el checkpoint cargado.

## Puerta C

`experiments.validate_baseline_resume` verifica tres casos:

| Caso | Pausa | Continuación local y otro proceso | Splits antes → después |
|---|---:|---:|---:|
| Sintético separable | 12 decisiones | 128 decisiones | 1 → 3 |
| Pong visual | 100 decisiones | 128 decisiones | 0 → 0 |
| Pong OCAtari | 100 decisiones | 128 decisiones | 0 → 0 |

Las pausas son a mitad de episodio. Los casos Atari usan horizonte 220 para
incluir también truncación y reset durante la continuación. Se comparan hashes
por decisión de acciones, hechos, reward/raw, flags, Q, estadísticas, splits,
RNG, seeds, historial y episodio parcial; se excluyen tiempos de perfilado.
Los parámetros Atari son los defaults del control; el sintético usa eta=1,
gamma=0, min_sample_size=6 y significancia=.05 para producir divisiones.

Pasaron los tres casos. Se carga el checkpoint final para evaluar con seed de
validación 100001 y horizonte 16; su SHA-256 permanece intacto. Esto valida la
ruta de evaluación y no mide rendimiento de Pong. Las pruebas incluyen archivos
truncados, formatos H0/H5, contratos alterados, estadísticas/counters incompatibles,
fallos simulados de publicación y guardado durante una actualización.
La suite previa a C pasó 317 pruebas; la final pasó **350** (33 nuevas), sin
fallos ni skips. Se mantienen las 13 advertencias de Matplotlib/Pyparsing. Ruff
y `git diff --check` pasaron.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest tests/test_baseline_checkpoint.py -q
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.validate_baseline_resume --output results/comparison_gate_c_new
```

La evidencia de esta entrega está en `step_C` dentro de
[validation_pong_comparison.json](validation_pong_comparison.json); se conservan
las mediciones previas de A y B.

C implementa el checkpoint del **baseline común**. D integra la persistencia
del reporting bayesiano mediante `comparison_checkpoint.py`, sobre el payload H5
sin cambiar su implementación. Se valida el tipo de recolector de comparación,
acciones/catálogo, runtime, historial, contadores y snapshot. El formato nuevo
no se confunde con un checkpoint de evaluación H0. Se mantiene el rechazo de
OCAtari en el runner histórico H0.

## Paso D: campaña pareada

La CLI `experiments.compare_pong` expande las cuatro condiciones antes de crear
runs y congela configuración resuelta, orden y hashes de fuentes en
`protocol.json`/`protocol.sha256`. El orden se permuta por bloques de training seed
con una semilla independiente; un run por proceso, secuencialmente. Los cambios
de código de entrenamiento bloquean resume/test. Reporting puede regenerarse
con código nuevo y registra sus hashes, conservando la identidad de la evidencia.

`configs/comparison_pong_smoke.json` fija cuatro celdas, training seed 71,
2.048 decisiones, horizonte 1.000, validación en 0/2.048 con dos seeds y los
hiperparámetros smoke del plan. El control usa min_sample_size=128 y no el
100000 histórico. `configs/comparison_pong_pilot.json` prepara 12 runs con
71/72/73, 20.480 decisiones, horizonte 27.000, cinco puntos de curva y diez
episodios de validación por punto. El piloto no se ejecuta con dry-run.

La auditoría previa inspeccionó 822 JSON/CSV de configs, docs y resultados,
incluidos campos de semillas anidados y columnas de logs. No encontró uso
registrado de 71/72/73, validación 320101–320110 ni test 420101–420120. Se eligió
71/72/73 como conjunto de desarrollo nuevo; 51/52 también habían aparecido como
RNG de espacios en pruebas anteriores. La evidencia conserva inventario/hash y
alcance de búsqueda; los documentos de planificación no se consideran ejecuciones.
El smoke consume solo 71 y 320101/320102. El test reservado sigue sin consumirse.

Cada run guarda manifiesto, configuración, encoder, logs de transiciones y
episodios, checkpoints de progreso, evaluaciones, curva/AUC y resumen con
complejidad, cobertura, tiempos separados y RSS pico. Las evaluaciones cargan el
checkpoint publicado y se vinculan a sus hashes de artefacto/protocolo/código.
Un checkpoint se guarda en cada punto de curva y periódicamente entre puntos.
El final conserva exactamente los bytes del último checkpoint evaluado.

`resume` valida identidad y checksum, conserva runs terminados y continúa solo
incompletos. Usa el historial del checkpoint para evitar filas duplicadas y
omite evaluaciones compatibles ya publicadas. Estados `partial`/`failed` nunca
se anuncian como `completed`. Un error de setup no evalúa ese run. Los topes
smoke son 300 segundos de entrenamiento por run y 1.200 segundos de evaluación
para la campaña; se conserva el último checkpoint al agotar presupuesto. Los
checks cooperativos terminan actualizaciones completas; una operación en curso
puede sobrepasar ligeramente el umbral y hay además un timeout de proceso.

```bash
# Ejecutar únicamente el smoke autorizado, en una carpeta nueva.
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.compare_pong --protocol configs/comparison_pong_smoke.json --output results/pong_comparison_smoke_new --stage train
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.compare_pong --output results/pong_comparison_smoke_new --stage resume
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.compare_pong --output results/pong_comparison_smoke_new --stage report-validation

# Preparación del piloto: no crea emuladores ni entrena.
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.compare_pong --protocol configs/comparison_pong_pilot.json --dry-run --cost-reference results/pong_comparison_smoke_sequential
```

`test` exige `--confirm-test`, protocolo congelado y todas las celdas completas.
Hace preflight de los resultados existentes, conserva archivos completos y evalúa
solo runs pendientes. Cada JSON se publica atómicamente y no se sobrescribe.
Un resultado incompatible bloquea el test antes de consumir nuevas semillas.
`report-test` solo lee resultados; no completa los faltantes. Esta ruta se probó
con resultados sintéticos y una interrupción tras el primer run, sin evaluar
las semillas reservadas reales de Pong.

Los contrastes se calculan **dentro de cada pipeline**. Se promedian episodios
dentro de cada training seed y `paired_bootstrap` remuestrea esas seeds. Los
intervalos de una o tres seeds son descriptivos; no sustentan superioridad
confirmatoria ni un efecto causal del extractor.

## Paso E: readiness y límites

`experiments.audit_pong_readiness` comprueba la capacidad de split del control
con sus defaults del piloto. La señal fija alterna bloques de 256 terminales con
x=less/reward=-1 y x=more/reward=+1: produce un split en la transición **292**, sin
cambiar F-test, eta, umbral ni herencia. También informa por acción candidatos
con n>128/varianza positiva y splits efectivamente producidos en el smoke.

La auditoría BART selecciona un fit de desarrollo por pipeline. Para OCAtari
intenta reutilizar el checkpoint anterior si coinciden hiperparámetros, catálogo
y contrato ambiental; la seed puede diferir porque se diagnostica un dataset
fijo. Para visual usa el checkpoint smoke visual, sin fabricar datos RAM.
Ambos kernels reciben exactamente IDs, X, targets, escala y prior de ruido de
ese fit. No se altera sampler, priors ni calibración para mejorar resultados.

Se seleccionan dos estados más frecuentes y dos más raros entre los estados
únicos del replay, con desempate determinista por códigos, sin consultar test.
Por acción/kernel se ejecutan cuatro cadenas independientes, 500 burn-in y
1.000 draws. Se conservan arrays de predicciones, sigma², hojas, profundidad,
dataset/referencias, R-hat rank-normalized, ESS bulk/tail y ESS/segundo.
Los observables constantes se señalan y no acreditan convergencia.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.audit_pong_readiness --campaign results/pong_comparison_smoke_sequential --output results/pong_readiness_new --budget-seconds 1200
```

El límite conjunto es 20 minutos: se conservan cadenas terminadas y estados
`partial`/`not_run` cuando no cabe el resto. No se reducen longitudes para
presentar la auditoría como completa. Incluso una auditoría favorable de un fit
no acredita la mezcla de todos los fits online de 20/50 draws.

El costo del piloto incluye **12 entrenamientos y 600 episodios de validación**,
además del guardado. La estimación central extrapola las duraciones observadas
del smoke; la alternativa por horizonte usa 27.000 decisiones por episodio.
Se explicitan factores de escala de fits/árboles/sweeps/batch y de inferencia
árboles×draws. Son aproximaciones del host actual, no garantías de runtime;
geometría de árboles y replay pueden aumentar el costo. Los topes operativos
provisionales del JSON piloto son 7.200 s de entrenamiento por run y 14.400 s
de evaluación conjunta: agotar esos topes produciría un piloto parcial.

El piloto queda preparado y **pendiente de autorización**. Tampoco se ejecutó
test reservado. Los resultados medidos, límites y siguiente diagnóstico se
registran en [pong_comparison_readiness.md](pong_comparison_readiness.md) y en `step_D_E` de
[validation_pong_comparison.json](validation_pong_comparison.json).

## Ampliar un presupuesto agotado

Los presupuestos son límites **acumulados**, no asignaciones por intento. Un
`resume` sin ampliación conserva los límites y los tiempos ya consumidos. Para
continuar una campaña detenida por presupuesto, declarar nuevos límites totales
mayores que los vigentes y el motivo:

```bash
python -m experiments.compare_pong --output results/mi_campana --stage resume \
  --training-budget-seconds 600 --evaluation-budget-seconds 2400 \
  --budget-reason "Completar el smoke detenido por presupuesto"
```

Se puede ampliar solo uno de los límites. El primero aplica a cada run; el
segundo a toda la evaluación de validación de la campaña. No son segundos
adicionales. Estas opciones solo se admiten en `resume`; no cambian decisiones,
horizontes, seeds, hiperparámetros ni la identidad del protocolo congelado.

`operational_budgets.json` se publica atómicamente y conserva las revisiones,
fecha UTC, motivo, límites anteriores/nuevos y una instantánea de contadores.
Cada revisión incluye el hash de la anterior y del protocolo. Los workers y
reportes registran el límite efectivo y la revisión aplicada. No se reinician
contadores, no se sobrescriben evaluaciones válidas y se omiten runs completos.
Los checks de compatibilidad de código siguen activos: una ampliación de tiempo
no autoriza migrar una campaña creada con fuentes incompatibles.

## Métricas de evaluación por episodio

Las evaluaciones nuevas (`pong_evaluation_v2`) guardan `points_won` y
`points_lost` durante cada episodio a partir de recompensa raw positiva y
negativa, antes de transformarla. Para Pong, cada punto corresponde a +1/-1;
si un paso agrupa varios puntos del mismo signo, se suma su magnitud. No se
infieren puntos a partir del retorno neto y no se incluyen los de entrenamiento.

Cada evaluación informa `mean_raw_return`, `median_raw_return`,
`std_raw_return` (desviación estándar poblacional, ddof=0) e `iqr_raw_return`
(cuartiles con interpolación lineal), además de duración, truncación y alcance.
La dispersión describe los episodios de esa evaluación, no incertidumbre entre
training seeds. Los contrastes pareados mantienen training seed como unidad.
El reporte de campaña conserva estas estadísticas por run y los puntos por seed.

Los resultados históricos sin estos campos se pueden seguir reportando: mediana
y dispersión se calculan desde los retornos por episodio, mientras
`evaluation_points` queda en `null`. No se inventan puntos ni se reevalúan semillas
para rellenarlos. Las evaluaciones nuevas validan el esquema, los contadores y
las estadísticas antes de aceptar un archivo como completo.
