# OCAtari Pong — pasos 0–5

Implementados el entorno reproducible, el protocolo de backend, el registro de
capacidades y el adaptador RAM para Pong de OCAtari 2.2.1. Se validaron con ALE
real: reset, step, acciones, objetos, truncación y terminal natural. El paso 2
añade `PongRelationsV1`, registrado como constructor de `pong_relational_v1`,
con catálogo fijo y memoria relacional serializable.

El paso 3 conecta OCAtari a `AtariRelationalEnvironment`, `QTrainingSession` y
`train_atari.py`: entrenamiento, evaluación y reanudación. Incluye el codec de
snapshot necesario para que la reanudación funcione, con pruebas de continuidad.
El registro declara `atari_relational_v2` y `exact_resume=True`; se conserva la
lectura de snapshots v1 legacy. Los pasos 4 y 5 completan la matriz de
reanudación entre procesos, el smoke de 2.048 interacciones y los diagnósticos
MCMC con cuatro cadenas. La integración pasa; la mezcla MCMC es insuficiente
en cinco de seis acciones. No se ejecutó test, piloto ni campaña de eficacia.

## Instalación aislada

Entorno verificado: CPython **3.11.7**, macOS arm64, OCAtari **2.2.1**,
Gymnasium **1.1.1** y ALE **0.10.2**. El intérprete usado fue
`/private/tmp/rrtl-ocatari-step01-env/bin/python`; no hereda paquetes globales.
`requirements-ocatari-lock.txt` registra la resolución completa. Todos los pins
del lock previo del proyecto se conservaron. Las dependencias PyObjC que
OCAtari incorpora transitivamente vía `keyboard` están condicionadas a macOS;
no se ha validado esta resolución en otras plataformas.

Para reproducir en otro directorio aislado con Python 3.11.7:

```bash
python3.11 -m venv /tmp/rrtl-ocatari-repro
/tmp/rrtl-ocatari-repro/bin/python -m pip install -r requirements-ocatari-lock.txt
/tmp/rrtl-ocatari-repro/bin/python -m pip check
/tmp/rrtl-ocatari-repro/bin/python -m pytest tests/test_ocatari_backend.py tests/test_ocatari_pipeline.py -q
/tmp/rrtl-ocatari-repro/bin/python -m experiments.validate_ocatari_backend --output results/ocatari_backend_new
```

El archivo directo `requirements-ocatari.txt` incluye los pins existentes y
`ocatari==2.2.1`; el lock es la opción para reproducir también las transitivas.
No se modificó el Python global ni `.venv`.

La ROM Pong estaba disponible en el paquete oficial `ale-py==0.10.2`.
`ale_py.roms.get_rom_path("pong")` comprobó su checksum; el informe también guarda
su SHA-256. No se descargaron ROMs de fuentes adicionales ni se añadieron al
repositorio. Si falta la ROM, reinstalar el paquete ALE fijado en el entorno
usado, comprobar `ALE_ROMS_DIR` si está definido y seguir las instrucciones del
paquete oficial. El validador falla ante dependencias/ROMs ausentes; no simula
aceptación real ni omite esas comprobaciones silenciosamente.

## Contratos implementados

- `atari_backend.py`: `ObjectBackend` y `ObjectStep` inmutable con frame,
  recompensa original, `terminated` y `truncated`.
- `game_registry.py` y `games/`: registro estático de Pong RAM y las tres rutas
  visuales existentes. Declara slots, acciones, esquema, opciones, políticas y
  capacidades de snapshot. Un juego OCAtari no registrado se rechaza.
- `games/pong.py`: asignación específica Player→player, Ball→ball y Enemy→enemy.
  Exige tres slots, categorías correctas, dimensiones positivas y coordenadas
  finitas. Los objetos falsos producen ausencia explícita; coordenadas cero o
  negativas válidas se conservan. Copia todo a los contratos existentes de
  `DetectedObject` y `ObjectFrame`, sin guardar referencias mutables.
- `ocatari_backend.py`: importa OCAtari solo al construir el backend. Exige las
  versiones verificadas y normaliza el retorno invertido de 2.2.1 en un único
  punto. El helper público `get_action_meanings()` de esa versión presupone
  dos wrappers y falla con `PassiveEnvChecker`; la compatibilidad lee las
  acciones del entorno subyacente `unwrapped`, sin crear otro emulador.

La instancia usa `ALE/Pong-v5`, RAM, `hud=False`, `obs_mode="ori"`, sin render,
lista nueva de buffers vacía, `frameskip=4`, probabilidad de repetir acciones
cero y espacio mínimo de seis acciones. No ejecuta FIRE/no-ops ocultos al reset
ni transforma recompensas. La fachada aplica `sign(raw_reward)` una sola vez,
conserva la recompensa original y exige cotas `[-1,1]`; la ruta visual histórica
de Pong sigue sumando +0,1 y usando cotas `[-0,9,1,1]`. El orden exigido es
`NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE`; falla antes de reset si difiere.

Uso directo del backend, independiente del aprendizaje:

```python
from bayesian_rrtl.ocatari_backend import OCAtariBackend

backend = OCAtariBackend(max_episode_steps=1000)
try:
    frame = backend.reset(seed=41)
    step = backend.step(0)
    # step.frame, step.raw_reward, step.terminated, step.truncated
finally:
    backend.close()
```

## Evidencia y pruebas

La evidencia inicial de pasos 0/1 se conserva en `docs/validation_ocatari_step01.json`.
La evidencia actual completa se guarda en `docs/validation_ocatari_pong.json`.
Los artefactos locales están en `results/ocatari_step01/`: logs y XML de pytest,
línea base, suite nueva, suite final y `real/validation.json`. El informe registra
intérprete, paquetes, hashes de `core.py`, `ram/pong.py` y `ram/game_objects.py`,
checksum de ROM, comandos, flags, conteos y tiempos. Las fuentes instaladas
`core.py` y `ram/pong.py` coinciden byte por byte con el sdist 2.2.1 de PyPI.

- Línea base H0–H7: **148 pruebas**, cero fallos/omisiones, 118,96 s.
- Nuevas pruebas de pasos 0/1: **36 pruebas**, cero fallos/omisiones, 2,64 s.
- Suite completa final: **184 pruebas**, cero fallos/omisiones, 36,06 s.
  Persisten los 13 avisos de deprecación Matplotlib/Pyparsing de la línea base.
  Ruff y `git diff --check` pasan.
- Límite real de un paso: `terminated=False`, `truncated=True`.
- Rollout aleatorio independiente, semilla 41: terminal natural en **931**
  decisiones, retorno original **−21**, sin truncación. Hubo movimiento de los
  tres objetos y cobertura de las seis acciones. Es evidencia del contrato,
  no una medición de aprendizaje.
- Una llamada subyacente por decisión y avance de cuatro frames ALE; reset sin
  acciones ocultas; frames previos intactos tras avanzar y terminar.
- Los cuatro pares de flags se probaron con dobles y también ejecutando el
  método `step` instalado de OCAtari sobre un entorno Gym simulado.

Las pruebas reales llevan las marcas `ocatari` y `atari`. Las pruebas de dobles
no necesitan OCAtari ni ROM. Para un entorno sin la dependencia opcional:

```bash
python -m pytest tests/test_ocatari_backend.py -q
python -m pytest -m 'not atari' -q
```

Para aceptación real obligatoria, usar la suite `test_ocatari_pipeline.py` o el
validador dedicado con las dependencias fijadas; no hacen skip automático.
El nombre `pipeline` reserva las pruebas de integración que se incorporarán en
pasos posteriores; sus pruebas actuales cubren únicamente el backend.

## Próxima extensión

La matriz de reanudación y el smoke están completados. El piloto queda preparado
para una ejecución posterior explícita; antes de interpretar sus resultados debe
revisarse la mezcla insuficiente encontrada en los diagnósticos del smoke.

Para registrar otro juego más adelante, crear su `GameSpec` y adaptador de slots,
fijar categorías y acciones verificadas, implementar sus relaciones y añadirlo
explícitamente en `games/__init__.py`. Hay una prueba de registro ficticio con
backend doble que verifica la selección sin añadir condicionales a este backend.
No acredita soporte real de otro juego. Mantener `exact_resume=False` hasta
implementar su codec y demostrar continuidad futura entre procesos.

## Paso 2: relaciones y memoria temporal

`PONG_OCATARI.relation_factory` construye `PongRelationsV1`. Solo admite
`representation="comparative"` e `include_incomplete_states=True`; rechaza las
otras opciones. No llama al constructor de relaciones visual legacy.

El encoder está disponible antes de observar datos y tiene **20 claves fijas**:
6 espaciales, 6 temporales, 6 presencias y 2 contactos. Usa centros de bbox,
origen arriba a la izquierda, X hacia la derecha e Y hacia abajo. Las relaciones
espaciales comparan player/ball, player/enemy y ball/enemy, con `same` dentro de
±4 píxeles inclusive. Las temporales comparan cada objeto actual con el de la
decisión inmediatamente anterior y tolerancia cero.

La ausencia omite comparaciones y contactos dependientes del objeto, pero emite
`present=False` explícito. Incluso un frame completamente ausente contiene seis
hechos de presencia. Los contactos player/ball y ball/enemy incluyen solapamiento
o toque en ambos ejes; no reconstruyen colisiones entre observaciones.

`transform(frame, raw_reward=...)` recibe la recompensa **original**. Si es
distinta de cero, construye el estado sin historial anterior y después guarda
el frame actual. `reset()` también borra el historial. Una desaparición y
reaparición nunca produce desplazamiento respecto de una posición anterior al
hueco; el resto de objetos conserva su memoria consecutiva.

```python
from bayesian_rrtl.game_registry import get_game_spec

relations = get_game_spec("Pong", "ocatari_ram_v1").relation_factory()
encoder = relations.encoder()  # catálogo completo antes del primer reset
relations.reset()
facts = relations.transform(frame)
# Después de backend.step(action):
facts = relations.transform(step.frame, raw_reward=step.raw_reward)
encoded = encoder.transform([facts])
memory = relations.snapshot()  # compatible con JSON, sin objetos del emulador
restored = type(relations)()
restored.restore(memory)
```

`manifest()` identifica el esquema, las convenciones y tolerancias semánticas,
la política de historial y el hash del catálogo. El snapshot relacional incorpora
ese manifiesto y rechaza versiones/configuraciones incompatibles, incluso cuando
el catálogo no cambió. El paso 2 probó solamente continuidad de hechos; las
pruebas de continuidad del emulador y H5 se describen abajo.

Las **47 pruebas nuevas** de `tests/test_pong_relations.py` cubren límites
espaciales/temporales, contacto, presencia, reset, punto, reaparición, orden del
encoding y restauración JSON. Corren sin ALE/OCAtari. Evidencia de esta entrega:
`docs/validation_pong_relations.json`; logs/XML: `results/ocatari_step2/`.
Suite completa tras el paso 2: **231 pruebas pasaron**, cero fallos/omisiones,
36,15 s; se mantienen los 13 avisos Matplotlib/Pyparsing previos. Ruff y
`git diff --check` pasan; el código visual legacy permanece intacto.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest tests/test_pong_relations.py -q
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest -q --junitxml=results/ocatari_step2/tests.xml
```

## Paso 3: fachada, CLI y reanudación

`AtariConfig` resuelve `env_id`, `relation_schema`, `reward_transform` y
`reset_policy` desde el registro. `to_dict()` conserva la forma histórica de una
configuración legacy cuando esos campos no estaban presentes; `resolved_dict()`
expone los defaults efectivos para manifiestos nuevos. La restauración compara
configuraciones resueltas, sin reescribir archivos anteriores ni sus hashes.

La fachada selecciona las factories registradas, valida el orden real de
acciones antes de construir el agente y entrega los mismos cinco valores a H5.
Un punto limpia la memoria relacional, pero no termina un episodio por sí solo.
La truncación entrega su observación final para bootstrap; agotar el presupuesto
del runner pausa la sesión. BART, replay, targets, epsilon-greedy, el calendario
de fits y `QTrainingSession` no se modificaron. Guardar no fuerza un fit.

`--variant corrected` con OCAtari se rechaza antes de crear la salida. H7 conserva
su selección de extractores visuales y su protocolo; esta integración no convierte
el baseline H0 ni la campaña H7 en controles OCAtari.

Configuración ejecutable de integración:
`configs/bayesian_pong_ocatari_integration.json`. Declara todos los parámetros;
usa 64 interacciones, 2 árboles, 2 burn-in/4 draws y fits cada 16 transiciones.
**No sustituye el smoke de 2.048 interacciones del paso 5 ni acredita mezcla o
aprendizaje.** Usa validación 310001/310002 y reserva 410001/410002 sin evaluarlas;
se comprobó que esas semillas no figuraban en los artefactos existentes antes
de ejecutar esta integración.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --config configs/bayesian_pong_ocatari_integration.json --output results/pong_integration_new --eval-split none
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --checkpoint results/pong_integration_new/final_checkpoint.rrtl --eval-split validation
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --resume results/pong_integration_new/final_checkpoint.rrtl --output results/pong_resume_new --steps 16 --eval-split none
```

El manifiesto registra entorno resuelto y real, versiones, fuentes externas y del
repo, checksum de ROM, acciones, catálogo y semántica relacional. El profiling
separa `backend_step_seconds` (**emulación y detección juntas**),
`backend_reset_seconds`, `object_conversion_seconds` y `relations_seconds`.

Para habilitar la reanudación de la CLI se adelantó el codec necesario del paso 4:

- Snapshot de fachada `atari_relational_v2`, dentro del contenedor de checkpoint
  existente, con lectura de `atari_relational_v1` preservada para legacy.
- Estado completo ALE con RNG, RNG de Gym y de spaces inicializados, contadores
  de wrappers, objetos RAM con campos temporales explícitos, frame actual,
  historial relacional, `done` y profiling. Los buffers deben estar desactivados.
- Se serializan datos explícitos, sin picklear el entorno vivo. ALE 0.10.2 expone
  su estado binario mediante su codec `__getstate__/__setstate__`; su método
  `serialize()` intenta decodificar esos bytes como UTF-8 y falla en esta versión.
- La restauración inicializa Gym sin acciones ocultas ni una nueva detección
  OCAtari, restaura ALE y los datos guardados, y rechaza incompatibilidades de
  versiones, fuentes, ROM, configuración, wrappers o semántica relacional.

Pruebas realizadas: 100 transiciones futuras idénticas en otro entorno, contador
correcto a un paso de truncación, RNG de spaces, continuidad H5 de replay,
exploración, targets y draws, y reanudación CLI en otro proceso contrastada con
continuación local. Esa evidencia inicial del paso 3 se amplía con la matriz
del paso 4 descrita a continuación.

Evidencia de la entrega: `docs/validation_ocatari_step3.json`; artefactos locales:
`results/ocatari_step3_final/`. **247 pruebas pasaron**, cero fallos/omisiones,
46,63 s; persisten los 13 avisos Matplotlib/Pyparsing conocidos. Ruff pasa.
Entrenamiento medido: 64 transiciones/4 fits en 0,503 s; reanudación de 16
transiciones adicionales hasta 80/5 fits en 0,121 s. La evaluación independiente
conservó el SHA-256 del checkpoint; no se ejecutó test ni campaña de eficacia.


## Pasos 4 y 5: aceptación final

El snapshot valida los campos del backend, la configuración y los contadores de
wrappers, los hashes/versiones y la coherencia entre `done`, frame actual e
historial relacional. Datos incompatibles se rechazan antes de restaurar el
emulador. Solo se admiten los buffers desactivados del contrato RAM actual.

La matriz real compara el avance original con restauraciones en otro entorno y
en otro proceso. Compara objetos, campos temporales del detector, hechos,
recompensas y flags: **128 decisiones idénticas** en cada uno de siete casos
(intercambio, antes/después de punto, antes/durante ausencia y antes/después de
reaparición); el octavo caso verifica la truncación en la siguiente decisión.
El máximo desvío fue cero. La continuidad H5 entre procesos compara otras 128
transiciones, replay/IDs, RNG, targets, modelos y draws, excluyendo tiempos.
Se prueban snapshots v1 de los tres juegos legacy con defaults resueltos.

El validador escribe en un directorio nuevo, guarda un checkpoint de progreso
por bloque y aplica presupuestos independientes de 600 segundos al entrenamiento
y a los diagnósticos. Un timeout conserva el último bloque, logs y curva, marca
lo pendiente y no reduce los parámetros. El checkpoint de progreso puede
reanudar la sesión usando `train_atari.py --resume`.

```bash
/tmp/rrtl-ocatari-step01-env/bin/python -m pytest tests/test_ocatari_resume.py tests/test_ocatari_validation.py -q
/tmp/rrtl-ocatari-step01-env/bin/python -m experiments.validate_ocatari_pong --output results/ocatari_pong_validation_new
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --config configs/bayesian_pong_ocatari_smoke.json --output results/ocatari_pong_smoke_new --eval-split none
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --resume results/ocatari_pong_smoke_new/final_checkpoint.rrtl --output results/ocatari_pong_resume_new --steps 256 --eval-split none
/tmp/rrtl-ocatari-step01-env/bin/python train_atari.py --checkpoint results/ocatari_pong_smoke_new/final_checkpoint.rrtl --eval-split validation
```

Los archivos `configs/bayesian_pong_ocatari_smoke.json` y
`configs/bayesian_pong_ocatari_pilot.json` declaran todos los parámetros del plan.
El piloto prepara 20.480 interacciones por semilla (41, 42, 43); **no se ejecutó**.
Como 310001/310002 ya se evaluaron en el paso 3, las configuraciones nuevas usan
310011/310012 para desarrollo. La auditoría previa no encontró esas semillas ni
410001/410002 consumidas en resultados; estas últimas siguen reservadas para
test. La evaluación de desarrollo es opcional y no se ejecutó en el validador.

Ejecución medida en `results/ocatari_pong_validation_step45/`:

- Smoke: **2.048 transiciones, 8 fits**, en 23,99 s de entrenamiento
  (25,51 s del proceso completo), dentro del presupuesto de 600 s.
- Q, targets y recompensas finitos; **217 estados codificados distintos** y
  cobertura de las seis acciones: 368, 354, 308, 344, 321 y 353 decisiones.
- 1 punto positivo, 42 negativos, 1 terminal natural y 1 truncación.
- Checkpoint final recargable de 1.082.539 bytes. Se guardaron manifiesto,
  configuración, CSV, curva de retorno acumulado de entrenamiento, fits,
  conteos de presencia/hechos ausentes y tiempos por etapa.
- Diagnósticos sobre los batches/targets congelados del fit final: **4 cadenas,
  500 burn-in y 1.000 draws por cadena**, para cada acción; 172,76 s de cómputo
  (173,82 s del proceso completo). El checkpoint permaneció intacto.

| Acción | R-hat máximo | ESS bulk mínimo | ESS tail mínimo | Todos los observables cumplen |
|---|---:|---:|---:|---|
| NOOP | 2,274 | 5,12 | 5,09 | No |
| FIRE | 2,799 | 4,62 | 4,62 | No |
| RIGHT | 1,126 | 20,69 | 22,21 | No |
| LEFT | 1,052 | 63,64 | 91,69 | No |
| RIGHTFIRE | 2,849 | 4,63 | 8,60 | No |
| LEFTFIRE | 1,003 | 1.113,45 | 1.156,27 | Sí |

Umbrales: R-hat ≤1,01 y ESS bulk/tail ≥400. No hubo observables constantes en
estas cadenas; el informe conserva su estado explícitamente. Solo LEFTFIRE
cumple en todos los observables comprobados. **La integración funciona, pero
estos resultados no acreditan convergencia global ni aprendizaje para ganar
Pong.** No se cambiaron priors ni sampler para mejorar las métricas.

El JSON principal contiene la evidencia final, auditoría de semillas, comandos,
pruebas y diagnósticos por observable. Los NPZ y checkpoints permanecen en
`results/` (ignorado por Git); no se añadieron ROMs al repositorio.

Suite final: **265 pruebas pasaron**, sin fallos ni omisiones, en 56,83 s.
Persisten los 13 avisos Matplotlib/Pyparsing conocidos. Ruff y `git diff --check`
pasan. Además se verificaron los datos/targets/IDs de los NPZ contra el fit
guardado y la continuación del checkpoint smoke hasta la transición 2.049,
manteniendo 8 fits y sin modificar el archivo original.
