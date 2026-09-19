# Plan de implementación: OCAtari extensible, primer juego Pong

## Encargo para el agente developer

Implementar en `capstone/rrl_bayesian/` un backend OCAtari que ejecute Pong de
extremo a extremo con el pipeline actual: relaciones → encoder → replay →
targets congelados → BART por acción → checkpoint → evaluación/reanudación.
Preparar una extensión por registro de juegos, sin añadir todavía otros juegos
OCAtari. Entregar código, pruebas, configuraciones, evidencia reproducible y una
guía para incorporar el siguiente juego.

No considerar terminado el trabajo con un demo de `env.objects`. Debe funcionar
con `QTrainingSession` y `train_atari.py`, incluidos guardado y reanudación exacta.
No prometer que una ejecución corta aprenda a ganar Pong.

Todas las rutas que siguen son relativas a `capstone/rrl_bayesian/`.

## 1. Decisiones cerradas y límites

- Backend inicial: OCAtari RAM, versión objetivo `ocatari==2.2.1`, con versiones
  compatibles de Python/Gymnasium/ALE fijadas y verificadas. No depender de master.
- Juego inicial: `ALE/Pong-v5`; `hud=False`; sin render durante entrenamiento;
  `obs_mode="ori"`; `create_buffer_stacks=[]` nuevo por instancia. Los RGB no
  entran al agente: las relaciones se construyen desde objetos.
- Una única instancia OCAtari avanza la partida. No crear otro ALE para detectar
  objetos ni llamar a `step` dos veces por interacción.
- `frameskip=4`, `repeat_action_probability=0.0`, `full_action_space=False`.
  Ningún wrapper adicional debe repetir acciones o saltar frames.
- Usar las seis acciones mínimas de Pong, en el orden que ALE expone; validar
  contra `NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE`. No renombrar RIGHT/LEFT a
  UP/DOWN ni asumir los cuatro índices de Breakout. Si el orden difiere, fallar
  antes de recolectar, no entrenar con un remapeo implícito.
- Inicio OCAtari: `reset_policy="none"`, sin FIRE/acciones ocultas ni no-ops
  aleatorios. El agente puede elegir FIRE. Verificar con el smoke que hay partida
  y movimiento; un cambio de política de inicio debe quedar versionado y probado.
- Recompensa OCAtari: `reward_transform="sign"`, sin el bonus histórico +0.1.
  Conservar recompensa original para retorno de evaluación. Cotas `[-1,1]`.
- Relación nueva `pong_relational_v1`, representación comparativa y presencia
  explícita; conservar estados incompletos. Rechazar representación lógica para
  esta versión inicial en vez de ignorar la opción. El contrato admite futuros
  constructores de relaciones sin implementar ahora una segunda representación.
- Conservar completamente la ruta visual legacy, incluyendo el +0.1 histórico
  de Pong y sus relaciones. No reinterpretar checkpoints ni resultados H6/H7.
- No modificar priors, sampler, General BART, actualización TD, replay, política
  epsilon-greedy o mecanismo de targets para hacer funcionar el adaptador.
- No lanzar campaña de eficacia, usar test histórico ni integrar más juegos,
  vision/both de OCAtari, seguimiento genérico de objetos o redes neuronales.

## 2. Estado actual y puntos a modificar

`atari.py` ya soporta Breakout/Pong/DemonAttack con extractores visuales. El
constructor, recompensas y snapshots siguen ligados a esa ruta. `objects.py`
contiene `DetectedObject`, `ObjectFrame`, extractores legacy y relaciones
históricas. Reutilizar sus objetos inmutables; no duplicar esos contratos.

`train_atari.py` valida acciones/cotas por juego y usa el adaptador común para
BART. Su `run_corrected` construye el entorno histórico dentro de H0: NO hereda
automáticamente el backend nuevo. `campaign_h7.settings` también selecciona
explícitamente extracción visual. No presentar esos caminos como OCAtari.

Archivos propuestos:

| Archivo | Responsabilidad |
|---|---|
| `bayesian_rrtl/atari_backend.py` | Protocolo de backend y resultado de paso tipado |
| `bayesian_rrtl/ocatari_backend.py` | Compatibilidad OCAtari, normalización de flags, objetos y snapshot |
| `bayesian_rrtl/game_registry.py` | Registro explícito de capacidades por juego/backend |
| `bayesian_rrtl/games/__init__.py` | Registro/importación explícita de juegos soportados |
| `bayesian_rrtl/games/pong.py` | Especificación de slots y `PongRelationsV1` |
| `bayesian_rrtl/atari.py` | Fachada existente, configuración y selección de ruta |
| `bayesian_rrtl/objects.py` | Reutilizar contratos; solo cambios compatibles si hacen falta |
| `train_atari.py` | CLI, manifiesto, validación y selección de checkpoint v1/v2 |
| `requirements-ocatari.txt` | Dependencia opcional fijada, compatible con las existentes |
| `requirements-ocatari-lock.txt` | Resolución comprobada del entorno OCAtari |
| `configs/bayesian_pong_ocatari_smoke.json` | Prueba end-to-end acotada |
| `configs/bayesian_pong_ocatari_pilot.json` | Piloto posterior, no ejecutado automáticamente |
| `tests/test_ocatari_backend.py` | Backend/flags/objetos con dobles sin ROM |
| `tests/test_pong_relations.py` | Semántica, catálogo y memoria temporal sin ALE |
| `tests/test_ocatari_pipeline.py` | Integración real, CLI y reanudación |
| `experiments/validate_ocatari_pong.py` | Validación acotada que produce evidencia JSON |
| `docs/ocatari_pong.md` | Instalación, ejecución, límites y receta del siguiente juego |
| `docs/validation_ocatari_pong.json` | Evidencia medida; no valores estimados |

Se permiten nombres equivalentes si se conserva esta separación. No convertir el
proyecto en un framework de plugins: registro estático, pequeño y probado.

## 3. Contratos de arquitectura

### 3.1 Backend y fachada

Contrato mínimo conceptual, usando tipos inmutables para los datos devueltos:

```python
class ObjectBackend(Protocol):
    backend_id: str
    actions: tuple[str, ...]
    def reset(self, *, seed: int) -> ObjectFrame: ...
    def step(self, action: int) -> ObjectStep: ...
    def snapshot(self) -> dict: ...
    def restore(self, snapshot: dict) -> None: ...
    def close(self) -> None: ...

# ObjectStep contiene frame, raw_reward, terminated y truncated.
```

`AtariRelationalEnvironment` conserva su API hacia H5:

```text
reset(seed=...) -> hechos relacionales
step(action) -> hechos, reward, raw_reward, terminated, truncated
encoder() / snapshot() / restore() / close()
```

El backend posee OCAtari y sus wrappers. La fachada posee relaciones, su memoria
temporal, transformación de recompensa y profiling. No hacer pasar OCAtari por
`ObjectExtractor.extract(rgb)`: RAM no es una función del RGB recibido.

Mantener el constructor legacy y su inyección `extractor=` operativos. Se puede
encapsular su implementación internamente, pero solo tras pruebas de equivalencia.
Usar importación perezosa de OCAtari: H1–H5 y el backend visual deben funcionar
sin ese paquete instalado.

### 3.2 Registro de juegos

Una especificación por juego/backend debe declarar:

- ID Gym/ALE, acciones esperadas, tamaños/identidades de slots y asignador.
- Constructor y versión de relaciones; opciones soportadas.
- Política de inicio y transformaciones de recompensa admitidas.
- Versión de snapshot y capacidad de reanudación exacta.

Claves iniciales: rutas legacy existentes y `(Pong, ocatari_ram_v1)`.
La fachada no debe acumular nuevos `if game == ...` para detección/relaciones.
Las peculiaridades de Pong viven en `games/pong.py`. Un juego no registrado
debe rechazarse aunque OCAtari lo soporte.

### 3.3 Configuración y compatibilidad

Conservar campos actuales y añadir discriminadores explícitos, por ejemplo:

```json
{
  "game": "Pong",
  "extractor": "ocatari_ram_v1",
  "env_id": "ALE/Pong-v5",
  "relation_schema": "pong_relational_v1",
  "representation": "comparative",
  "include_incomplete_states": true,
  "reward_transform": "sign",
  "reset_policy": "none",
  "frameskip": 4,
  "repeat_action_probability": 0.0,
  "max_episode_steps": 1000
}
```

Resolver campos nuevos ausentes de configuraciones legacy a su comportamiento
histórico. Separar lectura/migración de configuración vieja de configuración
resuelta: no invalidar hashes históricos por añadir defaults. No reescribir
archivos antiguos. Comparar snapshots legacy mediante su contrato/versionado,
no exigir que un dict antiguo tenga las claves de la nueva dataclass.

## 4. Implementación, en orden y con puertas de aceptación

### Paso 0 — Entorno reproducible y línea base

1. Leer instrucciones del repo y revisar cambios existentes sin sobrescribirlos.
2. Registrar intérprete (`sys.executable`) y versiones. La instalación global
   observada previamente de OCAtari no garantiza disponibilidad en `.venv`.
3. Crear/verificar entorno aislado; instalar dependencias opcionales allí, no
   actualizar Python global ni romper los pins del proyecto. Ejecutar `pip check`.
4. Identificar disponibilidad de ROM Pong sin descargarlas de fuentes arbitrarias;
   si falta un requisito/licencia, documentar el bloqueo y procedimiento oficial.
5. Fijar OCAtari 2.2.1 y comprobar su fuente instalada; registrar hash de `core.py`
   y del extractor RAM Pong. Si no es compatible, explicar la incompatibilidad
   antes de sustituir versiones y volver a verificar el contrato completo.
6. Ejecutar pruebas H0–H7 disponibles y registrar fallos/skips previos.

Puerta: imports y reset/step de Pong real funcionan en el mismo intérprete de
las pruebas; dependencias no conflictivas; evidencia de línea base. No afirmar
validación ALE si solo funcionaron dobles.

### Paso 1 — Compatibilidad OCAtari y slots Pong

1. Implementar backend/protocolo/registro con OCAtari opcional.
2. Normalizar el orden de flags en un único punto de compatibilidad versionado.
   La fuente inspeccionada de 2.2.1 retorna
   `(obs, reward, truncated, terminated, info)`, contrario al contrato Gymnasium
   que consume H5. No inferir el orden mirando valores booleanos; no invertirlo
   globalmente para futuras versiones. Rechazar versiones no verificadas.
3. Prueba con entorno subyacente simulado: los cuatro pares de flags, incluyendo
   solo terminal, solo truncación y ambos. Prueba real con `max_episode_steps=1`
   debe producir truncación, no terminal. Probar terminal real por fin de partida
   con un rollout aleatorio acotado independiente del entrenamiento.
4. Mapear slots RAM de Pong: índice 0 Player→player, 1 Ball→ball, 2 Enemy→enemy.
   Validar número de slots y categoría de cada objeto presente. Esta asignación
   es específica de la versión/juego, no una regla universal para otros juegos.
5. `NoObject`/objeto falso → slot ausente con bbox=None. No usar coordenada cero
   como prueba de ausencia; no descartar un objeto válido situado en un borde.
6. Copiar bbox a floats finitos `(x,y,w,h)` y validar tamaños positivos. No retener
   referencias a instancias mutables OCAtari ni recortar coordenadas sin regla
   explícita. Emitir exactamente tres slots en `ObjectFrame` de 160×210.
7. Capturar objetos tras cada reset y tras cada paso, también en terminal/truncación.

Puerta: flags correctos, acciones correctas, una llamada subyacente por paso,
ausencias explícitas y frames previos inmutables tras avanzar OCAtari.

### Paso 2 — Relaciones Pong versionadas

No llamar a `get_state_comparative_pong` para la representación nueva. Mantener
esa función para legacy. Definir `PongRelationsV1` con estas reglas exactas:

- Centros calculados por `DetectedObject.center`; origen superior izquierdo,
  X crece a la derecha, Y hacia abajo. No invertir Y en unos predicados y en
  otros no. `less/same/more` siempre comparan coordenadas del primer objeto
  con el segundo según esta convención.
- Pares espaciales ordenados: `(player_t,ball_t)`, `(player_t,enemy_t)`,
  `(ball_t,enemy_t)`. Para cada par, relaciones comparativas X/Y si ambos existen;
  `same` si diferencia absoluta ≤4 píxeles, `less` si diferencia <−4 y `more`
  si diferencia >4. Ausencia implica relación omitida, nunca `same` por defecto.
- Seis hechos lógicos `present`: cada uno de player/ball/enemy en t y t−1,
  con True/False explícitos. Tras reset no existe observación anterior.
- Seis relaciones temporales X/Y: cada objeto en t frente a t−1, solo si está
  presente en ambas observaciones consecutivas. Tolerancia temporal cero.
  Son desplazamientos entre decisiones, no velocidad física por frame ALE.
- Dos contactos: `(player_t,ball_t)` y `(ball_t,enemy_t)`. True si los intervalos
  de bbox se superponen o tocan en ambos ejes; False si ambos objetos existen
  y no se tocan; omitir si falta uno. No afirmar que esto detecta toda colisión
  física: con frameskip puede ocurrir entre observaciones.
- Al recibir recompensa original distinta de cero, limpiar el historial temporal
  antes de construir ese estado; sigue siendo transición no terminal salvo flag
  ALE. Después guardar el frame actual como anterior. Si la pelota falta y
  reaparece, nunca compararla con una posición anterior al hueco.
- Conservar relaciones de paletas/presencia aunque falte pelota. No devolver
  estado vacío por no reunir una cantidad histórica de hechos.

El catálogo fijo contiene 20 claves: 6 espaciales, 6 temporales, 6 de presencia
y 2 contactos. Se fija antes de reset/replay y no crece con datos de evaluación.
La versión de relaciones y su configuración son parte del manifiesto/snapshot;
un hash de catálogo por sí solo no detecta cambios de semántica.

Puerta: fixtures cubren arriba/abajo, izquierda/derecha, tolerancias exactas,
contacto/no contacto, primer frame, objeto ausente, punto, reaparición y reset.
El orden de objetos/hechos no cambia el encoding. Guardar/restaurar memoria
relacional produce los mismos próximos hechos.

### Paso 3 — Integrar en H5 y la CLI, sin modificar el aprendizaje

1. Seleccionar el backend y constructor de relaciones desde `AtariConfig`.
2. Validar acciones de config contra backend real antes de construir el agente.
   `validate_pair` valida cotas contra transformación configurada, no solo juego.
3. Transformar recompensa una sola vez. Conservar flags originales normalizados:
   un punto de Pong no termina episodio; límite temporal sí trunca; presupuesto
   del runner solo pausa. Entregar observación final para bootstrap de truncación.
4. Mantener `QTrainingSession` y su calendario de fits sin modificaciones
   algorítmicas. No forzar fit al guardar y no reutilizar un checkpoint Breakout
   como modelo inicial Pong.
5. `train_atari.py`: soportar entrenar, evaluar y reanudar el backend OCAtari.
   Manifiesto registra backend, ID real, versiones/hashes, ROM checksum si ALE
   lo expone, catálogo/semántica, acciones, recompensa, reset y frame skip.
6. Separar tiempos: OCAtari step incluye emulación y detección; no atribuirlo todo
   a emulación ni decir que detección cuesta cero. La conversión a ObjectFrame
   y construcción de relaciones pueden medirse aparte.
7. En esta entrega `--variant corrected` con OCAtari debe fallar explícitamente
   antes de crear salida/entrenar: `run_corrected` sigue ligado al entorno H0.
   No dejar que use visual legacy silenciosamente ni etiquetarlo como pareado.
   BatchTreeAgent/IncrementalBinaryAgent pueden usarse con el adaptador común;
   no es obligatorio ejecutar una campaña de controles en esta entrega.
8. No cambiar el protocolo H7 existente ni sus artefactos. Adaptar su factory a
   backend configurable y el baseline H0 común será una tarea posterior si se
   solicita una comparación de eficacia OCAtari vs baseline.

Puerta: CLI ejecuta Pong BART, guarda resultados coherentes y rechaza opciones
no soportadas; siguen pasando configuraciones/pruebas visuales existentes.

### Paso 4 — Snapshot completo y compatibilidad histórica

Para OCAtari usar una versión nueva de snapshot de entorno, por ejemplo
`atari_relational_v2`, dentro del contenedor de checkpoint actual. Mantener
lectura/restauración de v1 legacy. El runner no puede exigir exclusivamente v1.

Guardar sin handles/renderers:

- Estado completo ALE con RNG (`cloneSystemState`, no solo RAM).
- RNG Gym/ALE y action/observation spaces si se utilizan; configuración de ROM,
  acciones, wrappers y flags/contadores del límite temporal y comprobadores Gym.
- Estado mutable de detección OCAtari: objetos y sus campos temporales; buffers
  habilitados y demás campos que puedan cambiar las próximas detecciones.
- Frame actual copiado, historial y versión del constructor de relaciones,
  `done`, contadores de inicio/profiling y todos los campos del recolector H5.
- Versiones exactas OCAtari/Gymnasium/ALE, hashes de código relevante, configuración
  resuelta y versión de cada contrato. Rechazar incompatibilidades con mensaje útil.

Implementar serialización explícita de datos, no pickle del entorno vivo. Toda
dependencia de atributos privados OCAtari debe quedar confinada al módulo de
compatibilidad y probada para la versión fijada. No inventar campos ni usar
`vars(env)` completo como sustituto del análisis de estado.

Restaurar en un entorno nuevo: inicializar estructuras, restaurar ALE/RNG/wrappers,
restaurar objetos/buffers sin una detección adicional que avance su memoria,
restaurar relaciones/recolector. No ejecutar acciones ocultas durante restore.

Puerta: guardar en mitad de un intercambio, continuar y restaurar en otro entorno
y en otro proceso. Una secuencia fija de al menos 100 acciones produce los mismos
objetos, hechos, flags y recompensas hasta terminar/truncar. Repetir a un paso de
truncación y alrededor de ausencia/reaparición de pelota. En una sesión pequeña
comparar continuación H5: replay/IDs, exploración/RNG, próximos targets, modelos
y draws idénticos, excluyendo tiempos. No declarar reanudación exacta basándose
solo en igualdad de RAM o imagen inmediata.

### Paso 5 — Evidencia y prueba acotada

Crear configuraciones completas, no overrides ambiguos:

| Parámetro | Smoke ejecutable | Piloto preparado, ejecución posterior |
|---|---:|---:|
| Interacciones | 2.048 | 20.480 por semilla |
| Semillas de entrenamiento | 41 | 41, 42, 43 |
| Límite por episodio | 1.000 | 27.000 |
| m | 5 | 20 |
| Burn-in / draws por fit | 20 / 20 | 50 / 50 |
| Replay capacity | 5.000 | 50.000 |
| Batch máximo por acción | 64 | 128 |
| Warmup / fit_interval | 256 / 256 | 512 / 512 |
| max_depth / min_child | 4 / 2 | 4 / 2 |
| gamma / reward bounds | 0.99 / [-1,1] | iguales |
| epsilon inicial / mínimo | 1.0 / 0.1 | iguales |
| epsilon_decay_steps | 50.000 | 50.000 |
| Kernel | grow_prune | grow_prune |
| Remuestreo de acciones repetidas | false | false |

Mantener otros priors explícitos con los defaults actuales, incluido k=2. Usar
validación de desarrollo 310001/310002 y reservar 410001/410002 para test futuro
SIN ejecutarlo. Antes de fijar esas semillas, comprobar que no estén consumidas
en artefactos existentes; si lo están, elegir/documentar un conjunto nuevo.
No usar las semillas de test de H7 ya observadas para reclamar test nuevo.

El developer ejecuta smoke y validación de contratos, no el piloto completo por
defecto. Presupuesto operativo: hasta 10 minutos de entrenamiento smoke; si se
excede, conservar estado/evidencia, medir costo y reportar pendiente, sin rebajar
parámetros silenciosamente. Ajustar el plazo de tests según su costo observado,
sin cancelar una prueba correcta por un timeout arbitrariamente pequeño.

Para diagnóstico MCMC, usar un batch/targets congelados del checkpoint: cuatro
cadenas, 500 burn-in y 1.000 draws por cadena, en un presupuesto separado y
explícito (máximo 10 minutos iniciales). Si no cabe, dejarlo pendiente con tiempos.
Reportar R-hat, ESS bulk/tail y estados constantes; no condicionar el éxito de la
integración a la convergencia. No usar 80/160 muestras totales para exigir ESS≥400.
Ni 50 draws del piloto ni 20 del smoke acreditan calidad de inferencia.

`validate_ocatari_pong` escribe en un directorio nuevo:

- Manifiesto y configuración resuelta con hashes de fuentes.
- Resumen de reset/step, slots, flags y resume; máximo desvío o igualdad exacta.
- Contadores de transiciones, puntos (+/−), terminales/truncaciones, presencia
  por slot, hechos faltantes, distribución por acción y fits realizados.
- Checkpoint final, logs/curva y evaluación de desarrollo opcional, nunca test.
- Tiempo por etapa, tamaño de checkpoint y resultados MCMC disponibles.
- Estado de cada criterio: passed/failed/not_run, motivo y comando reproducible.

Smoke exige 2.048 transiciones registradas, 8 fits con el calendario anterior,
Q/targets finitos, cobertura no nula de las seis acciones, varios estados
codificados y un checkpoint recargable. Si una semilla no contiene puntos,
comprobar puntos/ausencias en el rollout de contrato separado; no inventar
evidencia ni interpretar un retorno cero como fallo de integración.

## 5. Matriz mínima de pruebas

| Grupo | Casos obligatorios |
|---|---|
| Dependencia opcional | Importar núcleo/legacy sin OCAtari; error claro al pedirlo sin instalar |
| Compatibilidad | Versión admitida/rechazada; cuatro pares de flags; una llamada por step |
| Slots | NoObject, bordes, duplicados/categoría inválida, bbox inválido, copia inmutable |
| Relaciones | Las 20 claves; tolerancias; Y coherente; contacto; reset; punto; hueco temporal |
| Config | Juego/backend incompatible, acciones erróneas, recompensa/cotas incompatibles |
| Targets | Terminal sin bootstrap; truncación con observación final; punto no terminal |
| Snapshot | Igualdad futura real entre procesos; límite temporal; objetos/relaciones/RNG |
| CLI | Train/resume/evaluate; evaluación no modifica checkpoint; corrected OCAtari rechazado |
| Legacy | Configs antiguas, snapshots v1, recompensas y hechos de los tres juegos intactos |
| Extensibilidad | Registro de juego ficticio con backend doble, sin cambiar fachada/agente |

Marcar pruebas reales con `ocatari` y/o `atari` e incluir el registro en pytest.ini.
Dobles deben correr sin OCAtari/ROMs. En un entorno básico se puede omitir la
integración con motivo; el comando de aceptación dedicado exige dependencias
y ROM y falla, no hace skip silencioso. No contar tests omitidos como aprobados.

## 6. Comandos que la entrega debe dejar operativos

Desde la raíz del repo, usando siempre el mismo intérprete:

```bash
.venv/bin/python -m pip install -r requirements-ocatari.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests/test_ocatari_backend.py tests/test_pong_relations.py -q
.venv/bin/python -m pytest tests/test_ocatari_pipeline.py -q
.venv/bin/python -m experiments.validate_ocatari_pong --output results/ocatari_pong_validation_new
.venv/bin/python train_atari.py --config configs/bayesian_pong_ocatari_smoke.json --output results/ocatari_pong_smoke_new --eval-split none
.venv/bin/python train_atari.py --resume results/ocatari_pong_smoke_new/final_checkpoint.rrtl --output results/ocatari_pong_resume_new --steps 256 --eval-split none
.venv/bin/python train_atari.py --checkpoint results/ocatari_pong_smoke_new/final_checkpoint.rrtl --eval-split validation
```

Los comandos son el contrato de la implementación futura, no funcionalidades
que este plan afirme ya existentes. No sobrescribir directorios de resultados.

## 7. Incorporar el siguiente juego

La guía debe mostrar que añadir un juego consiste en:

1. Crear su `GameSpec`, ID ALE y catálogo de acciones comprobado.
2. Declarar categorías, cardinalidades máximas y slots estables. Si hay múltiples
   objetos iguales, definir asignación/identidad temporal; no usar orden espacial
   como si fuera identidad física sin declararlo. No reutilizar slots Pong.
3. Implementar constructor de relaciones con catálogo fijo y semántica versionada.
4. Registrar soporte del backend y estado adicional del detector si lo tiene.
5. Añadir fixtures y validación ALE de flags, ausencias, recompensa y resume.
6. Añadir configuración y ejecutar la misma suite de contratos/smoke.

No tocar BART, replay, targets ni el ciclo H5. Si el detector nuevo requiere estado
no cubierto por el snapshot genérico, proporcionar su codec y pruebas; no marcar
ese juego como reanudable hasta demostrarlo. Registrar un juego ficticio en tests
prueba extensibilidad, pero no acredita soporte real de un segundo juego.

## 8. Definición de terminado y entrega

- [ ] Backend opcional OCAtari Pong registrado, versionado y probado con ALE real.
- [ ] Contrato de flags corregido y cubierto por prueba de truncación y terminal.
- [ ] Representación Pong explícita, fija e independiente de funciones legacy.
- [ ] Entrenamiento/evaluación/reanudación por la CLI existente.
- [ ] Reanudación exacta entre procesos, incluidos objetos y memoria temporal.
- [ ] Smoke completo con artefactos y conteos; suite relevante ejecutada.
- [ ] Ruta legacy y checkpoints v1 preservados; regresiones documentadas.
- [ ] Ningún fallback silencioso de corrected/H7 a visual al pedir OCAtari.
- [ ] Guía de instalación y receta del siguiente juego; pins/lock reproducibles.
- [ ] Informe final diferencia pruebas ejecutadas, pendientes, calidad de mezcla
      y aprendizaje. Si falta una puerta obligatoria, entrega parcial, no completa.

Entregar lista de archivos, comandos exactos, conteos de pruebas/skips, directorio
de evidencia, tiempos y problemas conocidos. No subir ROMs ni resultados pesados
al repositorio. Actualizar README con enlace a la guía, sin declarar eficacia ni
soporte OCAtari para juegos aún no implementados.

## Fuentes y alcance de las observaciones

El plan combina inspección del repo local con estas fuentes primarias. La API
instalada y sus pruebas mandan sobre ejemplos README; master puede cambiar:

- [OCAtari: construcción, step y estado interno](https://raw.githubusercontent.com/k4ntz/OC_Atari/master/ocatari/core.py).
- [Pong RAM: slots, categorías y ausencia de objetos](https://raw.githubusercontent.com/k4ntz/OC_Atari/master/ocatari/ram/pong.py).
- [Repositorio OCAtari](https://github.com/k4ntz/OC_Atari).

Las decisiones de arquitectura, relaciones, presupuesto, recompensa y pruebas
de este documento son especificaciones del proyecto, no recomendaciones
atribuidas al repositorio OCAtari.
