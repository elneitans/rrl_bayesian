# Auditoría de detectores OCAtari: bloque C

Auditoría del 14/10/2026, anterior a los adaptadores C13/C14. Se inspeccionaron las fuentes de **OCAtari 2.2.1**, con **Gymnasium 1.1.1 y ALE 0.10.2**. Las observaciones reales y trazas por acción están en [ocatari_detector_audit_c.json](ocatari_detector_audit_c.json). No se actualizaron dependencias ni detectores upstream.

La wheel inspeccionada tiene SHA256 `b343da0272d90446ac373c2bc9d2cf6cb3484e96f57cb786809f2a7e96291d93`. Fuentes RAM:

| Módulo | SHA256 |
| --- | --- |
| `ocatari/ram/breakout.py` | `cd4de44322ae4df792e3a5db0bd482a37cb4766206af5392ae2620329d18d12f` |
| `ocatari/ram/demonattack.py` | `bc96524f70f5a025471fc504f64ca86d25cd15bcc0ea97a76ac63d2720a10a70` |

## C13: Breakout

Con `hud=False` hay **110 slots**: 0 `Player`, 1 `Ball`, y 2–109 `Block`. Los bloques se indexan `2 + fila*18 + columna`; algunos objetos representan segmentos contiguos de ladrillos y tienen anchura variable. El adaptador valida los 110 slots, pero publica solamente `player` y `ball` para el agente. Los ladrillos completos se conservan en el snapshot.

`NoObject` es una ausencia certificada y es false-y. Otra categoría false-y no se acepta como ausencia. En el detector, la pelota desaparece si RAM[101] es cero o su posición calculada supera el límite inferior; al reaparecer se crea una `Ball`. No se inventan coordenadas para ausencias.

El catálogo ALE observado, en orden, es **NOOP, FIRE, RIGHT, LEFT**. La prueba real verifica que RIGHT y LEFT desplazan horizontalmente al jugador y que FIRE lanza la pelota; reset no ejecuta FIRE. Cada acción ocupa una decisión y pasa una sola vez por frameskip=4.

La representación `breakout_primary_v1` tiene **2 objetos y 11 relaciones**: 2 posiciones relativas jugador–pelota, 4 movimientos, 4 presencias actual/anterior y 1 contacto. Las comparativas toman `less/same/more`; las lógicas `True/False`. Además, cualquiera puede quedar **no observada**: no es un cuarto valor comparativo ni equivale a falso. Se usan centros, tolerancia espacial 4 píxeles y temporal 0, explícitas en configuración/manifiesto. Contacto significa solapamiento o toque de cajas en ambos ejes, no colisión certificada por ALE.

Al iniciar no hay historia anterior. Desaparición/reaparición impide calcular movimiento de pelota; una caída del contador de vidas invalida sólo su historia, incluso si el salto de frameskip oculta la desaparición. La historia de la paleta se conserva. Destruir un ladrillo no borra historia por producir recompensa. Limitación deliberada: el agente no observa directamente la distribución de ladrillos.

## C14: Demon Attack

La inicialización y los máximos declarados coinciden en **30 slots**. Un comentario upstream menciona 12 proyectiles enemigos, pero el máximo declarado y el array incluyen **14**; se usa el contrato real:

| Índices | Categoría | Identidad publicada |
| --- | --- | --- |
| 0 | Player | player |
| 1 | PlayerMissile | player_missile |
| 2–9 | Enemy | enemy_0…enemy_7 |
| 10–15 | EnemyPart | enemy_part_0…enemy_part_5 |
| 16–29 | EnemyMissile | enemy_missile_00…enemy_missile_13 |

Se conservan incluso slots poco utilizados. El detector reutiliza índices según fases RAM; un índice no certifica identidad física. No se ordenan enemigos por posición ni se publican velocidades de enemigos/proyectiles. Su presencia anterior describe ocupación del slot; la desaparición impide cualquier continuidad física. `EnemyPart` ocupa slots propios. Una categoría inesperada falla con el índice y la categoría esperada: no se recorta ni se convierte silenciosamente.

`PlayerMissile` se inicializa como objeto y su presencia por sí sola **no prueba que haya un disparo activo**. El detector tiene peculiaridades de escritura/limpieza de slots de proyectiles; no se corrigen upstream en este bloque. La prueba de disparo compara posiciones posteriores bajo FIRE y NOOP, además de comprobar los nombres de acciones.

Acciones reales, en orden: **NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE**. En el primer step el jugador cambia de su posición inicial provisional a la posición RAM. Por ello el test de movimiento compara ramas con el mismo estado y NOOP de referencia, en vez de interpretar esa inicialización como desplazamiento por LEFT. Tras estabilizar con acciones explícitas se observó x=87, RIGHT=119 y LEFT=55 en ocho decisiones; FIRE también cambia la trayectoria del proyectil respecto a NOOP.

`demonattack_slots_v1` tiene **30 objetos y 134 relaciones**: 58 posiciones relativas del jugador a los otros 29 slots, 2 movimientos del jugador, 60 presencias actual/anterior y 14 contactos jugador–proyectil enemigo. Valores comparativos/lógicos y ausencia de observación siguen C13. El crecimiento es lineal. Sólo se calcula movimiento del jugador cuando hay dos observaciones presentes y tamaño consistente. No se interpreta un cambio de slot como velocidad del mismo enemigo.

No hay reset de historia por recompensa, oleada ni pantalla sin enemigos. Perder una vida no impone terminal. No se reportan “puntos ganados/perdidos de Pong”.

## Recompensa, flags y estado mutable

Ambos juegos usan `sign(raw_reward)` en entrenamiento y recompensa ALE bruta en evaluación, sin shaping. Con magnitudes distintas, maximizar recompensa recortada no equivale exactamente a maximizar puntaje bruto. `terminated` elimina bootstrap; truncación lo conserva según el parámetro común A/B. La pérdida de vida no cambia esa regla. El core fijado devuelve los dos flags en orden distinto al de Gymnasium; el backend conserva la normalización auditada, comprobada por las pruebas de targets.

Para las categorías sin HUD, los campos mutables observados son `_rgb`, `_xy`, `wh`, `_prev_xy`, `_orientation`, `hud`, `_visible`; `NoObject` añade `nslen`. No se admiten campos HUD como `_value` ni campos nuevos no auditados. El codec de cada juego guarda **todos** los slots y esos campos, no sólo los `Fact`. No se identificó memoria mutable adicional del módulo detector que deba seguirse fuera de esos objetos.

## Snapshots y alcance de la validación

Los nuevos juegos usan `ocatari_ram_v2`, `ocatari_ram_state_v2`, `slot_relations_memory_v1` y el entorno `atari_common_v3`. El manifiesto vincula acciones, catálogo, ROM, versiones, fuentes del detector, objetos comunes, helpers, codec y adaptador. Se preservan estado ALE, wrappers, RNG, contadores, objetos completos y memoria relacional. Se rechazan hashes incompatibles; el contrato Pong v1 y el visual legacy conservan sus identificadores.

Se habilitó `exact_resume=True` **después** de pasar continuidad real en proceso separado para ambos juegos: prefijo de 32 decisiones y 64 decisiones posteriores, con igualdad de objetos y campos, `Fact`, recompensa bruta/recortada, vidas y flags. Se repitió la prueba con el código final. La evidencia local corresponde a macOS; `slurm/14_atari_preflight.sbatch` reproduce esta comprobación en Linux. Un snapshot no se declara portable entre arquitecturas o runtimes distintos.
