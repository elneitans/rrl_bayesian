# Diagnóstico de estabilidad de BART cada 1.024 decisiones en Breakout

El reajuste cambia con frecuencia cuál acción maximiza Q. Esto causa dos fallas observables: bloqueo antes de sacar y pérdida de rendimiento con la pelota en juego. En la semilla 733102, repetir solamente el muestreo del ajuste de 400.000, con las transiciones y objetivos TD guardados, cambió el puntaje de 34 a 2–3. En otros checkpoints se observa además que el bootstrap reutiliza Q por encima de la cota teórica. La importancia de la inflación varía por semilla y momento.

## Alcance

Revisé las diez corridas pareadas a 500.000 decisiones, los registros íntegros de 5 millones de interacciones y sus 7.320 reajustes, las 180 evaluaciones de desarrollo y los 25 checkpoints de la variante cada 1.024. Verifiqué los hashes de los registros, los checkpoints y los lotes de ajuste de esa variante. Contrasté las 180 evaluaciones individuales con sus identidades, estados y resúmenes. En las dos evaluaciones intermedias y las diez finales por política, coincidieron el retorno, las acciones agregadas, los pasos, el registro de `FIRE` y los demás resultados salvo la semilla y el tiempo de cómputo. ALE usó cuatro frames por decisión, sin acciones pegajosas ni acciones aleatorias al iniciar. La unidad para comparar replicaciones es la semilla de entrenamiento. El entrenamiento conservó la exploración con ε = 0,1 desde 50.000 decisiones y la evaluación usó la acción greedy.

| Semilla | 100k | 200k | 300k | 400k | 500k |
| --- | ---: | ---: | ---: | ---: | ---: |
| 733101 | 5 | 1 | 15 | 0 | 0 |
| 733102 | 1 | 5 | 0* | 34 | 4 |
| 733103 | 0 | 0 | 0 | 3 | 3 |
| 733104 | 1 | 11 | 0* | 0* | 2 |
| 733105 | 2* | 0 | 0 | 3 | 0 |

\* Episodio truncado a 27.000 decisiones. En 733105 a 100k hubo puntos antes de quedar bloqueado después de perder una vida. Por checkpoint, tres de las 25 políticas de 1.024 jamás sacaron y otras ocho terminaron con cero después de sacar. Al final, 1.024 consiguió 1,8 puntos de media y 512 consiguió 0. Con 1.024 las cinco políticas finales completaron sus partidas; con 512 se bloquearon cuatro de cinco. Durante el entrenamiento con exploración, los retornos medios por episodio y ventana de 100.000 decisiones se mantuvieron entre 2,78 y 3,33 con 1.024. No hubo truncaciones de entrenamiento en esas cinco corridas.

[Campaña verificada](../results_cluster/rrl_breakout_bart_cadence_500k_20261112_01/pilot/report_verified.json) · [Registros verificados y reducidos](../../tmp/bart_cadence_diagnostico_20261114/logs_resumen.json)

## Qué cambió dentro de las partidas

Congelé los modelos y repetí las evaluaciones en el emulador original. Verifiqué la caída 34 → 4 con una sola semilla de evaluación, que apliqué a ambos checkpoints. En la prueba de intercambio, uno de los modelos decide cuando el detector ve la pelota y el otro decide el resto de la partida. Los modelos reciben las mismas observaciones y no aprenden durante la evaluación. Las columnas indican qué checkpoint toma cada grupo de decisiones.

| Semilla | Checkpoints | Ambos del primero | Ambos del segundo | Sin pelota del primero y con pelota del segundo | Sin pelota del segundo y con pelota del primero |
| --- | --- | ---: | ---: | ---: | ---: |
| 733101 | 300k / 400k | 15 | 0 | 0 | 15 |
| 733102 | 400k / 500k | 34 | 4 | 5 | 53 |
| 733105 | 400k / 500k | 3 | 0 | 4 | 3 |

En 733101, los 15 puntos siguieron íntegramente al modelo que controlaba la pelota visible. En 733102, cambiar solo ese componente recuperó el puntaje y sostuvo peloteos más largos. Con el modelo original de 400k, la pelota fue visible durante 196, 113, 180, 159 y 229 decisiones en sus cinco vidas. En 500k fueron 65, 19, 147, 19 y 19. En 733105, ambos intercambios recuperaron puntos. El saque en la primera decisión es una métrica insuficiente: en 500k sacó inmediatamente en las cinco vidas y obtuvo cero.

El bloqueo de 733102 a 300k se aisló directamente. El reajuste publicado, en la decisión 299.520, cambió el margen inicial de `FIRE` frente a su mejor rival de +0,98 a −0,77 unidades Q. Eligió `LEFT` y luego se quedó en `NOOP`. Al usar el modelo inmediatamente anterior, de 298.496, la partida terminó con dos puntos. También se obtuvieron dos puntos al cambiar solo la política sin pelota visible y conservar la de 300k en juego.

[Trazas de las partidas](../../tmp/bart_cadence_diagnostico_20261114/policy_traces.jsonl) · [Resultados del intercambio](../../tmp/bart_cadence_diagnostico_20261114/intercambio.jsonl)

## De dónde sale el cambio en la política

Cada reajuste toma hasta 128 transiciones por acción de un replay de 50.000, calcula objetivos con el modelo inmediatamente anterior y ajusta desde árboles de una hoja con 50 sweeps de burn-in y 50 retenidos. Se publica el nuevo conjunto de modelos y la política elige el mayor Q medio. Al comparar el modelo publicado con el de un reajuste antes, usando el mismo conjunto de estados de su replay, cambió la acción greedy en una mediana del **77,2 %** de las visitas con pelota visible en los 25 checkpoints. En los cinco checkpoints finales cambió entre **69,3 % y 84,5 %**. Cada comparación pondera por frecuencia de visita en su propio replay. La mediana de las 25 brechas medianas entre las dos mejores acciones fue **0,18** unidades Q.

Para aislar el azar MCMC, recuperé los lotes exactos por acción de la semilla 733102, validé sus hashes y reconstruí sus objetivos contra el modelo previo para las transiciones todavía presentes en el replay. Repetí tres reajustes desde una hoja por checkpoint con el mismo presupuesto, la misma calibración del ruido y semillas MCMC nuevas. Evalué cada nuevo modelo en el mismo episodio de desarrollo.

| Checkpoint | Puntaje publicado | Puntajes de los tres reajustes | Acción inicial publicada | Acciones iniciales de los reajustes |
| --- | ---: | --- | --- | --- |
| 300k | 0 | 0, 0, 0* | `LEFT` | `RIGHT`, `NOOP`, `RIGHT` |
| 400k | 34 | 3, 2, 3 | `FIRE` | `RIGHT`, `FIRE`, `FIRE` |
| 500k | 4 | 2, 3, 1 | `NOOP` | `NOOP`, `NOOP`, `NOOP` |

\* En los tres reajustes de 300k no hubo `FIRE` en las primeras 1.500 decisiones, el límite administrativo de la prueba. La evaluación original sí llegó a 27.000.

En 400k, las nuevas cadenas cambiaron la acción en **50,7–76,9 %** de las 877 visitas con pelota visible del episodio que originalmente dio 34. Dos cadenas mantuvieron `FIRE` como primera acción y terminaron igualmente con 2–3 puntos. El modelo anterior al de 400k también sacó y llegó a dos puntos antes de bloquearse dentro de las 1.500 decisiones observadas. El mismo lote y los mismos objetivos pueden producir políticas de calidad muy distinta bajo el presupuesto MCMC usado. Estos contrastes fueron elegidos después de observar los puntajes y cuantifican el mecanismo en esa semilla.

[Resultados de los nueve reajustes](../../tmp/bart_cadence_diagnostico_20261114/reajuste_fijo.jsonl) · [Auditoría de los 25 checkpoints](../../tmp/bart_cadence_diagnostico_20261114/checkpoints_resumen.jsonl) · [Ajuste en línea](../bayesian_rrtl/online/learners.py#L251)

## Cobertura y sobreestimación

- **Cobertura del saque.** En los 25 checkpoints, la mediana fue de 198 visitas al estado codificado de inicio en el replay y dos coincidencias exactas en el lote de 512 filas entre las cuatro acciones. En seis checkpoints, el lote de `FIRE` no tuvo ninguna coincidencia. En 733102 a 300k había 123 transiciones de `FIRE` desde el estado inicial en el replay y se usó una. Los objetivos TD reconstruidos para ese estado apenas favorecían `NOOP` sobre `FIRE`, 60,46 frente a 60,41, mientras que el modelo ajustado favoreció `LEFT`. El conteo exacto mide la cobertura del estado, y los árboles también generalizan con filas de otras configuraciones. Los tres reajustes del mismo lote quedaron bloqueados durante la ventana observada, por lo que ese caso no se resolvió cambiando la semilla MCMC.
- **Información para devolver la pelota.** Las 11 relaciones discretizan dirección y comparaciones de posición, sin magnitud de distancia o velocidad, ni posición absoluta. En la traza de 733102 a 500k, las decisiones 175 y 242 tuvieron las mismas 11 entradas y los mismos Q, aunque la pelota estaba a alturas y = 112 y y = 184, y la separación horizontal entre centros era −39 y −7 píxeles respectivamente. El modelo eligió `NOOP` en ambas. El ejemplo verifica pérdida de información; su efecto causal sobre el puntaje requeriría cambiar la representación con los demás componentes fijos.
- **Inflación por bootstrap.** Con la recompensa de aprendizaje en [−1, 1] y γ = 0,99, el retorno descontado verdadero está dentro de [−100, 100]. BART puede salir de ese rango y el objetivo `r + γ max Q_anterior` transmite el exceso al reajuste siguiente. En las últimas 100.000 decisiones de 1.024, quedaron fuera de la cota el **57,83 %** de los objetivos en 733104 y el **28,47 %** en 733105. En su último ajuste, los 512 objetivos de cada una superaban 100 y el máximo Q sobre los estados visitados del replay llegó a 126,05 y 144,18. En 733102, todos los objetivos de los tres lotes auditados estaban dentro de la cota y el bloqueo de 300k ocurrió antes de que esa semilla registrara un objetivo fuera de escala. La inflación está confirmada en algunas semillas. El rol del azar MCMC se identifica directamente en el cambio de 34 a 2–3 con el mismo lote.

[Predicciones y soporte por checkpoint](../../tmp/bart_cadence_diagnostico_20261114/checkpoints_resumen.jsonl) · [Construcción de los objetivos](../bayesian_rrtl/targets.py#L38) · [Representación de Breakout](../bayesian_rrtl/games/breakout.py#L15)

El diagnóstico usó checkpoints existentes, evaluaciones congeladas y nueve reajustes de datos fijos, con menos de 20.000 decisiones nuevas de evaluación en total. No inicié entrenamientos nuevos a 1 millón. El diseño aísla el efecto del muestreo sobre el puntaje y de la acción greedy dentro de la partida. El efecto individual de cambiar la cobertura del lote, los objetivos TD o la representación sobre una corrida completa queda abierto.
