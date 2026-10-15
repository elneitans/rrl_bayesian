# Guía computacional: Pong, RRTL y BART en `bayesian_rrtl`

Estado de esta guía: 29 de septiembre de 2026. Describe **el código y el piloto existentes**, no un agente que ya sepa jugar. El resultado empírico más importante es que, al final del piloto, ni el baseline ni BART anotaron un punto en las 120 partidas de validación final. Eso **no identifica todavía la causa** del fallo.

## 1. El problema, sin asumir conocimientos previos

Pong enfrenta nuestra paleta (la derecha) con la paleta de la máquina (la izquierda). Una decisión elige una de seis acciones y el emulador avanza cuatro fotogramas. Queremos aprender una política \(\pi\) que, al ver el estado, escoja una acción útil. En aprendizaje por refuerzo no entregamos la trayectoria correcta de la paleta: el agente prueba acciones y recibe una recompensa cuando se marca o se pierde un punto. [ALE documenta la paleta, las seis acciones y el sistema de puntos](https://ale.farama.org/environments/pong/).

La cadena computacional es:

```text
acción → emulador Atari → objetos detectados → relaciones Fact → estado relacional
       → estimaciones Q(s,a) para seis acciones → siguiente acción
       ↘ recompensa y transición → actualización del modelo
```

Una **interacción** es una decisión del agente, no una partida ni un fotograma. Una partida termina cuando termina el juego; una interacción puede ocurrir centenares de veces dentro de ella.

## 2. Qué recibe el agente de Pong

### Entorno y OCAtari

La ruta nueva crea `OCAtari("ALE/Pong-v5", mode="ram", hud=False, obs_mode="ori")`, con `frameskip=4` y probabilidad de acciones repetidas (*sticky actions*) cero. OCAtari usa la RAM para localizar objetos; aunque configura una observación visual original, el aprendiz **no usa los píxeles como entrada**. Consume los objetos de `env.objects`. `hud=False` excluye el marcador de la lista de objetos. Véanse [el adaptador del emulador](../ocatari_backend.py), [el registro de Pong](../games/pong.py) y la [documentación de OCAtari](https://oc-atari.readthedocs.io/en/latest/ocatari/core.html).

Conservamos tres posiciones estables: `player`, `ball`, `enemy`. Cada objeto presente trae una caja \((x,y,w,h)\), donde \((x,y)\) es la esquina superior izquierda, **no el centro**. Calculamos el centro como \((x+w/2,\,y+h/2)\). Cuando un objeto falta, su casilla permanece identificable pero no inventamos una caja. Véanse [objetos](../objects.py), [Pong](../games/pong.py) y la [definición de cajas de OCAtari](https://oc-atari.readthedocs.io/en/latest/ocatari/ram.html).

Las seis acciones, en el orden que exige ALE, son `NOOP`, `FIRE`, `RIGHT`, `LEFT`, `RIGHTFIRE` y `LEFTFIRE`. Los nombres son los del joystick de ALE; no conviene interpretar `RIGHT` como «mover la paleta hacia la derecha de la pantalla» sin comprobar su efecto en Pong. El adaptador usa exactamente ese catálogo y valida que coincida con el del entorno. [ALE enumera esos índices](https://ale.farama.org/environments/pong/).

### De objetos a `Fact`

Un `Fact` representa una relación con una **clave** (tipo, nombre y objetos implicados) y un **valor**. Por ejemplo:

```text
comparative / y / player_t / ball_t   = less
comparative / y / ball_t / ball_t-1   = more
logical     / present / ball_t        = True
logical     / incontact / player_t / ball_t = False
```

En coordenadas de pantalla, `y` crece hacia abajo. Por tanto, el primer ejemplo dice que el centro de la paleta está por encima del centro de la pelota; el segundo dice que la pelota bajó desde la decisión anterior. `incontact=False` es una observación negativa explícita; si falta la pelota, esa relación no se emite y está **ausente**, que es distinto de falso.

El catálogo fijo de Pong contiene 20 posibles relaciones: seis espaciales entre pares de objetos, seis temporales (desplazamiento en `x`/`y` de cada objeto), seis indicadores de presencia actual/anterior y dos contactos entre cajas. Las comparaciones espaciales tienen tolerancia de 4 píxeles alrededor de `same`; las temporales, tolerancia cero. El contacto comprueba solapamiento o toque de cajas en ambos ejes, **sin** margen extra de 4 píxeles. Por `frameskip=4`, un rebote puede haber ocurrido entre dos observaciones aunque `incontact` nunca aparezca verdadero. Al anotarse un punto, se reinicia la memoria temporal para no comparar posiciones de rallies diferentes. Todo esto está en [PongRelationsV1](../games/pong.py).

[El `RelationalEncoder`](../encoding.py) convierte esos `Fact` a columnas categóricas de orden estable y guarda además una máscara de observado. No trata `less/same/more` como magnitudes numéricas ordenadas. Así, permutar el orden de los `Fact` no cambia la entrada del modelo, y «falso» no se confunde con «desconocido».

La comparación del piloto incluye otra ruta llamada `visual`: usa la extracción visual heredada y `PongNoFrameskip-v4`. No cambia únicamente el detector: también difieren el esquema de relaciones, la admisión de estados incompletos, el reinicio mediante `FIRE` y la transformación de recompensa. Por ello, una diferencia entre `visual` y `ocatari` **no sería una estimación causal aislada** del efecto de OCAtari. Véanse [adaptador Atari](../atari.py) y [protocolo del piloto](../../configs/comparison_pong_pilot.json).

## 3. Qué intenta optimizar y qué recompensa usamos

Para una política \(\pi\), el valor de tomar \(a\) en \(s\) es el retorno descontado esperado:

\[
Q^{\pi}(s,a)=\mathbb E_{\pi}\!\left[\sum_{k=0}^{T-t-1}\gamma^k\,\widetilde r_{t+k}\mid s_t=s,a_t=a\right],\qquad \gamma=0.99.
\]

Los dos agentes aproximan el valor óptimo \(Q^*(s,a)=\max_\pi Q^\pi(s,a)\), y la política de evaluación elige una acción con el mayor \(\widehat Q(s,a)\). Durante entrenamiento usan exploración \(\varepsilon\)-greedy: con probabilidad \(\varepsilon\) prueban una acción uniforme; si no, eligen una acción de valor máximo, rompiendo empates al azar. \(\varepsilon\) decae de 1 a 0,1 en 50.000 decisiones; a las 20.480 es aproximadamente 0,389. Véanse [comparación](../comparison.py), [agente BART](../agent.py) y [adaptador baseline](../baseline_adapter.py).

La recompensa **cruda** de ALE es positiva al anotar, negativa al perder un punto y cero entre puntos. Para aprender usamos una recompensa transformada \(\widetilde r\):

| Ruta | Recompensa de entrenamiento | Entre puntos | En un punto a favor / en contra |
| --- | --- | --- | --- |
| OCAtari | \(\operatorname{sign}(r_{\rm ALE})\) | 0 | +1 / −1 |
| Visual heredada | \(\operatorname{sign}(r_{\rm ALE})+0.1\) | +0,1 | +1,1 / −0,9 |

La evaluación informa **retorno crudo** \(\sum_t r_{\rm ALE,t}\), no la suma de recompensas transformadas. No hay recompensa explícita por tocar la pelota, acercarse a ella, ganar una partida ni terminar rápido. «Nunca perder» y «ganar en el menor tiempo posible» **no son objetivos codificados literalmente**. El descuento prefiere, a igualdad de lo demás, recompensas positivas más tempranas y penalizaciones negativas más tempranas; sus efectos combinados sobre duración no equivalen a una penalización explícita por tiempo. Además, el +0,1 visual premia sobrevivir cada paso y puede incentivar rallies largos. La ruta OCAtari no da ese incentivo intermedio. Véanse [transformaciones](../atari.py) y [configuración](../../configs/comparison_pong_pilot.json).

En una partida completa normal, el juego termina al llegar uno de los lados a 21 puntos; no existe un resultado de **empate** como clase de salida de este protocolo. Un 0–21 da retorno −21; un 21–20 da +1: el retorno crudo cuenta diferencia de puntos, no directamente tasa de victorias. Una partida cortada por el límite de pasos puede tener marcador igualado, pero se registra como `truncated`/`horizon_limited`, **no** como empate ni victoria. El umbral de 21 está descrito en el [artículo original de RRTL](https://arxiv.org/abs/2203.13599), y los resultados observados del piloto son consistentes con ello. Véanse [comparación](../comparison.py) y [reporte de validación](../../results_cluster/pong_pilot_linux_01/report_validation.json).

## 4. Baseline: árbol relacional con Q-learning

### Intuición

El *baseline* es el RRTL corregido que heredamos del repositorio original. No es BART ni una regresión supervisada convencional. Tiene **un árbol por acción**. Una hoja guarda un número \(Q\): «qué tan conveniente ha parecido esta acción en estados que llegan aquí». Al comienzo, todos los árboles son una única hoja con valor cero. Una regla del árbol pregunta algo como «¿la paleta está por encima de la pelota?»; las respuestas encaminan el estado hacia hojas distintas. Las reglas comparativas históricas tienen ramas `less/same/more`, y las lógicas ramas `True/False`.

Ejemplo hipotético: si el árbol de `LEFT` se divide por `y(player,ball)`, puede asignar valores diferentes a «paleta arriba», «alineada» y «paleta abajo». **No afirmamos que ésa sea una regla aprendida por el piloto**: sirve para explicar el mecanismo.

### Actualización y construcción del árbol

Por cada transición \((s,a,\widetilde r,s')\), sólo se modifica el árbol de la acción ejecutada. Su hoja actual realiza la actualización temporal (*TD/Q-learning*):

\[
y_t=\widetilde r_t+\gamma b_t\max_{a'}\widehat Q(s_{t+1},a'),\qquad
Q_{\rm nuevo}=Q_{\rm viejo}+\eta(y_t-Q_{\rm viejo}),\quad\eta=0.025.
\]

Aquí \(b_t=0\) al terminar realmente la partida; ante un corte artificial por tiempo, \(b_t=1\) si existe observación final, porque el estado no es terminal del juego. Después de actualizar \(Q\), el árbol acumula estadísticas y considera una división. Véanse [targets](../targets.py), [correcciones del baseline](../baseline.py) y [adaptador de Pong](../baseline_adapter.py).

Para cada relación candidata, el nodo lleva contadores y sumas de cuadrados \(J\) de los valores \(Q\) observados. Compara la dispersión interna de las futuras ramas con la dispersión total, de forma esquemática:

\[
F_\ell=\frac{\sum_{c\in\text{categorías de }\ell}J_c}{J_{\rm total}}.
\]

Un \(F_\ell\) menor sugiere que separar por esa relación deja ramas más homogéneas. El código selecciona por el menor valor \(p=F_{n-1,n-1}^{\rm CDF}(F_\ell)\) y sólo divide si supera el mínimo de muestras y \(p\le 10^{-4}\), hasta profundidad 4. Este «F-test» es **la heurística histórica del algoritmo**, no una prueba confirmatoria bien calibrada: los \(Q\) provienen de actualizaciones sucesivas correlacionadas, se exploran muchos candidatos y no se corrige multiplicidad. Una relación ausente en un nodo interno devuelve su \(Q\) de respaldo; el árbol no crea una rama de «ausente». Véanse [árbol original](../../relational_regresion_tree.py) y [corrección](../baseline.py).

La variante `rrtl_legacy` se preserva para reproducibilidad, pero la comparación de Pong usa `corrected_common`, que corrige la semántica de TD, terminal/truncación y registros sin cambiar la familia de árbol heredada. El [adaptador](../baseline_adapter.py) alimenta al árbol los mismos `Fact` de OCAtari que se envían a BART y normaliza `True/False` sin convertir «ausente» en falso.

## 5. BART: suma bayesiana de árboles

### De un árbol a muchos

Un árbol único puede representar reglas por partes, pero una decisión estructural temprana puede condicionar mucho sus predicciones. BART (*Bayesian Additive Regression Trees*) modela un valor continuo como **suma de muchos árboles pequeños**:

\[
z_i=\sum_{j=1}^{m}g(x_i;T_j,M_j)+\epsilon_i,\qquad
\epsilon_i\sim\mathcal N(0,\sigma^2),\qquad m=20.
\]

\(x_i\) es el estado relacional codificado, \(T_j\) la estructura del árbol, \(M_j\) sus valores de hoja y \(z_i\) el *target* normalizado. Cada árbol aporta una corrección pequeña. Se ajusta **un BART independiente por acción**: no hay un único modelo compartido que reciba la acción como variable. La idea original de suma de árboles y *backfitting* bayesiano proviene de [Chipman, George y McCulloch](https://arxiv.org/abs/0806.3286). Nuestra implementación concreta está en [árbol](../tree.py), [reglas](../rules.py) y [ensemble](../ensemble.py).

Para mantener la escala del prior, los valores objetivo se normalizan como \(z=(y-c)/(U-L)\). En OCAtari, \(r\in[-1,1]\) y \(\gamma=0,99\) dan cotas teóricas \(L=-100,U=100,c=0\); por ejemplo, un target de +1 se convierte en 0,005. Al predecir, la escala se revierte **una vez para la suma completa**, no por árbol. Véase [FixedScale](../priors.py).

### Qué significa «bayesiano» aquí

No fijamos un árbol «ganador». Especificamos distribuciones previas y muestreamos configuraciones plausibles dados los datos. Para un nodo a profundidad \(d\), la probabilidad previa de dividir es \(0,95/(1+d)^2\) cuando hay reglas admisibles; máximo de profundidad 4 y al menos dos observaciones por hijo. Los valores de hoja tienen prior \(\mu\sim\mathcal N(0,\tau^2)\) con \(\tau=0,5/(k\sqrt m)\), \(k=2\). La varianza residual tiene prior gamma inversa, \(\sigma^2\sim IG(\nu/2,\nu\lambda/2)\), con \(\nu=3\) y \(\lambda\) calibrada del primer lote de entrenamiento de cada acción. Véanse [priors](../priors.py) y [verosimilitud gaussiana](../likelihood.py).

Su objetivo estadístico en un lote fijo es aproximar la distribución posterior:

\[
p(T_{1:m},M_{1:m},\sigma^2\mid X,z)\propto
\prod_i\mathcal N(z_i;\sum_j g(x_i;T_j,M_j),\sigma^2)
\left[\prod_j p(T_j)p(M_j\mid T_j)\right]p(\sigma^2).
\]

Esto es distinto del **objetivo RL** de maximizar retorno futuro: la posterior intenta ajustar los *targets* de Bellman que le entrega el agente. No minimizamos directamente la cantidad de partidas perdidas al ejecutar MCMC.

En una hoja que recibe \(n\) residuos parciales con suma \(S\), la posterior gaussiana condicional es:

\[
v=\left(\frac{n}{\sigma^2}+\frac{1}{\tau^2}\right)^{-1},\qquad
\mu\mid\text{datos},T,\sigma^2\sim
\mathcal N\!\left(v\frac{S}{\sigma^2},v\right).
\]

Para buscar estructuras, el piloto propone `grow` (abrir una hoja) y `prune` (cerrar dos hojas hermanas) y acepta/rechaza con Metropolis–Hastings usando verosimilitud marginal, prior de estructura y probabilidades de propuesta directa/inversa. Por cada barrido, se actualiza cada árbol usando residuos que excluyen su propia contribución; luego se muestrea \(\sigma^2\). `change/swap` y la variante General BART lineal existen en el repositorio, pero **no fueron el kernel/modelo del piloto Pong**. Véanse [propuestas](../proposals.py), [sampler](../sampler.py), [ensemble](../ensemble.py) y [General BART](../general_bart.py).

El piloto hace 50 barridos de *burn-in* y conserva 50 muestras posteriores en cada ajuste. Para decidir, estima \(\widehat Q(s,a)\) promediando la suma de los 20 árboles de cada una de esas 50 muestras. No ejecuta *Thompson sampling*: la exploración sigue siendo \(\varepsilon\)-greedy sobre la **media posterior**. Tampoco se debe interpretar esta posterior como incertidumbre plenamente calibrada sobre \(Q^*\): los objetivos proceden de *bootstrapping* con un modelo estimado. Véanse [BARTPosterior](../ensemble.py) y [agente](../agent.py).

### Cómo lo conectamos con RL

El agente almacena transiciones en un *replay* de hasta 50.000. Cada 512 interacciones, desde la interacción 512, congela una instantánea del modelo y construye para cada transición muestreada un target de tipo Bellman:

\[
y_i=\widetilde r_i+\gamma b_i\max_{a'}\widehat Q_{\rm congelado}(s'_i,a').
\]

Toma hasta 128 ejemplos por acción, ajusta desde cero un BART por acción y publica los seis modelos juntos sólo si el bloque completo termina bien. El *replay* guarda transiciones, **no targets viejos**: los targets se recalculan en cada ajuste. Esto es una versión de *fitted Q iteration* aproximada, no una actualización TD online por hoja como el baseline. Véanse [replay](../replay.py), [targets](../targets.py), [agente](../agent.py) y [entrenamiento](../training.py).

## 6. Horizonte y costo computacional

El piloto entrena cada combinación durante **20.480 decisiones**: tres semillas de entrenamiento × dos rutas (`visual`, `ocatari`) × dos aprendices = 12 corridas. Cada decisión avanza nominalmente cuatro fotogramas del emulador, de modo que 20.480 decisiones representan aproximadamente 81.920 fotogramas por corrida, aparte de particularidades de reinicio. Hay evaluación en 0, 5.120, 10.240, 15.360 y 20.480 decisiones; cada evaluación usa diez semillas separadas. El límite de una partida es 27.000 decisiones (hasta unos 108.000 fotogramas), pero las partidas observadas terminan mucho antes. No confundamos ese **horizonte máximo por partida** con el presupuesto de **entrenamiento total**. Véanse [protocolo](../../configs/comparison_pong_pilot.json) y [comparación](../comparison.py).

Para BART, 20.480/512 = 40 bloques de ajuste. El orden de trabajo por bloque es aproximadamente \(6\) acciones × \(20\) árboles × \(100\) barridos × hasta \(128\) ejemplos, más búsqueda de reglas, aceptación de propuestas y muestreo. Sólo la predicción de una decisión puede requerir recorrer seis acciones × 50 muestras × 20 árboles. Es un **orden de operaciones**, no una cota formal universal, pues depende de tamaños y profundidad de árboles, cachés y reglas admisibles. El baseline actualiza una hoja por decisión y suele ser mucho más barato. En este piloto, el entrenamiento BART tomó aproximadamente 18–19 veces el tiempo del baseline por corrida; también fue mucho más costosa la evaluación. Véase [reporte](../../results_cluster/pong_pilot_linux_01/report_validation.json). Una corrida 100 veces más larga no debe presupuestarse extrapolando linealmente sin medir memoria y costo de evaluación.

Como referencia de escala, el [trabajo original de RRTL](https://arxiv.org/abs/2203.13599) entrenó Pong durante alrededor de **dos millones de iteraciones**, casi 98 veces las 20.480 decisiones del piloto (las unidades y configuración experimental deben revisarse antes de una comparación cuantitativa estricta). Nuestro piloto sirve principalmente para verificar el *pipeline* y obtener una señal inicial, no para declarar incapacidad definitiva del método.

## 7. Qué pruebas tenemos y qué dicen los resultados

### Pruebas de implementación

Hay pruebas unitarias para codificación, relaciones de Pong y OCAtari; transición terminal/truncación; conjugación gaussiana; posterior exacta en un problema diminuto; balance detallado de `grow/prune`; consistencia de cachés; recuperación de señales sintéticas; un MDP pequeño de \(Q\) conocido; integración, checkpoints y reanudación. Entre los puntos de entrada están [pruebas de Pong](../../tests/test_pong_relations.py), [OCAtari](../../tests/test_ocatari_backend.py), [conjugación](../../tests/test_bayesian_conjugacy.py), [kernel](../../tests/test_bayesian_kernel.py), [ensemble](../../tests/test_bart_ensemble.py), [fitted Q](../../tests/test_fitted_q.py) y [pipeline de comparación](../../tests/test_pong_comparison_pipeline.py). Estas pruebas detectan errores de implementación específicos; **no prueban que la política juegue bien**.

### Pruebas estadísticas y diagnóstico

El baseline usa el «F-test» de división descrito arriba. Para BART registramos trazas de \(\sigma^2\), estructura de árboles, aceptación de movimientos y diagnósticos de varias cadenas, incluidos \(\widehat R\) y tamaños efectivos de muestra (ESS) sobre funciones predichas. La [auditoría de datos fijos](../../docs/pong_comparison_readiness.md) examinó 24 combinaciones con cuatro cadenas y cadenas más largas: **0/24 cumplieron todos los umbrales/observables**. No basta añadir más interacciones del emulador para reparar automáticamente una cadena que mezcla mal en un dataset fijo. Ese resultado tampoco demuestra por sí solo un bug del sampler, y no certifica cada ajuste online corto. Véase [diagnósticos](../diagnostics.py).

El protocolo separa semillas de entrenamiento, validación y test reservado. El informe usa retorno crudo, dispersión, proporción de truncamientos y área de la curva de aprendizaje. Compara BART contra baseline **dentro de cada ruta** mediante *bootstrap* apareado que remuestrea semillas de entrenamiento, no partidas individuales tratadas como independientes. Hay sólo tres semillas de entrenamiento por celda: cualquier intervalo es exploratorio; el test reservado no se ha usado para una conclusión final. Véanse [métricas](../evaluation.py), [protocolo](../comparison_protocol.py) y [reporte](../../results_cluster/pong_pilot_linux_01/report_validation.json).

### Observación del piloto completo

Las **12/12 celdas** figuran completas en [estado de campaña](../../results_cluster/pong_pilot_linux_01/campaign_status.json). En el *checkpoint* final, las diez partidas de validación de cada celda terminaron 0–21: **120/120 partidas, retorno −21, cero puntos anotados**. Esto vale para ambos aprendices y ambas rutas. Por eso la diferencia de retorno BART − baseline es 0 y el intervalo *bootstrap* colapsa en \([0,0]\): es un **efecto de suelo**, no evidencia de equivalencia o ausencia real de ventaja. Hubo algunos puntos anotados durante entrenamiento, pero ninguna mejora sostenida en la política codiciosa evaluada. Algunos árboles baseline ni siquiera se dividieron; otros registraron pocas divisiones. Las trazas de política disponibles muestran casos de colapso hacia una sola acción, pero cubren una campaña diagnóstica anterior de 11 corridas y no identifican la causa general. Véanse [validación](../../results_cluster/pong_pilot_linux_01/report_validation.json) y [diagnóstico de política](../../results_cluster/pong_policy_diagnostic_01/report.json).

## 8. ¿Por qué falla? Evidencia, hipótesis y cómo distinguirlas

**Lo que sí sabemos:** con 20.480 decisiones, exploración aún alta, rewards escasos por punto y ajuste BART costoso, la política final evaluada no anotó. La auditoría BART en datasets fijos encontró problemas de mezcla. No sabemos todavía cuál de estos factores es dominante.

**No tenemos evidencia suficiente para afirmar que «BART sobreajusta y por eso juega mal».** Sobreajuste significaría, por ejemplo, que en transiciones de entrenamiento aprende una regla aparentemente excelente —«cuando `ball_y` es `more` que `player_y`, pulsar `LEFT`»— pero en estados nuevos de la misma situación predice acciones erróneas. El ejemplo es *hipotético*: las trazas actuales no establecen que tal regla se haya aprendido ni que exista una brecha cuantificada entre entrenamiento y datos retenidos. Además, corridas baseline con cero divisiones también fracasan; allí una explicación por «árbol demasiado complejo» sería especialmente débil.

Hipótesis comprobables, sin confundirlas con hallazgos:

1. **Representación insuficiente o aliasing.** Dos posiciones con el mismo signo `less/more` pueden requerir velocidades o anticipaciones distintas. La entrada carece de distancia precisa al punto de intercepción, aceleración y, a veces, contacto observado. Una política heurística que sólo vea los mismos `Fact` debe jugar antes de asumir que el aprendiz puede hacerlo.
2. **Crédito y recompensa escasos.** OCAtari sólo entrega señal ±1 al anotar/perder; entre esos eventos el target depende casi por completo de predicciones previas. Con pocas partidas, el agente puede no relacionar movimiento temprano de la paleta con el punto posterior. La recompensa visual +0,1 cambia la tarea y puede favorecer duración más que puntuación.
3. **Exploración y presupuesto.** A 20.480 decisiones, \(\varepsilon\approx0,389\). Entrenar unas decenas de partidas, con muchas acciones aleatorias y 128 observaciones por acción/ajuste BART, puede no bastar para extraer una política. **Correr más tiempo es una prueba necesaria, pero no suficiente**.
4. **Modelo/ajuste.** La posterior BART podría mezclarse mal, ser sensible al prior o a la calibración de \(\sigma^2\), o ajustarse a targets bootstrapped ruidosos. También puede infraajustar por lotes pequeños y regularización fuerte. Para atribuir causalidad necesitamos trazas fijas, cambios controlados del prior y medición fuera de muestra.
5. **Política de acción.** El signo de movimiento físico y la respuesta de la paleta a las seis acciones deben verificarse empíricamente. Una política que selecciona `NOOP` o una única dirección durante una partida puede perder 0–21 sin que ello indique por sí solo *overfitting*.

La secuencia de controles más informativa es: (i) validar con una política heurística que las relaciones y acciones permiten al menos devolver la pelota; (ii) congelar un dataset y comparar curvas de \(\sigma^2\), mezcla, sensibilidad al prior y error de predicción fuera de muestra; (iii) comparar *fitted Q* BART con árboles más baratos manteniendo datos, targets y evaluación fijos; (iv) sólo después escalar interacciones y semillas. Para una afirmación de mejora en Pong se necesitarían partidas completas con puntos a favor, curvas sobre más entrenamiento, varias semillas independientes y un test reservado ejecutado una sola vez bajo un protocolo fijado.

## 9. Mapa rápido de archivos

| Pregunta | Archivo principal |
| --- | --- |
| ¿Cómo se crea y se guarda OCAtari? | [ocatari_backend.py](../ocatari_backend.py) |
| ¿Qué objetos, acciones y relaciones usa Pong? | [games/pong.py](../games/pong.py) |
| ¿Cómo pasan los `Fact` a columnas y se distingue ausente de falso? | [encoding.py](../encoding.py) |
| ¿Cómo aprende el baseline? | [baseline_adapter.py](../baseline_adapter.py), [baseline.py](../baseline.py), [árbol heredado](../../relational_regresion_tree.py) |
| ¿Dónde están la verosimilitud, los priors y el MCMC de BART? | [likelihood.py](../likelihood.py), [priors.py](../priors.py), [proposals.py](../proposals.py), [sampler.py](../sampler.py), [ensemble.py](../ensemble.py) |
| ¿Cómo se forman los targets RL y se ajustan seis BART? | [targets.py](../targets.py), [replay.py](../replay.py), [agent.py](../agent.py) |
| ¿Qué se ejecutó y con qué resultado? | [protocolo](../../configs/comparison_pong_pilot.json), [estado](../../results_cluster/pong_pilot_linux_01/campaign_status.json), [validación](../../results_cluster/pong_pilot_linux_01/report_validation.json) |

## 10. Próximos pasos: revisión bibliográfica y métodos candidatos

Esta sección propone **experimentos futuros, no funcionalidades ya implementadas**. Ningún artículo demuestra que un método vaya a resolver *nuestro* Pong: 20.480 decisiones y cero puntos finales aún no distinguen falta de datos, representación insuficiente, error de crédito temporal o un estimador de \(Q\) inadecuado. La prioridad es cambiar **una causa candidata por vez** y conservar retornos crudos, semillas separadas y partidas completas como criterio final.

### Paso previo: demostrar que la entrada y las acciones permiten jugar

El [artículo original de RRTL](https://arxiv.org/pdf/2203.13599) ya informa de splits subóptimos en Pong y advierte que la posición relativa simple no bastaba para ganar consistentemente; propone describir la trayectoria de la pelota con mayor granularidad. Eso respalda una **hipótesis de representación**, pero no prueba que ésa sea la causa de nuestras 12 políticas fallidas. Además, el artículo omitió `RIGHTFIRE` y `LEFTFIRE` por considerarlas equivalentes a `RIGHT` y `LEFT` en su configuración; nuestro piloto mantuvo las seis acciones de ALE, así que la equivalencia debe verificarse en **nuestra** versión antes de reducir el espacio de acciones.

Implementar dos controles deterministas, sin aprendizaje:

1. `H_fact`: sólo ve los 20 `Fact` actuales de [Pong](../games/pong.py). Mueve la paleta según alineación y dirección vertical de la pelota, con reglas explícitas para pelota ausente y saque. Sirve para preguntar si **esta representación discreta** permite al menos devolverla; no esperamos una política óptima.
2. `H_geom`: ve los mismos objetos detectados, pero usa centros y diferencias \(\Delta x,\Delta y\) para estimar dónde cruzará la pelota la línea de la paleta, incluyendo rebotes de pared si la física observada lo permite. No debe leer la RAM cruda ni el estado interno del emulador fuera de los objetos. Si `H_geom` anota y `H_fact` no, hay evidencia específica de pérdida de información al discretizar. Si ninguno funciona, revisar primero detección, memoria temporal, signos de las acciones y efecto de `frameskip=4`.

Medir puntos a favor/en contra, proporción de devoluciones verificadas, retorno crudo y partidas completas en semillas de **desarrollo/validación**, no en el test reservado. Verificar experimentalmente el efecto físico de cada acción, en particular los pares con y sin `FIRE`. Si se cambia catálogo de relaciones o acciones, crear una versión de esquema y protocolo nuevos: no reetiquetar checkpoints anteriores.

### Primer competidor recomendado: Fitted Q Iteration con ExtraTrees

[Ernst, Geurts y Wehenkel (2005)](https://jmlr.org/papers/v6/ernst05a.html) formulan el aprendizaje de \(Q\) como una secuencia de regresiones supervisadas sobre transiciones y estudian ensambles de árboles, incluidos los *extremely randomized trees*. Éste es el reemplazo **más directo y barato de probar** para el estimador BART, manteniendo nuestro `replay`, los targets congelados y el protocolo de evaluación. A diferencia del árbol RRTL incremental, los árboles se reajustan sobre un lote y pueden elegir de nuevo sus divisiones; no quedan comprometidos de por vida con un split temprano basado en un F-test. A diferencia de BART, no necesitan MCMC ni prometen una posterior bayesiana.

Con un conjunto fijo \(D=\{(s_i,a_i,r_i,s'_i,b_i)\}\), la iteración \(k\) ajusta un regresor \(f_{k+1}\) a:

\[
y_i^{(k)}=r_i+\gamma b_i\max_{a'}Q_k(s'_i,a'),\qquad
f_{k+1}=\arg\min_{f\in\mathcal F}\sum_i\bigl(y_i^{(k)}-f(s_i,a_i)\bigr)^2.
\]

En nuestro código, construir un modelo por acción y reutilizar [replay](../replay.py), [targets](../targets.py), [codificación](../encoding.py) y la interfaz de [comparación](../comparison.py). Convertir cada categoría y su indicador `observed` a variables binarias: pasar `less/same/more` como números 0/1/2 impondría una distancia artificial, y convertir «ausente» en `False` repetiría un error de representación. Primero comparar los **mismos** \(X,y\) congelados en una regresión de un paso para aislar el estimador; después correr FQI completa sobre el mismo dataset, donde cada método recalcula sus propios targets con idéntica regla de Bellman. En la prueba online, conservar semillas, límites de entrenamiento y evaluación, y registrar tiempo de ajuste e inferencia por decisión. Sólo si no hay errores de cobertura y el costo es viable, pasar a entrenamiento largo.

Riesgo: el conjunto actual proviene de políticas que casi siempre pierden y puede carecer de estados/acciones útiles. Un FQI aparentemente bueno en error de regresión puede escoger acciones no respaldadas por datos: la [literatura sobre error de extrapolación en RL offline](https://proceedings.mlr.press/v97/fujimoto19a/fujimoto19a.pdf) explica por qué no basta minimizar error en el lote. Separar datos por **partida/seed**, medir cobertura por acción y validar la política dentro del emulador; no tratar transiciones consecutivas como observaciones independientes para el intervalo estadístico. Si falta cobertura, recoger datos con `H_geom` o con una mezcla controlada de políticas, documentando la experiencia adicional.

### Segundo competidor barato: true-online Sarsa(\(\lambda\)) con rasgos relacionales

El baseline propaga cada recompensa con una actualización TD de **un paso**. [Van Seijen y colaboradores (2016)](https://www.jmlr.org/papers/v17/15-599.html) estudian *true-online Sarsa*(\(\lambda\)), un método de bajo costo con trazas de elegibilidad y aproximación lineal que incluye experimentos en ALE. La intuición es mantener un rastro decreciente de rasgos activados: \(e_t\approx\gamma\lambda e_{t-1}+\phi(s_t,a_t)\). Cuando llega un punto, no se corrige sólo el estado inmediatamente anterior, sino una secuencia reciente. La regla exacta *true-online* tiene términos adicionales; esta expresión es sólo una explicación conceptual, **no** una especificación de implementación.

Usar \(Q_\theta(s,a)=\theta^\top\phi(s,a)\), donde \(\phi\) codifique `Fact` observados, ausencias y quizá pequeñas interacciones/intervalos de geometría. `Sarsa` aprende el valor de la política exploratoria que realmente actúa, con target \(r+\gamma Q(s',a')\), no el máximo de Q-learning; así evita ese máximo durante entrenamiento, aunque no garantiza mejor política. Añadirlo como agente separado, con pruebas de trazas en un MDP conocido y \(\lambda\in\{0,0.8,0.95\}\); conservar el baseline original como control. Si entre dos puntos median cientos de decisiones, una traza corta puede ser insuficiente y una muy larga elevar varianza: medir la distancia temporal de cada punto antes de fijar \(\lambda\). Su bajo costo permitiría una corrida más larga antes de gastar horas de MCMC.

### Cambios ortogonales, sólo si el diagnóstico apunta a ellos

| Cambio | Qué problema podría atacar | Condición y cautela |
| --- | --- | --- |
| **Double Q-learning** | El \(\max_a\widehat Q\) del baseline puede sobrevalorar acciones por ruido; [van Hasselt (2010)](https://papers.nips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) propone desacoplar selección y evaluación. | Mantener dos estimadores \(Q^A,Q^B\) y alternar targets como \(r+\gamma Q^B(s',\arg\max_a Q^A(s',a))\). Probar primero sesgo y estabilidad en MDP sintético; **no hemos medido sobreestimación en Pong**. Duplica estados del modelo y no arregla `Fact` insuficientes. |
| **Recompensa potencial** | Podría dar señal intermedia cuando los puntos son escasos. [Ng, Harada y Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) muestran la forma \(r'=r+\gamma\Phi(s')-\Phi(s)\) que preserva políticas óptimas bajo sus supuestos de MDP. | Diseñar \(\Phi\) sólo tras validar una heurística de intercepción; fijar \(\Phi=0\) en terminales reales y tratar truncaciones correctamente. Nuestros `Fact` pueden no ser Markovianos, por lo que **no se puede invocar automáticamente la garantía teórica**. El `+0,1` visual actual no es esta construcción y cambia el objetivo. Evaluar siempre con recompensa cruda. |
| **Muestreo de eventos o replay priorizado** | El [replay actual](../replay.py) muestrea uniformemente hasta 128 transiciones por acción; un punto raro puede quedar fuera del lote. [Schaul y colaboradores (2016)](https://arxiv.org/abs/1511.05952) proponen repetir transiciones de mayor relevancia según error TD. | Probar por separado estratos alrededor de puntos y prioridad TD, siempre contra muestreo uniforme. Registrar cobertura efectiva y sesgo de muestreo; usar corrección de importancia cuando corresponda. Repetir una derrota no crea ejemplos de acierto y puede sobreajustar los pocos eventos disponibles. |
| **Imitación/DAgger desde una heurística útil** | Puede proporcionar acciones informativas antes de que el agente aprenda de puntos raros. [Ross, Gordon y Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html) motivan consultar al experto en estados visitados por la propia política para corregir el cambio de distribución. | Sólo si `H_geom` juega mejor que aleatorio y sus decisiones pueden registrarse; primero *behavior cloning*, luego DAgger como experimento separado. Declarar cuántas etiquetas/demostraciones extra recibió; no atribuir esa ventaja al algoritmo RL ni imitar ciegamente una heurística incapaz de anotar. |

No recomiendo empezar por una red profunda o combinar simultáneamente estas técnicas: elevaría costo y confundiría qué corrigió el efecto de suelo. Tampoco recomiendo **repetir sin más** el piloto BART durante dos millones de decisiones: su mezcla en datos fijos y su costo de inferencia siguen sin resolverse.

### Orden experimental y decisión de avance

1. **Auditar interfaz:** terminar la traza faltante de `visual/bart/73`, verificar acciones y correr `H_fact`/`H_geom`. Éxito mínimo: al menos algunas devoluciones y puntos a favor reproducibles; si falla, corregir detector/rasgos antes de otra comparación de regresores.
2. **Dataset fijo y comparación justa:** guardar transiciones con procedencia de política y seed; dividir por episodios; ajustar baseline, BART y FQI-ExtraTrees con targets y datos comparables. Reportar error retenido, rankings de acciones, cobertura, sensibilidad a seeds/hiperparámetros, \(\sigma^2\) y mezcla de BART, tiempo por ajuste y por acción. El error de Bellman o supervisado es diagnóstico, **no** sustituto de jugar Pong.
3. **Crédito temporal:** comparar true-online Sarsa(\(\lambda\)) con \(\lambda=0\) y \(\lambda>0\) bajo la misma representación. Añadir Double Q o *shaping* sólo después de medir, respectivamente, sobreestimación o escasez de señal; cada uno requiere su propia ablación.
4. **Corrida larga escalonada:** llevar primero el método barato prometedor a más decisiones y semillas, con checkpoints y evaluación de partidas completas. Comparar en cada ruta usando retornos crudos, puntos a favor, tasa de victorias, área de aprendizaje y costo; remuestrear **semillas de entrenamiento** para intervalos exploratorios. Mantener intacto el test reservado hasta fijar método, presupuesto, representación y regla de selección de checkpoint.

La pregunta de decisión no es «¿qué método logra menor error de entrenamiento?», sino «¿qué combinación de representación, datos, actualización y costo produce **puntos y mejora sostenida fuera de entrenamiento**?». Si todos los estimadores fallan con los mismos `Fact` pero `H_geom` funciona, priorizar la representación. Si FQI barato progresa y BART no, investigar MCMC/prior/costo antes de ampliar BART. Si ni las heurísticas funcionan, ninguna comparación BART-versus-baseline será todavía una prueba informativa del algoritmo.

## Actualización C13–C15: contratos multijuego y comparación (14/10/2026)

La propuesta anterior expresa rutas de investigación; este bloque implementa únicamente la extensión del experimento común A/B a **Breakout y Demon Attack**. No añade Sarsa, Double Q, shaping ni destilación, ni modifica el algoritmo BART. RRTL corregido, fitted Q con CART y BART usan el mismo runner, cada uno con estado independiente por juego y semilla. ExtraTrees sigue opcional y sólo entra si se declara antes de congelar.

Breakout usa paleta/pelota, 11 relaciones y acciones NOOP/FIRE/RIGHT/LEFT; los ladrillos se guardan para continuidad del detector pero no se publican al agente. Demon Attack utiliza 30 slots auditados, 134 relaciones y las seis acciones reales NOOP/FIRE/RIGHT/LEFT/RIGHTFIRE/LEFTFIRE. Su identidad es de slot, por lo que sólo el jugador tiene movimiento publicado. En ambos, ausencia certificada es falsa y relación incalculable queda no observada. No se reinicia toda la historia por recompensa. Vida perdida no impone terminal; Breakout invalida la historia de pelota. FIRE siempre se contabiliza como decisión explícita.

Entrenamiento recibe `sign(raw_reward)` y evaluación conserva puntaje bruto. En estos juegos, magnitudes diferentes hacen que ambos objetivos no sean exactamente equivalentes. Contacto de cajas no certifica colisión/daño. Los snapshots v2 incluyen todos los objetos/campos auditados, ALE, wrappers, RNG y memoria relacional, con hashes y versiones estrictos. Se comprobaron continuaciones reales en procesos distintos para los dos juegos.

El smoke local completó las nueve celdas y cuatro ajustes por CART/BART a 64 decisiones; las heurísticas Fact también produjeron recompensa de desarrollo. Son comprobaciones de integración, separadas de aprendizaje útil. La comparación estadística toma como unidad la semilla de entrenamiento, declara alcance de cada episodio y AUC sobre el intervalo completo, y no promedia puntajes brutos entre juegos. Las exportaciones BART combinan reglas, frecuencia posterior, contribuciones y recuperación de escala Q; una lista de splits no constituye una política equivalente.

La [guía del bloque C](../../docs/atari_comparison_guide.md) contiene reproducción y Slurm; la [auditoría](../../docs/ocatari_detector_audit_c.md) y el [informe de preparación](../../docs/atari_comparison_preparacion.json) documentan la evidencia. La campaña final está preparada como template inactivo de 45 celdas. Faltan revisión A6/A7, costos Linux y autorización de la ejecución larga; luego corresponde congelar selección y autorizar test sellado. No se ha realizado esa campaña ni consumido test real. Las pruebas locales son macOS, no una validación del despliegue Linux.
