# Plan de implementación: RRTL con árboles bayesianos aditivos

Fecha: 2026-09-11. Rutas actualizadas: 2026-09-17. Base de código inspeccionada originalmente: `rrl`, commit `cf58dc7`; actualmente ubicada en `capstone/rrl_bayesian/`.

Esta copia versionada está en `capstone/rrl_bayesian/`. Los enlaces a fuentes son relativos a esta carpeta; las rutas de módulos, configuración, pruebas y resultados mencionadas en el plan se resuelven bajo `capstone/rrl_bayesian/`, salvo indicación explícita.

## 1. Objetivo y decisiones de alcance

Implementar **RRTL-B**, una variante que aproxima Q mediante una suma de árboles relacionales por acción y aprende sus estructuras y parámetros con inferencia bayesiana. El núcleo seguirá el BART gaussiano explicado por Tan y Roy (2019): prior sobre árboles, regularización de hojas, Bayesian backfitting, propuestas Metropolis-Hastings y muestreo de la varianza residual.

La propuesta de `p1/` busca reducir la dependencia de divisiones tempranas, permitir poda y revisión, y comparar rendimiento, estabilidad, complejidad y costo. Estos beneficios son hipótesis experimentales; BART no garantiza por sí solo mejoras en retorno, estabilidad ni detección de eventos raros.

Decisiones para una primera implementación verificable:

1. Conservar inicialmente los juegos y el preprocesamiento visual del repositorio.
2. Implementar BART como un módulo de regresión independiente del entorno.
3. Usar árboles **binarios** con predicados relacionales; los árboles comparativos ternarios se tratarán como extensión posterior.
4. Mantener un ensemble independiente por acción. Empezar con `m=1` para validar inferencia y pasar a `m>1` para BART aditivo.
5. Integrar el regresor mediante fitted Q-iteration por lotes, con transiciones y objetivos TD congelados durante cada ajuste.
6. Incluir `grow` y `prune` desde el MVP; agregar `change` y `swap` después de verificar el kernel básico.
7. Implementar el contrato modular de General BART y validar una extensión semiparamétrica sencilla en datos sintéticos. La primera política Atari usará su caso particular `h=0` y errores gaussianos.
8. Añadir OCAtari y los operadores de Ramon como etapas diferenciadas, después de obtener un baseline y un BART verificables.

Este documento especifica el plan original. La implementación y validación de H0/H1 se registra en [docs/h0_h1.md](docs/h0_h1.md), H2 en [docs/h2.md](docs/h2.md), H3 en [docs/h3.md](docs/h3.md), H4 en [docs/h4.md](docs/h4.md), H5 en [docs/h5.md](docs/h5.md), H6 en [docs/h6.md](docs/h6.md) y la infraestructura y campaña de desarrollo de H7 en [docs/h7.md](docs/h7.md). La campaña de eficacia a escala y H8 siguen pendientes. Se usa el extractor visual actual; OCAtari está diferido y no se presenta como implementado.

## 2. Fuentes y trazabilidad

Las decisiones se apoyan en estos materiales:

- [Presentación de P1](../p1/p1_latex_source/p1_capstone.tex), especialmente problema, BART, objetivos y entregables.
- [Guion preliminar](../p1/speech_presentacion_preliminar_rrtl.txt): variantes RRTL, RRTL-B, RRTL-R y RRTL-BR; conservación inicial de la representación relacional.
- Tan, Y. V. y Roy, J. (2019), *Bayesian additive regression trees and the General BART model*, Statistics in Medicine 38:5048-5069. [DOI](https://doi.org/10.1002/sim.8347). El PDF local del mismo título está en la carpeta superior `capstone/`.
- [Código RRTL](relational_regresion_tree.py) y [preprocesamiento](preprocessing.py).
- [Ramon, Driessens y Croonenborghs, documento local](../ramon-restructuring.pdf), sección 3, para delimitar la extensión futura de reestructuración.
- Ernst, Geurts y Wehenkel (2005), [Tree-Based Batch Mode Reinforcement Learning](https://www.jmlr.org/beta/papers/v6/ernst05a.html): referencia adicional para convertir el aprendizaje Q en una secuencia de problemas de regresión. La combinación concreta con BART propuesta aquí es una decisión del proyecto.

| Componente a implementar | Referencia en Tan y Roy | Adaptación a este proyecto |
|---|---|---|
| Suma de árboles | §2.1.2-2.1.3, ecuación (1), pp. 5050-5051 | Una suma por acción, con estados relacionales como entrada |
| Priors y backfitting | §2.1.5, ecuaciones (2)-(6), pp. 5054-5056 | Reglas relacionales en lugar de cortes numéricos generales |
| `grow`, `prune`, `change`, `swap` | §2.1.5, p. 5056 | Reglas válidas y ruteo total de estados incompletos |
| Hiperparámetros y escalamiento | Apéndice A, p. 5066 | Escala fija durante aprendizaje RL |
| Posteriores de hojas y ruido | Apéndice B, pp. 5066-5067 | Objetivos TD tratados como respuestas de regresión |
| Razón de aceptación | Apéndice C, pp. 5067-5069 | Probabilidades explícitas de propuestas relacionales |
| General BART | §4, ecuaciones (9)-(12), p. 5060 | Separar árboles, componente adicional y modelo de ruido |

Las ecuaciones y detalles del artículo se contrastaron con el PDF local, incluyendo sus apéndices. Las decisiones sobre replay, objetivos congelados, normalización fija, tratamiento de faltantes y organización del código son propuestas de implementación, no resultados atribuidos a Tan y Roy.

## 3. Punto de partida y verificación previa

| Código actual | Responsabilidad | Tratamiento propuesto |
|---|---|---|
| `train_and_test.py` | CLI, hiperparámetros, entrenamiento y evaluación | Conservar ejecución histórica y añadir un runner configurable |
| `Node` | Q por hoja, estadísticas, F-test y crecimiento irreversible | Mantener para baseline; construir un árbol bayesiano separado |
| `RRLAgent` | Entorno, política, aprendizaje, persistencia | Extraer interfaces pequeñas para modelo Q y representación |
| `preprocessing.py` | Objetos y conjuntos de `Fact` | Reutilizar; añadir un adaptador determinista a características |
| `utilities.Buffer` | Detección de acciones repetidas | Conservar; crear un replay buffer específico de transiciones |
| `datasets/`, `plots/` | Agregación y gráficos | Ampliar para variantes, costo y diagnósticos bayesianos |

Antes de comparar algoritmos, resolver estos puntos con pruebas pequeñas:

- En `relational_q_learning()`, líneas 970-976 del checkout inspeccionado, el bootstrap se calcula con `relational_state`, aunque ya existe `next_relational_state`. Además, el delta resta el Q de la acción greedy, no necesariamente el Q de la acción ejecutada, y no anula explícitamente el bootstrap en un terminal real. Verificar contra la actualización esperada:

  $$Q(s,a)\leftarrow Q(s,a)+\eta_Q[r+\gamma b\max_{a'}Q(s',a')-Q(s,a)].$$

  Usar `eta_Q` para el learning rate; reservar `alpha_tree` para el prior de profundidad. Registrar por separado `terminated`, `truncated` y la máscara `b`: terminal real implica `b=0`; una truncación por límite temporal puede permitir bootstrap si se dispone de la observación final correcta y el protocolo representa una tarea continuada.
- El corte por máximo de iteraciones ocurre antes del registro de la última transición. Verificar conteos, retorno final y guardado de un checkpoint cuando no termina un episodio.
- La selección actual del mejor checkpoint depende de un récord de retorno de entrenamiento. Introducir evaluación de validación con semillas separadas o usar el checkpoint final como comparación primaria.
- `datasets/make_train_dataset_per_game.py` busca `results_sig_0_0001_decay_500000/`, mientras el runner escribe en `results/`. Unificar mediante configuración.
- No hay pruebas automatizadas ni dependencias fijadas. Crear un entorno reproducible, verificar Gymnasium/ALE, disponibilidad de ROMs y Graphviz, y registrar las versiones que efectivamente funcionen. Incluir dependencias de análisis usadas por el código, como seaborn y pingouin.

Conservar dos identificadores experimentales: `rrtl_legacy` para el commit original y `rrtl_corrected` para la actualización verificada. Las comparaciones principales usarán el baseline corregido. No atribuir a BART una mejora causada por corregir TD o cambiar el protocolo de evaluación.

## 4. Modelo probabilístico

### 4.1 Un ensemble por acción

Para una acción `a`, formar un conjunto fijo de pares `(x_i, y_i)`, donde `x_i=phi(s_i)` codifica relaciones y `y_i` es un objetivo TD. En unidades normalizadas:

$$\widetilde y_i=f_a(x_i)+\epsilon_i,\qquad
f_a(x)=\sum_{j=1}^{m}g(x;T_{aj},M_{aj}),\qquad
\epsilon_i\sim\mathcal N(0,\sigma_a^2).$$

Cada hoja almacena una contribución `mu`, no un Q completo. Para una muestra posterior `d`:

$$Q_a^{(d)}(s)=c+W\sum_{j=1}^{m}g(\phi(s);T_{aj}^{(d)},M_{aj}^{(d)}).$$

Se **suman árboles dentro de cada muestra** y se **promedian muestras posteriores** para estimar el Q usado por la política. No confundir ambas operaciones.

La independencia entre ensembles de acciones será una simplificación explícita. La incertidumbre resultante es condicional a los objetivos, la representación y los datos utilizados; no es una posterior exacta sobre el MDP ni una garantía de incertidumbre bien calibrada sobre Q óptimo.

### 4.2 Prior sobre estructuras

Usar profundidad de raíz cero y:

$$p_{\mathrm{split}}(d)=\alpha_{\mathrm{tree}}(1+d)^{-\beta_{\mathrm{tree}}},
\qquad \alpha_{\mathrm{tree}}=0.95,\quad\beta_{\mathrm{tree}}=2.$$

La probabilidad de un árbol incluirá tanto su forma como sus reglas:

$$p(T\mid X)=
\prod_{u\in I(T)}p_{\mathrm{split}}(u)\,p(\mathrm{rule}_u\mid u,X)
\prod_{v\in L(T)}[1-p_{\mathrm{split}}(v)].$$

El catálogo de reglas y su distribución se fijarán para el conjunto `X` de cada ajuste. Seleccionar uniformemente una relación elegible y luego una regla elegible de esa relación. Así, una relación con más codificaciones no recibe automáticamente más masa a priori.

Una regla elegible debe ser compatible con sus ancestros y separar al menos una observación hacia cada hijo. Si no existe ninguna, o se alcanza un límite explícito de profundidad, definir `p_split=0`. Esto modifica el soporte del prior y debe aparecer de igual forma en `log_prior`, en las propuestas y en los tests. Empezar con mínimo de una observación por hijo; investigar mínimos mayores como regularización adicional, porque podrían eliminar señales infrecuentes.

No reutilizar el requisito de 100.000 muestras del F-test como condición de aceptación bayesiana. Una propuesta puede aceptarse o rechazarse con poca evidencia según likelihood, prior y kernel.

### 4.3 Priors de hojas, ruido y escala

Usar priors independientes:

$$\mu_{aj\ell}\mid T_{aj}\sim\mathcal N(0,\tau^2),
\qquad \tau=\frac{0.5}{k\sqrt m},\quad k=2,$$

$$\sigma_a^2\sim\operatorname{IG}(\nu/2,\nu\lambda_a/2),\qquad\nu=3.$$

Fijar la convención de implementación mediante la densidad:

$$p(z\mid A,B)\propto z^{-A-1}\exp(-B/z),\quad z>0.$$

Esto evita ambigüedad entre nombres `rate` y `scale`; para un generador gamma con parámetro escala, se puede obtener `z = 1 / Gamma(shape=A, scale=1/B)`.

Tan y Roy normalizan usando el mínimo y máximo de la respuesta. En RL cambian los objetivos entre ajustes; recalcular esa escala silenciosamente haría incompatibles hojas, prior y modelo objetivo. Para la primera versión usar:

$$\widetilde y=(y-c)/W,\quad c=(L+U)/2,\quad W=U-L,$$

con `L,U` fijados antes de entrenar a partir de cotas de recompensa transformada y `gamma<1`. Para recompensas dentro de `[r_min,r_max]`, usar cotas conservadoras que incluyan cero y `r_min/(1-gamma)`, `r_max/(1-gamma)`. Por ejemplo, con `gamma=0.99`, `[-1,1]` da `[-100,100]`; el shaping de Pong `sign(r)+0.1` da `[-90,110]`.

No recortar automáticamente predicciones ni objetivos fuera de estas cotas: el prior gaussiano tiene soporte no acotado. Registrar excesos para diagnóstico. Una escala empírica fijada con un piloto de entrenamiento puede evaluarse después; no usar test para calibrarla.

Calibrar `lambda_a` una sola vez con datos piloto en la misma escala, siguiendo `P(sigma_a^2 < s_a^2)=0.9`. Con la convención anterior:

$$\lambda_a=s_a^2\,F^{-1}_{\chi^2_\nu}(0.1)/\nu.$$

Estimar `s_a^2` con residuos de un modelo lineal auxiliar cuando sea identificable; documentar un fallback de varianza de respuestas y un piso positivo si hay datos insuficientes o varianza nula. El piso y el fallback son decisiones del proyecto. Persistir `c,W,k,nu,lambda_a` en cada checkpoint.

### 4.4 Estado relacional y valores ausentes

Implementar `RelationalEncoder` con un catálogo ordenado por claves semánticas, no por el orden de un `set`. Mantener nombres de objetos, dimensión y tiempo. Guardar un hash/versionado del catálogo.

- Para una relación comparativa, conservar las categorías `less`, `same`, `more`; generar tests binarios como `relation == more`. No interpretar códigos numéricos arbitrarios como distancias.
- Para una relación lógica, distinguir `True`, `False` y `MISSING`.
- Añadir una máscara de observación y una regla explícita `is_missing` cuando corresponda. La ausencia de un `Fact` no demuestra su falsedad.
- Para reglas ordinarias, fijar una convención de ruteo de ausentes, por ejemplo hacia la rama falsa, conservando la máscara que permite separarlos. Es una decisión de representación, no imputación probabilística.
- Todo estado, incluido el conjunto vacío, debe llegar exactamente a una hoja por árbol. No detenerse en nodos internos con Q heredado: eso rompería la likelihood particionada por hojas.
- El adaptador debe aprovechar los `Fact` realmente devueltos por el preprocesador. Si este ya descartó información al producir un estado vacío, una máscara no la recupera. Cambiar `include_incomplete_states` será una ablación explícita.

La binarización permite usar los movimientos de Tan y Roy sin derivar un kernel ternario. No preserva la profundidad ni el prior inducido por un split comparativo ternario: comparar complejidad teniendo esto presente. Añadir un baseline frecuentista binario con el mismo encoder para separar cambios de representación de cambios bayesianos.

## 5. Inferencia: especificación implementable

### 5.1 Estadísticas y distribuciones conjugadas

Para actualizar el árbol `j`, calcular residuos parciales usando los demás árboles, incluyendo sus actualizaciones más recientes:

$$r_j=\widetilde y-\sum_{h\ne j}g_h(X).$$

Para una hoja con residuos `r_1,...,r_n`, guardar `n`, `S=sum(r_i)` y `C=sum(r_i^2)`. Con prior centrado:

$$V=(n/\sigma^2+1/\tau^2)^{-1},\quad b=V S/\sigma^2,
\qquad\mu\mid r,T,\sigma^2\sim\mathcal N(b,V).$$

La log-verosimilitud marginal de esa hoja, integrando `mu`, es:

$$\log p(r\mid T,\sigma^2)=
-\frac n2\log(2\pi\sigma^2)
-\frac12\log\left(1+\frac{n\tau^2}{\sigma^2}\right)
-\frac{C}{2\sigma^2}
+\frac{\tau^2S^2}{2\sigma^2(\sigma^2+n\tau^2)}.$$

Implementar primero esta expresión completa y sumar sobre hojas. Optimizar después mediante diferencias locales. Las estadísticas de `Node.refinements_stats` actualizan valores Q históricos y no son sustitutos de estos residuos parciales.

Después de actualizar todos los árboles de una acción:

$$\sigma_a^2\mid\cdots\sim\operatorname{IG}\left(
\frac{\nu+N_a}{2},\frac{\nu\lambda_a+\sum_i(\widetilde y_i-f_a(x_i))^2}{2}\right).$$

No incluir términos adicionales de hojas en este conditional: aquí sus priors son independientes de `sigma_a^2`.

### 5.2 Metropolis-Hastings sobre estructuras

Para una propuesta `T -> T*`, calcular:

$$\log R=
\log p(r_j\mid T^*,\sigma^2)-\log p(r_j\mid T,\sigma^2)
+\log p(T^*\mid X)-\log p(T\mid X)
+\log q(T\mid T^*)-\log q(T^*\mid T).$$

Aceptar si `log(U) < min(0, log_R)`. Integrar las hojas para decidir la estructura y muestrearlas después, incluso cuando se rechaza el cambio estructural.

Primera versión: probabilidades fijas `p_grow=p_prune=0.5`. Si el movimiento elegido no es posible, producir una transición a sí mismo; no renormalizar silenciosamente. Para crecer una hoja `u`:

$$q(T^*\mid T)=p_{grow}\,\frac{1}{|G(T)|}\,
\frac{1}{|V(u)|}\,\frac{1}{|R(u,v)|},$$

$$q(T\mid T^*)=p_{prune}\,\frac{1}{|P(T^*)|},$$

donde `G(T)` son hojas elegibles, `V(u)` relaciones elegibles, `R(u,v)` reglas elegibles y `P(T*)` padres con dos hijos terminales. Recalcular estas cantidades en la estructura pertinente. Para prune, calcular la razón de la transición inversa; se invierte la razón sin truncar, no la probabilidad de aceptación ya limitada por uno.

`stay` puede resultar de rechazo o de una propuesta imposible; no necesita ser un operador adicional. Grow/prune permiten volver a la raíz y construir otra estructura dentro del soporte elegido, aunque la mezcla puede ser lenta.

Segunda versión: `grow`, `prune`, `change`, `swap` con probabilidades registradas, inicialmente 0.25 cada una. `change` reemplaza una regla interna; `swap` intercambia reglas de un par padre-hijo interno elegido. Recalcular ruteo y soporte de todos los descendientes afectados. Rechazar cambios incompatibles y contabilizar correctamente propuestas inversas, elecciones duplicadas y movimientos imposibles.

### 5.3 Backfitting y posterior predictiva

```text
fit(X, y, configuration):
    congelar X, y, escala, prior y catálogo de reglas
    inicializar m árboles raíz con contribución media(y_normalizada)/m
    inicializar sigma2 positiva
    para cada sweep:
        para j = 1,...,m:
            r_j = y_normalizada - (predicción_total - predicción_j)
            proponer estructura y calcular aceptación MH marginal
            aceptar o conservar estructura
            muestrear todas las hojas de j
            actualizar predicción_j y predicción_total
        muestrear sigma2 con residuos del ensemble completo
        registrar diagnósticos
        después de burn-in, conservar muestras completas
    devolver muestras y diagnósticos de este ajuste
```

Separar `predict_mean`, `predict_latent_draws` y `predict_observation_draws`. Los intervalos sobre la función latente usan draws de la suma; los predictivos de una nueva respuesta añaden ruido gaussiano. La política usará la media latente; Thompson sampling será una ablación posterior y no añadirá ruido residual como exploración.

## 6. Integración con aprendizaje por refuerzo

### 6.1 Fitted Q-iteration por bloques

El MCMC de Tan y Roy está definido para un dataset fijo. Ejecutarlo mientras cambian respuestas o minibatches dentro de la cadena no conserva automáticamente esa posterior. Adoptar el siguiente procedimiento aproximado de RL:

1. Recolectar transiciones `(s,a,r,s_next,terminated,truncated,episode_id,step_id)` con una política epsilon-greedy.
2. Mantener un replay buffer de capacidad finita. Guardar también recompensa original y transformada.
3. Cada `fit_interval` interacciones, seleccionar una muestra uniforme sin reemplazo por acción, y congelarla durante todo el ajuste. Cada transición tiene un ID único.
4. Congelar una copia de la media posterior del modelo anterior, `Q_target`. Todos los objetivos de todas las acciones se construyen con esa misma copia:

   $$y_i=r_i+\gamma b_i\max_{a'}\overline Q_{target}(s'_i,a').$$

5. Ajustar cada ensemble al dataset de su acción usando MCMC. Durante este ajuste no actualizar respuestas ni volver a muestrear el replay.
6. Publicar los nuevos ensembles juntos cuando todos hayan terminado. Continuar recolección con la media posterior actualizada.

Inicializar `Q_target=0` antes del primer ajuste. Si una acción no tiene datos, conservar su modelo anterior o una distribución predictiva del prior si aún no existe; marcar explícitamente que no fue actualizada. No inventar observaciones ni convertir su falta de evidencia en varianza cero.

La primera implementación reiniciará las cadenas en cada ajuste para simplificar su validación. Posteriormente puede reutilizar árboles como inicialización, con nuevo burn-in y reconstrucción de cachés. Si un árbol previo deja de ser válido para el nuevo `X`, reinicializarlo. Una cadena previa sirve como warm start; **no se convierte su posterior en prior para volver a contar las mismas transiciones**. No mezclar draws de distintos datasets/objetivos como si pertenecieran a una misma posterior.

Esta integración sustituye la actualización incremental de las hojas por ajustes de regresión periódicos. No aplicar además `eta_Q * delta` a las hojas BART. Registrar la diferencia de régimen de aprendizaje como parte del método.

### 6.2 Política, escala y evaluación

- Conservar epsilon-greedy y su calendario por número de interacciones, no por sweeps MCMC.
- Mantener recompensa transformada para entrenamiento y recompensa original para evaluación, como en el código base.
- Para `terminated=True`, no evaluar innecesariamente el estado terminal para construir el target.
- Reutilizar el tratamiento de acciones repetidas solo bajo una configuración compartida por las variantes comparadas.
- Evaluar en entornos separados, sin mutar replay, RNG de entrenamiento ni posterior.
- Versionar datasets, objetivos y modelos con `fit_id`, `target_model_id` y hash de configuración.

No prometer convergencia de Q-learning tabular: hay aproximación funcional, datos correlacionados, cobertura desigual y objetivos estimados. Diagnosticar extrapolación, magnitud de Q y cobertura por acción.

## 7. Qué significa implementar General BART aquí

La sección 4 plantea:

$$y=f(x)+h(w,\Theta)+\epsilon,\qquad\epsilon\sim G(\Sigma).$$

Construir tres componentes: `BARTMean`, `AdditionalMean` y `NoiseModel`. La primera configuración será `ZeroMean` y `GaussianNoise`. Validar además un caso sintético `h(w,theta)=w^T theta`, con `theta ~ Normal(0,V0)`, columnas de `w` diferenciadas de las de `x` y centrado que reduzca confusión entre componentes.

Un sweep General BART realizará:

1. Actualizar árboles con respuesta `y-h(w,theta)` y ruido condicional vigente.
2. Actualizar `theta` con respuesta `y-f(x)`.
3. Actualizar ruido con residuo `y-f(x)-h(w,theta)`.

Para la extensión lineal gaussiana:

$$V_\theta=(V_0^{-1}+W_{design}^T W_{design}/\sigma^2)^{-1},$$

$$b_\theta=V_\theta W_{design}^T(y-f(X))/\sigma^2,
\quad\theta\mid\cdots\sim\mathcal N(b_\theta,V_\theta).$$

Usar factorizaciones y sistemas lineales, no inversas explícitas. Validar que `h=0` reproduce el motor base y que, con la contribución de árboles fijada en cero, se recupera regresión lineal bayesiana.

Esto implementa el mecanismo modular y una extensión comprobable. No incluye todas las variantes del artículo: probit, efectos espaciales o mezclas Dirichlet quedan fuera del primer alcance. Cambiar `G` por ruido arbitrario puede exigir nuevas distribuciones condicionales o variables latentes; una interfaz común no hace válidas automáticamente las fórmulas gaussianas.

## 8. Organización propuesta del código

Añadir los módulos dentro de `capstone/rrl_bayesian/`, sin renombrar los archivos históricos en el primer cambio. `bayesian_rrtl/` será el paquete interno de esta carpeta:

```text
capstone/rrl_bayesian/
├── train_and_test.py                 # Ejecución histórica
├── relational_regresion_tree.py      # Baseline original
├── preprocessing.py                 # Extracción existente
├── train_bayesian.py                 # Nuevo runner configurable
├── bayesian_rrtl/
│   ├── encoding.py                  # Fact -> características y máscaras
│   ├── tree.py                      # Nodos binarios, ruteo, invariantes
│   ├── rules.py                     # Catálogo, elegibilidad, probabilidades
│   ├── priors.py                    # Estructura, hojas, ruido y escala
│   ├── likelihood.py                # Estadísticas y marginal gaussiana
│   ├── proposals.py                 # Grow/prune y luego change/swap
│   ├── sampler.py                   # MH, backfitting, draws
│   ├── ensemble.py                  # Predicción y estado del ensemble
│   ├── general_bart.py              # h(w,theta) y actualización por bloques
│   ├── replay.py                    # Transiciones e IDs
│   ├── targets.py                   # Construcción TD y terminales
│   ├── agent.py                     # Ensembles por acción, fit por bloques
│   ├── diagnostics.py               # Mezcla, estructuras, tiempos
│   └── checkpoint.py                # Serialización versionada
├── configs/                         # Parámetros completos de experimentos
├── tests/                           # Inferencia, representación y RL
└── experiments/                     # Sintéticos, ablaciones y evaluación
```

Contratos mínimos:

- `Encoder.transform(states) -> EncodedBatch`: valores, máscaras y versión del catálogo.
- `Rule.route(batch) -> boolean_mask`: ruteo determinista y total.
- `Tree.predict(batch) -> vector`: contribución de una muestra de árbol.
- `Proposal.propose(tree, X, rng) -> candidate, log_q_forward, log_q_reverse, metadata`.
- `BARTRegressor.fit(X, y, rng) -> PosteriorState` y predicciones por draw/media.
- `BayesianQModel.predict(states) -> [n_states, n_actions]`.
- `Agent.observe(transition)` y `Agent.fit_if_due()`.

Usar RNG explícitos separados para entorno, exploración, replay y MCMC; orden estable de acciones y reglas. Los checkpoints incluirán árboles, muestras retenidas, ruido, encoder, escala, configuración, contadores y estados RNG. Para reanudar entrenamiento, guardar también replay y modelo objetivo o referencias verificables a sus datos.

El núcleo puede implementarse con NumPy/SciPy y objetos simples. Antes de elegir dependencias adicionales, verificar sus APIs y soporte de reglas relacionales. Una implementación BART externa será un contraste opcional en regresión supervisada; no reemplaza validar el kernel propio.

## 9. Hitos, dependencias y criterios de aceptación

Las estimaciones son esfuerzo orientativo para una persona con familiaridad con Python y RL; excluyen colas de cómputo y campañas largas.

| Hito | Trabajo y entregable | Criterio para avanzar | Esfuerzo |
|---|---|---|---|
| H0. Baseline | Entorno reproducible, auditoría TD, `legacy` y `corrected`, registros | Pruebas de una transición, terminal/truncación y smoke test Atari; discrepancias documentadas | 2-4 días |
| H1. Representación | Encoder y árbol binario independiente | Todo estado llega a una hoja; resultados invariantes al orden de los `Fact`; faltantes diferenciados | 2-3 días |
| H2. Bayes con un árbol | Priors, marginal, posterior de hojas y ruido, grow/prune | Posterior enumerada y balance detallado en espacio pequeño; pruebas numéricas conjugadas | 4-6 días |
| H3. Ensemble BART | Residuos parciales, backfitting, draws y predicciones | Suma/cache correcta; recuperación de señal y diagnósticos de varias cadenas en datos fijos | 3-5 días |
| H4. General BART y revisión | Componente lineal sintético; change/swap verificados | Reducción a casos conocidos; kernels nuevos preservan la distribución objetivo | 3-5 días |
| H5. Integración Q | Replay, targets congelados, ensembles por acción y checkpoints | MDP pequeño con Q conocido; no bootstrap terminal; reanudación reproducible | 3-5 días |
| H6. Piloto Atari | Breakout, pocas semillas, perfil de costo y baseline pareado | Flujo completo finito, sin NaN, datos íntegros y costo medido; diagnóstico de mezcla | 3-5 días más cómputo |
| H7. Evaluación | Ablaciones y campaña multijuego | Informe con retorno, dispersión, complejidad y tiempo; test reservado | 4-7 días más cómputo |
| H8. Extensiones P1 | OCAtari; RRTL-R y posible RRTL-BR | Evaluación separada por representación y kernel; nuevas derivaciones revisadas | Estimar tras H6 |

Dependencia principal: `H0 -> H1 -> H2 -> H3 -> H5 -> H6 -> H7`. H4 depende de H3 y puede incorporarse antes de la campaña final. Un primer RRTL-B ejecutable se alcanza en H5 con grow/prune; el bloque General BART y los movimientos adicionales son entregables propios, no requisitos para comprobar la primera integración.

### Pruebas que justifican la inferencia

1. Comparar la marginal analítica de una hoja con integración numérica en casos pequeños; verificar también el caso vacío como utilidad matemática, aunque los splits del MVP no produzcan hijos vacíos.
2. Comparar momentos de draws de `mu` y `sigma2` con sus distribuciones conjugadas conocidas, usando tolerancias de error Monte Carlo.
3. Enumerar todos los árboles de profundidad acotada para un dataset diminuto, con `m=1` y `sigma2` fija. Calcular posterior exacta y comprobar frecuencias MCMC.
4. Verificar `pi(T)K(T,T*) = pi(T*)K(T*,T)` para transiciones vecinas, incluyendo árboles raíz y límites de profundidad. Verificar normalización de filas y estados de rechazo.
5. Comprobar que rechazo no muta el árbol original; grow seguido de su prune restaura la estructura; la actualización de hojas ocurre también tras rechazo.
6. Contrastar predicción incremental/cache con recomputación completa después de cada árbol en una prueba pequeña.
7. Validar suma dentro del ensemble y promedio entre draws; impedir que se promedien los `m` árboles por error.
8. Usar sintéticos de señal nula, función por tramos, interacción entre relaciones y evento raro. La cobertura se medirá sobre datasets repetidos; no exigir que una sola realización tenga exactamente 95%.
9. En un MDP finito, calcular Q óptimo por programación dinámica y verificar targets, escala e identificación de la acción óptima con datos suficientes.
10. Probar guardado/reanudación en un entorno controlado y comparar estado RNG, targets y próximas actualizaciones.

## 10. Presupuesto computacional y configuración inicial

No ejecutar MCMC completo después de cada frame. El costo aproximado por ajuste es proporcional a `sweeps * m * N * profundidad`, sumado sobre acciones y al costo de propuestas. `N` representa el total de transiciones efectivamente ajustadas. El número de draws retenidos también afecta memoria y latencia al seleccionar acciones.

Configuración inicial de desarrollo, sujeta a perfilado; no representa parámetros validados para Atari:

| Parámetro | Valor piloto |
|---|---|
| Árboles por acción | 1 en H2; 20 en integración; estudiar 50 y 100 |
| `alpha_tree`, `beta_tree`, `k`, `nu` | 0.95, 2, 2, 3 |
| Profundidad máxima | 6, documentando truncamiento del prior |
| Mínimo por hijo | 1; sensibilidad posterior a 5 |
| Burn-in / sweeps retenidos | 200 / 200 por ajuste inicial |
| Replay | 50.000 transiciones |
| Lote fijo de ajuste | Hasta 2.000 transiciones por acción, sin reemplazo |
| Warm-up de interacción | 5.000 pasos |
| Frecuencia de ajuste | Cada 20.000 pasos, configurable |
| Diagnóstico supervisado | 4 cadenas independientes; ampliar longitud según mezcla |
| Exploración inicial | Calendario epsilon-greedy del baseline |

Medir tiempo de encoding, target, propuestas, barrido y predicción por acción; memoria máxima; propuestas aceptadas/imposibles por tipo; hojas/profundidad total; sigma y error residual. Calcular ESS y R-hat sobre predicciones en estados de referencia, sigma y resúmenes identificables del ensemble. Los índices de árboles intercambiables no son buenos parámetros para evaluar convergencia individual.

Como guía de validación supervisada, buscar R-hat cercano a 1 (por ejemplo, <=1.01) y ESS suficiente para la precisión deseada; una tasa de aceptación alta sola no prueba mezcla. Si los presupuestos cortos de RL no alcanzan calidad comparable, reportarlos como inferencia aproximada y cuantificar su sensibilidad.

Perfilar primero en datasets fijos pequeños y luego extrapolar a los 2-3 millones de interacciones por run del código original. Si resulta inviable, reducir frecuencia, tamaño de lote o `m`, y medir el efecto. No introducir minibatches cambiantes dentro del MH estándar para acelerar sin reformular el método.

## 11. Diseño experimental

### 11.1 Comparaciones prioritarias

| Variante | Qué permite medir |
|---|---|
| RRTL legacy | Resultado del código original, separado del análisis causal principal |
| RRTL corregido | Baseline de aprendizaje incremental con TD verificado |
| RRTL corregido binario con encoder común | Efecto del cambio de representación/ruteo |
| Regresor frecuentista por lotes con los mismos targets | Control del cambio de incremental a fitted Q; declarar algoritmo e hiperparámetros |
| Árbol bayesiano `m=1` | Reversibilidad y selección bayesiana sin efecto aditivo |
| RRTL-B `m>1`, grow/prune | Método principal inicial |
| RRTL-B con change/swap | Efecto de movimientos adicionales sobre mezcla, costo y retorno |
| RRTL-B con Thompson sampling | Efecto de usar incertidumbre para explorar, solo después |

Un método que deshabilite prune puede servir como heurística de crecimiento para comparación, pero no debe etiquetarse como el mismo sampler de la posterior BART: se pierde la reversibilidad necesaria.

### 11.2 Protocolo

1. Empezar por regresión sintética y un MDP pequeño; luego Breakout.
2. Usar inicialmente 3 semillas de entrenamiento para verificar ejecución y costo, sin conclusiones estadísticas fuertes.
3. Realizar un piloto de aproximadamente 10 semillas y estimar variabilidad/costo; definir el número final necesario para medir el efecto mínimo relevante. Usar 30 semillas como punto de planificación, sujeto al presupuesto medido.
4. Comparar con igual presupuesto de interacción y reportar además rendimiento frente a tiempo de cómputo. No equiparar un sweep a una interacción.
5. En regresión offline usar exactamente las mismas transiciones. En evaluación online las políticas producirán trayectorias diferentes; compartir listas de semillas no implica datos idénticos.
6. Separar semillas de entrenamiento, validación y test. Fijar hiperparámetros y regla de selección de checkpoint antes del test.
7. Evaluar 100 episodios por modelo, como referencia del runner actual. La unidad estadística principal será la semilla de entrenamiento, no cada episodio como réplica independiente.
8. Extender a Pong y Demon Attack una vez estabilizado el protocolo; informar resultados por juego y evitar promediar retornos crudos de escalas diferentes.

Métricas: retorno por semilla, mediana e intervalo de incertidumbre, dispersión/IQR, frecuencia de runs deficientes con umbral predefinido, curva de aprendizaje y área bajo ella; número total de hojas, profundidad y reglas distintas; desacuerdo entre políticas posteriores; tiempo, memoria y costo por acción. Para comparaciones pareadas usar diferencias por semilla e intervalos bootstrap al nivel de semilla. Preespecificar contrastes y ajustar por comparaciones múltiples si se realizan pruebas confirmatorias.

La interpretabilidad debe evaluarse al nivel del ensemble: muchos árboles pequeños pueden ser menos legibles que un árbol único. Exportar reglas con frecuencia de aparición y ejemplos de contribuciones, además de Graphviz. Una política destilada a un solo árbol es una explicación aproximada y necesita medir fidelidad.

Para eventos raros, diseñar sintéticos controlados y cortes diagnósticos del replay; no concluir mejora solo por el prior. Si se añade muestreo prioritario, documentar que cambia la distribución ajustada y estudiar correcciones/ponderaciones antes de reutilizar las fórmulas actuales.

## 12. OCAtari y reestructuración de Ramon

### OCAtari

La presentación propone estandarizar la extracción de objetos. El checkout actual ya extrae objetos automáticamente con reglas visuales programadas a mano; no requiere etiquetado humano de cada frame durante la ejecución.

Definir un contrato común `ObjectExtractor -> objetos -> Fact` y añadir posteriormente un adaptador OCAtari. Verificar coordenadas, identidad temporal, presencia/ausencia, tamaño de objetos y catálogo de relaciones con ejemplos de los tres juegos. El acceso RAM puede cambiar la información disponible frente al preprocesamiento visual: comparar cada modelo bajo ambos extractores y mantener fijo el extractor en la comparación principal.

### RRTL-R y RRTL-BR

Ramon et al. describen cuatro operaciones en árboles relacionales de primer orden: dividir hojas, unir hojas hermanas, revisar un test interno insertando un nuevo test y duplicando el subárbol, y eliminar un nodo conservando un subárbol compatible. No equivalen simplemente a `change`/`swap` de BART. Además, TgR mantiene estadísticas en todos los nodos y evita almacenar todas las experiencias; este plan BART sí usa un replay acotado.

Para **RRTL-R**, implementar sus operadores y estadísticas en una línea separada sobre el baseline, documentando las diferencias entre las relaciones aterrizadas del repo y las dependencias de variables del formalismo de primer orden.

Para **RRTL-BR**, estudiar si esos operadores aportan algo adicional a grow/prune/change/swap. Una modificación estructural determinista durante la cadena no preserva automáticamente la posterior. Antes de integrarla, derivar su propuesta inversa, soporte y probabilidad MH; si no existe una inversa practicable, usarla solo como inicialización seguida de nuevo burn-in y etiquetarla como tal. No presentar una poda forzada como muestreo BART exacto.

La decisión de añadir BR dependerá de mezcla y rendimiento medidos. La primera entrega bayesiana ya debe permitir corregir divisiones mediante prune y crecimiento posterior.

## 13. Riesgos y decisiones de continuación

| Riesgo | Medida concreta |
|---|---|
| Confundir corrección de TD con efecto bayesiano | Baselines versionados y controles comunes |
| Romper MH al cambiar reglas/faltantes | Catálogo congelado por fit, log-propuestas y tests de balance |
| Duplicar evidencia entre bloques | IDs únicos; prior fijo; warm start solo como inicialización |
| Prior demasiado estrecho o amplio para Q | Escala fija, calibración piloto y sensibilidad a `k` |
| Cadena lenta por grow/prune | Diagnóstico predictivo, más sweeps y luego change/swap |
| Tiempo o memoria excesivos | Perfilado previo, replay acotado, registro de latencia por acción |
| Incertidumbre mal calibrada por dependencia temporal | Validación por episodios/bloques; no interpretar intervalos como posterior del MDP |
| Varias extensiones confunden la comparación | Separar BART, régimen por lotes, encoder, OCAtari y Ramon |
| Mejora de ajuste sin mejora de política | Evaluar decisiones en MDP conocido y retorno en entornos reservados |

## 14. Definición de terminado

- [x] Baseline original y corregido reproducibles, con cambios TD documentados (H0; pruebas cortas).
- [x] Encoder relacional binario con ruteo total y faltantes explícitos (H1).
- [x] Árbol bayesiano contrastado contra posterior exacta en un problema pequeño (H2; 20 estructuras y cuatro cadenas).
- [x] Ensemble con prior, backfitting, MH, hojas, ruido y draws implementados y probados (H3).
- [x] Grow/prune funcionales; change/swap incorporados con balance detallado verificado (H4).
- [x] General BART validado en el caso `h=0` y en el ejemplo lineal sintético (H4).
- [x] RRTL-B con objetivos congelados, terminales correctos y persistencia versionada (H5; MDP controlado y reanudación exacta).
- [x] Piloto Atari con medición de costo y calidad de inferencia (H6; tres semillas, baseline pareado y cadenas sobre datos congelados; extractor visual, OCAtari pendiente).
- [x] Infraestructura y campaña multijuego de desarrollo con controles y test separado (H7; presupuestos reducidos, sin conclusión de eficacia).
- [ ] Campaña de eficacia a escala, con inferencia calibrada y suficientes semillas/episodios según costo y variabilidad.
- [x] Informe que distinga resultados observados, limitaciones y extensiones pendientes de P1 (docs/h7.md).

La primera tarea de implementación será H0: fijar el entorno y establecer con pruebas la semántica TD. El primer entregable matemático será H2: un árbol bayesiano pequeño cuya distribución muestreada pueda contrastarse exactamente. Esas dos verificaciones habilitan integrar BART al agente con una base comprobable.
