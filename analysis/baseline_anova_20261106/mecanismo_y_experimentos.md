# Mecanismo de crecimiento y experimentos diagnósticos

Nota técnica del 6 de noviembre de 2026. Esta revisión se basa en las fuentes del repo: **no presupone resultados de la campaña ANOVA ni atribuye causalmente las caídas observadas a una división**. No se modificó código de entrenamiento.

## Qué cambia entre histórico y ANOVA

Ambos hacen Q-learning incremental con una hoja por estado y acción:

\[
y_t=r_t+\gamma m_t\max_a Q_t(s_{t+1},a),\qquad
q_{t+1}=q_t+\eta(y_t-q_t).
\]

Después de actualizar Q, acumulan momentos del **Q actualizado**, no del target TD, recompensa o error TD. Véanse [update_transition](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/baseline.py:101), [actualización TD](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/targets.py:26) y [estadísticas](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/relational_regresion_tree.py:296).

Para una candidata, sean T la suma de cuadrados total, W la suma dentro de categorías, B=T−W y K el número de categorías observadas no vacías:

| Criterio | Estadístico y puntuación nominal |
|---|---|
| Histórico | R=W/T; p=CDF_F(R;n−1,n−1) |
| ANOVA | F=[B/(K−1)]/[W/(n−K)]; p=SF_F(F;K−1,n−K) |

Fuentes: [razón histórica](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/relational_regresion_tree.py:129), [selección/prueba histórica](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/relational_regresion_tree.py:159), [ANOVA](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/baseline_anova.py:26).

ANOVA puede detectar efectos pequeños con suficiente n: F=((1−R)/R)(n−K)/(K−1). No es la misma regla con otro nombre. Hay que comparar tamaño del efecto B/T, cobertura y momento de las divisiones, además del número final de hojas. El histórico usa un cociente de dispersiones anidadas, no una ANOVA clásica. Aunque la fórmula ANOVA es la convencional, Q es temporalmente dependiente y cambia con la política; seleccionar el mínimo p y comprobar repetidamente invalida interpretar el umbral nominal como garantía de falsos positivos. Los p son puntuaciones operativas de crecimiento.

Ambos exigen n>128 por candidata/hoja/acción en esta receta; solo cuentan visitas donde la relación está presente. Un millón global no asegura soporte local. ANOVA puede aceptar dos categorías observadas y crear una tercera rama aún sin muestras. Registra errores explícitos para momentos inconsistentes y trata degeneraciones. Hay también una diferencia de desempate: el histórico retiene el último candidato empatado y ANOVA el primero; conviene medir empates, especialmente p numéricamente cero, antes de atribuirles un efecto.

## Qué conserva y qué no puede explicar una inspección aislada

La estructura solo crece: no hay poda, reemplazo ni revisión de una regla antigua. Los hijos eliminan la relación utilizada y comienzan con estadísticas nuevas. Con `inherit_q_values=true`, **todos heredan exactamente el Q del padre**. Por ello, dividir no cambia inmediatamente las predicciones: cualquier efecto surge de actualizaciones diferenciadas posteriores. Fuentes: [hijos históricos](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/relational_regresion_tree.py:430) y [hijos ANOVA](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/baseline_anova.py:94).

Una raíz centrada en movimiento de la paleta no demuestra un error: descendientes y valores pueden recuperar una política útil. Tampoco más hojas demuestran mejor aprendizaje, ni la ausencia de nuevos splits implica valores congelados. Cuando falta una relación, el árbol devuelve Q del nodo interno; [el adaptador cuenta estos fallbacks](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/baseline_adapter.py:139).

La suavización de Q y su tendencia temporal podrían ocultar separación de targets o producir asociaciones con el momento de visita. Es una hipótesis que exige medir targets y errores por categoría; no se deduce únicamente de un p grande o una política mala.

## Lectura correcta de las curvas

La exploración es **exponencial, no lineal**:

\[
\epsilon_t=\max(0.1,\;0.1^{t/50000}).
\]

Empieza en1, vale aproximadamente0,316 en25k y llega a0,1 en50k. Después permanece en0,1. Fuente real del runner: [OnlineSession.epsilon](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/online/session.py:61). El checkpoint0 tiene Q iniciales iguales a0 y ninguna regla; incluso su evaluación greedy desempata aleatoriamente entre acciones. [Inicialización](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/relational_regresion_tree.py:87), [raíces](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/baseline_adapter.py:102) y [GreedyController](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/experiments/compare_pong_progressive.py:217).

El entrenamiento recibe sign(puntaje); la evaluación suma puntaje bruto. Se usa frameskip4, sticky actions0 y reset sin FIRE automático, según [contrato del entorno](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/online/environments.py:47). Así, la exploración puede relanzar la pelota o escapar de ciclos que la política greedy repite. Diez semillas de evaluación no garantizan diez trayectorias distintas; verificar retornos, duraciones y trazas. La unidad independiente principal de comparación es la semilla de entrenamiento. Los episodios incompletos y los limitados por horizonte se reportan por separado; no imputarles partidas terminadas.

## Experimentos antes de modificar baseline, en prioridad

1. **Políticas congeladas y arranques diversos.** Reutilizar checkpoints0/100k/500k/1M y vecinos conservados. Evaluar ambas variantes desde el mismo panel de estados completos del emulador con diversidad verificada. Comparar greedy, epsilon pequeño y0,1, sin actualizar. Registrar retorno bruto, duración, vidas, FIRE, tiempo sin pelota, pérdidas, truncaciones y trayectoria. Distingue fragilidad de la política, dependencia de exploración y duplicación de episodios. Los cambios de arranque/exploración son una evaluación diagnóstica nueva, no reemplazan el benchmark congelado.
2. **Auditoría común de hojas y criterio.** Dataset de transiciones de comportamiento mixto, dividido por episodios. Por acción/hoja/candidata registrar n por categoría, B/T, ambos p, Q actualizado, target, error TD y fallback. Medir reducción del error de targets en episodios apartados al subdividir. Separa poca cobertura de señal insensible; no usar solo el error de entrenamiento ni confundir ajuste Bellman con retorno.
3. **Control heurístico y suficiencia de representación.** Comparar un controlador que usa únicamente las relaciones actuales con otro que usa coordenadas/movimiento físico. Buscar estados con igual codificación y acciones útiles distintas; medir éxito cerrado en el emulador. Un heurístico exitoso demuestra utilidad práctica; uno fallido no prueba imposibilidad representacional.
4. **Continuaciones desde el mismo checkpoint.** Control normal frente a crecimiento congelado, conservando TD, exploración y presupuesto; por separado, refinación puntual por posición en una rama problemática con Q heredado. Evaluar varias continuaciones y semillas, incluyendo buenas y malas seleccionadas por un criterio previo. Permite probar efectos del crecimiento y la partición. Observar justo antes/después del split no basta porque inicialmente Q no cambia.
5. **Ablaciones pequeñas posteriores.** Según lo anterior, variar por separado mínimo de muestras, calendario de exploración o señal para dividir (Q frente a target/error). Mantener presupuesto y evaluación comunes, medir calidad/cobertura de nuevas hojas y usar desarrollo para elegir; reservar test para la variante final. Comparar semillas pareadas y reportar diferencias e incertidumbre, sin tratar episodios repetidos de un agente como entrenamientos independientes.

Los experimentos1–3 determinan qué mecanismo merece intervención. Aumentar profundidad o entrenar más sin estos controles podría producir más árbol o más experiencia sin resolver la limitación concreta.
