# Análisis de los diagnósticos BART de Pong OCAtari

Fecha: 26 de octubre de 2026. Datos: `results_cluster/pong_analysis_v2_linux_20261025_01`. Esta revisión no cambió el algoritmo ni envió jobs.

## Conclusión

La ejecución completó lo solicitado, pero la inferencia MCMC es insuficiente con esta receta. El problema no se limita a métricas estructurales: las predicciones de Q también dependen de la cadena y de la inicialización, particularmente para dos estados raros ausentes del lote. No es defendible presentar estos draws como una aproximación posterior ya estabilizada ni autorizar a partir de ellos una campaña masiva de BART.

Los resultados no demuestran que BART sea incompatible con RL, que su matemática esté mal, que sigma² haya explotado, ni que el fallo sea sobreajuste. Sí proporcionan evidencia directa de baja eficiencia de mezcla, extrapolación poco informada y un coste relevante incluso en problemas diminutos.

## 1. Qué experimento estamos analizando

No es BART entrenado a 2M ni una nueva política jugando. Se tomó el último problema supervisado del piloto OCAtari/BART/71: interacción 20.480, ajuste 40, targets construidos con el modelo 39. X e y se congelaron y se volvieron a ajustar repetidamente.

- Acciones auditadas: NOOP (0), FIRE (1) y RIGHTFIRE (4), seleccionadas por cobertura del replay y recompensas, no por mejor resultado.
- 128 filas por acción; 20 relaciones con disponibilidad explícita.
- 20 árboles por ensemble, profundidad máxima 4 y mínimo de 2 observaciones por hijo.
- Kernel `grow_prune`; `change` y `swap` no se usaron.
- Prior estructural alpha=0,95, beta=2; k=2; prior de ruido con nu=3, calibración q=0,9 y floor=1e-8 en unidades normalizadas.
- Dos inicializaciones: árboles de una hoja centrados en la media del target (`stumps`) y árboles/hojas dispersos válidos (`dispersed`). Cambia el inicio, no el posterior objetivo.
- Cuatro cadenas por problema, 1.000 barridos de burn-in y 2.000 retenidos: seis problemas, 24 cadenas y 48.000 draws retenidos de ensemble.

Las semillas se reutilizan entre las dos inicializaciones para el contraste de inicios; no sumé esos grupos para afirmar ocho cadenas independientes ni un ESS conjunto.

Las fórmulas y la instrumentación están en [audit_bart_fixed_data.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/experiments/audit_bart_fixed_data.py), [ensemble.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/ensemble.py), [sampler.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/sampler.py) y [diagnostics.py](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/bayesian_rrtl/diagnostics.py). La ejecución usó la copia histórica preservada; no confundirla con una migración del piloto al código nuevo.

## 2. Verificación de artefactos y diagnóstico

Se verificaron los SHA-256 de los dos NPZ de datos y de las 24 trazas, los checksums de identidades y estados, la correspondencia con los metadatos completos, las longitudes y los hashes de X, disponibilidad e y por cadena. Se verificó que los grupos comparados conservan la misma escala y los targets registrados. No se deserializaron pickles de modelos.

Se recalcularon R-hat, ESS bulk y ESS tail de los 48 observables: ocho por problema, seis problemas. Coinciden numéricamente con los reportes. Para evitar un bloqueo local al importar SciPy, la comprobación adicional usó las mismas fórmulas verificadas con FFT de NumPy, rangos medios de NumPy y cuantiles normales de la biblioteca estándar; esta diferencia de implementación no cambió los diagnósticos. No se modificaron las librerías del proyecto.

La metodología rank-normalized/folded split R-hat y ESS se describe en [Vehtari et al.](https://arxiv.org/abs/1903.08008) y en su [apéndice reproducible](https://avehtari.github.io/rhat_ess/rhat_ess.html). Nuestro protocolo exige R-hat <= 1,01 y ESS bulk/tail >= 400. Son criterios operativos, no una prueba absoluta de convergencia.

| Acción | Inicio | R-hat de sigma² | ESS bulk de sigma² | R-hat de Q, referencia rara 3 |
| --- | --- | ---: | ---: | ---: |
| NOOP | stumps | 1,051 | 53,5 | 1,221 |
| NOOP | dispersed | 1,089 | 28,4 | 1,820 |
| FIRE | stumps | 1,132 | 19,7 | 1,998 |
| FIRE | dispersed | 1,128 | 20,2 | 2,324 |
| RIGHTFIRE | stumps | 1,718 | 6,1 | 1,208 |
| RIGHTFIRE | dispersed | 1,472 | 7,6 | 3,057 |

Ninguno de los seis problemas pasa el criterio global. Ninguna de las 24 comprobaciones predictivas pasa simultáneamente todos los umbrales. Esto no significa que las 24 sean igual de malas: FIRE/dispersed/referencia 1 está cerca, con R-hat=1,0099, ESS bulk=564 y ESS tail=378. El fallo es mucho más grave en estados raros y en algunos modos estructurales.

8.000 draws de sigma² por problema no equivalen a 8.000 observaciones independientes. Para RIGHTFIRE/stumps el ESS bulk de sigma² es sólo 6,1. Esa cifra no equivale a tener sólo seis barridos ni permite proyectar linealmente cuánto falta para converger.

Un diagnóstico exploratorio con sólo los últimos 500 draws no arregla el problema: R-hat de sigma² en RIGHTFIRE sigue en 1,81 y 1,63, y el de Q raro/dispersed llega a 3,28. No se sustituyó el burn-in oficial ni se eligió una ventana para declarar éxito.

## 3. Dónde aparece el problema predictivo

Se monitorearon cuatro estados: dos frecuentes, observados 926 y 918 veces en el replay, y dos raros, observados una sola vez cada uno. Los frecuentes aparecen entre 2 y 10 veces por lote según la acción. Los dos raros no aparecen en ninguno de los tres lotes de 128 filas.

Por tanto, son extrapolaciones fuera del soporte observado del lote, no necesariamente estados imposibles del juego. Tener incertidumbre amplia allí puede ser legítimo; lo problemático es no explorar de forma estable esa distribución.

En RIGHTFIRE, referencia rara 3, las medias por cadena son:

| Inicio | Cadena 0 | Cadena 1 | Cadena 2 | Cadena 3 |
| --- | ---: | ---: | ---: | ---: |
| stumps | 5,76 | 5,52 | 5,73 | 5,51 |
| dispersed | 11,89 | 8,63 | 15,14 | 5,86 |

El mayor valor observado del target de RIGHTFIRE es 6,26, aunque un Q legítimo de un estado no observado no tiene por qué estar acotado por el máximo de los targets de entrenamiento. La comparación no prueba que 15,14 sea un Q verdadero incorrecto: demuestra dependencia del inicio y falta de evidencia para estimarlo con precisión.

La diferencia persiste en la última ventana: la tercera cadena dispersa pasa de media 17,35 en sus primeros 500 draws retenidos a 14,34 en los últimos 500. No parece una fluctuación alrededor de una única media común ya establecida.

Las predicciones frecuentes también muestran R-hat insuficiente, normalmente con diferencias absolutas menores. Por eso no es correcto descartar todo como un problema puramente estructural o como permutación de etiquetas entre árboles: Q y sigma² son observables invariantes a permutar árboles.

En una política greedy, diferencias de estimación de Q pueden cambiar el argmax. Pero sólo auditamos tres de seis acciones y cuatro estados: no calculé probabilidades fiables de elegir la acción óptima ni afirmo haber demostrado cambios concretos de la política final.

Figura: [trazas RIGHTFIRE](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/results_cluster/pong_analysis_v2_linux_20261025_01/analysis_bart_20261026_v1/rightfire_traces.png). [Matriz R-hat](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/results_cluster/pong_analysis_v2_linux_20261025_01/analysis_bart_20261026_v1/rhat_matrix.png).

## 4. Sigma²: convertir las unidades cambia la lectura

La escala fija usa gamma=0,99 y recompensas en [-1,1]: límites teóricos [-100,100], centro 0 y ancho W=200. El sampler trabaja con y*=y/200, luego:

    sigma²_original = 200² × sigma²_normalizada = 40.000 × sigma²_normalizada.

| Acción | Mediana sigma² original, stumps | Mediana sigma² original, dispersed | Desviación residual aproximada |
| --- | ---: | ---: | ---: |
| NOOP | 0,0361 | 0,0370 | 0,19 |
| FIRE | 0,0209 | 0,0231 | 0,14–0,15 |
| RIGHTFIRE | 0,0923 | 0,0901 | 0,30 |

Los targets tienen desviaciones de 0,72–0,84. No estamos viendo ruido residual infinito o una varianza nula porque el número normalizado sea del orden de 1e-6. Las cifras posteriores son provisionales por la mala mezcla, pero bastan para rechazar esas lecturas numéricas simplistas.

También comprobé la actualización condicional gaussiana:

    E[sigma² | residuos] = (nu × lambda + SSE) / (nu + n - 2).

La media observada de sigma² dividida por la media condicional calculada está entre 0,9991 y 1,0013 en los seis problemas. Es una comprobación de consistencia de esa actualización, no una prueba de corrección del kernel completo ni de convergencia.

FIRE sí conserva una calibración histórica que recurrió al floor. Su varianza de calibración normalizada 1e-8 equivale a 0,0004 original, mientras su sigma² posterior mediana ronda 0,021–0,023: no quedó pegada a ese floor. NOOP y RIGHTFIRE también mezclan mal sin haber usado floor; no puede ser la explicación única.

## 5. Prior, estructuras y crítica de congelamiento

El prior de hojas tiene varianza normalizada tau²=(0,5/(k sqrt(m)))²=0,003125. Para un punto fijo, la desviación prior de la suma es 0,5/k=0,25, equivalente a **50 unidades Q**. La desviación de una hoja individual equivale a 11,18 unidades Q. El prior predictivo es muy amplio comparado con la dispersión de estos targets; no hay base para afirmar, sólo por k=2, que esté contrayendo todo excesivamente hacia cero.

Condicional a una estructura y a los otros árboles, la media posterior de una hoja pondera la media residual por:

    n_hoja × tau² / (sigma² + n_hoja × tau²).

Con dos observaciones y las medianas de sigma² registradas, ese peso supera 0,9996. Esto contradice la explicación de que una sigma² desmesurada haya hecho dominar al prior de medias en estas hojas observadas. No elimina los efectos del prior estructural, las penalizaciones marginales por nuevos splits ni la incertidumbre de extrapolación.

Las muestras retenidas tienen entre **28 y 56 hojas totales repartidas en 20 árboles**, no 28–56 hojas por árbol. Veinte stumps tendrían 20 hojas. Hay crecimiento real; los ensembles no quedaron vacíos ni todos reducidos a una sola hoja.

Las propuestas grow posibles se aceptan aproximadamente entre 1,31% y 4,96%, dependiendo de acción e inicio. Aun así hay cientos o miles de movimientos estructurales aceptados por cadena durante los 2.000 barridos retenidos. No es congelamiento absoluto. Es más preciso hablar de exploración local ineficiente y regímenes con predicciones distintas que se mantienen durante mucho tiempo.

Los inicios dispersos suelen tener más hojas y menores tasas de aceptación que los simples. `change/swap` tienen conteos cero porque el kernel solicitado no los incluye, no porque se hayan intentado y fallado. La implementación `RevisionKernel` existe, pero debe evaluarse como una variante explícita, no atribuirle resultados que no se midieron.

La calibración se calcula una vez por acción y se reutiliza entre ajustes; cada nuevo ajuste reinicia sigma² en lambda y las cadenas desde un inicio nuevo. La escala empírica de targets puede cambiar respecto de esa calibración inicial. Es una hipótesis razonable para estudiar, no un fallo causal demostrado por esta auditoría.

## 6. Calidad de los datos y targets

| Acción | Filas | Estados codificados distintos | SD target | Piso RMSE por duplicados X | RMSE mediano por draw BART, stumps |
| --- | ---: | ---: | ---: | ---: | ---: |
| NOOP | 128 | 57 | 0,763 | 0,119 | 0,189 |
| FIRE | 128 | 58 | 0,725 | 0,112 | 0,146 |
| RIGHTFIRE | 128 | 61 | 0,843 | 0,160 | 0,313 |

El piso es un cálculo descriptivo in-sample: asignar a cada X idéntico la media de sus targets. Ningún predictor determinista de ese X puede separar dos filas idénticas con targets distintos. No es un estimador del error de generalización ni prueba por sí solo de aliasing dañino para controlar el juego. Esas diferencias pueden venir de dinámica, historia omitida y targets bootstrapped.

Los promedios por encoding explican alrededor de 96,4–97,6% de la variación de los targets en estos lotes. Por eso los duplicados no autorizan concluir que la representación imposibilita ajustar el problema; queda margen entre el piso y el error actual. El RMSE por draw no es el RMSE de la media posterior y no debe confundirse con él.

En el replay de 20.480 transiciones hay sólo **5 recompensas positivas**, 479 negativas y 19.996 ceros. En los 384 ejemplos de las tres acciones auditadas hay **cero recompensas positivas, 13 negativas y cero terminales naturales**. Todos sus targets son positivos, aproximadamente entre 2,13 y 6,26, porque incorporan el bootstrap del modelo anterior.

Esto no prueba sobreestimación: desconocemos Q verdadero. Sí deja claro que ajustar esos targets no equivale a aprender directamente de partidas exitosas y que un estimador que reproduzca bien el target puede perpetuar errores del generador. Más muestras MCMC no añaden experiencia de juego.

La separación por episodios del último lote histórico no elimina la contaminación del generador de targets, que pudo entrenar con esos episodios. No presenté el ajuste histórico como una evaluación independiente de generalización. Para eso se necesitan los datasets nuevos, un generador ajustado sólo con train y episodios de evaluación separados.

## 7. Relación con las explicaciones previas

- **Confirmado:** esta receta MCMC no alcanza la estabilidad prevista, incluso con X/y completamente fijos. No se necesita deriva online de targets para que aparezca este problema.
- **Confirmado:** la inicialización importa mucho en predicciones poco informadas por el lote.
- **Confirmado:** la incertidumbre posterior calculada con estas cadenas aún no está validada para tomar decisiones.
- **No confirmado:** sigma² explota, los árboles quedan vacíos o las predicciones colapsan a cero. Los resultados observados contradicen esa descripción concreta.
- **No confirmado:** sobreajuste. Falta una comparación independiente train/holdout y de política; extrapolación sensible al inicio no lo demuestra.
- **No medido:** sensibilidad a alpha, beta, k o q. Sólo se ejecutó la condición base.
- **No demostrado:** incompatibilidad matemática general entre BART y fitted Q. Durante cada ajuste congelamos datos y targets; entre ajustes cambia el problema y esa incertidumbre no es una posterior exacta sobre el Q óptimo.

La crítica de coste y mezcla merece atención; su narrativa causal concreta debe contrastarse con estas mediciones, no asumirse como ya demostrada.

## 8. Coste: por qué no basta multiplicar los barridos

Los seis problemas consumieron unos 98,7 minutos de muestreo puro y 106,6 minutos sumando los tiempos registrados de las cadenas. Son 72.000 barridos de ensemble, con unos 0,082 segundos por barrido en este contexto. El Slurm agregado duró aproximadamente 108 minutos, sin timeout.

Sólo como escenario aritmético, no como estimación autorizada: si ese coste por barrido se mantuviera, 100 barridos por acción y seis acciones costarían unos 49 segundos por ajuste. Con un ajuste cada 512 decisiones, 2M implican unos 3.906 ajustes: aproximadamente **54 horas de muestreo por semilla**, sin emulación, guardado o evaluación. Adoptar 3.000 barridos en cada ajuste multiplicaría esa cifra por 30 y aun así aquí no aseguró mezcla.

La estructura aprendida, los datos futuros, el hardware, el kernel y la instrumentación cambian el coste. Esta cuenta no cumple las mediciones compatibles requeridas para autorizar una campaña larga ni se puede trasladar a Breakout. Su utilidad es mostrar por qué “darle mucho más MCMC” puede ser una estrategia inviable sin mejorar antes la eficiencia.

## 9. Qué haría a continuación

1. **Mantener baseline OCAtari y sus 15 semillas**, sin tocar sus manifiestos. BART sigue en diagnóstico, no en 15 entrenamientos largos.
2. **Comparar CART/ExtraTrees con estos mismos X/y** para separar dificultad del ajuste, capacidad y coste. Usar después un benchmark limpio con datos nuevos para comparar generalización y política; no confundirlo con la comparación histórica in-sample.
3. **Extender selectivamente RIGHTFIRE**, con los mismos datos y posterior, en ambos inicios, para ver si dejan de derivar. No extender todas las combinaciones ni declarar que un número prefijado de draws garantiza estabilidad.
4. **Probar un cambio de kernel declarado** con `change/swap` o una variante con menos árboles; comparar ESS por segundo y predicciones, no sólo aceptación o entrenamiento MSE. Cambiar m no estrecha automáticamente el prior de la suma porque la escala de hojas depende de m.
5. **Evaluar sensibilidad al prior/calibración**, un factor por vez y con identidad nueva cuando corresponda. Revisar especialmente escala amplia y calibración histórica. La calibración de una variante nueva sólo debe usar train. Registrar honestamente que la base fue insuficiente antes de habilitar sensibilidades; no relajar umbrales para conseguir un aprobado.
6. **Ampliar diversidad de referencias** y auditar las seis acciones antes de concluir estabilidad del argmax. Las cuatro referencias actuales no cubren todos los estados relevantes.
7. **Medir coste online compatible y fijar la receta** sólo cuando la variante candidata tenga diagnósticos defendibles. BART necesita superar ambos requisitos: inferencia razonable y presupuesto viable.

No recomiendo introducir inmediatamente una colección asistida como ventaja exclusiva de BART en la comparación online: los controles offline pueden usarla con etiqueta explícita; cualquier preentrenamiento asistido debe ser una condición experimental compartida o una ablación declarada.

## Artefactos de esta revisión

- [Resumen numérico completo](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/results_cluster/pong_analysis_v2_linux_20261025_01/analysis_bart_20261026_v1/summary.json).
- [Trazas RIGHTFIRE](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/results_cluster/pong_analysis_v2_linux_20261025_01/analysis_bart_20261026_v1/rightfire_traces.png).
- [Matriz de R-hat](/Users/natanzamorano/Documents/Universidad/capstone/rrl_bayesian/results_cluster/pong_analysis_v2_linux_20261025_01/analysis_bart_20261026_v1/rhat_matrix.png).

Los intervalos y medias guardados son descripciones de las muestras disponibles, no intervalos posteriores certificados como estables. Los artefactos de entrada y las corridas originales se conservaron sin cambios.
