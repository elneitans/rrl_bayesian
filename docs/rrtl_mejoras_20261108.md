# Mejoras de RRTL posteriores al diagnóstico de árboles

**Actualización del 9 de noviembre:** la evaluación declarada en este diseño está completa: 70 continuaciones, 7M transiciones y 4.200 evaluaciones. El [informe de cierre](../analysis/improvements_20261108/informe.md) contiene los resultados de las cinco semillas por criterio, la recuperación de la cuota y la auditoría de los 63 cambios aceptados. El texto siguiente conserva el diseño fijado antes de la corrida y el registro de su envío inicial.

## Hipótesis y alcance

El diagnóstico anterior mostró caídas con estructura fija y una poda que redujo MSE pero elevó FIRE, perjudicando el retorno. También mezclaba eliminar condiciones y reajustar Q. Se implementan dos mejoras experimentales, con controles que separan sus componentes. No se presupone que mejorarán todas las semillas.

1. **TD estabilizado:** paso 0.005, frente a 0.025, y un bosque objetivo copiado cada 5000 decisiones. La acción del bootstrap se selecciona con Q online y se valora con el objetivo. Es una adaptación a árboles de la idea de [Double DQN](https://arxiv.org/abs/1509.06461), no una red neuronal. Los objetivos periódicos también forman parte de [DQN](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf). No hemos demostrado sobreestimación respecto del Q verdadero en nuestro juego; esta es una hipótesis que ponemos a prueba.
2. **Revisión protegida:** las hojas propuestas se ajustan con targets, proyectando el nuevo valor al intervalo de predicciones previas observado en sus datos de ajuste. Después se comprueban pérdida Bellman, cambio de política en sondas independientes y retorno pareado desde snapshots de validación. Solo se instala una propuesta que pase los tres controles. La evaluación reportada utiliza otro banco, nunca consultado para aceptar cambios.

Para empates del bootstrap Double se promedia el objetivo sobre todas las acciones maximales online. Esto evita introducir otro stream que altere la exploración. Los terminales reales no consultan el siguiente estado; las truncaciones conservan el contrato actual. El bosque objetivo no se sincroniza al aceptar una revisión, sino en su cadencia declarada.

## Comparación fijada antes de las corridas

| Variante | Paso TD | Bootstrap | Mantenimiento |
|---|---:|---|---|
| baseline |0.025|online|Crecimiento original|
| slow_td |0.005|online|Crecimiento original|
| double_td |0.025|objetivo periódico Double|Crecimiento original|
| stable_td |0.005|objetivo periódico Double|Crecimiento original|
| guarded |0.025|online|Poda/reemplazo protegidos y crecimiento|
| stable_guarded |0.005|objetivo periódico Double|Poda/reemplazo protegidos y crecimiento|
| refit_only |0.025|online|Reestima Q de hojas existentes; conserva estructura y estadísticas|

`refit_only` es un control diagnóstico: permite saltos de valor con el gate de pérdida anterior, para estudiar el reajuste separado de la eliminación de condiciones. No se presenta como una mejora validada. La señal original del test de crecimiento se conserva para atribuir los cambios a TD y mantenimiento.

## Controles del mantenimiento

Buffer de entrenamiento:8192 transiciones; mantenimiento cada25k; mínimos128 de ajuste,64 de validación y4 bloques. Se revisan como máximo dos propuestas por mantenimiento y se acepta como máximo una. Las candidatas se ordenan usando únicamente datos de entrenamiento, mediante cambio de MSE más0.0001 por hoja adicional. La poda puede eliminar ramas; el reemplazo tiene profundidad local máxima2 y otra relación raíz. La espera para recrecer es10k.

El gate Bellman mantiene el margen de poda0.0025 y exige mejora por bloques de0.001 para reemplazar. La protección de política limita a0.05 la distancia media de variación total entre distribuciones greedy, con igual peso por bloque de sonda.

El gate de retorno compara candidato y referencia desde seis snapshots compartidos de validación, horizonte1500 y streams pareados. Exige media de diferencia no negativa, cota bootstrap unilateral90% no menor que−1 punto y ninguna caída por arranque mayor que el máximo de5 puntos y20% del retorno anterior. Son umbrales operativos de un piloto, no una garantía estadística de mejora. No se ajustarán tras mirar resultados.

Los bancos de evaluación, probes y validación de retorno tienen RAM disjunta. Hay un banco de retorno nuevo por mantenimiento, compartido entre variantes para comparar con las mismas situaciones. Los rollouts de validación no entrenan Q ni modifican RNG/estadísticas del agente. Se guardan los datos, referencias online/objetivo, propuestas, motivos de rechazo, retornos y árboles.

## Cómo se decide podar o reestructurar

La poda sustituye un subárbol por una hoja: elimina condiciones que ya no justifican su complejidad. El reemplazo propone otra relación raíz y un subárbol local de profundidad máxima 2. Ambas operaciones conservan el crecimiento posterior, con una espera de 10k decisiones en los nodos intervenidos. Por tanto, una decisión eliminada puede volver a aprenderse si los datos futuros la justifican.

No se acepta una poda simplemente porque la división deje de superar el ANOVA que la creó. El contraste de crecimiento y la comparación de dos árboles cumplen funciones distintas; además, no rechazar una hipótesis de igualdad no demuestra que las ramas sean equivalentes.

Para cada bloque de validación calculamos, con una referencia Q fija, la diferencia entre pérdidas:

\[
 d_b=\frac{1}{|b|}\sum_{i\in b}
 \left[(Q_{\mathrm{candidato}}(s_i,a_i)-y_i)^2
       -(Q_{\mathrm{actual}}(s_i,a_i)-y_i)^2\right].
\]

Un bootstrap por bloques produce una cota superior unilateral del 95% para la media de estas diferencias. Se exige que sea menor que 0.0025 para podar, o menor que −0.001 para reemplazar. La poda tolera así un pequeño aumento de error a cambio de simplificar; el reemplazo exige una mejora. La referencia usada para construir los targets es la misma para ambos árboles.

Ese primer filtro no controla el orden entre acciones. En el diagnóstico anterior, una poda de la rama FIRE redujo la MSE de 0.05960 a 0.02113 y, aun así, el retorno medio inmediato cayó de 63.33 a 17.83. La Q de FIRE subió por encima de los valores anteriores y pasó a ganar más comparaciones greedy. Por eso la nueva variante acota los valores propuestos, limita la distancia entre políticas y valida también el retorno. Los resultados y las condiciones de aquel contraste están en `analysis/revision_20261106/informe.md`.

Todos estos controles son exploratorios y dependen de las situaciones observadas. La repetición de revisiones, los pocos arranques y la dependencia temporal impiden presentarlos como garantía de significancia o mejora del retorno. Tampoco resuelven por sí solos las caídas causadas por actualizaciones TD con estructura fija; las variantes de TD separan esa hipótesis.

## Corrida nueva

Breakout/OCAtari, criterios histórico y ANOVA; semillas710602,710603,710607,710608,710610. Las tres usadas anteriormente se etiquetan desarrollo;710602/710608 son confirmación respecto del desarrollo de las revisiones, sin llamarlas un test ciego ni usar el test reservado del proyecto.

Cada variante continúa desde500k durante100k decisiones, manteniendo exploración0.1. Son70 continuaciones y7M transiciones. Se evalúa greedy a0/25k/50k/75k/100k, desde12 arranques alcanzables distintos, con horizonte3000:4200 rollouts de evaluación. Se reportarán todos los puntos, medianas/rangos por arranque, podas/reemplazos/refits efectivos y resultados por rol de semilla; no se seleccionará el mejor checkpoint retrospectivamente como comparación principal.

Antes del envío principal: pruebas unitarias, integración real reducida de las siete variantes para ambos criterios, reanudación y equivalencia del baseline con las transiciones originales. El código se congela en una carpeta nueva del cluster. Los modelos originales y la campaña del6 de noviembre son solo lectura. Se usa el entorno Conda existente, sin nuevas dependencias.

Los workers usan2CPU/8GB y hasta8h por grupo criterio/semilla; concurrencia inicial2. El ledger impide duplicar envíos. Se encadenan reporte y análisis; no se cancelan ni alteran jobs de otros proyectos. Los resultados de eficacia estarán pendientes cuando arranque la corrida; el envío no los demuestra.

## Código y reproducción

- `bayesian_rrtl/stable_revision.py`: variantes TD, bosque objetivo, propuestas acotadas, refit, protección de política y gate de retorno.
- `bayesian_rrtl/tree_revision.py`: hooks de bootstrap con comportamiento original por defecto; los nuevos forks regeneran IDs de nodos incluso si el padre ya estaba instrumentado.
- `experiments/diagnose_rrtl_improvements.py`: bancos disjuntos, entrenamiento, checkpoints, trazas, validación y reporte completo. Reutiliza almacenamiento y restauración de la campaña anterior con clase/schema explícitos.
- `experiments/analyze_rrtl_improvements.py`: curvas y comparación final pareada, separando desarrollo/confirmación y medias/medianas.
- `configs/breakout_rrtl_improvements_20261108.json`: parámetros completos de las siete variantes.
- `slurm/54_prepare_rrtl_improvements.sbatch`, `55_rrtl_improvement_worker.sbatch`, `56_submit_rrtl_improvements.sh`, `57_analyze_rrtl_improvements.sbatch`: DAG operativo.

Las campañas anteriores se reproducen con sus snapshots de código, en particular `~/archive/rrl_revision_code_20261106_v2`. El working tree ahora incluye hooks nuevos; las validaciones de procedencia de aquellos experimentos deben conservar sus fuentes originales. Esta campaña usa otra carpeta, manifiesto y esquema; no evita esas comprobaciones para reanudar resultados antiguos.

## Ejecución verificada el 8 de noviembre

La preparación Linux terminó correctamente en el job **37518**: 122 pruebas, integración Atari de las 14 combinaciones de variante y criterio, 14.000 transiciones y 140 rollouts cortos. Los dos controles reprodujeron exactamente sus 2.000 transiciones originales. También se comprobó que evaluar un clon idéntico desde los mismos arranques da diferencia de retorno cero y trayectorias idénticas, sin modificar el agente de entrenamiento.

El job **37603** completó una prueba adicional de reemplazo protegido y una auditoría independiente de los 13 cambios aceptados en la integración. Se reconstruyeron pérdidas, cambios de política, acotamiento de Q y conservación del bosque objetivo o de la estructura, según la variante. La integración usa horizontes y soportes reducidos: comprueba funcionamiento, no eficacia.

La comparación principal está enviada y ejecutándose:

| Job | Trabajo | Estado verificado al envío |
|---|---|---|
| 37604, array 0–9%2 | Diez grupos; siete variantes secuenciales por grupo | Dos tareas corriendo, ocho pendientes |
| 37605 | Reporte completo, integridad y equivalencia con controles originales | Espera a que termine el array, incluso si algún worker falla |
| 37606 | Curvas y comparación pareada por criterio/rol de semilla | Espera al reporte exitoso |
| 37611 | Auditoría independiente de las revisiones aceptadas | Espera al reporte exitoso |

En las primeras dos continuaciones de control ya se observaron 100.000 decisiones nuevas por semilla. El estado de entrenamiento y la evaluación final se registran por separado; un contador de 600k no implica que haya acabado todo el worker ni sus siete variantes. No hay todavía una conclusión de eficacia de esta campaña. La evidencia local de ejecución está en `analysis/improvements_20261108/main_progress/` y `implementation_review.json` dentro de la misma carpeta de análisis.

Código científico congelado: `~/archive/rrl_improvement_code_20261108_v1`. Datos, logs y ledger: `~/archive/rrl_improvements_20261108_01`. SHA-256 del archivo de fuentes: `f2edc165d455ca94435a932d1f49e6aa4d9ecab4bb39fb906116a55fd67ac604`; hash del manifiesto principal: `5c68a97ac3ddd52553a705703ef4faf12d014108b41ca16e7b3b84d7eff805b4`.

El lanzador operativo se actualizó fuera de las fuentes congeladas para comprobar mediante `sacct` las dependencias exitosas que Slurm ya había retirado de su controlador. Su SHA-256 es `eea048af376f0b5940a3adebd962ecbf0f0a30eded31bbc10de9e61d463192b0`, registrado en `SUBMISSION_SCRIPT_SHA256`. La auditoría adicional usa `slurm/58_audit_rrtl_improvement_guards.sbatch` y `analysis/improvements_20261108/audit_guards.py`, copiados al directorio de control. No se cambió el código científico ni los parámetros después de congelarlos.

Para consultar el estado sin volver a enviar:

```bash
squeue -j 37604,37605,37606,37611
sacct -X -j 37518,37603,37604,37605,37606,37611 \
  --format=JobID,State,ExitCode,Elapsed
```

El comando usado para el envío principal fue el siguiente. El ledger reutiliza los IDs existentes; no crea otra campaña ni sustituye una recuperación de workers fallidos:

```bash
export RRL_ROOT="$HOME/archive/rrl_improvement_code_20261108_v1"
export RRL_IMPROVEMENT_CONTROL="$HOME/archive/rrl_improvements_20261108_01"
export RRL_CONDA_ENV="$HOME/archive/conda_envs/rrl-pong-analysis"
bash "$RRL_IMPROVEMENT_CONTROL/56_submit_rrtl_improvements.sh"
```
