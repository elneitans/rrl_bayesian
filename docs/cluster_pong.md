# Pong en Slurm + Conda

Esta entrega prepara jobs; no envía ninguno. CPU únicamente, un coordinador
secuencial por campaña, sin job arrays ni ejecución de test reservado. El piloto
es exploratorio: los diagnósticos de mezcla anteriores no acreditan convergencia.

## 1. Subir e instalar

Copiar el repo completo `rrl_bayesian` (incluidos archivos nuevos no commiteados),
excepto `.venv`, cachés y resultados locales. Usar una copia de código estable
durante toda la campaña: los hashes de entrenamiento protegen resume. No mover
checkpoints macOS a Linux esperando continuidad exacta; iniciar campañas nuevas.

Desde la raíz del repo en el clúster, tras cargar Conda con el mecanismo local:

```bash
conda env create -f environment-cluster.yml
conda activate rrl-pong
python -m pip check
```

El YAML fija Python 3.11.7 y reutiliza `requirements-ocatari.txt`, que incluye
los pins directos de `requirements.txt`. Conda aporta libGL/GLib para importar
el OpenCV existente sin cambiar su distribución. No se instala CUDA.

Esto es una **receta de instalación, no un lock Linux ya validado**. La resolución
de transitivas debe pasar preflight. Cada job guarda `pip-freeze.txt` y
`conda-explicit.txt` en `results/cluster_jobs/JOB_ID/`; conservar ambos para
reproducir el entorno Linux validado (el segundo no sustituye los paquetes pip).
No reinstalar dependencias dentro de cada job ni actualizar el entorno entre runs.
Si el clúster carece de acceso a Internet, preparar paquetes por los mecanismos
permitidos por sus administradores. El preflight debe fallar si falta la ROM,
OCAtari o una biblioteca; no hay skips ni fallback al extractor visual.

## 2. Recursos y rutas

Enviar desde la raíz `rrl_bayesian`. `RRL_ROOT` permite indicar otra ruta absoluta;
`RRL_CONDA_ENV` permite un nombre o prefijo Conda distinto de `rrl-pong`.
Los resultados deben estar en almacenamiento persistente compartido, no en `/tmp`
ni en scratch efímero del nodo. El filesystem debe soportar `flock` entre nodos;
confirmarlo con el administrador. No mezclar estos wrappers con coordinadores
manuales sobre la misma campaña: esos coordinadores no adquieren el bloqueo.

Los batches omiten partición/cuenta/QoS: proporcionarlos a `sbatch` según las
políticas locales. El ejemplo recibido usaba `ialab-low`, `default-account` y
`regular`; confirmar que permiten jobs CPU antes de reutilizarlos. No se fija
el nodo `hydra` ni se pide GPU.

Recursos iniciales: 2 CPU, 16 GB preflight/smoke y 20 GB piloto/resume. BLAS usa
un hilo. Son solicitudes provisionales, no mediciones Linux. Revisar MaxRSS,
tiempo y espacio de checkpoints tras el smoke. Los límites Slurm de 1/2/12 horas
son editables mediante `sbatch --time=...`, sujetos a la partición.

## 3. Preflight y smoke

```bash
# Agregar --partition=... --account=... --qos=... si son necesarios.
sbatch slurm/00_preflight.sbatch
```

Esperar a que termine bien. El preflight ejecuta pruebas de ambos extractores,
inputs pareados, checkpoints en otro proceso, interrupción/reanudación del
runner, flags y protocolo. Usa seeds de desarrollo de las pruebas; no ejecuta
`--stage test`. Escribe un JUnit XML y expande el piloto con `--dry-run`.
No prueba por sí solo que el almacenamiento soporte locks entre dos nodos.

```bash
export RRL_OUTPUT="$PWD/results/pong_smoke_linux_01"
sbatch slurm/01_smoke.sbatch
```

La carpeta debe ser nueva. Se ejecutan las cuatro celdas del smoke existente.
Verificar `campaign_status.json`, los cuatro `summary.json` y
`report_validation.json`. Confirmar memoria/costo y que los 4 estados sean
`completed` antes del piloto.

Para verificar reanudación **entre jobs**, ejecutar después:

```bash
# Mantener RRL_OUTPUT apuntando al smoke.
sbatch slurm/03_resume.sbatch
```

Con el smoke completo no debe reentrenar ni reevaluar. Las pruebas preflight
verifican además continuación exacta de runs incompletos entre procesos. Para
probar un corte real de Slurm, usar una campaña smoke descartable, esperar a un
checkpoint publicado, cancelar ese job y enviar resume sobre esa carpeta.
No cancelar el piloto como prueba y no editar manualmente sus estados/hashes.

## 4. Piloto (autorización explícita)

```bash
export RRL_OUTPUT="$PWD/results/pong_pilot_linux_01"
export RRL_CONFIRM_PILOT=yes
sbatch slurm/02_pilot.sbatch
```

Usa exactamente `configs/comparison_pong_pilot.json`: 12 runs, 20.480 decisiones
por run, 600 episodios de validación en total. La confirmación es una protección
del batch, no una garantía científica. No se cambia el método ni se consume test.
El runner publica el reporte de validación al completar todas las celdas.

## 5. Reanudación y presupuestos

```bash
export RRL_OUTPUT="$PWD/results/pong_pilot_linux_01"
sbatch slurm/03_resume.sbatch
```

Si se agotaron los presupuestos del runner, autorizar límites acumulados mayores:

```bash
# Ejemplo de ampliación, NO presupuesto recomendado sin medir el smoke Linux.
export RRL_TRAINING_SECONDS=10800
export RRL_EVALUATION_SECONDS=21600
export RRL_BUDGET_REASON='Ampliación autorizada tras revisar el costo Linux'
sbatch slurm/03_resume.sbatch
# Evitar reutilizar accidentalmente esta ampliación en un envío posterior.
unset RRL_TRAINING_SECONDS RRL_EVALUATION_SECONDS RRL_BUDGET_REASON
```

Estos límites son totales acumulados: entrenamiento por run y evaluación por
campaña. No son horas adicionales ni alteran el protocolo científico. Cada nueva
ampliación debe superar el límite vigente y queda registrada por el runner.

El límite Slurm es independiente. Si Slurm mata el job, se recupera desde el
último checkpoint publicado; trabajo posterior y una evaluación aún no publicada
pueden repetirse. No se promete guardado durante SIGKILL. El checkpoint_interval
del protocolo limita cuánto entrenamiento se pierde; no modificarlo a mitad.
No activar requeue automática sin un mecanismo que seleccione `resume`.

El wrapper devuelve 3 si la campaña queda parcial aunque el runner retorne 0;
Slurm mostrará FAILED en ese caso. Consultar los estados antes de interpretarlo
como fallo de aprendizaje. Un timeout Slurm puede dejar estado de campaña antiguo:
los checkpoints/estados por run son la fuente para resume.

## 6. Validación local de los archivos de despliegue

```bash
python -m pytest tests/test_cluster_launch.py -q
bash -n slurm/common.sh
```

No se ha validado aquí una instalación Linux/Conda real ni se ha enviado un job.
No comparar costos concurrentes con los secuenciales originales, ni asumir que
las estimaciones del Mac se trasladan al clúster.
