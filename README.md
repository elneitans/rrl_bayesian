# Learning Rules from Rewards

## Implementación RRTL-B: H0–H7

Se añadió un baseline corregido y configurable, con pruebas de TD, terminales,
registros completos y evaluación del checkpoint final. El paquete
`bayesian_rrtl` incluye el encoder relacional, reglas y árboles binarios
independientes. H2 añade inferencia bayesiana con un árbol: priors, marginal
gaussiana, grow/prune, muestreo de hojas y ruido. H3 implementa el ensemble
BART aditivo, backfitting y diagnósticos de varias cadenas. H5 integra fitted Q
por bloques, replay, targets congelados y checkpoints con reanudación exacta
validada en un MDP controlado. H6 conecta Breakout real, perfila el costo y añade
un piloto con baseline pareado y diagnóstico de mezcla. Usa el extractor visual
actual mediante un contrato de objetos preparado para un futuro adaptador
OCAtari. H4 añade General BART lineal y change/swap verificados. H7 implementa
controles, ablaciones y campaña multijuego con test separado; se ejecutó una
campaña corta de desarrollo. OCAtari y el estudio de eficacia a escala siguen
pendientes; los diagnósticos cortos no prueban convergencia de las posteriores.

- [Guía de H0/H1 y validación](docs/h0_h1.md)
- [Guía de H2 y validación matemática](docs/h2.md)
- [Guía de H3: ensemble y diagnósticos](docs/h3.md)
- [Guía de H5: fitted Q, checkpoints y reanudación](docs/h5.md)
- [Guía de H6: piloto Atari y contrato para OCAtari](docs/h6.md)
- [Guía de H4: General BART y revisión estructural](docs/h4.md)
- [Guía de H7: controles, ablaciones y campaña multijuego](docs/h7.md)
- [Plan de implementación versionado](plan_implementacion_rrtl_bart.md)
- [Configuración de prueba corta](configs/breakout_smoke.json)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m experiments.validate_h2 --output results/h2_validation
.venv/bin/python train_baseline.py --config configs/breakout_smoke.json
.venv/bin/python train_bayesian.py --config configs/bayesian_toy.json --output results/h5_toy --steps 600
.venv/bin/python train_atari.py --config configs/bayesian_breakout_pilot.json --output results/breakout_bayes
```

Para otra ejecución, usar `--run 2` o cambiar `results_dir` en el JSON; no se
sobrescriben resultados existentes. Graphviz requiere el ejecutable `dot` para
exportar PDF. Las pruebas completas incluyen Atari y Graphviz.

## Código histórico (`rrtl_legacy`, commit `cf58dc7`)

El agente, el preprocesamiento y el runner siguientes permanecen sin cambios.

Codebase for the paper [Learning Rules from Rewards](https://arxiv.org/abs/2203.13599).

The `train_and_test.py` script allows to reproduce the primary results in the paper. The script requires three arguments:
1. Game name. Either 'Breakout', 'Pong' or 'DemmonAttack'
2. Model version. Either 'comparative' or 'logical'.
3. Number of runs.

For example, to replicate the results of the paper for the comparative version of RRTL on Breakout run:
```
train_and_test.py --game Breakout --model_version comparative --runs 100
```
This script uses the default values for the simulations reported in the paper.

The data used in the results of the paper can be found in: https://osf.io/m4yf9/

## Prerequisites

- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [PIL](https://pillow.readthedocs.io/en/3.1.x/installation.html)
- [Matplotlib](https://matplotlib.org/)
- [OpenCV-python](https://anaconda.org/conda-forge/opencv)
- [Graphviz-python](https://anaconda.org/conda-forge/graphviz)
