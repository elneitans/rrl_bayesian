# Corrección de mediciones B12 y diagnóstico A4 — 14/10/2026

## B12: intento sin cierre final

`measurements()` y la consolidación conservan los intentos incompletos, tanto nuevos como antiguos sin campos finales. Fin, duración y RSS desconocidos son `null`; no se sustituyen por cero ni por el reloj del siguiente intento. Se conserva aparte el máximo RSS medido y el tiempo acumulado registrado como límite inferior. La incertidumbre también se propaga al reporte multijuego. El lector no modifica el intento muerto.

El estimador no divide una duración desconocida ni calcula mínimos/máximos sobre `None`. Puede usar otras mediciones completas compatibles, indicando cuántos registros omitió; si no hay una tasa completa, el costo sigue desconocido. No se habilita una campaña larga con datos de costo inventados.

La prueba de regresión ejecuta un proceso con entorno sintético, envía SIGKILL a sí mismo después del checkpoint 16 y reanuda en otro proceso hasta 64. Verifica transiciones sin duplicar, conservación del intento original y exportación de mediciones. También comprueba lectura de registros antiguos sin campos finales y mezcla de mediciones completas/incompletas.

## A4: comparación condicionada por acción

La auditoría v2 separa colisiones de geometría y diferencias de evolución para `(estado codificado, acción)`. Un resultado siguiente distinto después de otra acción no es un ejemplo de pérdida de información. Geometría distinta demuestra una distinción omitida por el encoding, sin probar su relevancia para decidir; variación de evolución con la misma acción tampoco prueba aliasing por sí sola.

Las pruebas incluyen igual estado/geometría con acciones distintas, comparación de una acción que no aparece en el primer representante del estado, diferencias geométricas y JSON equivalente con claves en distinto orden.

## Validación e historia

Batería final: **183 passed, 1 skipped, 2 deselected**, en 47,61 segundos. Incluye B online, A runtime/blocks, C, protocolos, entrenamiento, baseline, auditoría BART y batches. La prueba omitida requiere habilitación explícita del emulador; los dos casos deseleccionados son Atari. Ruff y la comprobación de espacios de los archivos modificados pasaron. XML local: `results/review_sigkill_a4_20261014/pytest.xml`.

Antes de cambiar las fuentes se copiaron y verificaron los 88 hashes de la campaña C existente en `historical/atari_common_C_20261014_code/`. Esa copia local conserva el código necesario para abrir los resultados anteriores; no se migraron checkpoints ni se relajaron verificaciones. Los manifiestos anteriores permanecen intactos. No se ejecutó campaña larga ni test reservado real.
