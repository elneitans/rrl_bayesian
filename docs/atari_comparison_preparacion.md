# Bloque C: implementación y evidencia (14/10/2026)

C13 y C14 están integrados y verificados con emulador real. C15 tiene infraestructura, reportes y smoke completo; la campaña científica larga y el test siguen pendientes de revisión y autorización.

[Guía de reproducción](atari_comparison_guide.md) · [Auditoría](ocatari_detector_audit_c.md) · [Evidencia y hashes](atari_comparison_preparacion.json)

## Verificaciones realizadas

- Regresión común: 179 passed, 1 skipped, 2 deselected; ampliación sobre backend/pipelines previos: 142 passed. Las baterías se solapan.
- Emulador fijado: cuatro pruebas reales pasaron, incluyendo continuidad en procesos separados y acciones/FIRE de ambos juegos.
- Regresión adicional de test sellado: interrupción con un episodio completo, otro parcial y uno pendiente; reanudación sin duplicar pasos ni modificar entrenamiento. Usa sólo entorno y semillas sintéticos.
- Se conservaron y volvieron a verificar las 40 fuentes históricas del piloto y 77 fuentes de B.
- Ruff pasó y los 17 batches tienen sintaxis Bash válida y el comentario Slurm solicitado.

La validación real local corresponde a macOS arm64, Python 3.11.7, OCAtari 2.2.1, Gymnasium 1.1.1 y ALE 0.10.2. Un launcher temporal compiló las fuentes instaladas para evitar bloqueos de bytecode; no cambió detectores ni algoritmos. Linux queda pendiente del preflight preparado.

## Smoke común completado

Campaña local: `results/c15_three_games_smoke_02`. Nueve entrenamientos de 64 decisiones, 36 episodios de desarrollo de horizonte 128, cuatro ajustes por CART/BART y cero test. El wrapper pausó cada celda en 19 y la reanudó, comprobó checkpoints y reutilizó completos. Se regeneraron seis figuras y nueve exportaciones fieles sin ejecutar entornos.

| Juego | Learner | Ajustes | Retorno bruto de desarrollo final | RSS máximo (MiB) |
| --- | --- | ---: | ---: | ---: |
| Breakout | corrected_common | TD incremental | 1 | 373.5 |
| Breakout | cart | 4 | 1 | 374.6 |
| Breakout | bart | 4 | 0 | 374.6 |
| DemonAttack | corrected_common | TD incremental | 30 | 381.1 |
| DemonAttack | cart | 4 | 30 | 381.1 |
| DemonAttack | bart | 4 | 0 | 381.1 |
| Pong | corrected_common | TD incremental | -2 | 381.1 |
| Pong | cart | 4 | -2 | 381.1 |
| Pong | bart | 4 | -2 | 381.5 |

Todos esos retornos finales son limitados por horizonte. Una sola semilla por juego, modelos casi iniciales y BART reducido de smoke no permiten concluir superioridad, equivalencia ni aprendizaje útil. El JSON conserva complejidad, fases, reintentos, checkpoints, hashes, AUC y alcance. No se promedian estos puntajes entre juegos.

## Controles Fact de desarrollo

Breakout: heurística 41/41 a horizonte 1024, NOOP 0/0 al mismo horizonte, aleatorio 0/0 en partidas terminadas a 127/128 decisiones. Demon Attack: heurística 250/250 en partidas terminadas a 769, NOOP 0/0 al horizonte, aleatorio 80 (partida terminada a 982) y 100 (horizonte 1024). Todas las acciones y estados incompletos están registrados. Hay conducta útil en estas trazas; su alcance y determinismo impiden tratar esos números como prueba estadística de generalización.

## Lo que queda pendiente

La plantilla final enumera 45 celdas explícitas y continúa inactiva. Faltan revisión BART A6/A7, parámetros finales, presupuestos, calendarios, intervalo AUC y mediciones de costo Linux compatibles para todos los métodos/juegos. No se reemplazó ese costo desconocido con una extrapolación del smoke.

Después corresponde revisar costo y autorizar la larga, completar desarrollo, congelar selección y autorizar el test sellado. El test exige hashes finales y atestación de semillas intactas; sus pruebas usan datos sintéticos. No se lanzó una campaña larga, no se envió Slurm y no se consumió test real.
