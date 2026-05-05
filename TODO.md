# TODO.md - ROADMAP Phases 2-4 Progress Tracker

Proyecto: Continuar refactor/test/mejora SMO-Spatial per ROADMAP.

## Pendientes Originales (prior Phase 2: Correctness/Measurement)

- [x] 2. benchmarks/suites/optimizer_step/benchmark_step_time.py: GPU support (cuda), shapes variados, memory/step.  # shapes multi (64–4096), warmup, device-aware (CPU/CUDA/DirectML), memory reporting para CUDA; mean±std across seeds; GPU DirectML pendiente hasta que GPU local esté libre
- [x] 3. docs/ROADMAP.md: Update Phase 2→3 status, new profiles.  # Phase 2 section actualizado con avances (2026-05-05)
- [~] 4. smo/optimizers/spatial.py: Opt allocs (reuse tensors, fuse ops).  # Reclasificado: Phase 4 item. Requiere profiling GPU para identificar hotspots;pospuesto hasta Phase 3/4
- [x] 5. Align smo/optimizers/spatial_8bit.py (read first).  # test_spatial_and_8bit_match_when_quantization_is_exact pasa; consistencia algorítmica verificada
- [x] 6. benchmarks/timing.py: GPU timing extend.  # get_sync_fn() para CUDA/DirectML/CPU
- [x] 7. Run pytest, microbench pre/post.  # 3/3 tests pasan; microbench multi-shape con mean±std ejecutado en CPU
- [x] 8. Re-run MNIST baseline.  # Adam 99.04% (3.22 MB) vs SMO k=0.25 98.90% (0.35 MB) → 89.1% mem reduction, +0.14% acc gap. Results en benchmarks/results/
- [x] 9. Update CATALOG.md, results JSON.  # benchmark_step_time ya en CATALOG; results JSON actualizado con MNIST run

## Completados (Phase 2 - Correctness & Measurement)

### Benchmark Infrastructure
- `benchmarks/timing.py`: `get_sync_fn(device)` + `measure_steps` con sync opcional
- `benchmarks/suites/optimizer_step/benchmark_step_time.py`:
  - Shapes: 64–4096 (cuadrados)
  - Repeticiones con múltiples seeds (default 3)
  - Reporta mean±std para wall y CPU
  - Device-aware (CPU/CUDA/DirectML)
  - Memoria reportada para CUDA
  - GPU DirectML path preparado (comentado)
- Deterministic seeds en MNIST benchmark (torch.manual_seed + numpy + random)
- `benchmarks/results/` auto-generado vía `write_benchmark_bundle`

### Correctness Verified
- `tests/test_spatial_optimizers.py`: 3/3 ✅
- SMO vs SMO8bit paridad con `block_size=1`
- Segundo momento comprimido correcto (E[g²] vs E[g]²)
- MNIST end-to-end: SMO k=0.25 within 0.2% accuracy vs Adam, 89% memory savings

### Phase 2 Exit Criteria Status
- ✅ Benchmark runs repeatable (per-seed)
- ⚠️ Metrics consistent across hardware: infra ready, GPU execution pending
- ⚠️ Deterministic seed handling: implementado en MNIST y microbench; falta en otras suites (activation, spectral)
- ⚠️ Timing normalization: mean±std implemented en microbench; faltarepetir en otras suites

## Próximos Fases (Roadmap)

### Phase 3: Bottleneck Analysis ( aguardando Phase 2 completado)
- Requires GPU availability for profiling
- Profile optimizer-step: isolate pooling vs interpolation vs quantization overhead
- Tools: torch.profiler, memory_snapshot
- Target: SMO-Spatial y SMO-Spatial-8bit

### Phase 4: Optimization Work
- Prioridad 1: SMO-Spatial alloc optimization (reutilizar buffers temporales)
- Prioridad 2: SMO-Spatial-8bit alignment y block-size tuning
- Prioridad 3: Triton kernels (solo después de limpiar PyTorch path)
- Activation compression: separate track

### Phase 5: Benchmark Publication
- Multi-seed results tables (mean±std)
- Hardware/disclosure docs
- Honest discussion of failure modes

### Phase 6: Packaging & Documentation
- API cleanup, examples, reproducibility guide

## Tareas Inmediatas (Next Actions)

1. [Phase 2 finiquitación]:
   - [ ] Añadir `--seed` y `--repeats` a otras suites (activation, spectral)
   - [ ] Ejecutar benchmark_step_time en GPU cuando DirectML esté disponible
   - [ ] Documentar seed policy en `benchmarks/METHODOLOGY.md`

2. [Phase 4 - alloc opt]:
   - [ ] Agregar profiling script para identificar allocation hotspots en SMO-Spatial
   - [ ] Implementar buffer reuse en `smo/optimizers/spatial.py` (m_rec, v_rec buffers)
   - [ ] Medir impacto en microbench

3. [Documentation]:
   - [ ] Actualizar PROJECT_FOUNDATION.md con taxonomía final
   - [ ] Escribir changelog desde última baseline

Last update: 2026-05-05