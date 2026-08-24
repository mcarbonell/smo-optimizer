# SMO Paper/Post — Skeleton & Claims-to-Evidence Map

> Estado: estructura viva para el write-up. Cada sección lista su evidencia y su
> estado (`[DATOS OK]` / `[PENDIENTE run]`). No escribir prosa definitiva hasta que
> la cola experimental (k=0.5, scouting espectral) cierre.

---

## Título (candidatos)

1. *"What Does Compressing Adam's Moments Actually Buy? A Memory–Quality Study
   with Spatially Smoothed Optimizer States"*
2. *"SMO: bnb-class vision quality at 1 byte/parameter of optimizer state"*
3. *"Structured Smoothing, Not Generic Noise: Why Compressing Adam's Moments
   Helps Vision and Hurts Language"* (si el mecanismo pesa más que el benchmark)

## Abstract — dos variantes según k=0.5 `[PENDIENTE]`

- **vA (k=0.5 cierra el gap):** compresión con capacidad suficiente iguala a
  bitsandbytes-8bit en calidad a largo presupuesto con ~4× menos estado que bnb.
  Tesis central = compresión bien hecha no cuesta.
- **vB (k=0.5 no cierra):** trade-off Pareto explícito — ≤2 pts de calidad por
  11× menos estado + pico mínimo + estabilidad de seeds. Tesis central =
  cuantificar qué se compra y qué se paga.
- En ambas: mecanismo de localidad demostrado, hallazgo metodológico de
  no-transferencia de LR entre horizontes, anomalía bnb>+Adam-fp32 como pregunta
  abierta.

## Secciones

### 1. Introducción
- Problema: estado del optimizador domina la memoria de entrenamiento.
- Contribuciones (borrador):
  1. Familia SMO (spatial + 8-bit + low_peak banding) con contabilidad honesta
     persistente-vs-residente.
  2. Estudio presupuestos × optimizadores con tuning simétrico (protocolo reutilizable).
  3. Mecanismo: localidad necesaria (ablation), SGD-ness y ruido genérico descartados.
  4. Hallazgos secundarios: no-transferencia de LRs, anomalía bnb, colapso de varianza.

### 2. Método `[DATOS OK]`
- Operador de compresión: avg-pool exacto → EMA → bilinear; INT8 block-wise opcional.
- `low_peak`: actualización por bandas (exacta por localidad); pico ~21→~9 B/param.
- Flags: `protect_output` (grupos), `permute_basis` (ablation).
- Tabla de costes: bytes/param persistentes vs residentes vs pico de step.

### 3. Setup `[DATOS OK]`
T4 16GB, fp16-autocast fwd/bwd (estados fp32), wd=0, cosine+warmup idénticos,
clip 1.0, seeds {1234,5678,9012}, bundles JSON con git_commit+trayectorias.
Definiciones persistent-vs-resident (METHODOLOGY).

### 4. Resultados principales
- **Tabla 1 — Pareto por presupuesto** (3/10/30 ep, best-tuned): `[30ep n=3 casi OK;
  adamw s1234 regen pendiente]`
- **Figura 1 (principal) — frontera loss-matched**: `t4_loss_matched`, curvas
  test_acc vs train_loss por optimizador. `[DATOS OK @10ep; regenerar a 30ep si hay tiempo]`
- **Tabla 2 — curvas LR (H8)**: ventanas utilizables por optimizador. `[DATOS OK]`

### 5. Ingeniería de memoria
- Killer demo (~700M en T4): tabla OOM/ok + picos. `[DATOS OK]`
- Contabilidad: 94 MB ≈ teoría int8·k². `[DATOS OK]`
- Coste throughput: −5…−17% según suite.

### 6. Mecanismo
- H4 localidad: permutación colapsa a nivel-ruido. `[1 seed; 2 extra PENDIENTES]`
- H5 falsada (SGD-M peor frontera). `[DATOS OK]`
- Ruido genérico insuficiente (bnb solo compra una fracción). `[DATOS OK]`
- Colapso de varianza entre seeds (3 observaciones). `[DATOS OK]`
- H1 parcial (protect_output −⅓ del gap LM). `[DATOS OK, char-LM]`
- `[PENDIENTE]` k=0.5: ¿el gap restante es información perdida o dinámica?

### 7. Hallazgos metodológicos y abiertos
- No-transferencia de LR entre horizontes (nos revirtió el propio headline). `[DATOS OK]`
- Anomalía bnb > Adam-fp32 (+5…+7 tuneado): replicar antes de creer. `[ABIERTO]`
- Scouting espectral (Walsh/DCT-Pure @10ep): `[PENDIENTE]`

### 8. Limitaciones (honestas)
CIFAR-10/TinyViT + char-LM toy; conv-4D sin comprimir; residente≠persistente en
SMO-Spatial mono; tuning asimétrico documentado; resultados single-seed señalados.

### 9. Related work (a completar con lectura)
bnb (Dettmers+22), Adafactor, GaLore/Q-GaLore, torchao low-bit, Lookahead/SWA/EMA
(mismo perfil régimen), SAM/flatness, ruido benéfico (Neelakantan+15),
PID-optimizers, DeltaNet-family (solo si mencionamos cartera).

### 10. Reproducibilidad
Repo + notebook Colab + comandos canónicos + JSONs versionados + METHODOLOGY.md.

---

## Figuras/Tablas plan

| # | Tipo | Fuente | Estado |
|---|---|---|---|
| F1 | loss-matched curves | t4_loss_matched @10ep | datos OK |
| T1 | Pareto 3/10/30ep | tuned tables | 95% |
| T2 | LR windows | lrsweep.md extendido | datos OK |
| T3 | killer demo | big2 bundle | datos OK |
| T4 | mechanism matrix | perm/prot/hist bundles | 90% (H4 multiseed) |
| F2 | lr-curve shapes por optimizer | sweep bundles | datos OK |

## Orden de redacción propuesto

1. T1+F1 (espina dorsal empírica) → 2. Método → 3. Mecanismo → 4. Memoria →
5. Intro+Abstract (elegir vA/vB tras k=0.5) → 6. Related work → 7. Limitaciones.
