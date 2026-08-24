# LR sweep (vit)

```
LR sweep | suite=vit | metric=final_test_acc (higher better)

--- adamw ---
  lr=0.0006   @30ep    n=3  73.39±0.77  <-- best

--- bnb8bit ---
  lr=0.0003   @30ep    n=3  80.13±0.79  <-- best

--- smo ---
  lr=0.0015   @30ep    n=3  78.78±0.28  <-- best

--- smo8bit ---
  lr=0.001    @3ep     n=2  51.95±0.44
  lr=0.0015   @30ep    n=3  78.30±0.53  <-- best

RANKING by best-tuned configuration (check @horizon matches!):
  1. bnb8bit    best@lr=0.0003@30ep: 80.13
  2. smo        best@lr=0.0015@30ep: 78.78 (-1.35)
  3. smo8bit    best@lr=0.0015@30ep: 78.30 (-1.84)
  4. adamw      best@lr=0.0006@30ep: 73.39 (-6.74)
```
