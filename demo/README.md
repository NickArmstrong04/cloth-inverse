# Demo log (measurements; 2026-11-01)

## 128-vs-256 grid decision (step 2)

20 random `free` force configurations (`sample_forces`, seeds 100000-100019), simulated at 128 and 256 with the same seeds, M16-a scored on both (128 upsampled bilinear to 256). Means; impulse-rel median in brackets.

| path | pos cm | dir deg | impulse rel (median) | sim wall s |
|---|---|---|---|---|
| 128 -> 256 | 3.47 | 25.8 | 0.867 (0.588) | 4.4 |
| 256 | 1.37 | 6.4 | 0.251 (0.174) | 11.9 |

Ratios 128/256: pos 2.53x, dir 4.01x. **Decision: grid 256.**

Failed seeds (1, excluded): 100011.
