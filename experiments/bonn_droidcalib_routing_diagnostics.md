# Bonn Confidence-Recovery Repeatability and Mechanism Diagnostics

All runs use the same seed (43), the same stability-v2 base configuration, and the same
Omega confidence threshold (30 samples, mean/median ratio >= 1.05). ATE is RMSE in metres.

## ATE Repeatability

| Sequence | Stability v2 Full/KF | First confidence Full/KF | Repeat01 Full/KF |
| --- | ---: | ---: | ---: |
| balloon2 | 0.025676 / 0.028891 | 0.219136 / 0.218314 | 0.026742 / 0.029802 |
| person_tracking2 | 0.029748 / 0.030126 | 0.029594 / 0.030027 | 0.029942 / 0.030304 |

## Calibration-Path Statistics

| Run | Accepted / rows | Recovery / accepted | fx initial → final | |Δfx| | Σ|log step| | max |log step| | confidence median [min,max] | max pose t / r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balloon2 first | 163 / 241 | 188 / 157 | 55.797 → 54.491 | 2.34% | 0.198560 | 0.004044 | 1.0573 [1.0441, 1.0991] | 11.0384 / 2.3041 |
| balloon2 repeat01 | 164 / 242 | 189 / 158 | 55.797 → 54.542 | 2.25% | 0.065162 | 0.004000 | 1.0575 [1.0441, 1.0991] | 0.4022 / 0.1665 |
| person_tracking2 first | 118 / 341 | 100 / 89 | 56.271 → 54.898 | 2.44% | 0.059327 | 0.003166 | 1.0345 [0.9933, 1.1114] | 0.1764 / 0.1323 |
| person_tracking2 repeat01 | 117 / 341 | 100 / 88 | 56.271 → 54.896 | 2.44% | 0.059555 | 0.003234 | 1.0345 [0.9933, 1.1114] | 0.1765 / 0.1324 |

## Rejection Reasons

| Run | accepted | unstable trajectory | loss increase | low observability |
| --- | ---: | ---: | ---: | ---: |
| balloon2 first | 163 | 39 | 31 | 8 |
| balloon2 repeat01 | 164 | 39 | 31 | 8 |
| person_tracking2 first | 118 | 191 | 32 | 0 |
| person_tracking2 repeat01 | 117 | 191 | 33 | 0 |

## Interpretation

The balloon2 failure is not repeatable with the fixed seed: its repeat01 ATE returns near the
stability-v2 baseline even though the confidence statistic, recovery duration, final focal, and
rejection counts are nearly identical.  The remaining difference is the accumulated accepted
Schur focal-step path, indicating numerical/multithreaded BA sensitivity rather than a robust
sequence-level routing signal.  Person_tracking2 is repeatable in ATE and focal evolution,
confirming that the same branch can be stable on another sequence while still not providing a
safe universal trigger.
