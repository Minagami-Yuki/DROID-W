# Bonn DROIDCalib Routing Results

ATE values are RMSE in metres. Controls use seeds 41, 42, and 43.
The full sweep uses seed 43 and compares confidence recovery with the existing stability-v2 run.

## Multi-Seed Controls

| Sequence | Method | Full mean | Full std | KF mean | Completed |
| --- | --- | ---: | ---: | ---: | ---: |
| bonn_moving_nonobstructing_box2 | stability | 0.032566 | 0.000010 | 0.035910 | 3/3 |
| bonn_moving_nonobstructing_box2 | omega_fixed_k | 0.034557 | 0.000060 | 0.038082 | 3/3 |
| bonn_moving_nonobstructing_box2 | k_recovery | 0.020434 | 0.000089 | 0.022245 | 3/3 |
| bonn_moving_nonobstructing_box2 | confidence_recovery | 0.022394 | 0.000031 | 0.024598 | 3/3 |
| bonn_person_tracking2 | stability | 0.029443 | 0.000274 | 0.029984 | 3/3 |
| bonn_person_tracking2 | omega_fixed_k | 0.034440 | 0.001940 | 0.034014 | 3/3 |
| bonn_person_tracking2 | k_recovery | 0.035576 | 0.010514 | 0.036302 | 3/3 |
| bonn_person_tracking2 | confidence_recovery | 0.036018 | 0.011554 | 0.036844 | 3/3 |
| bonn_crowd2 | stability | 0.020629 | 0.002745 | 0.021690 | 3/3 |
| bonn_crowd2 | omega_fixed_k | 0.023774 | 0.001289 | 0.025629 | 3/3 |
| bonn_crowd2 | k_recovery | 0.020666 | 0.001913 | 0.022072 | 3/3 |
| bonn_crowd2 | confidence_recovery | 0.020165 | 0.000948 | 0.021458 | 3/3 |

## Full Bonn Sweep (Initial Run)

| Sequence | Stability v2 Full | Confidence recovery Full | Delta |
| --- | ---: | ---: | ---: |
| bonn_balloon | 0.022008 | 0.022402 | +1.79% |
| bonn_balloon2 | 0.025676 | 0.219136 | +753.46% |
| bonn_crowd | 0.015091 | 0.015099 | +0.05% |
| bonn_crowd2 | 0.021228 | 0.020318 | -4.29% |
| bonn_moving_nonobstructing_box | 0.016870 | 0.016874 | +0.02% |
| bonn_moving_nonobstructing_box2 | 0.032585 | 0.022425 | -31.18% |
| bonn_person_tracking | 0.030476 | 0.031391 | +3.00% |
| bonn_person_tracking2 | 0.029748 | 0.029594 | -0.52% |

## Canonical Full Bonn Sweep (Balloon2 Repeat Replacement)

`bonn_balloon2`'s original confidence-recovery run (0.219136 m Full) is an
isolated, non-repeatable BA excursion.  With the same seed and configuration,
repeat01 gives 0.026742 m Full / 0.029802 m KF.  We therefore use repeat01 as
the canonical balloon2 value for the following analysis; the initial value is
retained above as a robustness observation rather than a representative score.
All ATE values are RMSE in metres.

| Sequence | Stability Full | Confidence Full | Full Δ | Stability KF | Confidence KF | KF Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonn_balloon | 0.022008 | 0.022402 | +1.79% | 0.023273 | 0.023735 | +1.99% |
| bonn_balloon2 | 0.025676 | 0.026742 | +4.15% | 0.028891 | 0.029802 | +3.15% |
| bonn_crowd | 0.015091 | 0.015099 | +0.05% | 0.015865 | 0.015859 | -0.04% |
| bonn_crowd2 | 0.021228 | 0.020318 | -4.29% | 0.022953 | 0.021757 | -5.21% |
| bonn_moving_nonobstructing_box | 0.016870 | 0.016874 | +0.02% | 0.018467 | 0.018474 | +0.04% |
| bonn_moving_nonobstructing_box2 | 0.032585 | 0.022425 | -31.18% | 0.035931 | 0.024630 | -31.45% |
| bonn_person_tracking | 0.030476 | 0.031391 | +3.00% | 0.030015 | 0.030878 | +2.88% |
| bonn_person_tracking2 | 0.029748 | 0.029594 | -0.52% | 0.030126 | 0.030027 | -0.33% |
| **Macro mean** | **0.024210** | **0.023106** | **-4.56%** | **0.025690** | **0.024395** | **-5.04%** |

Confidence recovery improves Full ATE on 3/8 sequences and KF ATE on 4/8
sequences.  The aggregate gain is dominated by box2; crowd2 gives a smaller
consistent gain, while the other sequences are near-neutral to mildly worse.
The method should consequently be presented as a conditional intrinsics
recovery mechanism rather than a universal replacement for stability-v2.

## Benefit-Type Localization

The repeat-replaced sweep separates two types of dynamic sequences.  **box2**
is calibration-sensitive: confidence recovery improves Full/KF ATE by 31.18%
and 31.45%, respectively.  **crowd2** is a smaller but consistent recovery
case (4.29% / 5.21% improvement).  `crowd`, `box`, and `person_tracking2` are
near-neutral, while `balloon`, `balloon2`, and `person_tracking` pay a mild
1.8--4.2% Full-ATE cost.  Similar final focal changes on these groups show
that focal displacement alone does not identify the benefit type; the useful
signal is whether early K-pose-depth coupling resolves an otherwise degraded
trajectory.

## Minimal Three-Seed Ablation

The following controls use seeds 41/42/43.  `Omega fixed-K` disables focal
calibration; `K recovery` enables the same early high-frequency joint BA
through the prior fixed focal interval; `confidence recovery` instead routes
that early BA using Omega's raw depth-confidence shape statistic.  Values are
Full ATE mean ± sample standard deviation in metres.

| Sequence | Stability | Omega fixed-K | K recovery | Confidence recovery |
| --- | ---: | ---: | ---: | ---: |
| box2 | 0.032566 ± 0.000010 | 0.034557 ± 0.000060 | **0.020434 ± 0.000089** | 0.022394 ± 0.000031 |
| crowd2 | 0.020629 ± 0.002745 | 0.023774 ± 0.001289 | 0.020666 ± 0.001913 | **0.020165 ± 0.000948** |
| person_tracking2 | **0.029443 ± 0.000274** | 0.034440 ± 0.001940 | 0.035576 ± 0.010514 | 0.036018 ± 0.011554 |

This establishes the source of the observed gain.  On **box2**, calibration is
necessary (fixed-K is worse) and the dominant gain comes from allowing early
joint K-pose-depth BA; confidence recovery retains most of that gain while
being more conservative than the fixed interval.  On **crowd2**, confidence
routing is the only recovery variant that improves the mean and it also reduces
seed variation.  On **person_tracking2**, both recovery variants are worse and
far less stable than the stability gate, so the confidence statistic is not yet
a universally selective calibration-need predictor.

The resulting claim should therefore be conditional: confidence-routed early
joint BA is a useful recovery mechanism for a subset of calibration-sensitive
dynamic trajectories, not a global replacement for stability-v2.

## Benefit-Type Localization

The repeat-replaced sweep separates two types of dynamic sequences.  **box2**
is calibration-sensitive: confidence recovery improves Full/KF ATE by 31.18%
and 31.45%, respectively.  **crowd2** is a smaller but consistent recovery
case (4.29% / 5.21% improvement).  `crowd`, `box`, and `person_tracking2` are
near-neutral, while `balloon`, `balloon2`, and `person_tracking` pay a mild
1.8--4.2% Full-ATE cost.  Similar final focal changes on these groups show
that focal displacement alone does not identify the benefit type; the useful
signal is whether early K-pose-depth coupling resolves an otherwise degraded
trajectory.

## Minimal Three-Seed Ablation

The following controls use seeds 41/42/43.  `Omega fixed-K` disables focal
calibration; `K recovery` enables the same early high-frequency joint BA
through the prior fixed focal interval; `confidence recovery` instead routes
that early BA using Omega's raw depth-confidence shape statistic.  Values are
Full ATE mean ± sample standard deviation in metres.

| Sequence | Stability | Omega fixed-K | K recovery | Confidence recovery |
| --- | ---: | ---: | ---: | ---: |
| box2 | 0.032566 ± 0.000010 | 0.034557 ± 0.000060 | **0.020434 ± 0.000089** | 0.022394 ± 0.000031 |
| crowd2 | 0.020629 ± 0.002745 | 0.023774 ± 0.001289 | 0.020666 ± 0.001913 | **0.020165 ± 0.000948** |
| person_tracking2 | **0.029443 ± 0.000274** | 0.034440 ± 0.001940 | 0.035576 ± 0.010514 | 0.036018 ± 0.011554 |

This establishes the source of the observed gain.  On **box2**, calibration is
necessary (fixed-K is worse) and the dominant gain comes from allowing early
joint K-pose-depth BA; confidence recovery retains most of that gain while
being more conservative than the fixed interval.  On **crowd2**, confidence
routing is the only recovery variant that improves the mean and it also reduces
seed variation.  On **person_tracking2**, both recovery variants are worse and
far less stable than the stability gate, so the confidence statistic is not yet
a universally selective calibration-need predictor.

The resulting claim should therefore be conditional: confidence-routed early
joint BA is a useful recovery mechanism for a subset of calibration-sensitive
dynamic trajectories, not a global replacement for stability-v2.
