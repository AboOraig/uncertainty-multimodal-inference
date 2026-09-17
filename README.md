# uncertainty-multimodal-inference

## 1. Problem setup

A 2D position must be estimated from **3 heterogeneous sensors**, each with its own
noise level, dropout probability, occasional bias faults, and a noisy
self-reported "quality" signal (an SNR-like proxy for its own current noise —
this is the observable feature any adaptive method is allowed to use; the *true*
noise level is never given to a model, only to the evaluation code).

Degradation is **localized and random per-sample**: on each sample, one randomly
chosen sensor is the "stressed" modality, and a scalar `d ∈ [0, ∞)` controls how
badly that one sensor is currently degraded (more noise, more bias-fault
probability, more dropout). The other two sensors stay at baseline quality. This
is what makes a **fixed** weighting scheme genuinely suboptimal: the *identity* of
the degraded sensor changes from sample to sample, so any static weighting
(however well-tuned) can't track it — only a fusion rule that looks at
per-instance evidence can.
