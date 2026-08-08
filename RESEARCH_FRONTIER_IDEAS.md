# Research Frontier — Three Unconventional Directions

**Status: brainstorm only. Nothing here is a registered hypothesis.**
Read alongside `ARCHITECTURE_DOCS.md` → *Production Methodology* and `CLAUDE.md` → *Invariants*.

---

## 0. The framing constraint that shapes all three ideas

Your repo already contains the two facts that should govern any "bleeding edge" attempt:

1. **The first moment is empty.** ROC-AUC ≈ 0.50, the return regressor shrinks to the mean, and nine
   direction-family hypotheses have produced nine non-clearances (`results/feature_hypothesis_log.csv`).
   The family bar is now α = 0.05/9 ≈ 0.00556. Any new *feature* is fighting a bar that its own
   registration tightens.
2. **The one thing that cleared turned out to be a clock artifact.** The volatility ensemble beat GARCH
   out-of-sample — and the 2026-08-07 verification showed the entire aggregate edge lives in **Friday
   rows**, i.e. the Fri→Mon weekend gap, and a 6-number day-of-week lookup table beats the LSTM
   outright (`results/volatility_hypothesis_log.csv` #3).

Read together, those say something sharp: **your daily bar is the wrong object.** A "day" is an
arbitrary, non-uniform slice of an information process, and the only measurable structure you found was
the model partially rediscovering that fact. The three ideas below all attack the *representation of the
problem* rather than the function approximator — because a better approximator of a zero-signal target
is worth nothing.

**Register each of these as its OWN hypothesis family** (as you did for volatility), with its own log
file and its own α ladder. Folding them into the direction family would be self-defeating.

---

## Idea 1 — Precision-first forecasting via hierarchical predictive coding / active inference

*Novel training paradigm. The model's primary output is its own uncertainty, and the point forecast is
a by-product.*

### Core concept

Predictive coding (Rao–Ballard; Friston's free-energy formulation) builds a hierarchical generative model
in which each level predicts the level below, and only the **precision-weighted prediction error**
propagates upward. The objective is variational free energy:

```
F = E_q[-log p(x | z)]  +  KL[ q(z) || p(z) ]
    \_____ accuracy _____/    \___ complexity ___/
```

with the accuracy term explicitly precision-weighted: `Π·ε²`, where `Π = Σ⁻¹` is a *learned, state-dependent*
inverse covariance.

Why this fits a near-efficient market better than anything you have: in an efficient market the conditional
mean `E[r_{t+1} | F_t] ≈ 0`, so a point-forecast architecture is spending its entire capacity estimating a
quantity that is provably tiny. But `Var[r_{t+1} | F_t]` is strongly predictable — you have already proven
this with CI-confirmed evidence. Predictive coding is the one paradigm where **precision is a first-class
learned variable in the loss**, not a bolted-on second model. Two consequences follow for free:

- **Surprise becomes a native quantity.** `-log p(x_t)` under the learned generative model is a calibrated
  regime-break detector, not a heuristic z-score. Structural breaks are exactly the events where a fixed
  hierarchical model's free energy spikes.
- **Learning is local and online.** PC updates use local error signals rather than backprop-through-time.
  For a non-stationary series that is not an aesthetic preference: it means the model can be updated
  continually as sessions settle, without re-optimising across a regime boundary the way your current
  fixed `[0:80%]` fit implicitly does.

The active-inference extension makes the trade decision part of the same functional: expected free energy
`G(π) = pragmatic value + epistemic value` gives you a principled reason to size down when the model's
epistemic uncertainty (parameter uncertainty) is high rather than only when its aleatoric uncertainty
(vol forecast) is high — a distinction no GARCH-plus-classifier stack can make.

### The gap

Quant finance does *mean model + separate variance model* (ARIMA-GARCH, or your GBM/LSTM + separate
`models/volatility/`). The two are estimated with different losses, on different targets, and joined by
hand. Predictive coding is a neuroscience/perception framework — the finance literature has borrowed
Bayesian filtering (Kalman, particle filters) and, rarely, "surprise" as a feature, but essentially nobody
has built the *hierarchical precision-weighted generative model with iterative inference at test time* as
the production forecaster. The reason it hasn't crossed over is sociological: PC's selling point in
neuroscience is biological plausibility (local learning rules), which quants correctly regard as irrelevant.
The property that actually matters here — that precision is the optimised object — was never the headline,
so nobody read the paper for it.

### Roadmap

1. **Substrate: JAX + Optax.** You need `jax.grad` on an energy functional with a fast inner loop; TF/Keras
   is the wrong shape. Keep it entirely out of `src/inference.py` until it clears a gate.
2. **Generative model.** Three levels over a state `z = (latent trend, latent log-vol, latent activity)`,
   emitting `x_t = (log return, realized range, tick/volume proxy)`. Gaussian transitions with
   **state-dependent precision** `Π_θ(z_t)` — a small MLP head, not a scalar.
3. **Inference = inner gradient descent.** At each time step, minimise `F` w.r.t. `z` for K steps (K ≈ 8–20)
   holding θ fixed; then a single outer step on θ. This is the "thinking at inference time" that makes PC
   different from a feedforward net — implement with `jax.lax.scan` and `optax.apply_updates`, checkpoint
   the inner loop with `jax.checkpoint` to keep memory sane.
4. **Evaluate with proper scoring rules, not MAE.** CRPS and NLL on `r_{t+1}`, plus PIT-histogram uniformity
   and a Diebold–Mariano test against your existing champion. **Critical:** the honest benchmark is not
   GARCH alone — your own log says the benchmark to beat is `GARCH(1,1) × day-of-week lookup table` fitted
   on `[0:80%]`. Anything that cannot beat that is rediscovering the calendar.
5. **Arbiter.** New family, own log (`results/predictive_coding_hypothesis_log.csv`), scored on
   `validation[70:80]` via the `src/ablation.py` fit-on-`[0:70]` discipline. Pre-register the primary
   quantity (I would make it **CRPS delta vs the GARCH×DoW benchmark**) before you look.
6. **Only then** consider the active-inference layer, and only as a *sizing* signal fed to
   `src/paper_trading.py` — never as order execution.

### Biggest hurdle

**Free energy is not a proper scoring rule, and the model can cheat by becoming uncertain.** `F` can always
be reduced by inflating predicted variance: the accuracy term shrinks faster than the complexity term grows
over a wide region of parameter space. You will get a model that says "I know nothing" with beautiful
convergence curves. This is the variance-collapse failure mode and it is *silent* — training loss looks
excellent. You must (a) constrain training with a strictly proper scoring rule (CRPS or NLL) as either the
outer objective or a hard-monitored early-stopping criterion, and (b) check PIT calibration every epoch.
Secondary hurdle: the inner-loop inference makes each training step 10–20× more expensive, and hyper-tuning
K interacts with the learning rate in a way that is genuinely nasty to diagnose.

---

## Idea 2 — Gauge-equivariant geometric deep learning on the currency graph (discrete Hodge decomposition of FX)

*Unorthodox mathematical framework. Stop modelling EUR/USD as a time series; model the currency system as
a cochain complex and read EUR/USD off it.*

### Core concept

FX is not a collection of series — it is a **groupoid with an exact cocycle condition**. For log rates,

```
log(EUR/USD) = log(EUR/JPY) + log(JPY/USD)
```

Written as a 1-cochain `ω` on the complete graph over currencies `C`, no-arbitrage says `ω` is **closed**
(`dω = 0` on every triangle) and, on a connected graph, therefore **exact**: `ω = dφ` for a 0-cochain
`φ: C → ℝ` — a per-currency scalar *potential*. Rates are potential differences. This is discrete Hodge
theory, and it has a real gauge symmetry: `φ → φ + c` leaves every rate invariant.

The architectural consequence is the whole point. Instead of predicting one scalar for one pair, you build
a message-passing network on the currency graph that emits `Δφ_c(t+1)` for each of ~8 majors, and define

```
r_EURUSD(t+1) := Δφ_EUR(t+1) − Δφ_USD(t+1)
```

You have replaced `N(N−1)/2 = 28` free predictions with `N−1 = 7` free parameters. In a regime where the
signal-to-noise ratio is ≈ 0, **that variance reduction is the entire game** — it is a hard structural prior,
not another feature fighting your Bonferroni bar. Every EUR/JPY and USD/CHF observation now contributes
statistical strength to the EUR/USD forecast, legitimately, by an accounting identity rather than by an
assumption.

The Hodge decomposition then gives you a second, more interesting object. Decompose the observed flow into

```
ω = dφ  ⊕  harmonic  ⊕  δψ (curl)
```

At daily close the curl component is ~0 (arbitrage is enforced). At 1-minute — and you have `results/eurusd_m15.csv`
and H1 infrastructure already — the curl is *not* zero: it is the latency/liquidity residual. **Curl energy is
a microstructure stress index derived from first principles**, with no free parameters and no fitting. That is
a genuinely new observable, and it costs you nothing to measure before committing to the architecture.

### The gap

Three separate communities each hold one piece and none hold two. Practitioners know triangular arbitrage as
a latency trade and "currency strength indices" as a retail chart overlay (unprincipled, usually just averaged
returns). Econophysics has studied FX correlation networks — but descriptively, as topology of a correlation
matrix, not as a constrained hypothesis class. Geometric deep learning has developed gauge-equivariant and
simplicial/Hodge-aware networks and pointed them at meshes, molecules and physics simulations. The FX cochain
complex is arguably the cleanest real-world example of an exact 1-cochain in all of applied ML — the constraint
is an *identity*, not a modelling approximation, which is rarer than anything in the mesh literature — and it
has been sitting unexploited because the people who know Hodge theory don't trade FX and vice versa.

### Roadmap

1. **Measure before you build (cheap, one day, no model).** Pull 8 majors from MT5 (you have `src/mt5_coverage.py`),
   build the 28-edge log-rate cochain, compute `dω` per triangle at D1, H1, M15. Confirm curl ≈ 0 at D1 and
   > 0 at M15. Plot curl energy against known stress dates. **If the curl residual at M15 is pure noise with no
   persistence, stop here — you've saved yourself a month.**
2. **Data hygiene is the load-bearing step.** Different pairs have different close conventions, holidays and
   quote sources. Misalignment injects *fake* curl that will look exactly like signal. Build a strict
   simultaneity filter and unit-test that synthetic aligned data yields machine-epsilon curl.
3. **Architecture.** PyTorch Geometric (or hand-rolled JAX; the graph has 8 nodes, this is tiny). Nodes = currencies
   with per-currency features; edges = pair features. `k` rounds of message passing → head emits `Δφ_c`. Fix the
   gauge by projecting `φ` to zero-mean (do **not** anchor to USD — that silently privileges one currency and breaks
   equivariance).
4. **Unit test the symmetry.** `assert model(φ + c) == model(φ)` to float tolerance. This is the equivalent of your
   no-look-ahead tests: it's the invariant that makes the architecture mean what you say it means.
5. **Train multi-task on all pairs, evaluate on EUR/USD only.** The pre-registered comparison is:
   *node-potential model trained on 8 currencies* vs *the identical model restricted to the EUR/USD pair alone*.
   That isolates the value of the geometric prior from the value of the extra data.
6. New family, own log, `validation[70:80]` arbiter. Then the `src/paper_trading.py` forward ledger.

### Biggest hurdle

**The constraint is an identity, so it may carry exactly zero information about dynamics.** `ω = dφ` is true
by construction of the quote convention — you can always write today's rates as potential differences. The
danger is that you build something mathematically beautiful that is a *reparameterisation* of the same
function class, with the same predictive content, and then mistake the elegance for evidence. The bet is
narrower than it first appears: it's that constraining *predictions* to the exact subspace (which the
unconstrained model would violate) reduces estimation variance enough to matter, and that cross-pair
message passing shares real information. You must design the ablation to test precisely that and nothing else.
The close second hurdle: as above, timestamp misalignment manufactures curl, and it will be *the most
exciting-looking result you get* right before you discover it's a data bug.

---

## Idea 3 — Liquid time-constant dynamics in learned business time, with a spiking abstention readout

*Emerging architecture + the direct answer to what your own volatility verification found.*

### Core concept

Two ideas that belong together and have never been combined.

**(a) Learn the subordinator.** Mandelbrot–Taylor / Clark: financial prices are a Brownian motion
*time-changed* by a stochastic clock driven by information arrival. Returns are near-Gaussian in business
time and fat-tailed in calendar time. Your daily bar samples the process on a calendar clock — and your own
verification found that the ensemble's edge was a Friday effect, i.e. the model half-learning that a "day"
containing a weekend is a bigger chunk of business time. **Liquid Time-Constant Networks** (Hasani/Lechner;
use the closed-form CfC variant) have *input-dependent* time constants:

```
dh/dt = −[ 1/τ + f(x,h) ] · h  +  f(x,h) · A
```

The relaxation rate `τ_eff(x)` is modulated by the input. That is, structurally and exactly, **a learned
subordinator**: the network integrates fast when information arrives and slowly when it doesn't. It's a
continuous-time ODE, so irregular sampling (weekends, holidays, news gaps) is handled natively by passing
`Δt` to the solver instead of `ffill`-ing a synthetic bar — which removes a known distortion from your
pipeline rather than adding a feature to it.

And it yields a **falsifiable scientific claim**, which is rare in this field: *if the subordination
hypothesis is right, the learned `τ_eff(x_t)` should correlate with realized tick/volume intensity.* You can
test that directly against your H1/M15 data. A learned time constant that reproduces the volume clock is a
publishable result independent of whether it makes money.

**(b) Spike, don't predict.** A near-efficient market has exploitable structure on a *small subset* of days.
Forcing a prediction every session guarantees the edge is diluted in noise — and it's exactly what your
architecture does now, with `CONFIDENCE_THRESHOLD = 0.52` as a hand-set patch over the symptom. Replace the
readout with a leaky integrate-and-fire neuron: membrane potential accumulates evidence, and the model emits
a trade only when it crosses a **learned** threshold. This makes abstention an end-to-end trained decision
rather than a constant, gives you a spike rate that *is* a risk budget, and lets you put transaction cost
directly into the loss as a firing-rate penalty:

```
L = Σ_{t: spike} CRPS(r_t, ŷ_t)  +  λ · (spike rate)      # λ ≡ cost per round trip
```

### The gap

LTC/CfC networks live in robotics and autonomous drone control, where their selling point is smooth
continuous-time control with tiny parameter counts. SNNs live in neuromorphic vision, where the selling
point is energy per inference on edge hardware. **Both fields sell the property that doesn't matter here,
and neither sells the property that does** — for LTCs it's the input-dependent clock as a subordination
model; for SNNs it's the threshold as a principled, learnable *selective prediction* mechanism. Quant
finance meanwhile has a rich subordination literature (variance-gamma, NIG, Clark's model) that is entirely
*generative and parametric*, never a learned neural clock. The three-way intersection —
learned-clock + learned-abstention + cost-in-the-loss — is empty.

### Roadmap

1. **Establish the null first.** Fit `GARCH(1,1) × day-of-week table` and a plain CfC on calendar time. Your
   own log says the DoW table is the honest benchmark; make it the pre-registered comparator from day one.
2. **CfC in JAX + diffrax** (closed-form approximation — avoid stiff adaptive solvers, they will dominate
   your wall-clock and introduce solver-tolerance nondeterminism on top of the TF nondeterminism you already
   documented). Feed `Δt` explicitly. Seed-ensemble it: you already learned that lesson the hard way in
   `models/volatility/`.
3. **Probe the clock.** Extract `τ_eff(x_t)` and correlate against realized H1 tick counts / range. Pre-register
   this as the *primary* hypothesis — it is the scientifically interesting one and it is testable independent
   of P&L.
4. **Spiking readout** with surrogate gradients (fast-sigmoid or SLAYER-style; `snntorch`/`spikingjelly` if you
   want a reference implementation, though a 1-neuron LIF readout is ~30 lines). Add a **Lagrangian floor on
   firing rate** — see hurdle.
5. **Evaluate with risk-coverage curves and AURC**, not accuracy. Once the model chooses when to predict, AUC
   and MAE are not comparable across models: they're computed on different, model-selected row sets.
6. **`src/paper_trading.py` is the real arbiter here.** An abstaining model is precisely the kind of thing that
   looks fine in-sample and only reveals itself over months of forward, cost-net ledger.

### Biggest hurdle

**Selection bias plus a degenerate optimum: the model's globally best strategy is to never fire.** Zero trades
means zero loss on the accuracy term, so with any `λ > 0` the trivial solution wins, and gradient descent will
find it — usually early, and it will look like "converged nicely." Worse, once firing is model-dependent, your
evaluation set is chosen by the thing being evaluated, so every metric you're used to becomes non-comparable and
your existing `src/ablation.py` McNemar machinery does not directly apply (McNemar assumes paired predictions on
a common row set; two abstaining models don't share one). You need: a hard minimum-coverage constraint or a
dual-ascent Lagrangian on trade count; a selective-prediction evaluation framework (risk-coverage, AURC); and a
paired comparison restricted to the *intersection* of both models' firing sets, with the coverage difference
reported separately. The surrogate-gradient bias is a real but comparatively boring second problem.

---

## Recommended order

If you run only one, run **Idea 2's step 1** — the curl measurement. It is a day of work, requires no
training, no new hypothesis registration, and no model, and it returns a hard yes/no on whether an entire
research direction is alive. It is the highest information-per-hour experiment available to you.

**Idea 3** is the most likely to produce a real result, because it targets the exact defect your own
verification already exposed (calendar time is wrong) rather than a speculative one, and its primary
hypothesis (`τ_eff` ~ tick intensity) is falsifiable without reference to P&L.

**Idea 1** is the most intellectually ambitious and the most likely to consume three months and produce a
beautifully-converged model that knows nothing. Attempt it after one of the other two has taught you what
the honest benchmark looks like on a proper scoring rule.

And the standing constraint from `CLAUDE.md`: none of these touches `src/inference.py`, `models/`, or the
serving path until it clears a pre-registered gate in its own family log. The paper-trading ledgers, over
months, remain the only thing that decides.
