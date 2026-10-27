# Step 1 — Reading of Duan et al. (2016), "Denoising optical coherence tomography using second order total generalized variation decomposition"

*Biomedical Signal Processing and Control* 24 (2016) 120–127.

This document records (a) my own summary of the method, (b) the derivations I had to
redo because the text/equations are garbled or inconsistent, and (c) every ambiguity I
found *before* writing code. Items marked **[ASSUMPTION]** are choices I made and state
in the report; items marked **[GAP]** are facts the paper simply does not contain.

---

## 1.1 Noise model and the decomposition idea

Speckle is modelled multiplicatively, `f = u·ξ` (Eq. 2.3–2.4; the additive term `n` is
dropped as negligible). Because OCT is displayed log-compressed, the *log* of the data
turns multiplicative speckle into **additive** noise, and the denoising problem becomes
"remove an additive, highly oscillatory component". The authors therefore adopt an
**image-decomposition** (not filtering) view: split the log-image `f` into a piecewise
smooth *structure* `u` (the denoised image) and an *oscillatory texture* `div(g)`
(the speckle), with `g = (g1,g2)` a vector field.

Vese–Osher (VO) model, Eq. (2.5):

    E(u,g) = ½‖f − u − div g‖²₂ + β‖g‖₁ + α∫|∇u|

TGV² model, Eq. (2.6):

    min_{u,g}  ½‖f − u − div g‖²₂ + β‖g‖₁ + TGV²_α(u)

with (Eq. 2.2)

    TGV²_α(u) = min_p  α₁∫|∇u − p| + α₀∫|ε(p)|,
    ε(p) = ½(∇p + ∇pᵀ).

Eliminating the inner minimum `p` (Eq. 2.11) gives the problem actually solved:

    min_{u,g,p} ½‖f − u − div g‖²₂ + β‖g‖₁ + α₁‖∇u − p‖₁ + α₀‖ε(p)‖₁.

Interpretation: `½‖f − u − div g‖²` ties structure + texture to the data; `β‖g‖₁` makes
`div(g)` a *sparse-oscillation* field (small L1 norm = highly oscillatory); `α₁‖∇u − p‖₁`
lets `p` explain the gradient of `u`, so `u` may be **piecewise affine** rather than
piecewise constant — that is the whole point, and it is why TGV does not staircase.
`α₀‖ε(p)‖₁` keeps `p` itself smooth. α₀/α₁ is the TGV "order split": α₁ large ⇒ TV-like,
α₀ large ⇒ stronger penalisation of the second derivative.

## 1.2 Split-Bregman formulation (Eq. 3.1)

Three split variables and three Bregman variables are introduced:

| split var | meaning | constraint | Bregman var | penalty |
|---|---|---|---|---|
| `m` | copy of `g` | `m = g` | `b1` | λ₁ |
| `w` | copy of `∇u − p` | `w = ∇u − p` | `b2` | λ₂ |
| `v` | copy of `ε(p)` (3 distinct entries, symmetric 2×2) | `v = ε(p)` | `b3` | λ₃ |

Augmented functional actually minimised (Eq. 3.1):

    E(u,g,p,v,m,w;b1,b2,b3) =
      ½‖f − u − div g‖²
    + β‖m‖₁ + (λ₁/2)‖m − g − b1‖²
    + α₁‖w‖₁ + (λ₂/2)‖w − ∇u + p − b2‖²
    + α₀‖v‖₁ + (λ₃/2)‖v − εp − b3‖²

The `‖m‖₁, ‖w‖₁, ‖v‖₁` are *isotropic* (pointwise vector 2-norms summed over pixels),
not entry-wise absolute values; `‖v‖₁` uses all four entries of the symmetric matrix,
i.e. `√(v₁₁² + v₂₂² + 2v₁₂²)`.

**Algorithm 1** (Gauss–Seidel / alternating minimisation, one sweep per iteration,
all variables initialised to 0, fixed penalties):

1. `u` ← FFT closed form (3.2)
2. `g` ← FFT closed form (3.3)–(3.4), a 2×2 Hermitian system solved pointwise in k
3. `p` ← FFT closed form (3.5)–(3.6), another 2×2 system
4. `v` ← isotropic shrinkage (3.7)
5. `m` ← isotropic shrinkage (3.8)
6. `w` ← isotropic shrinkage (3.9)
7. `b1 ← b1 + (g − m)`; `b2 ← b2 + (∇u − p − w)`; `b3 ← b3 + (εp − v)`
8. stop when the relative change of `u` is < 10⁻³

`h = 1`, `∇` forward differences, `div` backward differences, periodic boundary
conditions (the paper invokes "circulant boundary condition" for the `p` step and
implicitly for all FFT steps). Because the penalty parameters are **fixed** (not
continued/adapted), this is exactly the standard split Bregman iteration of Goldstein &
Osher and it converges monotonically in the energy for fixed λ's.

## 1.3 What I had to derive myself (and why)

The text layer of the PDF mangles subscripts, Greek letters and the signs in
Eqs. (3.2)–(3.6), and — more importantly — **the printed equations are not all mutually
consistent**. I therefore derived every closed form directly from the energy (3.1) and
then *validated each one numerically against a direct sparse linear solve* of the same
sub-problem (`tests/test_operators.py`, `src/verify.py`). Derivations:

**u-step.** Stationarity of (3.1) in `u`:
`(f − u − div g) − λ₂ div(w + p − b2) − λ₂ Δu = 0` ⇒

    (1 − λ₂Δ) u = f − div g − λ₂ div(w + p − b2)

    u = ℜ F⁻¹[ F(f − div g − λ₂ div(w + p − b2)) / (1 − λ₂ F(Δ)) ]

With `F(Δ) = −(|d̂ₓ|² + |d̂_y|²)` this denominator is `1 + λ₂(|d̂ₓ|²+|d̂_y|²) > 0`, i.e.
strictly positive — which confirms the printed `1 − λ₂F(Δ)` is a *sign-carrying* form and
not a typo.

**g-step.** `(λ₁ I − ∇div) g = λ₁(m − b1) + ∇(u − f)`, with symbol matrix
`A = [[λ₁+|d̂ₓ|², d̂ₓ d̂ȳ], [d̂ȳ d̂ₓ̄, λ₁+|d̂_y|²]]` (Hermitian), `det A = λ₁(λ₁+|d̂ₓ|²+|d̂_y|²)`.
The paper's `h₁ = ∂ₓ(u−f) − λ₁(b1 − m)` is exactly `∇(u−f) + λ₁(m − b1)`; ✓ consistent.

**p-step.** `(λ₂ I + λ₃ εᵀ_W ε) p = λ₂(∇u + b2 − w) − λ₃ D^B(v − b3)` where
`D^B σ = (∂ₓ^B σ₁₁ + ∂_y^B σ₁₂, ∂ₓ^B σ₁₂ + ∂_y^B σ₂₂)` and the Hessian uses the *weighted*
(1,1,2) inner product because the tensor norm double-counts the off-diagonal. Symbol matrix

    b₁₁ = λ₂ + λ₃|d̂ₓ|² + (λ₃/2)|d̂_y|²
    b₂₂ = λ₂ + (λ₃/2)|d̂ₓ|² + λ₃|d̂_y|²
    b₁₂ = +(λ₃/2) d̂ₓ conj(d̂_y) ,  b₂₁ = conj(b₁₂)

which reproduces the printed diagonal entries `F(λ₂ − λ₃∂ₓ∂ₓ − (λ₃/2)∂_y∂_y)` and the
printed off-diagonals `F(−(λ₃/2)∂_y∂ₓ)`, `F(−(λ₃/2)∂ₓ∂_y)` **exactly**, provided those
`∂` are read as the *backward* differences (`F(∂^B∂^F) = −|d̂|²`, and `∂_y^B∂_x^F` has
symbol `−conj(d̂_y)d̂ₓ`). I initially mis-signed this off-diagonal; the direct-solve check
(`src/verify.py`) caught it, and the corrected matrix agrees with the paper.

The one place where the printed text is genuinely wrong is `d₁, d₂`: the paper's version
is only consistent with the printed `b` matrix if the differences inside the bracket are
**backward** ones, i.e.

    d₁ = λ₂(∂ₓu + b2₁ − w₁) − λ₃[∂ₓ^B(v₁₁−b3₁₁) + ∂_y^B(v₁₂−b3₁₂)]

which is what I implement. All three closed forms now agree with an explicit sparse solve
of the corresponding quadratic to ~1e-16 relative error.

**Shrinkage steps.** `v, m, w` are entry-wise isotropic soft thresholds with the convention
`0·(0/0) = 0` (stated in the paper, implemented as an explicit mask):

    m = (g+b1)/|g+b1| · max(|g+b1| − β/λ₁, 0)          (pointwise vector modulus)
    w = (∇u−p+b2)/|·| · max(|·| − α₁/λ₂, 0)
    v = (εp+b3)/|·| · max(|·| − α₀/λ₃, 0)              (4-component modulus)

I verified all three are the exact proximal maps of `γ‖·‖₁ + (λ/2)‖·−z‖²` for the
isotropic norm.

## 1.4 Experimental protocol (Section 4)

* **Synthetic data** (4.1): `f(x,y) = Σᵢ (aᵢ/2) exp(−((x−xᵢ)² + (y−yᵢ)²)/(2σᵢ²)) − bᵢ`,
  9 retinal layers, then **zero-mean Gaussian noise with variance 0.02…0.1** (step 0.02).
  Metrics: PSNR, SNR, RMSE, SSIM (Table 2, Fig. 3), visual comparison (Fig. 4).
* **Real data**: in-vivo human retina B-scans (ENVISU C-class 2300, 3.3 µm axial,
  500 A-scans × 100 B-scans, 3 subjects), visual comparison only (Figs. 5–7).
* **Baselines** (Table 1): HMF, HWT (hard), DTCWT (soft), NCDF, ACED, VO.
* Hardware: MATLAB 2014b, Xeon E5-1620, 32 GB. No runtime is reported.

## 1.5 Ambiguities, typos, missing details

**[GAP] 1. No parameter values anywhere.** α₀, α₁, β, λ₁, λ₂, λ₃ are never given, nor are
the number of iterations, nor the threshold settings for HWT/DTCWT/NCDF/ACED. Table 2 is
therefore not reproducible as printed. I perform a documented grid search on a *single*
noise level, freeze the parameters, and use them for every other experiment (details in
`report/report.md` §4).

**[GAP] 2. Phantom parameters missing.** `aᵢ, bᵢ, σᵢ, (xᵢ,yᵢ)` in Eq. (4.1) are not
reported, only "by choosing appropriate parameters". Fig. 2 shows 9 layers with ILM and
RPE hyper-reflective. **[ASSUMPTION]** I built my own 9-layer phantom from Eq. (4.1) with a
documented table of `aᵢ, σᵢ, bᵢ` and a foveal-dip curvature; it is *qualitatively* the
same image class, not a bit-exact reproduction.

**[GAP] 3. Noise variance scale.** "Variance 0.02" is only meaningful given the image
range. Since their noisy-image PSNR is 17.53 dB, the image must be ≈ [0,1] with
peak ≈ 1.0–1.06. **[ASSUMPTION]** phantom in [0,1], peak = 1 for PSNR.

**[CONTRADICTION] 4. Table 2's own RMSE column is inconsistent with its PSNR/SNR columns.**
For the noisy image at σ²=0.02 they report PSNR = 17.53 dB (⇒ MSE ≈ 0.0177) and
SNR = 8.47 dB (⇒ MSE ≈ 0.0177 given mean(u²)≈0.125 — consistent), but RMSE = 9.9006,
whereas √MSE = 0.133. The ratio RMSE/√MSE is 74 at σ²=0.02 and 45 at σ²=0.08, so it is not
a constant unit scaling either. I report standard `RMSE = √MSE` on the [0,1] scale (and its
×255 equivalent) and flag that my RMSE column is not directly comparable with theirs.

**[GAP] 5. Metric definitions.** SSIM variant (window size, Gaussian vs uniform, data
range), and the PSNR peak (`max(u)` vs 1.0) are unspecified. **[ASSUMPTION]** SSIM per
Wang et al. via `skimage.metrics.structural_similarity` with `data_range=1.0`, 7×7 uniform
window (skimage default) — I also report the Gaussian-window variant in the sensitivity
section because the paper's near-zero noisy SSIM (0.13) is very sensitive to this choice.
SNR = `10log10(‖u‖²/‖u−û‖²)`.

**[CONTRADICTION] 6. The noise model is used inconsistently.** Eq. (2.3)→(2.4) *drops* the
additive term because it is "significantly small", and then the method relies precisely on
the additive post-log noise being the thing that is removed. Log-domain noise in OCT is
also not Gaussian. This is a modelling weakness, not an implementation blocker; I follow the
paper's protocol (additive Gaussian in the log domain).

**[AMBIGUITY] 7. Log transform on real data.** The paper says `f` in (2.6) is "the original
noisy OCT image after logarithmic transformation" but never says whether their B-scans
were linear or already display-compressed. The supplied dataset (`Sparsity_SDOCT_DATASET_2012`,
Duke/Hunan, uint8 0–255, 8-bit) is **already log/display-compressed**: intensities are
unimodal with a long right tail (skew ≈ 1.0–1.4 on the near-noise-free averaged images),
a mid-grey background pedestal (~70/255) and no zeros in the background, which is the
signature of a post-log display mapping, not of linear |signal|². I also test empirically
whether local noise std grows with local mean (multiplicative) or is flat (additive):
result in `results/inspection/noise_model.txt`. **[ASSUMPTION]** the TIFF values are
treated as log-domain and only linearly rescaled to [0,1] (`x/255`); an optional
`log_domain=False` path applies `log(1+x)` first if a linear-intensity file is supplied.

**[AMBIGUITY] 8. Stopping rule norm.** "Relative errors of u smaller than 0.001" — norm
unspecified. **[ASSUMPTION]** `‖uᵏ − uᵏ⁻¹‖₂ / ‖uᵏ‖₂ < 10⁻³`, plus `max_iter` as a hard cap.

**[AMBIGUITY] 9. "three L1 norms".** There are in fact *four* L1 terms in (2.11)
(`‖g‖₁, ‖∇u−p‖₁, ‖εp‖₁` = three, plus the squared data term) but *three* in the split
functional (3.1) after `‖g‖₁` is transferred to `‖m‖₁`. Minor, but worth noting because it
explains why only three split/Bregman pairs are used.

**[TYPO] 10.** In (3.1) the `w`-term is printed as `‖w − ∇u + p − b2‖²`; the ordering is
unusual but correct (`w` approximates `∇u − p`). Also `v` is called "4D" and listed as
`(v₁₁,v₂₂,v₁₂,v₁₃)` in the text — it is a *symmetric* 2×2 tensor, so only 3 independent
components exist (`v₁₃` is a doubled subscript `v₃` = the (1,2) entry). I store 3 components
and use the 4-entry Frobenius modulus.

**[OMISSION] 11.** ACED (Weickert coherence-enhancing diffusion) and NCDF parameter values,
the DTCWT decomposition level and threshold rule, and the HWT threshold rule are all
unreported. My baseline settings are documented in `src/baselines.py`; where a method has
a free threshold I use the standard universal/MAD rule rather than tuning it per noise
level, so the baselines are not unfairly handicapped *or* helped.

## 1.6 Consequence for the reproduction

Because α₀, α₁, β, λ₁–λ₃ are missing, "reproduce Table 2 to 4 decimal places" is not a
well-posed target. What I *can* reproduce is: (i) the algorithm as derived and validated,
(ii) the qualitative and quantitative *ordering* of methods, (iii) the shape of the
metric-vs-noise curves, and (iv) the absolute metric range under a documented parameter
choice. I report all four, and I do **not** tune parameters per noise level or per metric
to chase the printed numbers.
