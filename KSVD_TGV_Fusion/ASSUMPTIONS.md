# Assumptions and deviations

Every choice that could change a result. All numeric values also live in the config cell of `ksvd_tgv_fusion_analysis.ipynb`.

## Decisions on ambiguous points

| # | Question | Decision | Why |
|---|---|---|---|
| A1 | What is `X_T`: TGV on the noisy image, or TGV on the K-SVD output? | **TGV on the raw image `Y`.** `X_K` is K-SVD on `Y`. | The brief says to reuse the outputs in `comparison/`, and `Comparison/outputs/denoised` stores exactly these. The sequential K-SVD then TGV pipeline of the pasted benchmark isn't in this repo (`adaptive_oct` and `scripts/run_adaptive_benchmark.py` don't exist here). |
| A2 | Base parameters | TGV `alpha=0.2, beta=1.0` (`lam=2, max_iter=300, tol=1e-3`). K-SVD `omp_C=1.6, lambda_scale=3.0`. | The frozen "balanced tuning" of `Comparison/comparison.ipynb`. Section 2 asserts them against `Comparison/outputs/run_log.txt`. They differ from the pasted benchmark, which used `omp_C=1.15`. |
| A3 | Which cases are scored? | **13 cases**: held-out 13-17 and other 2, 3, 5, 6, 8, 9, 11, 12. Cases 1, 4, 7, 10 are **training only**. | Those four tuned the base parameters, so scoring them would favour the base methods. This also matches `Comparison/`, and every training fold can still use them. |
| A4 | Missing outputs for cases 1, 4, 7, 10 | Generated with the repo's `methods.py` and the frozen settings, then rounded to uint8 like the stored files, and cached in `cache/`. | Section 2 tests that the same code reproduces the stored case 13 output. |
| A5 | Stored outputs are uint8 | Used as stored, so the small rounding difference from the floats scored by `Comparison/` remains. | Section 2 tests that the PSNR difference stays below 0.02 dB. |
| A6 | Data domain | Fusion runs on the stored 8-bit log-compressed scale, with no extra log. | `src/io_utils.py` documents the compression. |
| A7 | Primary endpoint | **Edge-band SSIM**, fixed in the config cell before any result existed. | The brief lists several structure metrics; one has to decide the winner. |
| A8 | How is the winner named? | Highest mean primary endpoint among W1 to W10. Because that reads test results, **WS** is also reported: in each fold, the spatial method with the lowest inner-CV MSE, chosen without the test case. | WS is the honest estimate of "pick a method and deploy it". |
| A9 | "Best single method" | Per metric, whichever of K-SVD and TGV has the better mean over the scored images. | This is the hardest baseline available, so it favours the null. |

## Deviations from the brief

| # | The brief says | What was done | Why |
|---|---|---|---|
| D1 | Fused output must always lie pointwise between `X_K` and `X_T`; assert it. | Asserted for W0 to W8, WS and controls. **Not asserted for W9 and W10.** | They blend frequency bands, so their output can leave that interval by construction. |
| D2 | Compute an oracle upper bound, show it prominently, and test that it beats every method. | **Removed**, including its tables, figure, macros and tests. | Requested after the full run. The study therefore has no stated ceiling. Two things that use the reference remain: the winner-map diagnostic (section 3), which no method sees, and W8's training target, the best local weight on training cases only. |
| D3 | W10: Laplacian pyramid. | **Non-decimated** pyramid: Gaussian differences at sigma 1, 3, 10 plus the residual, 4 levels. Weights are fitted jointly. | Exact reconstruction, simple per-pixel weights, and the cuts match the band diagnostics. |
| D4 | Post-processing for every spatial `w`. | Applied to W2 to W8. **Not applied to W10b**, whose per-level maps are already smooth through `g`. | W9, W10a and W1 have no spatial map to smooth. |
| D5 | "Sigmoid scale" is a hyperparameter. | The slope `a` is fitted directly, so there is no separate scale. Feature scale and post-smoothing are selected by inner CV. Bounds are `|a|, |b| <= 30`. | Avoids two parameters for the same thing. The bounds were widened from 10 after the quick run hit them. |
| D6 | Hyperparameters by inner CV over all combinations. | Two steps: hyperparameters by held-back sample MSE, then post-smoothing on held-back images with those hyperparameters fixed. W9's cutoff is evaluated on all training cases because it has nothing to fit. | Keeps the runtime feasible. Still training cases only. |
| D7 | Edge preservation: Laplacian-based correlation. | **Both** are reported: the repo's Sobel EPI (`methods.epi`) and a Laplacian version with a Gaussian pre-blur of sigma 1. | The repo's K-SVD notebook shows the unblurred Laplacian version ranks noisy images above denoised ones. |
| D8 | The repo's CNR, MSR, ENL. | CNR: `cnr`/`select_rois` of `PSO_Denoising.ipynb`, on a smoothed reference. ENL and `sigma_flat`: `flat_stats` of the TGV project's `04_real_data.py`. **MSR isn't defined in the repo**, so it is mean/std over the CNR foreground windows, as in Fang et al. (2012). | Reuse where possible. The ENL here is measured on log-compressed data, so it is only a relative smoothness score. |
| D9 | Reuse `comparison/error_correlation/roi_analysis/` if it exists. | It doesn't exist, so masks are built here. | |
| D10 | `pdflatex`/`latexmk` if available. | **Neither is installed**, so the report was not compiled. Section 11 checks statically that every macro, table and figure it references exists. | The brief's fallback: `.tex` and figures ready for Overleaf. |
| D11 | Python: numpy, scipy, scikit-image, scikit-learn, matplotlib, pandas. | Also uses `tifffile` for loading and `pywt` through the repo's `methods.py`. **`scikit-learn` was missing and was installed into `venv/`**; `requirements.txt` is unchanged. | W8 needs it. The repo's own code needs the other two. |
| D12 | Work only in `fusion_study/` (since renamed `KSVD_TGV_Fusion/`). | True for the notebook. Adding results to `Comparison/` was requested separately, afterwards. | |

## Other fixed choices

* **Seeds:** `SEED = 42`. Pixel sampling uses `default_rng([SEED, case])`, controls `default_rng([SEED, case, 1])`, bootstraps `default_rng(SEED)`, HGB `random_state=SEED`.
* **ROI:** Gaussian sigma 3, Otsu, closing with a disk of radius 15, largest component, **holes filled** (the brief doesn't list this, but dark layers otherwise leave holes), first 15 columns dropped.
* **Edge band:** top 10 % of the ROI's reference gradient (Gaussian sigma 1.5), searched in the ROI dilated by 6 px so both sides of the top retinal boundary count, then dilated by 2 px.
* **Features:** `depth` is the row divided by the image height, not depth below the retina. Variance, gradient and `dmag` use `log1p` because they are heavy-tailed. Coherence uses derivative scale 1 and integration scale `2s`. Everything is standardised with training-fold statistics.
* **Training pixels:** 20 000 random ROI pixels per case, the same for every fold and method. W1 uses all ROI pixels. Weights are fitted on the ROI but applied to the whole image.
* **W7b:** ridge fixed at 1.0. The `[0, 1]` constraint is enforced on training samples by cutting planes; test pixels are clipped.
* **W8:** 100 trees, learning rate 0.1, at least 200 samples per leaf, target is the best local weight in a sliding 15 px window, sample weight is the local mean of `d^2`.
* **Controls:** applied to WS's map. C1 permutes 32x32 blocks and leaves incomplete border blocks in place. C2 is smooth noise (sigma 16) through a sigmoid, with the offset bisected to match the mean.
* **Boundary tracer:** DP on a derivative of Gaussian (sigma 2 rows, 4 columns), at most 2 rows per column, with three anatomically anchored paths: the top retinal edge (within 15 px of the ROI's top), then the brightest band at least 20 px below it, whose upper and lower edges (within 15 px of the band) are the other two boundaries. The same tracer runs on the reference and on every output. The dataset has no manual layer annotations.
* **The tracer changed twice, and the second change came after seeing full results.**
  1. Three strongest paths, matched 3 against 3: replaced after the quick run, when an image nearly identical to the reference scored 37 px because the third-strongest edge changed rank.
  2. Five candidates per output, matched to the reference paths: replaced after the first full run. The per-pixel oracle, an image built to hug the reference, still scored about 20 px on cases 5, 13 and 14, per-case values jumped between about 1 and about 20 px, and a very small image change moved case 14 from 1.4 to 23 px. Paths could follow one layer and cross to another, so the metric was measuring path jumps. Means under that version were K-SVD 6.42, TGV 8.50, W1 10.13, W8 11.28 px, and no comparison with the better single method or W1 survived Holm.
  3. The anchored tracer above. It was chosen on validity only, not on method ranking: on the reference plus noise of s.d. 2 and 5 it moves by at most 0.19 and 0.80 px on scored cases, the image built to hug the reference scored best on every case, and the overlays sit on the anatomy. Noise stability is now a permanent test in section 7.
* **Known tracer failures, left in place:** in case 6 a large pathology means the ROI misses the elevated layer on the left, so the "top edge" follows the lower tissue there. In case 9 the top path jumps at the far right. Every method is scored against the same paths on those cases.
* **Rise distance:** profiles of +/-15 px aligned on each output's own trace and averaged over 32-column segments; the median over segments, then the mean over boundaries. Lower is sharper, but sharper than the reference isn't automatically better, so the reference value is reported too.
* **Statistics:** Wilcoxon two-sided with zeros dropped, Holm per metric and baseline, 10 000 bootstrap resamples over images, effect size `d_z` = mean / s.d. of the paired differences.
* **Crops and figures:** case 16, the example case of the earlier benchmark. The boundary crop is the 110x200 window with the highest mean reference gradient in the ROI. The speckle crop has the same width and start column, centred three quarters of the way down the retina.
* **Threads:** `OPENBLAS_NUM_THREADS=1` with 6 worker threads, because the machine has about 16 GB of RAM. HGB runs outside the pool with its own OpenMP threads.

## Known limitations

* The reference is a different realisation with its own residual grain, and it is smoother than any single frame, so a smoothing method is rewarded by construction. Differences between methods are meaningful; absolute PSNR is capped.
* 13 scored images from one dataset; 17 cases in total.
* The K-SVD dictionary saw the references of cases 1-12, so in training folds K-SVD may look slightly better than it would on new data, which could push weights toward K-SVD. Section 10 tests this for W1 on the held-out cases.
