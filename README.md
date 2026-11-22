# HXEnsemble

**Version 1.0.0** — amide hydrogen exchange (HX) forward models and maximum-entropy reweighting of protein conformational ensembles using HX-NMR protection factors or hydrogen–deuterium exchange mass spectrometry (HDX-MS) uptake.

HXEnsemble calculates amide opening estimates from atomistic MD trajectories using the **M11, M12 or M14** structural models, including a Debye–Hückel (DH) electrostatic contribution. It then refines frame populations to reproduce experimental HX observations while penalising deviations from prior ensemble weights (Bayesian-maximum entropy reweighting).

## Installation

Python **3.10+** is required (Python 3.11 is used in `environment.yml`). From the repository root:

```bash
conda env create -f environment.yml
conda activate hxensemble
pip install .
```

Alternatively, use `pip install -e .` for an editable installation. The package depends on NumPy, SciPy, pandas, MDTraj, OpenMM, Matplotlib and threadpoolctl. OpenMM is used for automatic force-field charge assignment; you can bypass charge assignment by setting `--DH_scaling 0` or provide `--user-charges`.

The installation provides three programs:

```text
hxensemble-forward   Compute the HX forward model from PDB/XTC trajectories
hxensemble-nmr       Reweight against residue-resolved HX-NMR protection factors
hxensemble-ms        Reweight against peptide-resolved HDX-MS uptake
```

Use `<command> --help` for the full current options.

## Forward-model definitions

An amide is classified as **structurally open** in a given frame only when all three descriptors satisfy the model-specific criteria:

| Model | Amide-H SASA | Hydrogen bonding | Front-side contacts ($\rho_{\mathrm{front}}$) | Peptide factor $c_{\mathrm{peptide}}$ |
|---|---|---|---|---|
| **M11** | > 0.05 Å² | No H-bond | < 0.5; exclude atoms in the same and preceding residues of that chain | 0.957319 |
| **M12** | > 0.05 Å² | No H-bond | < 0.5; no local-residue exclusion | 0.957089 |
| **M14** | > 0.5 Å² | No H-bond | < 0.5; no local-residue exclusion | 0.951914 |

The hydrogen-bond test considers an amide H bonded when an eligible acceptor is within 2.6 Å of H and the N–H–acceptor angle is at least 120°. The SASA and contact descriptors include non-protein atoms. $\rho_{\mathrm{front}}$ is the distance- and orientation-weighted contact density in front of the amide hydrogen.

The three models (M11, M12 and M14) generally produce similar results. M14 is used by default, with a Debye–Hückel electrostatic correction scaled by a default factor of 0.75. 

In explicit open-state models, the protection factor of amide $i$ is calculated from its ensemble-averaged effective exchange-competent population,

$$
PF_i = \frac{1}{\mu_i}, \qquad
\mu_i = \sum_n w_n
\frac{O_i^{(n)}}{c_{\mathrm{peptide}}PF_{\mathrm{DH},i}^{(n)}}.
$$

where $w_n$ is the statistical weight of conformation $n$, $O_i^{(n)}$ is a binary indicator equal to 1 when the amide is structurally open and 0 when closed, $c_{\mathrm{peptide}}$ is a peptide-based correction factor, and $PF_{\mathrm{DH},i}^{(n)}$ accounts for electrostatic modulation of exchange. Without these corrections, the protection factor reduces to the inverse of the ensemble-averaged open-state population. The explicit open-state models share this functional form but differ in the structural criteria used to classify amides as exchange competent.

The `O.npy` output stores the **full per-frame effective exchange-competent opening values**, $O_i^{(n)}/(c_{\mathrm{peptide}}PF_{\mathrm{DH},i}^{(n)})$, **not** the binary structural opening indicators $O_i^{(n)}$. Both the peptide factor and DH correction are already incorporated, so averaging `O.npy` over frames with weights $w_n$ directly gives $\mu_i$; these corrections must not be applied again during reweighting.

Setting `--DH_scaling 0` disables the DH electrostatic correction ($PF_{\mathrm{DH},i}^{(n)}=1$), but the peptide correction remains included in `O.npy`.

## 1. Calculate amide HX opening using a forward model

Supply either a directory containing **exactly one PDB** and one or more XTC files, or an explicit PDB and trajectory list:

```bash
hxensemble-forward \
    --directory MD_DATA \
    --model M14 \
    --DH_scaling 0.75 \
    --chunk 1000 \
    --n-cpus 4 \
    --out forward

# Equivalent explicit input form:
hxensemble-forward \
    --topology protein.pdb --trajectory run1.xtc run2.xtc \
    --model M14 --DH_scaling 0 \
    --out forward
```

The PDB and all XTC files must have consistent atom ordering. Trajectories are processed in the supplied order (or sorted XTC filename order for `--directory`). The program processes frames in chunks; `--stride` optionally subsamples frames, and `--n-cpus` parallelises forward-model chunks across processes (default **1**). The default chunk size is **1,000** frames.

The forward model identifies protein backbone `H`/`HN` atoms and their corresponding nitrogen atoms. It computes solvent accessibility, hydrogen-bond and front-side contact descriptors from the **complete topology**, including non-protein atoms; DH calculations also include charges on all atoms. Correct atom identities, chemically meaningful hydrogens and appropriate input structures are essential. In the current implementation, M11/M12/M14 differ in SASA threshold, peptide correction and whether the same/preceding residues are excluded from the front-side contact descriptor.

The default `--DH_scaling` is **0.75**. Set `--DH_scaling 0` to disable the electrostatic contribution. With DH enabled, charges are read from `--user-charges charges.txt` (or `.npy`) in PDB atom order if given; otherwise the program attempts OpenMM AMBER14 charges and then its bundled JSON fallback. Automatic assignment may fail for non-standard residues or complex topologies, in which case provide correctly ordered charges. In particular, the JSON fallback identifies terminal residues using the global minimum/maximum residue numbers, so it may not assign chain termini correctly in heterogeneous multichain systems; prefer explicitly validated charges for such systems. DH temperature and ionic strength default to `--temperature 298.0` K and `--ionic 0.15` M.

`--filter-frames` discards frames with any pair of atoms closer than **0.001 nm** and writes a record of retained trajectory/frame indices. `--save-components` saves the underlying structural and DH contributions in addition to the opening matrix.

### Forward-model outputs

| File | Meaning / shape |
|---|---|
| `O.npy` | Full effective exchange-competent opening values, $O_i^{(n)}/(c_{\mathrm{peptide}}PF_{\mathrm{DH},i}^{(n)})$, **including both peptide and DH corrections** (not binary opening indicators); **`(n_frames, n_amides)`** |
| `amideresidues.txt` | PDB residue number (`resSeq`) for each column of `O.npy` |
| `amidechainidx.txt` | MDTraj chain index (zero-based) for each amide column |
| `amideresidx.txt` | MDTraj topology residue index for each amide column |
| `open_sasa_hbonds.npy` | SASA/hydrogen-bond binary opening component, `(n_frames, n_amides)`; optional |
| `rho_front.npy` | Front-side contact descriptor, same shape; optional |
| `lnPF_DH.npy` | DH electrostatic ln(PF) contribution, same shape; optional (zeros when DH is disabled) |
| `retained_frames.csv` | Trajectory path and zero-based original frame index/number; only with `--filter-frames` |

All subsequent CLI commands load `O.npy` and the residue/chain metadata from the forward output. Internally, the reweighting functions transpose `O` to **`(n_amides, n_frames)`**. The optional component arrays have the original **frames × amides** layout.

## 2. Reweight against HX-NMR protection factors

Prepare a CSV containing at least `Residue`, `PF` and `PF_error` (or `Model_residue` in place of `Residue`). PFs and their uncertainties must be positive. Residue numbers refer to the same PDB numbering used by the forward model.

```csv
Residue,PF,PF_error
12,35,17.5
13,120,60
14,8,4
```

```bash
hxensemble-nmr \
    --O forward/O.npy --metadata forward \
    --experiment nmr.csv \
    --method log_lambda \
    --out nmr_fit
```

Choose either `--method log_lambda` (default; parameters correspond to experimental observables) or `--method log_softmax` (parameters correspond to trajectory frames). Both optimise agreement in **ln(PF)** space with KL regularisation relative to the prior ensemble. The default prior is a set of uniform weights, which can be replaced with user-specified weights using `--prior weights.npy` (or a text file of positive weights).

The default scan starts with **30 theta values from 1e5 to 1**, descending, and adaptively extends to smaller values if needed; it stops when the entropy effective fraction drops below `--min-neff` (default **0.1**) or reduced chi-squared is below **0.001**. Explicit `--theta-values`, or a range set with `--theta-min`, `--theta-max` and/or `--n-theta`, fits **exactly the requested grid** without adaptive extension. The NMR fit excludes experimental amides that never sample a nonzero opening value; these are reported separately, rather than assigned artificial finite ln(PF) predictions. **At least two matched, exchange-competent NMR observations are currently needed** because the retained legacy diagnostic uses Pearson correlation.

### NMR outputs

| File | Meaning |
|---|---|
| `scan.csv` | One row per retained theta: `theta`, `phi` (entropy-based), `chi2r` |
| `wopt_array.npy` | Optimised frame weights **`(n_theta, n_frames)`**; **no prior row** |
| `wopt.npy` | Selected posterior weights (last solution satisfying `phi >= min-neff`) |
| `prior_lnPF.txt` / `rew_lnPF.txt` | Fitted-amide predictions before/after reweighting, in the order of `matched_nmr.csv` |
| `matched_nmr.csv` | Fitted experimental observations, with `Chain` and `Residue` |
| `nmr_amides_no_open_state.csv` | Experimental amides excluded because all sampled opening values are zero |
| `amide_mapping.csv` | `Amide_column`, `Chain`, `Residue` for all forward-model amides after optional chain averaging |
| `wopt_elbow.npy`, `rew_lnPF_elbow.txt`, `elbow_summary.txt` | Geometric elbow solution and report, if at least three theta solutions were retained |

`phi = exp[-KL(w || w0)]` is the entropy-based effective fraction relative to prior weights `w0`. Elbow selection is an alternative posterior selection and does **not** necessarily equal the default `wopt.npy` posterior.

## 3. Reweight against HDX-MS uptake

The experimental CSV must contain the following exact column names, with **inclusive** peptide start/end residue numbers:

```csv
Start,End,Sequence,Time (s),Rel. Uptake,Rel. Uptake SD
30,35,AAAAAA,10,0.28,0.04
30,35,AAAAAA,100,0.48,0.05
```

`Sequence` is mandatory and is checked against the FASTA for **every** row. `Rel. Uptake` and `Rel. Uptake SD` are dimensionless fractional values, not absolute deuterium counts or percentages on a 0–100 scale. The supplied relative uptake should use the experiment's intended normalisation, such as its measured maximum uptake after back exchange. The fitter clips experimental uptake to [0, 1] and uses `Rel. Uptake SD` in the fitting objective; provide appropriate uncertainty values.

### Calculate intrinsic exchange rates from FASTA

```bash
hxensemble-ms \
    --O forward/O.npy --metadata forward \
    --experiment ms.csv --fasta sequence.fasta \
    --temperature 298.0 --pHread 7.0 --fraction-d2O 1.0 \
    --kint-model 3ALA \
    --out ms_fit
```

`--temperature` is in K. The pD used for intrinsic-rate calculation is **`pHread + 0.4 × fraction-d2O`**; for the example it is **7.4**. The default rate model is `3ALA` (Stofella2025); `PDLA` selects Nguyen2018. When calculating rates from FASTA, **temperature, pH meter reading and D₂O fraction are required**; there are no assumed defaults. The generated per-chain rates are saved as `kint_chains.txt`.

Alternatively, supply a precomputed rate table with `--kint rates.txt --fasta sequence.fasta` (no intrinsic-rate calculation is performed). The table contains **one row per FASTA position**, including prolines and the first sequence residue, and **one column per topology chain** in chain-index order. A single-chain file may contain a single column; different-length chains are padded with trailing `nan` entries. The FASTA remains mandatory for checking peptide identities and verifying missing proline amides. The first sequence position may use a `+inf` intrinsic-rate sentinel (excluded from fitted uptake); internal missing rates are not allowed.

**Residue numbering:** The FASTA need not begin at residue 1, but the mapping must be specified. For a FASTA whose first residue corresponds to PDB residue **27** (and MS `Start`/`End` values use that numbering), supply `--sequence-start 27`. For a peptide spanning residues **30–35**, the program compares its sequence against FASTA positions `3:9` (Python slice) and finds opening values by residue numbers **30–35** in the forward metadata. It does **not** assume that residue 30 occupies a particular column of `O.npy`. Use one `--sequence-start` value per topology chain for multichain systems; omission means 1 for each chain.

By default, `--n-skip 2` excludes the **first two residues of each experimental peptide** from the uptake prediction, and prolines contribute no exchangeable backbone amide. A peptide-covered non-proline amide missing from the forward-model metadata raises an error (except the first FASTA sequence position). Amides present in `O.npy` with zero opening across all frames remain in the calculation. Rows with invalid time/uptake/uncertainty or with no exchanging residues after the peptide skip are excluded and counted.

### Theta scan and prior-only mode

The default MS scan begins with **30 theta values from 1e7 to 1**, extending to smaller theta if needed, and stops when `phi < --min-neff` (default **0.1**) or `chi2r < 0.001`. Use `--theta-values 1e6 1e5 1e4`, or `--theta-min`/`--theta-max`/`--n-theta`, to run an explicit grid without early stopping or extension. The selected posterior is the **last fitted** theta that satisfies `phi >= min-neff`; for a descending grid this is the smallest acceptable theta.

To calculate only prior-ensemble uptake and statistics, without optimisation:

```bash
hxensemble-ms \
    --O forward/O.npy --metadata forward \
    --experiment ms.csv --fasta sequence.fasta \
    --temperature 298 --pHread 7 --fraction-d2O 1 \
    --n-theta 0 --out ms_prior
```

### MS outputs

| File | Meaning |
|---|---|
| `scan.csv` | Prior row (theta empty) followed by one row per optimised theta: `theta`, `chi2r`, `rmse`, `phi`, `phi_kish` |
| `wopt_array.npy` | Frame weights **`(1 + n_theta, n_frames)`**; **row 0 is the prior** |
| `pred_uptakes.npy` | Predicted fractional uptake **`(1 + n_theta, n_observations)`**, same row order as `wopt_array.npy` |
| `opt_convergence.npy` | Convergence flags in scan order, starting with the prior row |
| `wopt.npy` / `pred.npy` | Selected posterior weights / predictions (`pred.npy` is prior predictions in prior-only mode) |
| `pred_prior.npy` | Predictions from the prior frame weights |
| `prior_lnPF.txt` / `rew_lnPF.txt` | Prior / selected-posterior ln(PF) for **all forward-model amides** after any averaging; zero opening yields `inf` |
| `matched_ms.csv` | Retained experimental peptide/timepoint rows, in prediction-array column order |
| `amide_mapping.csv` | Forward-model amide column, chain and residue number after averaging |
| `ms_residue_axis.csv` | Chain/residue axis spanning measured peptides, including non-exchangeable prolines |
| `kint_residue_axis.npy` | Intrinsic exchange rates along `ms_residue_axis.csv` |
| `kint_chains.txt` | Generated per-sequence intrinsic rates, **only** when computed from FASTA |
| `wopt_elbow.npy`, `pred_elbow.npy`, `rew_lnPF_elbow.txt`, `elbow_summary.txt` | Optional elbow solution, if at least three theta values were fitted |

In **prior-only** mode, `scan.csv`, `wopt_array.npy`, `pred_uptakes.npy` and `opt_convergence.npy` contain only the prior row; `pred.npy` and `pred_prior.npy` both contain prior uptake predictions. No selected posterior weights, `rew_lnPF.txt` or elbow outputs are produced.

Note that the NMR and MS `wopt_array.npy` conventions differ: **NMR has one row per theta; MS has an additional leading prior row.**

## 4. Multiple chains and chain averaging

The forward model **always calculates each protein chain separately**; `amidechainidx.txt` records zero-based MDTraj chain indices, not PDB chain letters.

For different chains with independently measured observations, include a `Chain` column in the experimental CSV and provide chain-specific FASTA/rates as applicable. Repeated residue numbers on different chains are valid: matching uses the pair **`(Chain, Residue)`**.

For equivalent homomeric subunits, reweighting can average opening values **frame by frame, at corresponding residue numbers, before fitting**:

```bash
hxensemble-nmr --O forward/O.npy --metadata forward \
    --experiment nmr.csv --average-chains 0 1 2 --out nmr_avg
hxensemble-ms --O forward/O.npy --metadata forward \
    --experiment ms.csv --fasta trimer.fasta \
    --temperature 298 --pHread 7 --fraction-d2O 1 \
    --average-chains 0 1 2 --out ms_avg
```

The averaged group retains the **first supplied chain index** (chain 0 in this example). If no distinct chains remain, the experimental `Chain` column may be omitted. If other chains remain, `Chain` is required and measurements from the averaged group must use its representative index. Averaged chains must have identical sets of amide residue numbers. For MS, they must also have identical FASTA sequences, sequence starts and intrinsic rates; provide one FASTA entry and one rate column **per original topology chain**, even when averaging.

When averaging is requested, both fitting commands additionally save:

| File | Meaning |
|---|---|
| `O_averaged.npy` | Frame × amide opening matrix after averaging; unaveraged chains are retained |
| `amideresidues_averaged.txt` | Residue number per column of `O_averaged.npy` |
| `amidechainidx_averaged.txt` | Chain index per column; averaged group uses first requested index |

To trace every matched observation, inspect `amide_mapping.csv` and `matched_nmr.csv` / `matched_ms.csv`.

## Testing

Install test dependencies and run the included fast unit/integration tests:

```bash
pip install -e '.[test]'
python -m pytest -q
```

The test suite checks forward-model combination rules and atom selection, matching and averaging across chains, intrinsic-rate/sequence validation, prior-only and theta-scan output conventions.

## Citation

If you use HXEnsemble in your research, please cite:

Streit J.O., Peterle D., Tesei G., Invernizzi M., Bottaro S., Schulze T.K., Kunz P., Tamiola K., Lindorff-Larsen K. (2026). *Integrating hydrogen-exchange experiments with conformational ensembles*. **bioRxiv**. Link TBA.
