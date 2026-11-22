from pathlib import Path
import numpy as np
import pandas as pd
from .fastbme import HDXReweighterFast
from .utils import find_elbow
from threadpoolctl import threadpool_limits


def open_matrix(path):
    """On-disk convention: frames x amides, transposed for BME algorithms."""
    a = np.load(path, mmap_mode='r')
    if a.ndim != 2: raise ValueError('O.npy must be frames x amides')
    return np.asarray(a.T, dtype=np.float64)


def weights0(n, path=None):
    a = np.ones(n)/n if path is None else np.asarray(np.load(path) if str(path).endswith('.npy') else np.loadtxt(path),float)
    if a.shape != (n,) or np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f'Prior weights must be a positive finite length-{n} vector')
    return a/a.sum()


def load_metadata(folder, n):
    folder = Path(folder)

    def load_ints(path):
        values = np.atleast_1d(np.loadtxt(path))
        if not np.all(np.isfinite(values) & (values == np.round(values))):
            raise ValueError(f'Metadata must contain integer values: {path}')
        return values.astype(int)

    resid = load_ints(folder / 'amideresidues.txt')
    chainfile = folder / 'amidechainidx.txt'
    chain = load_ints(chainfile) if chainfile.exists() else np.zeros(len(resid), int)

    if len(resid) != n or len(chain) != n:
        raise ValueError('Residue/chain metadata mismatches O columns')

    return resid, chain


def average_chains(O, residues, chains, average):
    """Average equivalent subunits frame by frame, retaining other chains.

    The first chain index in `average` identifies the averaged observations.
    All grouped chains must have exactly the same exchangeable residue numbers.
    """
    if not average:
        return O, residues, chains
    average = list(map(int, average))
    if len(set(average)) != len(average) or len(average) < 2:
        raise ValueError('Provide at least two distinct chain indices to average')
    if any(c not in chains for c in average):
        raise ValueError('Requested chain not present in forward-model metadata')
    # The first chain is the representative label in the experimental CSV.
    # Averaging is performed per frame, BEFORE matching experimental data.
    ref = np.sort(residues[chains == average[0]])
    if len(ref) != len(set(ref)):
        raise ValueError('Duplicate residue numbers within a chain')
    for c in average[1:]:
        if not np.array_equal(ref, np.sort(residues[chains == c])):
            raise ValueError('Averaged chains must have identical amide residue numbers')
    rows = []
    out_res = []
    out_chain = []
    for i, (c, r) in enumerate(zip(chains, residues)):
        if c in average:
            if c != average[0]:
                continue
            indices = [np.flatnonzero((chains == cc) & (residues == r))[0]
                       for cc in average]
            rows.append(np.mean(O[indices], axis=0))
        else:
            rows.append(O[i])
        out_res.append(r)
        out_chain.append(c)
    return np.asarray(rows), np.asarray(out_res), np.asarray(out_chain)


def save_averaged_chains(out, O, residues, chains):
    """Save the chain-averaged forward model in the standard frames x amides layout."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'O_averaged.npy', O.T)
    np.savetxt(out / 'amideresidues_averaged.txt', residues, fmt='%d')
    np.savetxt(out / 'amidechainidx_averaged.txt', chains, fmt='%d')


def _chain_for_experiments(exp, chains):
    """Return an explicit chain index for every experimental row."""
    if 'Chain' in exp:
        if exp['Chain'].isna().any():
            raise ValueError('Chain contains missing values')
        return exp['Chain'].astype(int).to_numpy()
    unique = np.unique(chains)
    if len(unique) != 1:
        raise ValueError('For multiple distinct chains, add a Chain column to experimental CSV')
    return np.full(len(exp), int(unique[0]), dtype=int)

def fit_nmr(O, residues, chains, experiment, out, prior=None, theta_values=None,
            average=None, method='log_lambda', min_neff=0.1, verbose=False):
    """Match experimental residues then call the NMR optimizer.

    O is amides x frames (the CLI transposes the forward model's frames x amides).
    The functions use frame x observable and unique Model_residue labels.
    We give each matched chain/residue pair a unique local identifier, so the
    scientific optimisation remains unchanged while multichain matching works.
    """
    from .nmr_optimizers import reweight_log_lambda, reweight_log_softmax

    if method not in ('log_lambda', 'log_softmax'):
        raise ValueError('method must be log_lambda or log_softmax')

    O, residues, chains = average_chains(O, residues, chains, average)
    if average:
        save_averaged_chains(out, O, residues, chains)
        print(f"Averaged chains {list(average)}; use Chain={int(average[0])} for the averaged group in the experiment CSV")
    exp = pd.read_csv(experiment)
    if 'Model_residue' in exp and 'Residue' not in exp:
        exp = exp.rename(columns={'Model_residue': 'Residue'})
    if not {'Residue', 'PF', 'PF_error'}.issubset(exp.columns):
        raise ValueError('NMR CSV requires Residue (or Model_residue), PF and PF_error')
    experiment_chains = _chain_for_experiments(exp, chains)
    print(f"NMR forward model: {len(residues)} amides, {O.shape[1]:,} frames, "
          f"{len(np.unique(chains))} chain(s)")
    print(f"NMR experimental data: {len(exp)} rows; "
          f"{len(set(zip(experiment_chains, exp['Residue'].astype(int))))} "
          f"unique chain/residue pairs")

    indices, matched = [], []
    no_open_state = []
    for c, (_, row) in zip(experiment_chains, exp.iterrows()):
        rid = int(row['Residue'])
        mask = (residues == rid) & (chains == c)
        hits = np.flatnonzero(mask)
        if len(hits) > 1:
            raise ValueError(f'Duplicate amides for chain {c}, residue {rid}')
        if not len(hits):
            continue
        idx = int(hits[0])
        if not (np.isfinite(row.PF) and row.PF > 0 and
                np.isfinite(row.PF_error) and row.PF_error > 0):
            continue
        if not np.any(O[idx] > 0):
            no_open_state.append({'Chain': int(chains[idx]), 'Residue': rid,
                                  'PF': float(row.PF), 'PF_error': float(row.PF_error)})
            continue
        indices.append(idx)
        matched.append({'Chain': int(chains[idx]), 'Residue': rid,
                        'PF': float(row.PF), 'PF_error': float(row.PF_error)})

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(no_open_state, columns=['Chain', 'Residue', 'PF', 'PF_error']).to_csv(
        out / 'nmr_amides_no_open_state.csv', index=False)
    if no_open_state:
        print('Experimental NMR amides excluded (no sampled open state): ' +
              ', '.join(f"chain {r['Chain']} residue {r['Residue']}" for r in no_open_state))
    print(f"NMR matching: {len(indices)} experimental measurements retained, "
          f"{len(no_open_state)} excluded for no sampled open state, "
          f"{len(exp) - len(indices) - len(no_open_state)} unmatched or invalid")
    if not indices:
        raise ValueError('No matched exchange-competent NMR amides')
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate NMR observations for the same chain/residue')

    matched = pd.DataFrame(matched)
    local_ids = np.arange(len(indices))
    original_exp = pd.DataFrame({'Model_residue': local_ids,
                                 'PF': matched.PF, 'PF_error': matched.PF_error})
    original_data = [local_ids, np.asarray(O[indices].T, dtype=np.float64)]
    w0 = weights0(O.shape[1], prior)
    explicit_thetas = theta_values is not None
    theta_values = (list(np.geomspace(1e5, 1e0, 30)) if theta_values is None
                    else list(theta_values))
    print(f"NMR optimisation: {method}; {len(theta_values)} {'explicit' if explicit_thetas else 'initial'} theta values; "
          f"minimum phi={min_neff:.3g}")
    if explicit_thetas:
        print('Explicit theta grid: fitting all supplied values; no early stopping or additional theta values.')
    else:
        print('Default theta grid: adaptive stopping and fine scans enabled.')
    print('Progress (theta, chi2, phi, iterations, convergence):')
    optimizer = {'log_lambda': reweight_log_lambda,
                 'log_softmax': reweight_log_softmax}[method]
    # Limit BLAS to one thread during optimisation to avoid oversubscription.
    with threadpool_limits(limits=1, user_api='blas'):
        labels, stats, weights, best, values, posterior_lnPF, prior_lnPF = optimizer(
            original_exp, original_data, Neff_threshold=min_neff,
            w0=w0, theta_grid=theta_values,
            explicit_theta_grid=explicit_thetas, verbose=True,
        )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Amide_column': np.arange(len(residues)),
                  'Chain': chains, 'Residue': residues}).to_csv(out / 'amide_mapping.csv', index=False)
    np.save(out / 'wopt_array.npy', np.asarray(weights))
    np.save(out / 'wopt.npy', best)
    
    np.savetxt(out / 'rew_lnPF.txt', posterior_lnPF)
    np.savetxt(out / 'prior_lnPF.txt', prior_lnPF)
    matched.to_csv(out / 'matched_nmr.csv', index=False)
    # Remove obsolete files from earlier runs in the same output directory.
    for name in ('reweighting_chi2_phi_theta.txt', 'stats_df.csv'):
        (out / name).unlink(missing_ok=True)
    summary = pd.DataFrame({'theta': values[2], 'phi': values[1],
                            'chi2r': values[0]})
    summary.to_csv(out / 'scan.csv', index=False)

    # The optimiser fits ln(PF). Report per-theta RMSE in the
    # same space, using the exact posterior weights it returned.
    exp_lnPF = np.log(original_exp['PF'].to_numpy(dtype=float))
    print(f'NMR scan complete: {len(summary)} retained theta values')
    for i, (theta, phi, chi2r) in enumerate(
            zip(summary['theta'], summary['phi'], summary['chi2r'])):
        predicted_open = np.average(original_data[1], axis=0, weights=weights[i])
        predicted_lnPF = -np.log(np.clip(predicted_open, 1e-300, None))
        rmse_lnPF = np.sqrt(np.mean((predicted_lnPF - exp_lnPF)**2))
        print(f'  theta={theta:.3e}  chi2={chi2r:.4f}  '
              f'RMSE(lnPF)={rmse_lnPF:.4f}  phi={phi:.4f}')

    # Elbow criterion: maximum distance from the end-to-end line
    # on the (phi, chi2) scan. Its index refers to the saved wopt_array.npy.
    if len(summary) >= 3:
        elbow_idx, elbow_phi, elbow_chi2 = find_elbow(
            summary['phi'].to_numpy(), summary['chi2r'].to_numpy()
        )
        elbow_idx = int(elbow_idx)
        elbow_theta = float(summary['theta'].iloc[elbow_idx])
        elbow_weights = np.asarray(weights[elbow_idx])
        np.save(out / 'wopt_elbow.npy', elbow_weights)
        # Same fitted amides, same ordering, and same -ln(mean opening)
        # convention as prior_lnPF.txt and rew_lnPF.txt.
        elbow_open = np.average(original_data[1], axis=0, weights=elbow_weights)
        elbow_lnPF = -np.log(np.clip(elbow_open, 1e-300, None))
        np.savetxt(out / 'rew_lnPF_elbow.txt', elbow_lnPF)
        elbow_report = (
            f'Elbow index (0-based): {elbow_idx}\n'
            f'Theta: {elbow_theta:.8g}\n'
            f'Chi2 (lnPF): {elbow_chi2:.8g}\n'
            f'Phi (entropy): {elbow_phi:.8g}\n'
        )
        (out / 'elbow_summary.txt').write_text(elbow_report)
        print('NMR elbow selection:\n' + elbow_report, end='')
    else:
        # Prevent stale elbow results when reusing an output directory.
        for name in ('wopt_elbow.npy', 'rew_lnPF_elbow.txt', 'elbow_summary.txt'):
            (out / name).unlink(missing_ok=True)
        print('NMR elbow not computed: fewer than 3 fitted theta values.')

    return summary

def read_fasta_sequences(path):
    """Read one FASTA entry per topology chain, preserving entry order."""
    sequences = []
    sequence = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if sequence:
                sequences.append(''.join(sequence))
                sequence = []
            elif sequences:
                raise ValueError('Empty FASTA sequence')
        else:
            sequence.append(line)
    if sequence:
        sequences.append(''.join(sequence))
    if not sequences:
        raise ValueError('No sequences in FASTA')
    return sequences


def load_ms_rates(path):
    """Load rates as rows=residue positions, columns=topology chain indices."""
    path = Path(path)
    if path.suffix == '.npy':
        rates = np.load(path)
    else:
        rates = np.loadtxt(path, delimiter=',' if path.suffix == '.csv' else None, ndmin=2)
    rates = np.asarray(rates, dtype=float)
    if rates.ndim == 1:
        rates = rates[:, None]
    if rates.ndim != 2 or rates.size == 0:
        raise ValueError('kint should have one column per chain')
    return rates


def fit_ms(O, residues, chains, experiment, rates, out, prior=None, theta_values=None,
           n_skip=2, average=None, sequence_start=1, sequences=None, min_neff=0.1):
    """Fit experimental HDX-MS uptake using the HDXReweighterFast theta loop.

    O: (amide, frame), with columns in the accompanying forward-model metadata.
    rates: intrinsic rates by full-sequence position, one column per topology chain.
    The full-sequence path preserves nonmodelled/proline positions as zero-opening
    columns.
    """
    import warnings
    if not np.isfinite(min_neff) or not 0 < min_neff <= 1:
        raise ValueError('min_neff must be finite and in the interval (0, 1]')
    # Sequence starts are PDB residue numbers, independently specified per chain.
    if np.isscalar(sequence_start):
        starts = [int(sequence_start)]  # single-chain convenience for Python API
    else:
        starts = [int(x) for x in sequence_start]
    O, residues, chains = average_chains(O, residues, chains, average)
    if average:
        save_averaged_chains(out, O, residues, chains)
        print(f"Averaged chains {list(average)}; use Chain={int(average[0])} for the averaged group in the experiment CSV")
    exp = pd.read_csv(experiment)
    required = ['Start', 'End', 'Sequence', 'Time (s)', 'Rel. Uptake', 'Rel. Uptake SD']
    if not all(col in exp for col in required):
        raise ValueError(f'MS CSV requires {required}')
    # Rate column c corresponds to topology chain index c, including for
    # homomers. Use full-sequence residue positions, not amide positions.
    kint_matrix = load_ms_rates(rates) if isinstance(rates, (str, Path)) else np.asarray(rates, float)
    if kint_matrix.ndim == 1:
        kint_matrix = kint_matrix[:, None]
    if kint_matrix.ndim != 2 or kint_matrix.size == 0:
        raise ValueError('kint must be a matrix: residues x topology chains')
    if len(starts) != kint_matrix.shape[1]:
        raise ValueError('Provide one --sequence-start per topology chain/kint column')
    # Both intrinsic-rate models return +inf at the first sequence position,
    # a sentinel for the N-terminus (not an exchangeable backbone amide).
    # Keep that value unchanged in kint; never include it in the fit.
    unexpected_inf = np.isinf(kint_matrix)
    unexpected_inf[0, :] = False
    if np.any(unexpected_inf) or np.any(np.isneginf(kint_matrix)) or np.any(kint_matrix[np.isfinite(kint_matrix)] < 0):
        raise ValueError('Intrinsic rates must be nonnegative; +inf is permitted only at the first residue of each chain; use NaN only for trailing padding')
    n_amides, n_frames = O.shape
    print(f"MS forward model: {n_amides} amides, {n_frames:,} frames, "
          f"{len(np.unique(chains))} chain(s)")
    exp_chains = _chain_for_experiments(exp, chains)
    if np.any(exp_chains < 0) or max(np.unique(exp_chains)) >= kint_matrix.shape[1]:
        raise ValueError('Missing kint column for an experimental chain (columns follow topology chain indices)')
    for c in np.unique(exp_chains):
        if c not in np.unique(chains):
            raise ValueError(f'Experimental Chain {c} not found in metadata')

    if sequences is not None:
        if len(sequences) != kint_matrix.shape[1]:
            raise ValueError('FASTA must have exactly one sequence per kint column / topology chain')
        for c, seq in enumerate(sequences):
            if len(seq) > kint_matrix.shape[0]:
                raise ValueError(f'Chain {c} sequence exceeds the kint table length')
            # The first rate may be +inf by design; all others must be finite.
            if np.isnan(kint_matrix[0, c]) or np.any(~np.isfinite(kint_matrix[1:len(seq), c])):
                raise ValueError(f'Chain {c} has missing intrinsic rates within its FASTA sequence')
            if np.any(np.isfinite(kint_matrix[len(seq):, c])):
                raise ValueError(f'Chain {c} has non-padding kint rows beyond its FASTA sequence')
    if n_skip < 0:
        raise ValueError('n_skip must be nonnegative')
    if average:
        ref = int(average[0])
        for c in average[1:]:
            c = int(c)
            if starts[ref] != starts[c]:
                raise ValueError('Averaged chains must have identical sequence-start values')
            if sequences is not None and sequences[ref] != sequences[c]:
                raise ValueError('Averaged chains must have identical FASTA sequences')
            if not np.array_equal(kint_matrix[:, ref], kint_matrix[:, c], equal_nan=True):
                raise ValueError('Averaged chains must have identical intrinsic exchange rates')
    # FASTA and experimental peptide sequences are mandatory to verify identity
    # and distinguish missing prolines from missing exchangeable amides.
    if sequences is None:
        raise ValueError('MS fitting requires --fasta (FASTA sequences), even with --kint')
    if exp['Sequence'].isna().any() or exp['Sequence'].astype(str).str.strip().eq('').any():
        raise ValueError('Every experimental MS row must contain a non-empty Sequence')
    # Validate all peptide sequences before fitting.
    for c, (_, row) in zip(exp_chains, exp.iterrows()):
        start, end = int(row['Start']), int(row['End'])
        if end < start:
            raise ValueError(f'Invalid peptide range chain {c}: {start}-{end}')
        offset = starts[c if len(starts) > 1 else 0]
        i, j = start - offset, end - offset + 1
        if i < 0 or j > kint_matrix.shape[0]:
            raise ValueError(f'Peptide chain {c}, {start}-{end} falls outside intrinsic-rate table')
        if sequences is not None:
            seq = sequences[c].upper()
            if j > len(seq):
                raise ValueError(f'Peptide chain {c}, {start}-{end} falls outside FASTA sequence')
            observed = str(row['Sequence']).strip().upper()
            expected = seq[i:j]
            if observed != expected:
                raise ValueError(f'Peptide sequence mismatch: chain {c}, {start}-{end}: '
                                 f'CSV={observed}, FASTA={expected}')

    # Build a residue axis for all measured peptides (per chain), including
    # prolines and residues not present in the forward model.
    chain_for_row = exp_chains
    if len(exp) == 0:
        raise ValueError('No experimental rows')
    axis = sorted({(int(c), residue) for c, (_, row) in zip(chain_for_row, exp.iterrows())
                   for residue in range(int(row['Start']), int(row['End']) + 1)})
    axis_index = {key: i for i, key in enumerate(axis)}
    full_O = np.zeros((len(axis), n_frames), dtype=np.float64)
    full_k = np.zeros(len(axis), dtype=float)
    metadata_lookup = {}
    for i, (c, r) in enumerate(zip(chains, residues)):
        key = (int(c), int(r))
        if key in metadata_lookup: raise ValueError(f'Duplicate amide for {key}')
        metadata_lookup[key] = i
    # Proline has no backbone amide H, so its absent O column is expected.
    # All other peptide-covered residues must exist in the forward-model metadata.
    missing_nonproline = []
    for j, (c, r) in enumerate(axis):
        amide = metadata_lookup.get((c, r))
        if amide is not None:
            full_O[j] = O[amide]  # Keep columns even if every opening probability is zero.
        else:
            pos = r - starts[c]
            is_proline = sequences[c][pos].upper() == 'P'
            if not is_proline and pos != 0:
                # The first sequence residue need not have a backbone amide H.
                missing_nonproline.append((c, r))
        k_idx = r - starts[c]
        if not (0 <= k_idx < kint_matrix.shape[0]):
            raise ValueError(f'Chain {c}, residue {r} outside kint table '
                             f'(sequence_start={starts[c]})')
        full_k[j] = kint_matrix[k_idx, c]
        if not np.isfinite(full_k[j]) and not (k_idx == 0 and np.isposinf(full_k[j])):
            raise ValueError(f'No intrinsic rate for chain {c}, residue {r} (NaN padding)')
    if missing_nonproline:
        missing_text = ', '.join(f'chain {c} residue {r}' for c, r in missing_nonproline)
        raise ValueError('Experimental peptide covers non-proline residues absent from '
                         f'forward-model amide metadata: {missing_text}. '
                         'Check residue numbering, topology chain indices and O.npy metadata.')
    peptides, times, F_exp, sigma, kept = [], [], [], [], []
    for c, (_, row) in zip(chain_for_row, exp.iterrows()):
        start, end = int(row['Start']), int(row['End'])
        if end < start: raise ValueError('End must be >= Start')
        pep = [axis_index[(int(c), resid)] for resid in range(start, end + 1)]
        time, uptake, err = float(row['Time (s)']), float(row['Rel. Uptake']), abs(float(row['Rel. Uptake SD']))
        if not (np.isfinite(time) and time >= 0 and np.isfinite(uptake)
                and np.isfinite(err) and err > 0):
            continue
        if np.any(np.isinf(full_k[np.asarray(pep[n_skip:], dtype=int)])):
            raise ValueError(f'Peptide {start}-{end} includes the N-terminal +inf rate after n_skip; '
                             'exclude that position by increasing --n-skip')
        if not np.any(full_k[np.asarray(pep[n_skip:], dtype=int)] > 0):
            warnings.warn(f'Ignoring peptide {start}-{end}: no exchanging amides after n_skip', UserWarning)
            continue
        peptides.append(pep); times.append(time); F_exp.append(uptake); sigma.append(err)
        kept.append(row)
    print(f"MS experimental data: {len(exp)} peptide/timepoint measurements; "
          f"{len(peptides)} retained for fitting, {len(exp) - len(peptides)} excluded")
    # Summarise matches without counting peptide-start or N-terminal positions.
    matched_count = sum((c, r) in metadata_lookup for c, r in axis)
    missing_prolines = [(c, r) for c, r in axis
                        if (c, r) not in metadata_lookup
                        and sequences[c][r - starts[c]].upper() == 'P']
    print(f"MS matching: {matched_count} backbone amides matched in O.npy; "
          f"{len(missing_prolines)} sequence-confirmed proline(s) without an amide; "
          "no missing required non-proline amides")
    if not peptides: raise ValueError('No usable HDX-MS observations')
    F_exp = np.clip(np.asarray(F_exp, float), 0, 1)
    sigma = np.asarray(sigma, float)
    times = np.asarray(times, float)
    w0 = weights0(n_frames, prior)
    explicit_thetas = theta_values is not None
    thetas = np.geomspace(1e7, 1e0, 30) if not explicit_thetas else np.asarray(theta_values, float)
    if thetas.ndim != 1 or np.any(~np.isfinite(thetas)) or np.any(thetas <= 0):
        raise ValueError('Theta values must be a positive, finite 1D array')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'Amide_column': np.arange(len(residues)),
                  'Chain': chains, 'Residue': residues}).to_csv(out / 'amide_mapping.csv', index=False)
    pd.DataFrame(kept).to_csv(out / 'matched_ms.csv', index=False)
    pd.DataFrame(axis, columns=['Chain', 'Residue']).to_csv(out / 'ms_residue_axis.csv', index=False)
    np.save(out / 'kint_residue_axis.npy', full_k)

    # fitting/statistics/output loop 
    reweighter = HDXReweighterFast(full_k, full_O, peptides, times, F_exp, sigma,
                                  theta=thetas[0] if len(thetas) else 1.0, n_skip=n_skip)
    wopt_array = [w0.copy()]
    opt_conv = [True]
    chi2_values = [reweighter.calc_chi2(w0)]
    rmses = [reweighter.calc_rmse(w=w0)]
    phis = [reweighter.calc_neff(w=w0, w0=w0)]
    pred_values = [reweighter.predict_with_weights(w0)]

    # Kish effective fraction: N_eff,Kish / N_frames; unlike entropy phi,
    # this is not necessarily 1 for a nonuniform prior.
    phi_kish = lambda w: float(np.sum(w)**2 / (n_frames * np.sum(np.asarray(w)**2)))
    kish_phis = [phi_kish(w0)]
    if len(thetas):
        print(f"MS optimisation: {len(thetas)} {'explicit' if explicit_thetas else 'initial'} theta values"
              + (" (all will be fitted)" if explicit_thetas
                 else f"; stop at phi_entropy < {min_neff:g} or chi2r < 0.001"))
    else:
        print('MS prior-only predictions: 0 theta values; optimisation skipped')
    print(f"{'theta':>12}  {'chi2r':>12}  {'rmse':>10}  "
      f"{'phi_entropy':>12}  {'phi_kish':>10}  {'success':>7}  "
      f"{'nit':>8}  {'nfev':>8}")
    print(f"{'prior':>12}  {chi2_values[0]:12.5f}  {rmses[0]:10.5f}  "
      f"{phis[0]:12.4f}  {kish_phis[0]:10.4f}  {'-':>7}  "
      f"{'-':>8}  {'-':>8}")

    # Write prior results immediately; save the growing scan after every theta.
    with np.errstate(divide='ignore'):
        np.savetxt(out / 'prior_lnPF.txt', -np.log(O @ w0))
    completed_thetas = []

    def save_results():
        np.save(out / 'opt_convergence.npy', np.asarray(opt_conv, bool))
        np.save(out / 'wopt_array.npy', np.asarray(wopt_array))
        np.save(out / 'pred_uptakes.npy', np.asarray(pred_values))
        pd.DataFrame({'theta': np.r_[np.nan, completed_thetas],
                      'chi2r': chi2_values, 'rmse': rmses, 'phi': phis,
                      'phi_kish': kish_phis}).to_csv(out / 'scan.csv', index=False)

    save_results()

    if len(thetas) == 0:
        # Predict uptake with the supplied prior (or uniform weights) only.
        # scan.csv has one prior row; no posterior or elbow exists.
        np.save(out / 'pred_prior.npy', np.asarray(pred_values[0]))
        np.save(out / 'pred.npy', np.asarray(pred_values[0]))
        for name in ('wopt.npy', 'rew_lnPF.txt', 'wopt_elbow.npy',
                     'pred_elbow.npy', 'rew_lnPF_elbow.txt', 'elbow_summary.txt',
                     'chi2r.txt', 'rmse.txt', 'phis.txt'):
            (out / name).unlink(missing_ok=True)
        print('MS prior-only results saved; no reweighting performed.')
        return pd.DataFrame({'theta': [np.nan], 'chi2r': chi2_values,
                             'rmse': rmses, 'phi': phis, 'phi_kish': kish_phis})

    def run_theta_list(theta_list):
        for theta in theta_list:
            reweighter = HDXReweighterFast(full_k, full_O, peptides, times, F_exp,
                                          sigma, theta=theta, n_skip=n_skip)
            # Limit BLAS to one thread during optimisation to avoid oversubscription.
            with threadpool_limits(limits=1, user_api='blas'):
                result = reweighter.fit(w0=w0)
            wopt = reweighter.w_opt.copy()
            opt_conv.append(bool(result.success))
            chi2_values.append(reweighter.calc_chi2(wopt))
            rmses.append(reweighter.calc_rmse(w=wopt))
            phis.append(reweighter.calc_neff(w=wopt, w0=w0))
            kish_phis.append(phi_kish(wopt))
            pred_values.append(reweighter.predict_with_weights(wopt))
            wopt_array.append(wopt)
            completed_thetas.append(float(theta))
            print(f'{theta:12.3e}  {chi2_values[-1]:12.5f}  {rmses[-1]:10.5f}  '
                f'{phis[-1]:12.4f}  {kish_phis[-1]:10.4f}  '
                f'{str(bool(result.success)):>7}  '
                f'{result.nit:8d}  {result.nfev:8d}', flush=True)
            save_results()

            # Like NMR: stop the *default* scan after reaching either criterion.
            # Explicit theta grids always evaluate every requested value.
            if not explicit_thetas and (phis[-1] < min_neff or chi2_values[-1] < 0.001):
                return True
        return False

    stopped = run_theta_list(thetas)
    if not explicit_thetas and not stopped:
        # Same fine-scan ranges and sizes as the NMR theta scanner.
        stopped = run_theta_list(np.geomspace(completed_thetas[-1], 1e-2, 15)[1:])
    if not explicit_thetas and not stopped:
        run_theta_list(np.geomspace(completed_thetas[-1], 1e-6, 15)[1:])

    # 'phi' is the entropy-based effective fraction (same name as in NMR).
    table = pd.DataFrame({'theta': np.r_[np.nan, completed_thetas],
                          'chi2r': chi2_values, 'rmse': rmses, 'phi': phis,
                          'phi_kish': kish_phis})
    # scan.csv has already been written after the prior and after every theta.
    # Remove obsolete diagnostic files when reusing an existing output directory.
    for name in ('chi2r.txt', 'rmse.txt', 'phis.txt'):
        (out / name).unlink(missing_ok=True)
    # The prior is row 0 of wopt_array and pred_uptakes. Scan rows 1 onward
    # correspond to completed theta fits, in exactly the same order.
    np.save(out / 'pred_prior.npy', np.asarray(pred_values[0]))

    # As for NMR: select the last evaluated theta retaining phi >= min_neff.
    # On the default descending grid this is the smallest acceptable theta.
    valid = [i for i in range(1, len(wopt_array)) if phis[i] >= min_neff]
    if not valid:
        for name in ('wopt.npy', 'pred.npy', 'rew_lnPF.txt'):
            (out / name).unlink(missing_ok=True)
        raise ValueError(f'No fitted MS theta has phi_entropy >= {min_neff:g}. '
                         'The full theta scan and prior results were saved.')
    best_idx = valid[-1]
    best_w = np.asarray(wopt_array[best_idx])
    np.save(out / 'wopt.npy', best_w)
    np.save(out / 'pred.npy', np.asarray(pred_values[best_idx]))
    with np.errstate(divide='ignore'):
        np.savetxt(out / 'rew_lnPF.txt', -np.log(O @ best_w))
    print(f'MS selected posterior: theta={completed_thetas[best_idx - 1]:.3e}, '
          f'chi2r={chi2_values[best_idx]:.5f}, phi_entropy={phis[best_idx]:.4f}')

    # Exclude the prior from the elbow scan. Its index is
    # 0-based among the fitted theta solutions (wopt_array index = elbow + 1).
    if len(completed_thetas) >= 3:
        elbow_idx, elbow_phi, elbow_chi2 = find_elbow(
            np.asarray(phis[1:]), np.asarray(chi2_values[1:]))
        elbow_idx = int(elbow_idx)
        elbow_w = np.asarray(wopt_array[elbow_idx + 1])
        np.save(out / 'wopt_elbow.npy', elbow_w)
        np.save(out / 'pred_elbow.npy', np.asarray(pred_values[elbow_idx + 1]))
        with np.errstate(divide='ignore'):
            np.savetxt(out / 'rew_lnPF_elbow.txt', -np.log(O @ elbow_w))
        report = (f'Elbow theta index (0-based, excludes prior): {elbow_idx}\n'
                  f'wopt_array index (includes prior): {elbow_idx + 1}\n'
                  f'Theta: {completed_thetas[elbow_idx]:.8g}\n'
                  f'Chi2r: {elbow_chi2:.8g}\n'
                  f'Phi (entropy): {elbow_phi:.8g}\n')
        (out / 'elbow_summary.txt').write_text(report)
        print('MS elbow selection:\n' + report, end='')
    else:
        for name in ('wopt_elbow.npy', 'pred_elbow.npy',
                     'rew_lnPF_elbow.txt', 'elbow_summary.txt'):
            (out / name).unlink(missing_ok=True)
        print('MS elbow not computed: fewer than 3 fitted theta values.')

    return table
