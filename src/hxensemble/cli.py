import argparse
import numpy as np
from .forward import run_forward
from .reweight import open_matrix, load_metadata, fit_nmr, fit_ms


def forward_main():
    p=argparse.ArgumentParser(description='Calculate M11/M12/M14 open-state matrices from trajectories')
    p.add_argument('--directory', help='Directory containing one PDB and one or more XTC files')
    p.add_argument('--topology', help='Topology PDB (optional with --directory)')
    p.add_argument('--trajectory', nargs='+', help='XTC files (optional with --directory)')
    p.add_argument('--filter-frames', action='store_true', help='Skip frames containing overlapping atoms')
    p.add_argument('--save-components', action='store_true', help='Also save open_sasa_hbonds, rho_front and DH arrays')
    p.add_argument('--out', required=True)
    p.add_argument('--model', choices=('M11', 'M12', 'M14'), default='M14')
    p.add_argument('--chunk', type=int, default=1000)
    p.add_argument('--n-cpus', type=int, default=1, help='Number of CPU processes for forward-model chunks (default: 1)')
    p.add_argument('--DH_scaling', type=float, default=0.75, help='Charge scaling for DH calculations (default: 0.75)')
    p.add_argument('--user-charges', help='Optional atomic charges (.txt/.npy), in topology atom order')
    p.add_argument('--temperature', type=float, default=298.0, help='DH temperature (K)')
    p.add_argument('--ionic', type=float, default=0.15)
    p.add_argument('--stride', type=int, default=1)
    a = p.parse_args()
    if a.n_cpus < 1:
        p.error('--n-cpus must be a positive integer')
    if not a.directory and (not a.topology or not a.trajectory):
        p.error('Supply --directory OR both --topology and --trajectory')
    result = run_forward(a.topology, a.trajectory, a.out, a.model, a.chunk,
                         a.DH_scaling, a.user_charges, a.temperature, a.ionic, a.stride,
                         directory=a.directory, filter_overlaps=a.filter_frames,
                         save_components=a.save_components, n_cpus=a.n_cpus)
    print('Saved:', result)


def _shared(p):
    p.add_argument('--O',required=True,help='Frames x amides O.npy from forward model')
    p.add_argument('--metadata',required=True,help='Directory containing amideresidues.txt and amidechainidx.txt')
    p.add_argument('--experiment',required=True); p.add_argument('--out',required=True)
    p.add_argument('--prior',help='Optional prior frame weights, npy or txt')
    p.add_argument('--theta-min',type=float,default=1.0); p.add_argument('--theta-max',type=float,default=1e6)
    p.add_argument('--n-theta',type=int,default=50)
    p.add_argument('--average-chains',type=int,nargs='+',metavar='CHAIN',
                   help='Equivalent chain indices to average per frame (e.g. 0 1)')
    p.add_argument('--theta-values',type=float,nargs='+',metavar='THETA',
                   help='Explicit theta values, e.g. --theta-values 1e7 1e6 1e5 1e4; overrides theta-min/max/n-theta')


def _data(a):
    O=open_matrix(a.O); residues,chains=load_metadata(a.metadata,O.shape[0])
    if a.theta_values is not None:
        theta = np.asarray(a.theta_values, dtype=float)
    elif a.theta_min is None and a.theta_max is None and a.n_theta is None:
        # No NMR theta arguments: let fit_nmr use its original default grid
        # (30 values from 1e5 to 1), with adaptive stopping/fine scans.
        theta = None
    else:
        # An explicitly requested range is also an explicit theta grid.
        # Fill any omitted range arguments with the NMR defaults.
        theta = np.geomspace(
            a.theta_max if a.theta_max is not None else 1e5,
            a.theta_min if a.theta_min is not None else 1.0,
            a.n_theta if a.n_theta is not None else 30,
        )
    if theta is not None and (len(theta) == 0 or not np.all(np.isfinite(theta)) or np.any(theta <= 0)):
        raise ValueError('Theta values must be finite and strictly positive')
    return O,residues,chains,theta


def nmr_main():
    p=argparse.ArgumentParser(description='Reweight supplied O.npy against residue-level PF measurements')
    _shared(p)
    p.add_argument('--method', choices=('log_lambda', 'log_softmax'), default='log_lambda',
                   help='Original NMR optimizers: per-observable lambda or per-frame log-softmax')
    p.add_argument('--min-neff', type=float, default=0.1,
                   help='Minimum entropy effective fraction (original default: 0.1)')
    p.add_argument('--verbose', action='store_true')
    # Distinguish omitted NMR theta options from user-supplied grids.
    # For HDX-MS, _shared() retains its existing defaults.
    p.set_defaults(theta_min=None, theta_max=None, n_theta=None)
    a = p.parse_args()
    O, residues, chains, theta = _data(a)
    print(fit_nmr(O,residues,chains,a.experiment,a.out,a.prior,theta,a.average_chains,
                  a.method,a.min_neff,a.verbose).tail(3).to_string(index=False))


def ms_main():
    p=argparse.ArgumentParser(description='Fast HDX-MS reweighting of supplied O.npy')
    _shared(p)
    p.add_argument('--kint', help='Optional rates file: one column per topology chain (s^-1)')
    p.add_argument('--fasta', required=True, help='Required: one FASTA sequence per topology chain for peptide and amide validation')
    p.add_argument('--kint-model', choices=('PDLA', '3ALA'), default='3ALA',
                   help='PDLA = Nguyen2018; 3ALA = Stofella2025 (default)')
    p.add_argument('--temperature', type=float, help='Required with --fasta: temperature (K)')
    p.add_argument('--pHread', type=float, help='Required unless --kint supplied: measured pH meter reading')
    p.add_argument('--fraction-d2O', type=float, help='Required unless --kint supplied: D2O fraction between 0 and 1')
    p.add_argument('--min-neff', type=float, default=0.1,
                   help='Minimum entropy effective fraction for selecting the MS posterior and default theta stopping (default: 0.1)')
    p.add_argument('--n-skip',type=int,default=2,help='First peptide residues excluded from uptake')
    p.add_argument('--sequence-start',type=int,nargs='+',metavar='START',help='First residue number for each topology chain, in chain order (default: 1 for every chain)')
    p.set_defaults(theta_min=None, theta_max=None, n_theta=None)
    a = p.parse_args()
    if not np.isfinite(a.min_neff) or not 0 < a.min_neff <= 1:
        p.error('--min-neff must be finite and between 0 (exclusive) and 1 (inclusive)')
    if a.n_theta == 0 and (a.theta_values is not None or a.theta_min is not None or a.theta_max is not None):
        p.error('--n-theta 0 cannot be combined with other theta options')

    # Only explicitly supplied MS theta options disable the adaptive scan.
    # An explicitly supplied range uses the original MS defaults for omitted parts.
    if a.theta_values is None and any(v is not None for v in
                                      (a.theta_min, a.theta_max, a.n_theta)):
        if a.theta_min is None: a.theta_min = 1.0
        if a.theta_max is None: a.theta_max = 1e7
        if a.n_theta is None: a.n_theta = 30

    if not a.kint and (a.temperature is None or a.pHread is None or a.fraction_d2O is None):
        p.error('Calculating kint requires --temperature (K), --pHread, and --fraction-d2O; no defaults are assumed')
    if a.temperature is not None and (not np.isfinite(a.temperature) or a.temperature <= 0):
        p.error('--temperature must be a positive, finite value in Kelvin')
    if a.pHread is not None and (not np.isfinite(a.pHread) or not 0 <= a.pHread <= 14):
        p.error('--pHread must be finite and between 0 and 14')
    if a.fraction_d2O is not None and (not np.isfinite(a.fraction_d2O) or not 0 <= a.fraction_d2O <= 1):
        p.error('--fraction-d2O must be a finite fraction between 0 and 1')
    pD = None if a.pHread is None or a.fraction_d2O is None else a.pHread + 0.4 * a.fraction_d2O
    if pD is not None and pD > 14:
        p.error('Calculated pD must not exceed 14')
    if a.n_theta == 0:
        # Prior-only MS predictions: _data() rejects empty theta grids for NMR.
        O = open_matrix(a.O)
        residues, chains = load_metadata(a.metadata, O.shape[0])
        theta = np.array([], dtype=float)
    else:
        O, residues, chains, theta = _data(a)
    from .reweight import load_ms_rates, read_fasta_sequences
    sequences = None
    if a.fasta:
        from pathlib import Path
        from .kint import Nguyen2018, Stofella2025
        sequences = read_fasta_sequences(a.fasta)
        if len(sequences) <= int(max(chains)):
            p.error('FASTA must contain a sequence for each topology chain, in chain order')
    if a.kint:
        kint_file = load_ms_rates(a.kint)
    else:
        model = Nguyen2018 if a.kint_model == 'PDLA' else Stofella2025
        print(f'Intrinsic rates: T={a.temperature:g} K, pHread={a.pHread:g}, '
              f'D2O fraction={a.fraction_d2O:g}, calculated pD={pD:.4f}')
        kint_columns = [model(sequence=seq, temperature=a.temperature, pH=pD,
                              reference='poly', exchange_type='HD', hd_percentage=100.)
                        for seq in sequences]
        # A longer chain is padded only at its end, with NaNs. Padding is never
        # used as an exchange rate; measured residues must have actual rates.
        kint_file = np.full((max(map(len, kint_columns)), len(kint_columns)), np.nan)
        for c, values in enumerate(kint_columns):
            kint_file[:len(values), c] = values
        from pathlib import Path
        outdir = Path(a.out)
        outdir.mkdir(parents=True, exist_ok=True)
        np.savetxt(outdir / 'kint_chains.txt', kint_file)
    if sequences is not None and len(sequences) != kint_file.shape[1]:
        p.error('FASTA entry count must equal kint column count')
    starts = a.sequence_start if a.sequence_start is not None else [1] * kint_file.shape[1]
    if len(starts) != kint_file.shape[1]:
        p.error('--sequence-start requires exactly one integer per chain')
    if kint_file.shape[1] <= int(max(chains)):
        p.error('kint must have one column per topology chain, in chain order')
    print(fit_ms(O,residues,chains,a.experiment,kint_file,a.out,a.prior,theta,a.n_skip,a.average_chains,starts,sequences,min_neff=a.min_neff).tail(3).to_string(index=False))
