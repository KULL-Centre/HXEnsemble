"""Version 1.0.0 end-to-end and regression checks (small synthetic ensembles)."""

import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from hxensemble.reweight import (
    average_chains, fit_ms, fit_nmr, load_metadata, open_matrix,
)


def test_metadata_loads_integral_text_and_one_amide(tmp_path):
    (tmp_path / 'amideresidues.txt').write_text('27.0\n')
    residues, chains = load_metadata(tmp_path, 1)
    assert_array_equal(residues, [27])
    assert_array_equal(chains, [0])
    (tmp_path / 'amidechainidx.txt').write_text('0.0\n')
    assert_array_equal(load_metadata(tmp_path, 1)[1], [0])
    (tmp_path / 'amideresidues.txt').write_text('27.8\n')
    with pytest.raises(ValueError, match='integer values'):
        load_metadata(tmp_path, 1)


def test_open_matrix_is_frames_by_amides_on_disk(tmp_path):
    data = np.arange(12).reshape((4, 3)) / 10
    path = tmp_path / 'O.npy'
    np.save(path, data)
    assert_allclose(open_matrix(path), data.T)


def test_average_chains_rejects_unmatched_and_duplicated_metadata():
    O = np.ones((4, 3))
    with pytest.raises(ValueError, match='identical amide residue numbers'):
        average_chains(O, np.array([10, 11, 10, 12]),
                       np.array([0, 0, 1, 1]), [0, 1])
    with pytest.raises(ValueError, match='Duplicate residue numbers'):
        average_chains(O, np.array([10, 10, 10, 10]),
                       np.array([0, 0, 1, 1]), [0, 1])


@pytest.mark.parametrize('method', ['log_lambda', 'log_softmax'])
def test_nmr_predictions_weights_and_elbow(method, tmp_path):
    O = np.array([[.2, .3, .5, .8], [.4, .2, .3, .1], [.1, .2, .4, .6]])
    exp = tmp_path / 'nmr.csv'
    pd.DataFrame({'Residue': [10, 11, 12], 'PF': [3., 6., 5.],
                  'PF_error': [1.5, 3., 2.5]}).to_csv(exp, index=False)
    w0 = np.array([.1, .2, .3, .4])
    prior = tmp_path / 'prior.npy'
    np.save(prior, w0)
    out = tmp_path / method
    result = fit_nmr(O, np.array([10, 11, 12]), np.zeros(3, int),
                     exp, out, prior=prior, theta_values=[1e4, 1e2, 1.], method=method)
    assert result.theta.to_list() == [1e4, 1e2, 1.]
    weights = np.load(out / 'wopt_array.npy')
    assert weights.shape == (3, 4)
    assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    assert_allclose(np.loadtxt(out / 'prior_lnPF.txt'), -np.log(O @ w0))
    assert_allclose(np.loadtxt(out / 'rew_lnPF.txt'), -np.log(O @ np.load(out / 'wopt.npy')))
    assert_allclose(np.loadtxt(out / 'rew_lnPF_elbow.txt'),
                    -np.log(O @ np.load(out / 'wopt_elbow.npy')))
    assert (out / 'elbow_summary.txt').exists()
    assert pd.read_csv(out / 'scan.csv').shape == (3, 3)
    assert not (out / 'stats_df.csv').exists()


def test_nmr_excludes_never_open_amide(tmp_path):
    # At least two retained amides are needed by the original optimiser's
    # Pearson correlation diagnostics.
    O = np.array([[.1, .2, .3], [0., 0., 0.], [.3, .4, .5]])
    exp = tmp_path / 'nmr.csv'
    pd.DataFrame({'Residue': [10, 11, 12], 'PF': [5., 50., 3.],
                  'PF_error': [2.5, 25., 1.5]}).to_csv(exp, index=False)
    out = tmp_path / 'nmr'
    fit_nmr(O, np.array([10, 11, 12]), np.zeros(3, int),
            exp, out, theta_values=[1e3])
    assert pd.read_csv(out / 'matched_nmr.csv').Residue.to_list() == [10, 12]
    assert pd.read_csv(out / 'nmr_amides_no_open_state.csv').Residue.to_list() == [11]
    assert np.loadtxt(out / 'rew_lnPF.txt', ndmin=1).shape == (2,)


def test_nmr_one_retained_amide_diagnostic_limitation(tmp_path):
    # Known limitation: the legacy NMR diagnostic calls scipy.stats.pearsonr
    # unconditionally, which requires at least two matched observables.
    # PF=4 avoids an exact prior fit: PF=5 gives chi2 approximately zero,
    # which can instead trigger the legacy chi2>0 solution-retention filter.
    O = np.array([[.1, .2, .3], [0., 0., 0.]])
    exp = tmp_path / 'nmr.csv'
    pd.DataFrame({'Residue': [10, 11], 'PF': [4., 50.],
                  'PF_error': [2., 25.]}).to_csv(exp, index=False)
    with pytest.raises(ValueError, match='length at least 2'):
        fit_nmr(O, np.array([10, 11]), np.zeros(2, int),
                exp, tmp_path / 'out', theta_values=[1e3])


def ms_fixture(tmp_path):
    # FASTA starts at PDB residue 27; position 29 is a proline and has no O column.
    seq = 'AAPAAA'
    O = np.array([
        [.2, .3, .4, .5],   # 27: N-terminal position
        [.3, .4, .5, .6],   # 28
        [.4, .1, .6, .2],   # 30
        [.3, .4, .7, .5],   # 31
        [.6, .4, .5, .3],   # 32
    ])
    residues = np.array([27, 28, 30, 31, 32])
    chains = np.zeros(len(residues), int)
    rates = np.array([[np.inf], [1.], [0.], [.8], [1.6], [2.4]])
    exp = tmp_path / 'ms.csv'
    pd.DataFrame({
        'Start': [27, 27, 28], 'End': [32, 32, 32],
        'Sequence': [seq, seq, seq[1:]],
        'Time (s)': [1., 5., 2.],
        'Rel. Uptake': [.3, .5, .4],
        'Rel. Uptake SD': [.08, .08, .08],
    }).to_csv(exp, index=False)
    return O, residues, chains, rates, exp, seq


def test_ms_prior_only_offset_proline_and_zero_opening(tmp_path):
    O, residues, chains, rates, exp, seq = ms_fixture(tmp_path)
    O[4] = 0  # A non-proline amide present but never sampled open remains.
    prior = tmp_path / 'prior.npy'
    w0 = np.array([.1, .2, .3, .4])
    np.save(prior, w0)
    out = tmp_path / 'prior_only'
    result = fit_ms(O, residues, chains, exp, rates, out,
                    prior=prior, theta_values=[], n_skip=2,
                    sequence_start=27, sequences=[seq])
    assert len(result) == 1 and np.isnan(result.theta.iloc[0])
    assert_allclose(np.load(out / 'wopt_array.npy'), w0[None, :])
    assert np.load(out / 'pred_uptakes.npy').shape == (1, 3)
    assert_allclose(np.load(out / 'pred_prior.npy'), np.load(out / 'pred.npy'))
    assert (out / 'rew_lnPF.txt').exists() is False
    assert not (out / 'wopt.npy').exists()
    assert not (out / 'elbow_summary.txt').exists()
    assert_allclose(np.load(out / 'kint_residue_axis.npy'), rates[:, 0])
    assert pd.read_csv(out / 'ms_residue_axis.csv').Residue.to_list() == list(range(27, 33))
    assert pd.read_csv(out / 'matched_ms.csv').shape[0] == 3
    # Independent reference: first two peptide residues are skipped;
    # kint=0 for proline, and O=0 for the nonopening residue at 32.
    o_mean = O @ w0
    expected = []
    for first, end, t in zip([27, 27, 28], [32, 32, 32], [1., 5., 2.]):
        included = np.arange(first + 2, end + 1)  # n_skip = 2
        indices = included - 27
        k = rates[indices, 0]
        opening = np.zeros_like(k)
        for n, res in enumerate(included):
            if res in residues:
                opening[n] = o_mean[np.where(residues == res)[0][0]]
        nonzero = k > 0
        expected.append(1 - np.mean(np.exp(-k[nonzero] * t * opening[nonzero])))
    assert_allclose(np.load(out / 'pred_prior.npy'), expected, atol=1e-14)
    assert_allclose(result.phi.iloc[0], 1.)
    assert result.phi_kish.iloc[0] < 1.


def test_ms_posterior_predictions_elbow_and_prior_row(tmp_path):
    O, residues, chains, rates, exp, seq = ms_fixture(tmp_path)
    out = tmp_path / 'msfit'
    result = fit_ms(O, residues, chains, exp, rates, out, theta_values=[1e4, 100., 1.],
                    n_skip=2, sequence_start=27, sequences=[seq])
    assert len(result) == 4
    assert np.isnan(result.theta.iloc[0])
    assert_allclose(result.theta.iloc[1:].to_numpy(), [1e4, 100., 1.])
    weights = np.load(out / 'wopt_array.npy')
    pred = np.load(out / 'pred_uptakes.npy')
    assert weights.shape == (4, 4)
    assert pred.shape == (4, 3)
    assert_allclose(weights.sum(axis=1), 1)
    assert_allclose(weights[0], .25)
    assert_allclose(pred[0], np.load(out / 'pred_prior.npy'))
    assert_allclose(np.load(out / 'pred.npy'), pred[-1], atol=1e-12)
    assert (out / 'wopt_elbow.npy').exists()
    assert (out / 'pred_elbow.npy').exists()
    assert (out / 'elbow_summary.txt').exists()
    assert np.load(out / 'opt_convergence.npy').shape == (4,)
    assert_allclose(pd.read_csv(out / 'scan.csv').chi2r.to_numpy(), result.chi2r.to_numpy())
    assert not (out / 'chi2r.txt').exists()


def test_ms_cli_prior_only_matches_python_api(tmp_path, monkeypatch):
    from hxensemble.cli import ms_main

    O, residues, chains, rates, exp, seq = ms_fixture(tmp_path)
    np.save(tmp_path / 'O.npy', O.T)
    np.savetxt(tmp_path / 'amideresidues.txt', residues, fmt='%d')
    np.savetxt(tmp_path / 'amidechainidx.txt', chains, fmt='%d')
    np.save(tmp_path / 'rates.npy', rates)
    (tmp_path / 'sequence.fasta').write_text(f'>protein\n{seq}\n')
    out = tmp_path / 'cli_prior'
    monkeypatch.setattr(sys, 'argv', [
        'hxensemble-ms', '--O', str(tmp_path / 'O.npy'),
        '--metadata', str(tmp_path), '--experiment', str(exp),
        '--fasta', str(tmp_path / 'sequence.fasta'), '--kint', str(tmp_path / 'rates.npy'),
        '--sequence-start', '27', '--n-theta', '0', '--out', str(out),
    ])
    ms_main()
    assert np.load(out / 'wopt_array.npy').shape == (1, 4)
    assert_allclose(np.load(out / 'pred_uptakes.npy')[0], np.load(out / 'pred.npy'))
    assert not (out / 'wopt.npy').exists()
