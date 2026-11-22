import numpy as np
import pandas as pd
from hxensemble.reweight import fit_nmr,fit_ms

def test_nmr(tmp_path):
    O = np.array([[.1,.3,.5],[.02,.05,.1]])
    ex=tmp_path/'nmr.csv'; pd.DataFrame({'Residue':[10,11], 'PF':[5.,20.],'PF_error':[2.5,10.]}).to_csv(ex,index=False)
    res=fit_nmr(O,np.array([10,11]),np.zeros(2,int),ex,tmp_path/'nmr',theta_values=[1e3,1])
    assert len(res)==2  # an explicit NMR theta grid is not extended
    assert np.allclose(np.load(tmp_path/'nmr'/'wopt_array.npy').sum(axis=1),1)

def test_ms(tmp_path):
    O=np.array([[.1,.3,.5],[.1,.5,.7],[.2,.4,.6],[.2,.5,.8]])
    ex=tmp_path/'ms.csv'; pd.DataFrame({'Start':[7,7],'End':[10,10],'Sequence':['AAAA','AAAA'],'Time (s)':[.1,1.], 'Rel. Uptake':[.2,.5],'Rel. Uptake SD':[.1,.1]}).to_csv(ex,index=False)
    rates=tmp_path/'rates.txt'; np.savetxt(rates,np.array([0.,0.,0.,0.,0.,0.,1.,2.,3.,4.]))
    res=fit_ms(O,np.array([7,8,9,10]),np.zeros(4,int),ex,rates,tmp_path/'ms',n_skip=1,theta_values=[1e3,1],sequences=['AAAAAAAAAA'])
    assert len(res)==3
    assert np.load(tmp_path/'ms'/'wopt_array.npy').shape==(3,3)


def test_both_original_nmr_methods(tmp_path):
    O = np.array([[.1,.3,.5,.7],[.02,.05,.1,.2]])
    exp = tmp_path/'experiment.csv'
    pd.DataFrame({'Model_residue':[10,11], 'PF':[5.,20.],
                  'PF_error':[2.5,10.]}).to_csv(exp,index=False)
    for method in ('log_lambda', 'log_softmax'):
        path = tmp_path/method
        result=fit_nmr(O,np.array([10,11]),np.zeros(2,int),exp,path,
                       theta_values=[1e4,1e3],method=method)
        assert len(result) >= 1
        weights=np.load(path/'wopt_array.npy')
        assert np.allclose(weights.sum(axis=1),1)
        assert (path/'rew_lnPF.txt').exists()
        assert (path/'scan.csv').exists()
        assert not (path/'reweighting_chi2_phi_theta.txt').exists()


def test_ms_original_outputs_and_full_sequence_matching(tmp_path):
    # All exchangeable residues are represented in forward-model metadata.
    O = np.array([[.5, .6, .7], [.2, .3, .4], [.0, .0, .0], [.4, .5, .6]])
    residues = np.array([1, 2, 3, 4])
    exp = tmp_path/'ms_full.csv'
    pd.DataFrame({'Start':[1,1], 'End':[4,4], 'Sequence':['AAAA','AAAA'], 'Time (s)':[.2,2.],
                  'Rel. Uptake':[.15,.3], 'Rel. Uptake SD':[.05,.05]}).to_csv(exp,index=False)
    rates = tmp_path/'rates_full.txt'
    np.savetxt(rates,[1.,2.,3.,4.])
    out=tmp_path/'out'
    result=fit_ms(O,residues,np.zeros(4,int),exp,rates,out,theta_values=[1000.,100.],n_skip=2,sequences=['AAAA'])
    assert len(result)==3
    for filename in ['opt_convergence.npy','wopt_array.npy','scan.csv',
                     'pred_uptakes.npy','ms_residue_axis.csv',
                     'pred_prior.npy','pred.npy','wopt.npy','amide_mapping.csv']:
        assert (out/filename).exists()
    assert np.load(out/'wopt_array.npy').shape == (3,3)
    assert not (out/'chi2r.txt').exists()
    assert not (out/'rmse.txt').exists()
    assert not (out/'phis.txt').exists()
    assert np.allclose(np.load(out/'kint_residue_axis.npy'), [1,2,3,4])
    from hxensemble.fastbme import HDXReweighterFast
    r=HDXReweighterFast(np.array([1.,2.,3.,4.]), np.array([[.5,.6,.7],[.2,.3,.4],
        [0.,0.,0.],[.4,.5,.6]]), [[0,1,2,3]]*2,[.2,2.], [.15,.3],[.05,.05],n_skip=2)
    assert np.allclose(np.load(out/'pred_uptakes.npy')[0],r.predict_with_weights(np.ones(3)/3))


def test_average_chains_and_chain_identity():
    from hxensemble.reweight import average_chains
    O = np.array([[.1, .2], [.2, .4], [.3, .6], [.4, .8], [.9, .8]])
    residues = np.array([10, 11, 10, 11, 20])
    chains = np.array([0, 0, 1, 1, 2])
    av, r, c = average_chains(O, residues, chains, [0, 1])
    assert np.array_equal(r, [10, 11, 20])
    assert np.array_equal(c, [0, 0, 2])
    assert np.allclose(av, [[.2, .4], [.3, .6], [.9, .8]])


def test_heteromer_nmr_matches_same_residue_number_in_both_chains(tmp_path):
    O = np.array([[.3, .5, .7], [.1, .2, .4]])
    exp = tmp_path / 'heteromer_nmr.csv'
    pd.DataFrame({'Chain': [0, 1], 'Residue': [10, 10],
                  'PF': [3., 6.], 'PF_error': [1.5, 3.]}).to_csv(exp, index=False)
    fit_nmr(O, np.array([10, 10]), np.array([0, 1]), exp,
            tmp_path / 'nmr_multi', theta_values=[1e3, 1e2])
    matched = pd.read_csv(tmp_path / 'nmr_multi' / 'matched_nmr.csv')
    assert list(matched.Chain) == [0, 1]
    assert len(matched) == 2


def test_heteromer_nmr_requires_explicit_chains(tmp_path):
    exp = tmp_path / 'nmr.csv'
    pd.DataFrame({'Residue': [10], 'PF': [3.], 'PF_error': [1.5]}).to_csv(exp, index=False)
    import pytest
    with pytest.raises(ValueError, match='Chain column'):
        fit_nmr(np.array([[.1, .2], [.3, .4]]), np.array([10, 10]),
                np.array([0, 1]), exp, tmp_path / 'invalid', theta_values=[1e3])


def test_heteromer_ms_uses_chain_specific_rates(tmp_path):
    O = np.array([[.2, .4, .5], [.4, .5, .6], [.1, .2, .3], [.3, .4, .5]])
    exp = tmp_path / 'ms.csv'
    pd.DataFrame({'Chain': [0, 1], 'Start': [1, 1], 'End': [2, 2], 'Sequence':['AA','AA'],
                  'Time (s)': [1., 1.], 'Rel. Uptake': [.2, .3],
                  'Rel. Uptake SD': [.1, .1]}).to_csv(exp, index=False)
    rates = tmp_path / 'rates.txt'
    np.savetxt(rates, [[1., 3.], [2., 4.]])
    fit_ms(O, np.array([1, 2, 1, 2]), np.array([0, 0, 1, 1]),
           exp, rates, tmp_path / 'ms_multi', theta_values=[1e3], n_skip=0, sequence_start=[1,1],sequences=['AA','AA'])
    assert np.allclose(np.load(tmp_path / 'ms_multi' / 'kint_residue_axis.npy'), [1, 2, 3, 4])
    assert np.load(tmp_path / 'ms_multi' / 'pred_uptakes.npy').shape == (2, 2)


def test_nmr_saves_averaged_forward_model(tmp_path):
    O = np.array([[.1,.2,.3], [.2,.3,.4], [.3,.4,.5], [.4,.5,.6]])
    residues = np.array([10,11,10,11])
    chains = np.array([0,0,1,1])
    exp = tmp_path/'nmr_avg.csv'
    pd.DataFrame({'Chain':[0,0], 'Residue':[10,11], 'PF':[5.,4.],
                  'PF_error':[2.,2.]}).to_csv(exp,index=False)
    out = tmp_path/'nmr_avg'
    fit_nmr(O,residues,chains,exp,out,theta_values=[1e3],average=[0,1])
    assert np.allclose(np.load(out/'O_averaged.npy'), np.array([[.2,.3],[.3,.4],[.4,.5]]))
    assert np.array_equal(np.loadtxt(out/'amideresidues_averaged.txt',dtype=int), [10,11])
    assert np.array_equal(np.loadtxt(out/'amidechainidx_averaged.txt',dtype=int), [0,0])


def test_ms_saves_averaged_forward_model(tmp_path):
    O = np.array([[.2,.3,.4], [.3,.4,.5], [.4,.5,.6], [.5,.6,.7]])
    residues = np.array([1,2,1,2])
    chains = np.array([0,0,1,1])
    exp = tmp_path/'ms_avg.csv'
    pd.DataFrame({'Chain':[0], 'Start':[1], 'End':[2], 'Sequence':['AA'], 'Time (s)':[1.],
                  'Rel. Uptake':[.4], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    rates=tmp_path/'rates.txt'; np.savetxt(rates,[[1.,1.],[2.,2.]])
    out=tmp_path/'ms_avg'
    fit_ms(O,residues,chains,exp,rates,out,theta_values=[1e3],average=[0,1],n_skip=0,sequence_start=[1,1],sequences=['AA','AA'])
    assert np.allclose(np.load(out/'O_averaged.npy'), np.array([[.3,.4],[.4,.5],[.5,.6]]))
    assert np.array_equal(np.loadtxt(out/'amideresidues_averaged.txt',dtype=int), [1,2])
    assert np.array_equal(np.loadtxt(out/'amidechainidx_averaged.txt',dtype=int), [0,0])


def test_multisequence_fasta_order(tmp_path):
    from hxensemble.reweight import read_fasta_sequences
    fasta = tmp_path/'seq.fasta'
    fasta.write_text('>chain0\nAAAA\nGG\n>chain1\nPEPTIDE\n')
    assert read_fasta_sequences(fasta) == ['AAAAGG', 'PEPTIDE']


def test_kint_columns_different_lengths(tmp_path):
    from hxensemble.reweight import load_ms_rates
    rates = tmp_path/'rates.txt'
    np.savetxt(rates, [[1, 3], [2, 4], [np.nan, 5]])
    assert load_ms_rates(rates).shape == (3,2)
    assert np.isnan(load_ms_rates(rates)[2,0])


def test_ms_multichain_sequence_validation_different_starts(tmp_path):
    O = np.array([[.2, .3], [.3, .4], [.4, .5], [.5, .6]])
    res = np.array([101, 102, 501, 502])
    chains = np.array([0, 0, 1, 1])
    exp = tmp_path / 'peptides.csv'
    pd.DataFrame({'Chain':[0, 1], 'Start':[101, 501], 'End':[102, 502],
                  'Sequence':['AC', 'DE'], 'Time (s)':[1., 1.],
                  'Rel. Uptake':[.3, .4], 'Rel. Uptake SD':[.1, .1]}).to_csv(exp, index=False)
    rates = np.array([[1., 2.], [3., 4.]])
    out = tmp_path / 'fitted'
    fit_ms(O, res, chains, exp, rates, out, theta_values=[1e3], n_skip=0,
           sequence_start=[101, 501], sequences=['AC', 'DE'])
    assert pd.read_csv(out / 'matched_ms.csv').Sequence.tolist() == ['AC', 'DE']
    assert np.allclose(np.load(out / 'kint_residue_axis.npy'), [1., 3., 2., 4.])


def test_ms_rejects_wrong_peptide_sequence(tmp_path):
    import pytest
    exp = tmp_path / 'exp.csv'
    pd.DataFrame({'Start':[101], 'End':[102], 'Sequence':['AD'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp, index=False)
    with pytest.raises(ValueError, match='Peptide sequence mismatch'):
        fit_ms(np.array([[.2,.3],[.3,.4]]), np.array([101,102]), np.array([0,0]),
               exp, np.array([[1.],[2.]]), tmp_path/'out', theta_values=[1e3],
               sequence_start=[101], sequences=['AC'], n_skip=0)


def test_ms_rejects_sequences_without_fasta(tmp_path):
    import pytest
    exp = tmp_path / 'exp.csv'
    pd.DataFrame({'Start':[1], 'End':[2], 'Sequence':['AC'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='requires --fasta'):
        fit_ms(np.array([[.2,.3],[.3,.4]]),np.array([1,2]),np.array([0,0]),
               exp,np.array([[1.],[2.]]),tmp_path/'out',theta_values=[1e3],n_skip=0)


def test_ms_average_rejects_different_rates(tmp_path):
    import pytest
    exp = tmp_path / 'exp.csv'
    pd.DataFrame({'Chain':[0], 'Start':[1], 'End':[2], 'Sequence':['AA'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='identical intrinsic exchange rates'):
        fit_ms(np.ones((4,2))*.2,np.array([1,2,1,2]),np.array([0,0,1,1]),
               exp,np.array([[1.,1.],[2.,3.]]),tmp_path/'out',
               theta_values=[1e3],average=[0,1],n_skip=0,sequence_start=[1,1],sequences=['AA','AA'])


def test_ms_missing_proline_is_silent(tmp_path):
    import warnings
    exp = tmp_path / 'ms.csv'
    pd.DataFrame({'Start':[1], 'End':[3], 'Sequence':['APA'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3],
                  'Rel. Uptake SD':[.1]}).to_csv(exp, index=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit_ms(np.array([[.3,.4],[.5,.6]]), np.array([1,3]), np.array([0,0]),
               exp, np.array([[1.],[0.],[2.]]), tmp_path/'fit',
               theta_values=[1e3], n_skip=0, sequences=['APA'])
    assert not any('no forward-model amide' in str(w.message) for w in caught)
    assert not (tmp_path/'fit'/'missing_amides.csv').exists()
    assert np.allclose(np.load(tmp_path/'fit'/'kint_residue_axis.npy'), [1,0,2])


def test_ms_missing_nonproline_raises(tmp_path):
    import pytest
    exp = tmp_path / 'ms.csv'
    pd.DataFrame({'Start':[1], 'End':[3], 'Sequence':['AAA'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3],
                  'Rel. Uptake SD':[.1]}).to_csv(exp, index=False)
    with pytest.raises(ValueError, match='chain 0 residue 2'):
        fit_ms(np.array([[.3,.4],[.5,.6]]), np.array([1,3]), np.array([0,0]),
               exp, np.array([[1.],[2.],[3.]]), tmp_path/'fit',
               theta_values=[1e3], n_skip=0, sequences=['AAA'])
    assert not (tmp_path/'fit'/'wopt_array.npy').exists()


def test_ms_rejects_scalar_sequence_start_for_multichain(tmp_path):
    import pytest
    exp = tmp_path/'ms.csv'
    pd.DataFrame({'Chain':[0], 'Start':[101], 'End':[102], 'Sequence':['AA'], 'Time (s)':[1.],
                  'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='one --sequence-start per topology chain'):
        fit_ms(np.array([[.2,.3],[.3,.4],[.4,.5],[.5,.6]]),
               np.array([101,102,501,502]), np.array([0,0,1,1]), exp,
               np.array([[1.,2.],[3.,4.]]),tmp_path/'fit',theta_values=[1e3],
               sequence_start=101,n_skip=0,sequences=['AA','AA'])

def test_ms_zero_opening_amide_is_not_dropped(tmp_path):
    exp=tmp_path/'ms.csv'
    pd.DataFrame({'Start':[1], 'End':[3], 'Sequence':['AAA'], 'Time (s)':[1.],
                  'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    fit_ms(np.array([[.2,.4],[0.,0.],[.3,.5]]), np.array([1,2,3]),
           np.array([0,0,0]), exp, np.array([[1.],[2.],[3.]]),
           tmp_path/'fit', theta_values=[1e3],n_skip=0, sequences=['AAA'])
    assert pd.read_csv(tmp_path/'fit'/'ms_residue_axis.csv').Residue.tolist()==[1,2,3]


def test_ms_requires_sequence_column(tmp_path):
    import pytest
    exp = tmp_path/'missing_sequence.csv'
    pd.DataFrame({'Start':[1], 'End':[2], 'Time (s)':[1.],
                  'Rel. Uptake':[.3], 'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='Sequence'):
        fit_ms(np.ones((2,2))*.2,np.array([1,2]),np.array([0,0]),
               exp,np.array([[1.],[2.]]),tmp_path/'out', sequences=['AA'])

def test_ms_rejects_blank_sequence(tmp_path):
    import pytest
    exp = tmp_path/'blank_sequence.csv'
    pd.DataFrame({'Start':[1,1], 'End':[2,2], 'Sequence':['AA',''],
                  'Time (s)':[1.,2.], 'Rel. Uptake':[.3,.4],
                  'Rel. Uptake SD':[.1,.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='non-empty Sequence'):
        fit_ms(np.ones((2,2))*.2,np.array([1,2]),np.array([0,0]),
               exp,np.array([[1.],[2.]]),tmp_path/'out', sequences=['AA'])

def test_ms_requires_fasta_even_with_zero_kint(tmp_path):
    import pytest
    exp = tmp_path/'peptide.csv'
    pd.DataFrame({'Start':[1], 'End':[2], 'Sequence':['AA'],
                  'Time (s)':[1.], 'Rel. Uptake':[.3],
                  'Rel. Uptake SD':[.1]}).to_csv(exp,index=False)
    with pytest.raises(ValueError, match='requires --fasta'):
        fit_ms(np.ones((2,2))*.2,np.array([1,2]),np.array([0,0]),
               exp,np.array([[0.],[2.]]),tmp_path/'out')
