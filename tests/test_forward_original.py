"""Small, deterministic regression tests for the public forward-model API.

Scientific equivalence to external/original trajectories must also be checked
against independently stored reference results; these fixtures test internal
consistency, atom selection, metadata and output conventions.
"""

import numpy as np
import mdtraj as md
import pytest

from hxensemble.forward import (
    MODELS, match_backbone_amides, calc_sasa_hbonds, calc_rho_front,
    calc_PF_DH, calculate_chunk, run_forward,
)


def small_protein(n_frames=1):
    top = md.Topology()
    chain = top.add_chain()
    for i in (1, 2):
        res = top.add_residue('ALA', chain, resSeq=i)
        for name, element in [
            ('N', md.element.nitrogen), ('H', md.element.hydrogen),
            ('CA', md.element.carbon), ('C', md.element.carbon),
            ('O', md.element.oxygen),
        ]:
            top.add_atom(name, element, res)
    xyz = np.array([
        [0, 0, 0], [.1, 0, 0], [.1, .2, 0], [.2, .2, 0], [.25, .2, 0],
        [.4, 0, 0], [.5, 0, 0], [.5, .2, 0], [.6, .2, 0], [.65, .2, 0],
    ], dtype=np.float32)
    return md.Trajectory(np.repeat(xyz[None], n_frames, axis=0), top)


@pytest.mark.parametrize('model', ['M11', 'M12', 'M14'])
def test_forward_model_combination_without_dh(model):
    traj = small_protein()
    amides = match_backbone_amides(traj)
    H, N = amides[:2]
    threshold, correction, exclude = MODELS[model]
    _, _, _, open_sh = calc_sasa_hbonds(traj, sasa_threshold=threshold, amides=amides)
    rho = calc_rho_front(traj, N, H, exclude_neighbors=exclude)
    O, rho_found, DH, residues, _, chains, open_found = calculate_chunk(
        traj, model, DH_scaling=0, return_components=True, amides=amides,
    )
    np.testing.assert_array_equal(rho_found, rho)
    np.testing.assert_array_equal(open_found, open_sh)
    np.testing.assert_allclose(O, open_sh * (rho < 0.5) / correction)
    assert DH is None
    np.testing.assert_array_equal(residues, [1, 2])
    np.testing.assert_array_equal(chains, [0, 0])


def test_m11_rho_excludes_only_preceding_and_current_residue():
    top = md.Topology()
    chain = top.add_chain()
    for r in (1, 2, 3):
        res = top.add_residue('ALA', chain, resSeq=r)
        top.add_atom('N', md.element.nitrogen, res)
        top.add_atom('H', md.element.hydrogen, res)
        top.add_atom('CA', md.element.carbon, res)
    other = top.add_residue('ALA', top.add_chain(), resSeq=3)
    top.add_atom('CA', md.element.carbon, other)
    xyz = np.ones((1, top.n_atoms, 3), dtype=np.float32) * 5
    H = top.select('protein and (name H or name HN)')[-1]
    N = next(a.index for a in top.atom(int(H)).residue.atoms if a.name == 'N')
    xyz[0, N] = [0, 0, 0]
    xyz[0, H] = [0.1, 0, 0]
    xyz[0, 2] = [0.2, 0, 0]  # residue 1: included
    xyz[0, 5] = [0.2, 0, 0]  # residue 2: excluded
    xyz[0, 8] = [0.2, 0, 0]  # residue 3: excluded
    xyz[0, 9] = [0.2, 0, 0]  # separate chain: included
    rho = calc_rho_front(md.Trajectory(xyz, top), np.array([N]),
                         np.array([H]), exclude_neighbors=True)
    xyz2 = xyz.copy()
    xyz2[0, [5, 8]] = [5, 5, 5]
    reference = calc_rho_front(md.Trajectory(xyz2, top), np.array([N]),
                               np.array([H]), exclude_neighbors=True)
    np.testing.assert_allclose(rho, reference)
    xyz3 = xyz2.copy()
    xyz3[0, 2] = [5, 5, 5]
    reduced = calc_rho_front(md.Trajectory(xyz3, top), np.array([N]),
                             np.array([H]), exclude_neighbors=True)
    assert rho[0, 0] > reduced[0, 0]


def test_nonprotein_heavy_atom_contributes_to_rho():
    traj = small_protein()
    top = traj.topology
    ligand = top.add_residue('LIG', top.add_chain(), resSeq=100)
    top.add_atom('C1', md.element.carbon, ligand)
    xyz = np.concatenate([traj.xyz, np.array([[[0.2, 0, 0]]], dtype=np.float32)], axis=1)
    extended = md.Trajectory(xyz, top)
    h, n = 1, 0
    rho_with = calc_rho_front(extended, np.array([n]), np.array([h]))
    xyz[0, -1] = [5, 5, 5]
    rho_without = calc_rho_front(md.Trajectory(xyz, top), np.array([n]), np.array([h]))
    assert rho_with[0, 0] > rho_without[0, 0]


def test_dh_includes_nonprotein_charges_and_scales_linearly():
    traj = small_protein()
    top = traj.topology
    ligand = top.add_residue('LIG', top.add_chain(), resSeq=100)
    top.add_atom('C1', md.element.carbon, ligand)
    xyz = np.concatenate([traj.xyz, np.array([[[0.22, 0., 0.]]], dtype=np.float32)], axis=1)
    extended = md.Trajectory(xyz, top)
    amides = match_backbone_amides(extended)
    charges = np.zeros(top.n_atoms)
    charges[-1] = 1.0  # Only the non-protein ligand carries a charge.
    _, _, dh_full = calc_PF_DH(extended, charge_scaling=1., user_charges=charges, amides=amides)
    _, _, dh_scaled = calc_PF_DH(extended, charge_scaling=.75, user_charges=charges, amides=amides)
    _, _, dh_zero = calc_PF_DH(extended, charge_scaling=1., user_charges=np.zeros_like(charges), amides=amides)
    assert np.max(np.abs(dh_full)) > 0
    np.testing.assert_allclose(dh_scaled, .75 * dh_full, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(dh_zero, 0)


def test_match_backbone_amides_uses_spatial_chain_identity():
    top = md.Topology()
    coords = []
    for offset in (0.0, 3.0):
        chain = top.add_chain()
        res = top.add_residue('ALA', chain, resSeq=27)
        top.add_atom('N', md.element.nitrogen, res)
        top.add_atom('H', md.element.hydrogen, res)
        coords.extend([[offset, 0, 0], [offset + 0.1, 0, 0]])
    traj = md.Trajectory(np.asarray([coords], dtype=np.float32), top)
    H, N, residues, _, chains = match_backbone_amides(traj)
    np.testing.assert_array_equal(H, [1, 3])
    np.testing.assert_array_equal(N, [0, 2])
    assert residues == [27, 27]
    assert chains == [0, 1]


def test_component_saving_default_and_chunk_order(tmp_path):
    from inspect import signature
    assert signature(run_forward).parameters['chunk'].default == 1000
    assert signature(run_forward).parameters['save_components'].default is False
    trj = small_protein(n_frames=3)
    pdb, xtc = tmp_path / 'protein.pdb', tmp_path / 'trajectory.xtc'
    trj[0].save_pdb(str(pdb))
    trj.save_xtc(str(xtc))
    normal = tmp_path / 'normal'
    run_forward(pdb, [xtc], normal, chunk=1, dh_scaling=0)
    assert not (normal / 'rho_front.npy').exists()
    assert np.load(normal / 'O.npy').shape == (3, 2)
    assert np.loadtxt(normal / 'amideresidues.txt', dtype=int).tolist() == [1, 2]
    assert np.loadtxt(normal / 'amidechainidx.txt', dtype=int).tolist() == [0, 0]
    extra = tmp_path / 'extra'
    run_forward(pdb, [xtc], extra, chunk=2, dh_scaling=0, save_components=True)
    for name in ('O.npy', 'rho_front.npy', 'open_sasa_hbonds.npy', 'lnPF_DH.npy'):
        assert np.load(extra / name).shape == (3, 2)
    np.testing.assert_array_equal(np.load(extra / 'O.npy'), np.load(normal / 'O.npy'))
    np.testing.assert_array_equal(np.load(extra / 'lnPF_DH.npy'), 0)
