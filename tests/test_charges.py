import numpy as np
from hxensemble import charges


def test_user_charges_override(tmp_path, monkeypatch):
    path = tmp_path / 'charges.txt'
    np.savetxt(path, [0.1, -0.1])
    def no_auto(*args, **kwargs):
        raise AssertionError('Automatic assignment must not run when user charges are supplied')
    monkeypatch.setattr(charges, 'openmm_charges', no_auto)
    q, source = charges.resolve_charges('unused.pdb', user_charges=path, expected_atoms=2)
    np.testing.assert_allclose(q, [0.1, -0.1])
    assert source == str(path)


def test_amber14_then_json_fallback(monkeypatch):
    calls = []
    def openmm(path, forcefield):
        calls.append(('openmm', forcefield))
        raise ValueError('unsupported system')
    def json(path, source):
        calls.append(('json', source))
        return np.array([0.2, -0.2])
    monkeypatch.setattr(charges, 'openmm_charges', openmm)
    monkeypatch.setattr(charges, 'json_charges', json)
    q, source = charges.resolve_charges('fake.pdb', expected_atoms=2)
    assert calls == [('openmm', 'amber14'), ('json', charges.DEFAULT_JSON)]
    np.testing.assert_allclose(q, [0.2, -0.2])
    assert source.startswith('JSON:')
