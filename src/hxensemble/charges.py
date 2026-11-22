"""Atomic charge assignment.

fallback order: user file > OpenMM AMBER14 > residue JSON.
No atoms are added, removed or reordered during charge assignment.
"""
import json
from pathlib import Path
import numpy as np

DEFAULT_JSON = Path(__file__).resolve().parent / 'data' / 'partial_charges.json'


def _openmm():
    try:
        import openmm
        from openmm.app import PDBFile, ForceField, NoCutoff
    except ImportError as exc:
        raise ImportError('OpenMM is required for automatic force-field charges: pip install openmm') from exc
    return openmm, PDBFile, ForceField, NoCutoff


def openmm_charges(pdb_file, forcefield='amber14'):
    """Original OpenMM ForceField(...).createSystem method, unchanged atom ordering."""
    openmm, PDBFile, ForceField, NoCutoff = _openmm()
    pdb = PDBFile(str(pdb_file))
    ff = ForceField(f'{forcefield}-all.xml', f'{forcefield}/tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
    nbforce = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nbforce is None:
        raise ValueError('Force-field system has no NonbondedForce')
    return np.asarray([nbforce.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
                       for i in range(system.getNumParticles())], dtype=float)


def json_charges(pdb_file, json_file=DEFAULT_JSON):
    """JSON lookup.
    Terminal residue names use the minimum/maximum PDB residue numbers.
    """
    _, PDBFile, _, _ = _openmm()
    with open(json_file, 'r') as f:
        charge_dict = json.load(f)

    pdb_openmm = PDBFile(str(pdb_file))
    topology = pdb_openmm.topology

    residue_numbers = []
    residue_names = []
    atom_names = []
    for atom in topology.atoms():
        residue = atom.residue
        residue_numbers.append(int(residue.id))
        residue_names.append(residue.name)
        atom_names.append(atom.name)

    if not residue_numbers:
        raise ValueError('Cannot assign JSON charges to an empty topology')

    final_residue_names = []
    start_residue = np.min(residue_numbers)
    end_residue = np.max(residue_numbers)
    for resname, resid in zip(residue_names, residue_numbers):
        if resid == start_residue:
            final_residue_names.append('N' + resname)
        elif resid == end_residue:
            final_residue_names.append('C' + resname)
        else:
            final_residue_names.append(resname)

    charges = []
    for resname, atom, resnum in zip(final_residue_names, atom_names, residue_numbers):
        if resname[0] == 'N' and atom == 'H':
            atom = 'H1'
        if resname == 'HIS':
            try:
                ch = float(charge_dict['HID'][atom])
            except (KeyError, ValueError):
                ch = float(charge_dict['HIE'][atom])
        else:
            ch = float(charge_dict[resname][atom])
        charges.append(ch)

    return np.array(charges)


def resolve_charges(topology, user_charges=None, expected_atoms=None):
    """Use supplied atomic charges, otherwise OpenMM Amber14 then bundled JSON.

    Charges must correspond to the PDB atom order and are in elementary charge units.
    """
    if user_charges is not None:
        path = Path(user_charges)
        q = np.load(path) if path.suffix == '.npy' else np.loadtxt(path)
        source = str(path)
    else:
        try:
            q = openmm_charges(topology, 'amber14')
            source = 'OpenMM:amber14'
        except Exception as openmm_error:
            try:
                q = json_charges(topology, DEFAULT_JSON)
                source = f'JSON:{DEFAULT_JSON}'
            except Exception as json_error:
                raise RuntimeError(
                    f'Automatic charge assignment failed: OpenMM: {openmm_error}; '
                    f'JSON: {json_error}. Supply --user-charges for unsupported residues.'
                ) from json_error
    q = np.asarray(q, dtype=float).reshape(-1)
    if expected_atoms is not None and q.size != expected_atoms:
        raise ValueError(f'Number of charges ({q.size}) differs from topology atoms ({expected_atoms})')
    if not np.all(np.isfinite(q)):
        raise ValueError('Charges contain NaN/inf')
    return q, source
