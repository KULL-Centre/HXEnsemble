"""
Forward model calculation sequence, streamlined for chunked trajectories.

"""
from pathlib import Path
import json
import csv
import tempfile
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import numpy as np
import mdtraj as md
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from .charges import resolve_charges
from .utils import find_xtc_filenames

# Original peptide corrections from make_M*.py. M12 differs from M11.
MODELS = {"M11": (0.05, 0.957319, True), "M12": (0.05, 0.957089, False), "M14": (0.5, 0.951914, False)}

def match_backbone_amides(pdb):
    """Match H/HN to N using PDB coordinates, once before trajectory processing.

    Require the same PDB residue number and residue name. Use N-H distance
    to resolve split heavy/hydrogen residue objects and repeated chain sequences.
    """
    import warnings

    top = pdb.topology
    xyz = pdb.xyz[0]  # nm
    H_idx = top.select('protein and (name H or name HN)')
    N_candidates = top.select('protein and name N')
    N_idx = []
    for h in H_idx:
        h = int(h)
        h_res = top.atom(h).residue
        candidates = [int(n) for n in N_candidates
                      if top.atom(int(n)).residue.resSeq == h_res.resSeq
                      and top.atom(int(n)).residue.name == h_res.name]
        if not candidates:
            raise ValueError(f'No backbone N matching H atom {h} '
                             f'({h_res.name} {h_res.resSeq})')

        distances = np.linalg.norm(xyz[candidates] - xyz[h], axis=1)
        within = [candidates[j] for j in np.flatnonzero(distances < 0.15)]
        if not within:
            message = (f'No backbone N within 1.5 Å of H atom {h} '
                       f'({h_res.name} {h_res.resSeq}); '
                       f'nearest distance {min(distances) * 10:.2f} Å')
            warnings.warn(message, RuntimeWarning)
            raise ValueError(message)

        same_res = [n for n in within if top.atom(n).residue.index == h_res.index]
        same_chain = [n for n in within if top.atom(n).residue.chain.index == h_res.chain.index]
        if same_res:
            within = same_res
        elif same_chain:
            within = same_chain
        n = min(within, key=lambda k: np.linalg.norm(xyz[k] - xyz[h]))
        N_idx.append(n)

    N_idx = np.asarray(N_idx, dtype=int)
    if len(set(N_idx.tolist())) != len(N_idx):
        raise ValueError('Multiple amide hydrogens matched to the same backbone N')

    matched_res = [top.atom(int(n)).residue for n in N_idx]
    residues = [r.resSeq for r in matched_res]
    residx = [r.index for r in matched_res]
    chains = [r.chain.index for r in matched_res]
    return H_idx, N_idx, residues, residx, chains


def calc_sasa_hbonds(trj, sasa_threshold=0.5, distance_threshold=0.26,
                     angle_threshold=120.0, remove_closest=True, amides=None):
    """All complex atoms included.

    All O/N/F atoms can be acceptors,
    and SASA is computed on the complete (water-free) trajectory.
    """
    if amides is None:
        amides = match_backbone_amides(trj)
    amideH_idx, amideN_idx, amide_resid, amide_res_idx, amide_chain_idx = amides
    sasa = md.shrake_rupley(trj, probe_radius=0.14, n_sphere_points=960, mode='atom') * 100
    amideH_SASA = sasa[:, amideH_idx]

    hbonds_res = np.zeros((len(trj), len(amideH_idx)))
    oxy = trj.top.select('symbol O')
    fluor = trj.top.select('symbol F')
    nitro = trj.top.select('symbol N')
    for i, (h, n) in enumerate(zip(amideH_idx, amideN_idx)):
        residue = trj.top.atom(int(n)).residue
        resid = residue.index
        chain = residue.chain.index

        # Original selection-string logic: closest-residue exclusion ONLY for O;
        # F stays eligible and N excludes only backbone N of this amide.
        if remove_closest:
            oxygen_idx = np.array([j for j in oxy if not (
                trj.top.atom(int(j)).residue.chain.index == chain and
                trj.top.atom(int(j)).residue.index in (resid, resid - 1)
            )], dtype=int)
        else:
            oxygen_idx = oxy
        nitrogen_idx = np.array([j for j in nitro if not (
            trj.top.atom(int(j)).residue.index == resid and
            trj.top.atom(int(j)).residue.chain.index == chain and
            trj.top.atom(int(j)).name == 'N'
        )], dtype=int)
        acceptor_idx = np.concatenate((oxygen_idx, fluor, nitrogen_idx))
        neigh = md.compute_neighbors(trj, distance_threshold, np.array([h]), acceptor_idx, periodic=False)

        hbonds_count = []
        for frame, frame_neighbors in enumerate(neigh):
            count = 0
            frame_xyz = trj.xyz[frame]
            for h_idx, donor_idx in zip([h], [n]):
                vec_dh = frame_xyz[donor_idx] - frame_xyz[h_idx]
                vec_dh /= np.linalg.norm(vec_dh)
                for a_idx in frame_neighbors:
                    vec_ha = frame_xyz[a_idx] - frame_xyz[h_idx]
                    vec_ha /= np.linalg.norm(vec_ha)
                    angle_rad = np.arccos(np.clip(np.dot(vec_dh, vec_ha), -1.0, 1.0))
                    if angle_rad >= np.deg2rad(angle_threshold):
                        count += 1
            hbonds_count.append(count)
        hbonds_res[:, i] = hbonds_count

    bool_SASA = (amideH_SASA > sasa_threshold) * 1
    bool_hbond = (hbonds_res == 0) * 1
    bool_open = ((bool_SASA + bool_hbond) == 2) * 1
    return amide_resid, amide_res_idx, amide_chain_idx, bool_open

def switching(r, r_cut):
    """
    Smooth contact / distance weight.

    """
    x = r / r_cut
    return 1.0 / (1.0 + x**12)

def angular_weight(cos_theta, angular_power):
    """
    Directional/front-side angular weight.

    cos_theta is the cosine of the angle between N->H and H->heavy_atom.
    Only atoms in front of the amide H contribute.
    """
    return np.maximum(0.0, cos_theta) ** angular_power

def calc_rho_front(trj, N_idx, H_idx, exclude_neighbors=False,
                   r_cut=0.50, cutoff_nm=0.60, angular_power=8.0):
    """Rho_front calculation on one chunk.

    All heavy atoms of the complex contribute. M11 additionally excludes the
    preceding and same residue of the same chain, as in the original M11 variant.
    """
    top = trj.top
    heavy_idx = top.select('not element H')
    heavy_res_idx = np.array([top.atom(int(j)).residue.index for j in heavy_idx])
    heavy_chain_idx = np.array([top.atom(int(j)).residue.chain.index for j in heavy_idx])
    allowed_heavy_per_amide = []
    for N, h in zip(N_idx, H_idx):
        residue = top.atom(int(N)).residue
        if exclude_neighbors:
            exclude = ((heavy_chain_idx == residue.chain.index) &
                       ((heavy_res_idx == residue.index) | (heavy_res_idx == residue.index - 1)))
            allowed = heavy_idx[~exclude]
        else:
            allowed = heavy_idx
        allowed_heavy_per_amide.append(allowed)

    xyz = trj.xyz
    rho = np.zeros((trj.n_frames, len(N_idx)), dtype=np.float32)
    for k, (N, H, allowed_heavy) in enumerate(zip(N_idx, H_idx, allowed_heavy_per_amide)):
        if len(allowed_heavy) == 0:
            continue
        neighbours = md.compute_neighbors(trj, cutoff=cutoff_nm,
                                          query_indices=np.array([H], dtype=int),
                                          haystack_indices=allowed_heavy, periodic=False)
        for f, nbrs in enumerate(neighbours):
            if len(nbrs) == 0:
                continue
            NH = xyz[f, H] - xyz[f, N]
            NH_norm = np.linalg.norm(NH)
            if NH_norm < 1e-8:
                continue
            NH /= NH_norm
            HV = xyz[f, nbrs] - xyz[f, H]
            dist = np.linalg.norm(HV, axis=1)
            valid = dist > 1e-8
            if not np.any(valid):
                continue
            HV = HV[valid]
            dist = dist[valid]
            cos_theta = (HV @ NH) / dist
            rho[f, k] = np.sum(switching(dist, r_cut) * angular_weight(cos_theta, angular_power))
    return rho

def calc_PF_DH(trj, temperature=298.0, ionic=0.15, charge_scaling=1.0, user_charges=None, amide_block_size=64, amides=None):
    EPSILON_0 = 8.854187817e-12
    E_CHARGE = 1.602176634e-19
    NA = 6.02214076e23
    kB = 1.380649e-23
    F = 9.648533e4
    z = -1.0
    R = 8.314463

    def debye_length(ionic_strength, temperature):
        Im = ionic_strength * 1e3
        dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
        numerator = dielectric * EPSILON_0 * kB * temperature
        denominator = 2 * NA * E_CHARGE ** 2 * Im
        return np.sqrt(numerator / denominator)

    if amides is None:
        amides = match_backbone_amides(trj)
    amideH_indices, _, amide_resid, amide_res_idx, amide_chains = amides

    if user_charges is None:
        raise ValueError("Provide the charge vector; OpenMM/JSON/file assignment happens once before processing chunks")
    charges = np.asarray(user_charges, dtype=float)

    if len(charges) != trj.n_atoms:
        raise ValueError(f'Number of charges ({len(charges)}) does not match number of trajectory atoms ({trj.n_atoms}).')

    charges = np.asarray(charges * E_CHARGE * charge_scaling, dtype=np.float64)
    residues = np.fromiter((atom.residue.resSeq for atom in trj.topology.atoms), dtype=int, count=trj.n_atoms)
    chainids = np.fromiter((atom.residue.chain.index for atom in trj.topology.atoms), dtype=int, count=trj.n_atoms)
    n_amides = len(amideH_indices)
    n_frames = len(trj)

    dielectric = 5321 / temperature + 233.76 - 0.9297 * temperature + 1.417e-3 * temperature ** 2 - 8.292e-7 * temperature ** 3
    debye_len_m = debye_length(ionic_strength=ionic, temperature=temperature)
    debye_len_nm = debye_len_m * 1e9
    electrostatic_prefactor = 1.0 / (4.0 * np.pi * EPSILON_0 * dielectric)
    lnPF_prefactor = z * F / (R * temperature)
    total_prefactor = electrostatic_prefactor * 1e9 * lnPF_prefactor

    lnPF_frames = np.empty((n_frames, n_amides), dtype=np.float64)

    for start in range(0, n_amides, amide_block_size):
        end = min(start + amide_block_size, n_amides)
        block_idx = amideH_indices[start:end]
        block_resids = np.asarray(amide_resid[start:end])
        block_chains = np.asarray(amide_chains[start:end])
        keep = (chainids[None, :] != block_chains[:, None]) | ((residues[None, :] != block_resids[:, None]) & (residues[None, :] != (block_resids[:, None] - 1)))

        for frame in range(n_frames):
            xyz = trj.xyz[frame]
            distances = cdist(xyz[block_idx], xyz)
            distances[~keep] = np.inf
            kernel = np.exp(-distances / debye_len_nm) / distances
            electrostatic_sum = kernel @ charges
            lnPF_frames[frame, start:end] = total_prefactor * electrostatic_sum


    return amide_resid, amide_res_idx, lnPF_frames

def resolve_trajectories(topology, trajectories, directory):
    if directory:
        folder = Path(directory)
        if topology is None:
            pdbs = sorted(folder.glob("*.pdb"))
            if len(pdbs) != 1:
                raise ValueError("Expected exactly one PDB in directory; otherwise use --topology")
            topology = pdbs[0]
        if not trajectories:
            trajectories = [folder / f for f in sorted(find_xtc_filenames(folder))]
    if not topology or not trajectories:
        raise ValueError("Use --directory, or --topology and --trajectory")
    return Path(topology), [Path(t) for t in trajectories]


def calculate_chunk(trj, model, DH_scaling=0.75, charges=None, temperature=298.0, ionic=0.15, return_components=False, amides=None):
    """Original sequence: amide SASA/H-bonds, rho_front, optional DH, peptide correction."""
    sasa_threshold, peptide_factor, exclude_neighbors = MODELS[model]
    if amides is None:
        amides = match_backbone_amides(trj)
    H_idx, N_idx, _, _, _ = amides
    residues, residx, chains, open_sasa_hbonds = calc_sasa_hbonds(trj, sasa_threshold=sasa_threshold, amides=amides)
    rho_front = calc_rho_front(trj, N_idx, H_idx, exclude_neighbors=exclude_neighbors)
    O = open_sasa_hbonds * (rho_front < 0.5)
    DH = None
    if DH_scaling != 0:
        _, _, DH = calc_PF_DH(trj, temperature=temperature, ionic=ionic,
                               charge_scaling=DH_scaling, user_charges=charges, amides=amides)
        O = O / np.exp(DH)
    O = O / peptide_factor
    if return_components:
        return O, rho_front, DH, residues, residx, chains, open_sasa_hbonds
    return O, rho_front, DH, residues, residx, chains


def _process_chunk(trj, model, dh_scaling, charges, temperature, ionic, path, save_components, amides):
    """One independent chunk; worker writes its arrays to avoid returning large results."""
    O, rho, DH, residues, residx, chains, open_sasa_hbonds = calculate_chunk(
        trj, model, dh_scaling, charges, temperature, ionic, return_components=True, amides=amides)
    components = {"open_sasa_hbonds": open_sasa_hbonds, "rho": rho} if save_components else {}
    if save_components:
        components["DH"] = DH if DH is not None else np.zeros_like(O)
    np.savez(path, O=O, **components)
    return list(zip(chains, residues, residx)), trj.n_frames


def run_forward(topology, trajectories, output, model="M14", chunk=1000, dh_scaling=0.75,
                user_charges=None, temperature=298, ionic=0.15, stride=1,
                directory=None, filter_overlaps=False, save_components=False, n_cpus=None):
    topology, trajectories = resolve_trajectories(topology, trajectories, directory)
    if model not in MODELS: raise ValueError("model must be M11, M12 or M14")
    if chunk <= 0 or stride <= 0: raise ValueError("chunk and stride must be positive")
    if n_cpus is None:
        n_cpus = int(os.environ.get("HXENSEMBLE_CPUS", "1"))
    if n_cpus < 1: raise ValueError("n_cpus must be >= 1")
    pdb = md.load(str(topology))
    top = pdb.topology
    amides = match_backbone_amides(pdb)
    chains = list(top.chains)
    print(f"Topology: {topology}")
    print(f"Identified {top.n_atoms:,} atoms, {len(set(amides[4]))} protein chain(s), {len(amides[0])} protein amides")
    for chain in chains:
        n_chain = sum(c == chain.index for c in amides[4])
        if n_chain:
            print(f"  Chain {chain.index}: {n_chain} protein amides")
    print(f"Model: {model} | DH_scaling: {dh_scaling} | Chunk: {chunk:,} | Stride: {stride}")
    print(f"Using {n_cpus} CPU process(es)")
    print(f"Trajectories ({len(trajectories)}), in processing order:")
    for i, trajfile in enumerate(trajectories, 1):
        print(f"  {i}. {trajfile}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    charges = None
    if dh_scaling != 0:
        charges, source = resolve_charges(topology, user_charges=user_charges,
                                          expected_atoms=md.load_topology(str(topology)).n_atoms)
        print("Charges:", source)
    saved = []
    labels = None
    nframes = 0
    accepted = []
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        scratch = Path(scratch)
        # Limit outstanding chunks, so multi-CPU runs do not buffer the entire trajectory.
        executor = ProcessPoolExecutor(max_workers=n_cpus, mp_context=mp.get_context("spawn")) if n_cpus > 1 else None
        pending = deque()

        def show_progress(done, total):
            if total is None:
                sys.stdout.write(f"\r  Completed {done:,} frames")
            else:
                fraction = min(done / total, 1.0) if total else 1.0
                filled = round(30 * fraction)
                bar = "#" * filled + "-" * (30 - filled)
                sys.stdout.write(f"\r  [{bar}] {fraction:6.1%} ({done:,}/{total:,} frames)")
            sys.stdout.flush()

        def collect_oldest():
            nonlocal labels, nframes, completed
            path, job, n_input = pending.popleft()
            this_labels, count = job.result()
            if labels is None:
                labels = this_labels
            elif labels != this_labels:
                raise ValueError("Amide metadata changed between chunks")
            saved.append((path, count))
            nframes += count
            completed += n_input
            show_progress(completed, total_frames)

        try:
            for trajfile in trajectories:
                print(f"\nProcessing {trajfile}")
                try:
                    with md.open(str(trajfile), mode="r") as handle:
                        raw_frames = len(handle)
                    total_frames = (raw_frames + stride - 1) // stride
                except (OSError, ValueError, TypeError, AttributeError):
                    total_frames = None
                completed = 0
                retained = 0
                frame_no = 0
                show_progress(0, total_frames)
                for trj in md.iterload(str(trajfile), top=str(topology), chunk=chunk, stride=stride):
                    n_input = trj.n_frames
                    ids = frame_no + stride * np.arange(n_input)
                    frame_no += stride * n_input
                    if filter_overlaps:
                        keep = np.array([cKDTree(x).query(x, k=2)[0][:, 1].min() > 0.001 for x in trj.xyz])
                        trj = trj[keep]
                        ids = ids[keep]
                    if trj.n_frames == 0:
                        completed += n_input
                        show_progress(completed, total_frames)
                        continue
                    if filter_overlaps:
                        accepted.extend((str(trajfile), int(i)) for i in ids)
                    retained += trj.n_frames
                    path = scratch / f"part_{len(saved) + len(pending):06d}.npz"
                    args = (trj, model, dh_scaling, charges, temperature, ionic, path, save_components, amides)
                    if executor is None:
                        this_labels, count = _process_chunk(*args)
                        if labels is None:
                            labels = this_labels
                        elif labels != this_labels:
                            raise ValueError("Amide metadata changed between chunks")
                        saved.append((path, count))
                        nframes += count
                        completed += n_input
                        show_progress(completed, total_frames)
                    else:
                        pending.append((path, executor.submit(_process_chunk, *args), n_input))
                        if len(pending) >= 2 * n_cpus:
                            collect_oldest()
                # Collect remaining chunks for this trajectory before moving to the next.
                while pending:
                    collect_oldest()
                print()
                if filter_overlaps:
                    print(f"  Retained {retained:,}/{completed:,} frames after overlap filtering")
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        if not saved: raise ValueError("No frames to process")
        outputs = [("O.npy", "O")]
        if save_components:
            outputs += [("open_sasa_hbonds.npy", "open_sasa_hbonds"),
                        ("rho_front.npy", "rho"), ("lnPF_DH.npy", "DH")]
        for name, key in outputs:
            arr = np.lib.format.open_memmap(out/name, mode="w+", dtype=float, shape=(nframes,len(labels)))
            offset = 0
            for path,n in saved:
                with np.load(path) as part:
                    arr[offset:offset+n] = part[key]
                offset += n
            arr.flush()
            del arr
    np.savetxt(out/"amideresidues.txt", [r for c,r,i in labels], fmt="%d")
    np.savetxt(out/"amidechainidx.txt", [c for c,r,i in labels], fmt="%d")
    np.savetxt(out/"amideresidx.txt", [i for c,r,i in labels], fmt="%d")
    if filter_overlaps:
        with (out/"retained_frames.csv").open("w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["Trajectory","Frame"])
            w.writerows(accepted)
    print(f"\nFinished: {nframes:,} total frames processed, {len(labels)} amides. Saved to {out}")
    return out
