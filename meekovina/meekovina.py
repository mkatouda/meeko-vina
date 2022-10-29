#!/usr/bin/env python

"""
# meekovina

Python script Easy to use Autodock Vina basic docking simualation

## License

This package is distributed under the MIT License.

## Required libiraries

1. python: 3.7 or later
2. pyyaml
3. numpy
4. scipy
5. pandas
6. rdkit (https://www.rdkit.org/)
7. AutoDock-Vina 1.2.3 binary (https://github.com/ccsb-scripps/AutoDock-Vina )

## Optional libiraries

8. vina 1.2.3 python API (https://github.com/ccsb-scripps/AutoDock-Vina )

## Install

- Install from github
pip install git+https://github.com/mkatouda/meekovina.git

- Local install
git clone https://github.com/mkatouda/meekovina.git
cd meekovina
pip install -e .

## Usage

### Basic usage

- Ligand input from file

meekovina -l LIGAND -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y  -cz CENTER_Z

- Ligand input from SMILES

meekovina -input_smiles INPUT_SMILES -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y  -cz CENTER_Z

- Ligand input from file,
  box center (CENTER_X, CENTER_Y, CENTER_Z) is determined by center of mass of REFLIGAND

meekovina -l LIGAND -r RECEPTOR -o OUTPUT -rl REFLIGAND

### Optional arguments

Input:
  --ligand arg          ligand (PDBQT, SDF, MOL2, PDB)
  --input_smiles arg    ligand (SMILES)
  --receptor arg        rigid part of the receptor (PDBQT)

Search space (required option 1):
  --center_x arg       X coordinate of the center
  --center_y arg        Y coordinate of the center
  --center_z arg        Z coordinate of the center

Search space (required option 2):
  --refligand arg       reference ligand (PDBQT, SDF, MOL2, PDB) determining
                        box center values (CENTER_X, CENTER_Y, CENTER_Z) 
                        from center of mass of this ligand file

Search space (optional):
  --size_x arg          size in the X dimension (Angstroms)
  --size_y arg          size in the Y dimension (Angstroms)
  --size_z arg          size in the Z dimension (Angstroms)

Output (optional):
  --out arg             output models (PDBQT), MOL, the default is chosen based on 
                        the ligand file name
  --outdir arg          make output directry if argument is specified

Misc (optional):
  --cpu arg                 the number of CPUs to use (the default is to try to
                            detect the number of CPUs or, failing that, use 1)
  --seed arg                explicit random seed
  --exhaustiveness arg (=8) exhaustiveness of the global search (roughly 
                            proportional to time): 1+
  --num_modes arg (=9)      maximum number of binding modes to generate
  --energy_range arg (=3)   maximum energy difference between the best binding 
                            mode and the worst one displayed (kcal/mol)

Information (optional):
  --help                display usage summary

"""

import sys
import os
import shutil
import argparse

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from meeko import MoleculePreparation, PDBQTMolecule


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="python script easy to use Autodock Vina basic docking simulation"
    )
    parser.add_argument(
        "-i", "--inp", type=str,
        help="yaml style input file, overwriting argument values",
    )
    parser.add_argument(
        "-l", "--ligand", type=str,
        help="ligand (PDBQT, MOL, SDF, MOL2, PDB)"
    )
    parser.add_argument(
        "--input_smiles", type=str,
        help="SMILES string (Need to put the atom you want to extend at the end of the string)"
    )
    parser.add_argument(
        "-r", "--receptor", type=str,
        help="rigid part of the receptor (PDBQT)"
    )
    parser.add_argument(
        "-lr", "--refligand", type=str,
        help="reference ligand (PDBQT, MOL, SDF, MOL2, PDB) to determine the box center"
    )
    parser.add_argument(
        "-o", "--out", type=str,
        help="output models (PDBQT), the default is chosen based on the ligand file name"
    )
    parser.add_argument(
        "-od", "--outdir", type=str,
        help="make output directry if argument is specified"
    )
    parser.add_argument(
        "-cx", "--center_x", type=float,
        help="X coordinate of the center"
    )
    parser.add_argument(
        "-cy", "--center_y", type=float,
        help="Y coordinate of the center"
    )
    parser.add_argument(
        "-cz", "--center_z", type=float,
        help="Z coordinate of the center"
    )
    parser.add_argument(
        "-sx", "--size_x", type=float, default=22.5,
        help="size in the X dimension (Angstroms)"
    )
    parser.add_argument(
        "-sy", "--size_y", type=float, default=22.5,
        help="size in the Y dimension (Angstroms)"
    )
    parser.add_argument(
        "-sz", "--size_z", type=float, default=22.5,
        help="size in the Z dimension (Angstroms)"
    )
    parser.add_argument(
        "-c", "--cpu", type=int, default=4,
        help="the number of CPUs to use (the default is to try to"
        "detect the number of CPUs or, failing that, use 1)"
    )
    parser.add_argument(
        "--scoring", type=str, default='vina',
        help="force field name: vina(default), ad4, vinardo"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="explicit random seed"
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=8,
        help="exhaustiveness of the global search"
        "(roughly proportional to time): 1+"
    )
    parser.add_argument(
        "--max_evals", type=int, default=0,
        help="number of evaluations in each MC run (if zero,"
        "which is the default, the number of MC steps is"
        "based on heuristics)"
    )
    parser.add_argument(
        "--num_modes", type=int, default=9,
        help="maximum number of binding modes to generate"
    )
    parser.add_argument(
        "--min_rmsd", type=int, default=1,
        help="minimum RMSD between output poses"
    )
    parser.add_argument(
        "--energy_range", type=int, default=3,
        help="maximum energy difference between the best binding"
        "mode and the worst one displayed (kcal/mol)"
    )
    parser.add_argument(
        "--spacing", type=float, default=0.375,
        help="grid spacing (Angstrom)"
    )
    parser.add_argument(
        "--verbosity", type=int, default=1,
        help="verbosity (0=no output, 1=normal, 2=verbose)"
    )
    parser.add_argument(
        "--score_only", action='store_true',
        help="evaluate the energy of the current pose or poses without strucutre optimization"
    )
    parser.add_argument(
        "--local_only", action='store_true',
        help="evaluate the energy of the current pose or poses with local structure optimization"
    )
    parser.add_argument(
        "--exec", type=str, default='bin',
        help="select AutoDock-Vina executer"
    )
    parser.add_argument(
        "--bin_path", type=str, default='vina',
        help="AutoDock-Vina binary path"
    )
    parser.add_argument(
        "--boxauto", action='store_true',
        help = "enable automatic box determination algorithm"
    )
    parser.add_argument(
        "--gybox_ratio", type=float, default=2.5,
        help = "scaling factor of radius of gyration to determine of docking box\n"
               "with automatic box determination algorithm"
    )
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    args = parser.parse_args()

    print(args)

    if args.out is None:
        if args.ligand:
            args.out = os.path.splitext(os.path.basename(args.ligand))[0] + '_out.pdbqt'
        else:
            args.out = 'ligand_out.pdbqt'

    return args 

def set_config(args):
    # Read config yaml file
    if args.inp is not None and os.path.isfile(args.inp):
        with open(args.inp, 'r') as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}

    # Set up default config values from program arguments
    conf_def = vars(args).copy()
    [conf.setdefault(k, v) for k, v in conf_def.items()]

    return conf

def file2rdmol(file_path):
    root, ext = os.path.splitext(file_path)
    ext = ext.lower()
    #print('ext: ', ext)
    if ext == '.pdbqt':
        mol = PDBQTMolecule.from_file(file_path, skip_typing=True)[0].export_rdkit_mol()
    elif ext == '.mol' or ext == '.sdf':
        mol = Chem.MolFromMolFile(file_path)
    elif ext == '.mol2':
        mol = Chem.MolFromMol2File(file_path)
    elif ext == '.pdb':
        mol = Chem.MolFromPDBFile(file_path)
    else:
        print('Error: input file {} is not supported file format'.format(file_path))
        sys.exit(1)

    return mol

def get_ligand_com(ligand_path, debug=False):
    if os.path.isfile(ligand_path):
        mol = file2rdmol(ligand_path)
        center = list(rdMolTransforms.ComputeCentroid(mol.GetConformer(-1)))
    else:
        print('Error: no reference ligand input file: {}'.format(ligand_path))
        sys.exit(1)

    return center

def set_ligand_pdbqt(ligand_path, center, input_smiles=None, output_pdbqt_path=None, debug=False):

    if os.path.isfile(ligand_path):
        mol = file2rdmol(ligand_path)
    elif input_smiles:
        mol = Chem.MolFromSmiles(input_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
    else:
        print('Error: no ligand input file {} or SMILES'.format(ligand_path))
        sys.exit(1)
    if debug: print(Chem.MolToMolBlock(mol))

    mol_conf = mol.GetConformer(-1)
    com = list(rdMolTransforms.ComputeCentroid(mol_conf))

    #print('PM\n', rdMolTransforms.ComputePrincipalAxesAndMoments(mol_conf))

    if debug: print('com of ligand: ', com, 'reference center: ', center)
    tr = [center[i] - com[i] for i in range(3)]
    rmax = 0.0
    rg = 0.0
    for i, p in enumerate(mol_conf.GetPositions()):
        #print(i, p)
        mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
        r2 = (p[0]-com[0])**2 + (p[1]-com[1])**2 + (p[2]-com[2])**2
        rg += r2
        r = np.sqrt(r2)
        if r > rmax:
            rmax = r
    rg = np.sqrt(rg / mol.GetNumAtoms())
    print('rmax:', rmax, 'rg:', rg)
    if debug: print(Chem.MolToMolBlock(mol))

    preparator = MoleculePreparation(hydrate=True)
    preparator.prepare(mol)
    mol_pdbqt = preparator.write_pdbqt_string()
    if isinstance(output_pdbqt_path, str):
        preparator.write_pdbqt_file(output_pdbqt_path)

    if debug:
        preparator.show_setup()
        print(mol_pdbqt)

    return mol_pdbqt, rmax, rg

def dockscore_summary(output_ligand_path, output_csv_path):
    scores = []
    with open(output_ligand_path) as f:
        for l in f.readlines():
            if 'REMARK VINA RESULT' in l:
                scores.append([float(s) for s in l.split()[3:6]])

    df_score = pd.DataFrame(scores, columns=['Docking_score', 'dist_RMSD', 'bestmode_RMSD'])
    df_score.to_csv(output_csv_path)

    return scores

def vina_dock_bin(receptor_path, ligand_path, ref_ligand_path, outbasename, vina_bin_path,
                  boxcenter, boxauto=True, boxsize=[22.5, 22.5, 22.5], gybox_ratio=4.0,
                  scoring='vina', cpu=0, seed=0, exhaustiveness=8, 
                  max_evals=0, num_modes=9, min_rmsd=1, energy_range=3,
                  spacing=0.375, verbosity=1, maxtry=100, score_only=False, local_only=False,
                  debug=False):

    import subprocess

    #outbasename = os.path.splitext(os.path.basename(ligand_path))[0]
    input_ligand_path = './{}_vinain.pdbqt'.format(outbasename)
    output_ligand_path = './{}_vinaout.pdbqt'.format(outbasename)

    if ref_ligand_path is not None and os.path.isfile(ref_ligand_path):
        center = get_ligand_com(ref_ligand_path)
    else:
        center = boxcenter
    print('box_center:', center)
    mol_pdbqt_in, rmax, rg = set_ligand_pdbqt(ligand_path, center,
                                              output_pdbqt_path=input_ligand_path,
                                              debug=debug)
    if debug: print(mol_pdbqt_in)

    # Set box size
    if boxauto:
        bl = rg * gybox_ratio
        boxsize_mod = [bl, bl, bl]
    else:
        boxsize_mod = boxsize
        maxtry = 1

    for i in range(maxtry):
        print('traial', i+1, 'boxauto:', boxauto, 'boxsize:', boxsize_mod)
        cmd = [vina_bin_path,
               '--receptor', receptor_path,
               '--ligand', input_ligand_path,
               '--center_x', str(center[0]),
               '--center_y', str(center[1]),
               '--center_z', str(center[2]),
               '--size_x', str(boxsize_mod[0]),
               '--size_y', str(boxsize_mod[1]),
               '--size_z', str(boxsize_mod[2]),
               '--out', output_ligand_path,
               '--cpu', str(cpu),
               '--seed', str(seed),
               '--exhaustiveness', str(exhaustiveness),
               '--max_evals', str(max_evals),
               '--num_modes', str(num_modes),
               '--min_rmsd', str(min_rmsd),
               '--energy_range', str(energy_range),
               '--spacing', str(spacing),
               '--verbosity', str(verbosity)]
        if debug: print(' '.join(cmd))

        try:
            results = subprocess.run(cmd, capture_output=True, check=True, text=True)
            print(results.stdout, results.stderr)
            if results.returncode == 0: break 
        except subprocess.CalledProcessError:
            boxsize_mod[0] += 0.5; boxsize_mod[1] += 0.5; boxsize_mod[2] += 0.5

    pdbqt_out = PDBQTMolecule.from_file(output_ligand_path, skip_typing=True)
    if debug: print(Chem.MolToMolBlock(pdbqt_out[0].export_rdkit_mol()))
    for i, pose in enumerate(pdbqt_out):
        Chem.MolToMolFile(pose.export_rdkit_mol(), './{}_vinaout_{:02}.mol'.format(outbasename, i))

    output_csv_path = './{}_vinascore.csv'.format(outbasename)
    scores = dockscore_summary(output_ligand_path, output_csv_path)

    return scores

def vina_dock_lib(receptor_path, ligand_path, ref_ligand_path, outbasename,
                  boxcenter, boxauto=True, boxsize=[22.5, 22.5, 22.5], gybox_ratio=4.0,
                  scoring='vina', cpu=0, seed=0, exhaustiveness=8,
                  max_evals=0, num_modes=9, min_rmsd=1, energy_range=3,
                  spacing=0.375, verbosity=1,
                  score_only=False, local_only=False, debug=False):

    from vina import Vina

    if ref_ligand_path is not None and os.path.isfile(ref_ligand_path):
        center = get_ligand_com(ref_ligand_path)
    else:
        center = boxcenter
    print('box_center:', center)
    mol_pdbqt_in, rmax, rg = set_ligand_pdbqt(ligand_path, center, debug=debug)
    if debug: print(mol_pdbqt_in)

    # Set box size
    if boxauto:
        bl = rg * gybox_ratio
        boxsize_mod = [bl, bl, bl]
    else:
        boxsize_mod = boxsize
    print('boxauto:', boxauto, 'boxsize:', boxsize_mod)

    # Set up vina object
    #verbosity = 1 if debug else 0
    v = Vina(sf_name=scoring, cpu=cpu, seed=seed, verbosity=verbosity)
    v.set_receptor(rigid_pdbqt_filename=receptor_path)
    v.set_ligand_from_string(mol_pdbqt_in)
    v.compute_vina_maps(center=center, box_size=boxsize_mod)

    if debug: print(v.info())

    # Score the current pose
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    #outbasename = os.path.splitext(os.path.basename(ligand_path))[0]
    output_ligand_path = './{}_vinaout.pdbqt'.format(outbasename)

    if score_only:
        remarks='VINA RESULT:    {:>.3f}      0.000      0.000'.format(energy[0])
        v.write_pose(output_ligand_path, remarks=remarks, overwrite=True)
    else:
        # Minimized locally the current pose
        energy_minimized = v.optimize()
        print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])

        if local_only:
            v.write_pose(output_ligand_path, overwrite=True)
        else:
            # Dock the ligand
            v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes,
                   min_rmsd=min_rmsd, max_evals=max_evals)
            #print("Vina Docking energies:\n",
            #      v.energies(n_poses=(num_modes+1), energy_range=energy_range))
            v.write_poses(output_ligand_path, n_poses=num_modes, 
                          energy_range=energy_range, overwrite=True)

    pdbqt_out = PDBQTMolecule.from_file(output_ligand_path, skip_typing=True)
    if debug: print(Chem.MolToMolBlock(pdbqt_out[0].export_rdkit_mol()))
    for i, pose in enumerate(pdbqt_out):
        Chem.MolToMolFile(pose.export_rdkit_mol(), './{}_vinaout_{:02}.mol'.format(outbasename, i))

    output_csv_path = './{}_vinascore.csv'.format(outbasename)
    scores = dockscore_summary(output_ligand_path, output_csv_path)

    return scores

def vina_dock(receptor_path, ligand_path, ref_ligand_path, outbasename, vina_exec, vina_bin_path,
                  boxcenter, boxauto=True, boxsize=[22.5, 22.5, 22.5], gybox_ratio=4.0,
                  scoring='vina', cpu=0, seed=0, exhaustiveness=8, 
                  max_evals=0, num_modes=9, min_rmsd=1, energy_range=3,
                  spacing=0.375, verbosity=1, maxtry=100,
                  score_only=False, local_only=False, debug=False):

    if 'bin' in vina_exec.lower():
        print('Using AutoDock Vina binary')
        scores = vina_dock_bin(receptor_path,
                               ligand_path,
                               ref_ligand_path,
                               outbasename,
                               vina_bin_path,
                               boxcenter,
                               boxauto=boxauto,
                               boxsize=boxsize,
                               gybox_ratio=gybox_ratio,
                               scoring=scoring,
                               cpu=cpu,
                               seed=seed,
                               exhaustiveness=exhaustiveness,
                               max_evals=max_evals,
                               num_modes=num_modes,
                               min_rmsd=min_rmsd,
                               energy_range=energy_range,
                               spacing=spacing,
                               verbosity=verbosity,
                               maxtry=maxtry,
                               score_only=score_only,
                               local_only=local_only,
                               debug=debug)
    else:
        print('Using AutoDock Vina python API')
        scores = vina_dock_lib(receptor_path,
                               ligand_path,
                               ref_ligand_path,
                               outbasename,
                               boxcenter,
                               boxauto=boxauto,
                               boxsize=boxsize,
                               gybox_ratio=gybox_ratio,
                               scoring=scoring,
                               cpu=cpu,
                               seed=seed, 
                               exhaustiveness=exhaustiveness,
                               max_evals=max_evals,
                               num_modes=num_modes,
                               min_rmsd=min_rmsd,
                               energy_range=energy_range,
                               spacing=spacing,
                               verbosity=verbosity,
                               score_only=score_only,
                               local_only=local_only,
                               debug=debug)

    return scores

def vina_dock_main(conf):
    protein_pdbqt_path = os.path.abspath(conf['receptor'])
    ligand_path = os.path.abspath(conf['ligand'])
    ref_ligand_path = os.path.abspath(conf['refligand'])
    vina_outbasename = conf['out']
    vina_outdir = conf['outdir']
    vina_exec = conf['exec']
    vina_bin_path = conf['bin_path']
    vina_boxauto = conf['boxauto']
    vina_gybox_ratio = conf['gybox_ratio']
    vina_boxcenter = [conf['center_x'], conf['center_y'], conf['center_z']]
    vina_boxsize = [conf['size_x'], conf['size_y'], conf['size_z']]
    vina_scoring = conf['scoring']
    vina_cpu = conf['cpu']
    vina_seed = conf['seed']
    vina_exhaustiveness = conf['exhaustiveness']
    vina_max_evals = conf['max_evals']
    vina_num_modes = conf['num_modes']
    vina_min_rmsd = conf['min_rmsd']
    vina_energy_range = conf['energy_range']
    vina_spacing = conf['spacing']
    vina_verbosity = conf['verbosity']
    vina_score_only = conf['score_only']
    vina_local_only = conf['local_only']
    vina_debug = conf['debug']

    cwdir = os.getcwd()
    if vina_outdir is not None:
        if not os.path.isdir(vina_outdir): os.makedirs(vina_outdir)
        os.chdir(vina_outdir)
    
    scores = vina_dock(protein_pdbqt_path,
                       ligand_path,
                       ref_ligand_path,
                       vina_outbasename,
                       vina_exec,
                       vina_bin_path,
                       boxcenter=vina_boxcenter,
                       boxauto=vina_boxauto,
                       boxsize=vina_boxsize,
                       gybox_ratio=vina_gybox_ratio,
                       scoring=vina_scoring,
                       cpu=vina_cpu,
                       seed=vina_seed,
                       exhaustiveness=vina_exhaustiveness,
                       max_evals=vina_max_evals,
                       num_modes=vina_num_modes,
                       min_rmsd=vina_min_rmsd,
                       energy_range=vina_energy_range,
                       spacing=vina_spacing,
                       verbosity=vina_verbosity,
                       score_only=vina_score_only,
                       local_only=vina_local_only,
                       debug=vina_debug)

    os.chdir(cwdir)

    return scores
       
def main():
    args = get_parser()
    if args.debug: print(args)

    conf = set_config(args)

    print('======= Input configulations =======')
    for k, v in conf.items():
        print('{}: {}'.format(k, v))
    print('====================================')

    vina_dock_main(conf)

if __name__ == '__main__':
    main()
