#!/usr/bin/env python
import sys
import os
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule

"""
# meekovina

Python script Easy to use Autodock Vina basic docking simualation

## Requirements

*. python: 3.6 or later
* numpy
* scipy
* pandas
* rdkit
* meeko
* vina

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

### Optional arguments

Input:
  --ligand arg          ligand (PDBQT, SDF, MOL2, PDB)
  --input_smiles arg    ligand (SMILES)
  --receptor arg        rigid part of the receptor (PDBQT)

Search space (required):
  --center_x arg        X coordinate of the center
  --center_y arg        Y coordinate of the center
  --center_z arg        Z coordinate of the center
  --size_x arg          size in the X dimension (Angstroms)
  --size_y arg          size in the Y dimension (Angstroms)
  --size_z arg          size in the Z dimension (Angstroms)

Output (optional):
  --out arg             output models (PDBQT), MOL, the default is chosen based on 
                        the ligand file name

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

## License

This package is distributed under the MIT License.

"""

def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        usage=f"python {os.path.basename(__file__)} -l LIGAND -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y -cz CENTER_Z"
        "python {os.path.basename(__file__)} --input_smiles INPUT_SMILES -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y -cz CENTER_Z"
    )
    parser.add_argument(
        "-l", "--ligand", type=str,
        help="ligand (PDBQT, MOL, SDF, MOL2, PDB)"
    )
    parser.add_argument(
        "-r", "--receptor", type=str, required=True,
        help="rigid part of the receptor (PDBQT)"
    )
    parser.add_argument(
        "-o", "--out", type=str,
        help="output models (PDBQT), the default is chosen based on the ligand file name"
    )
    parser.add_argument(
        "--input_smiles", type=str,
        help="SMILES string (Need to put the atom you want to extend at the end of the string)"
    )
    parser.add_argument(
        "--smi", type=str,
        help="ligand (PDBQT, MOL, SDF, MOL2, PDB)"
    )
    parser.add_argument(
        "-cx", "--center_x", type=float, required=True,
        help="X coordinate of the center"
    )
    parser.add_argument(
        "-cy", "--center_y", type=float, required=True,
        help="Y coordinate of the center"
    )
    parser.add_argument(
        "-cz", "--center_z", type=float, required=True,
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
        "--seed", type=int,
        help="explicit random seed"
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=8,
        help="exhaustiveness of the global search"
        "(roughly proportional to time): 1+"
    )
    parser.add_argument(
        "--num_modes", type=int, default=9,
        help="maximum number of binding modes to generate"
    )
    parser.add_argument(
        "--energy_range", type=int, default=3,
        help="maximum energy difference between the best binding"
        "mode and the worst one displayed (kcal/mol)"
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
        "-d", "--debug", action='store_true',
        help="debug mode"
    )
    args = parser.parse_args()

    if not args.ligand and not args.input_smiles:
        print('Error: no ligand input file or SMILES')
        sys.exit(1)

    if not args.out:
        if not args.input_smiles:
            args.out = os.path.splitext(os.path.basename(args.ligand))[0] + '_out.pdbqt'
        else:
            args.out = 'ligand_out.pdbqt'

    return args 

def set_ligand_pdbqt(conf):
    if conf.ligand:
        root, ext = os.path.splitext(conf.ligand)
        ext = ext.lower()
        print('ext: ', ext)
        if ext == '.pdbqt':
            mol = PDBQTMolecule.from_file(conf.ligand, skip_typing=True)[0].export_rdkit_mol()
        elif ext == '.mol' or ext == '.sdf':
            mol = Chem.MolFromMolFile(conf.ligand)
        elif ext == '.mol2':
            mol = Chem.MolFromMol2File(conf.ligand)
        elif ext == '.pdb':
            mol = Chem.MolFromPDBFile(conf.ligand)
        else:
            print('Error: input file {} is not supported file format'.format(conf.ligand))
            sys.exit(1)
    elif conf.input_smiles:
        mol = Chem.MolFromSmiles(conf.input_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,AllChem.ETKDGv3())
        mol = Chem.RemoveHs(mol)
    else:
        print('Error: no ligand input file or SMILES')
        sys.exit(1)
    if conf.debug:
        print(Chem.MolToMolBlock(mol))

    mol_conf = mol.GetConformer(-1)
    com = list(rdMolTransforms.ComputeCentroid(mol_conf))
    center = [conf.center_x, conf.center_y, conf.center_z]
    print('com: ', com, 'center: ', center)
    tr = [center[i] - com[i] for i in range(3)]
    for i, p in enumerate(mol_conf.GetPositions()):
        #print(i, p)
        mol_conf.SetAtomPosition(i, Point3D(p[0]+tr[0], p[1]+tr[1], p[2]+tr[2]))
    if conf.debug:
        print(Chem.MolToMolBlock(mol))

    preparator = MoleculePreparation(hydrate=True)
    preparator.prepare(mol)
    preparator.show_setup()
    mol_pdbqt = preparator.write_pdbqt_string()
    if conf.debug:
        print(mol_pdbqt)

    return mol_pdbqt

def vina_dock(conf):
    center = [conf.center_x, conf.center_y, conf.center_z]
    box_size = [conf.size_x, conf.size_y, conf.size_z]

    mol_pdbqt_in = set_ligand_pdbqt(conf)
    #print(mol_pdbqt_in)

    # Set up vina object
    verbosity = 1 if conf.debug else 0
    v = Vina(sf_name=conf.scoring, cpu=conf.cpu, seed=conf.seed, verbosity=verbosity)
    v.set_receptor(rigid_pdbqt_filename=conf.receptor)
    v.set_ligand_from_string(mol_pdbqt_in)
    v.compute_vina_maps(center=center, box_size=box_size)

    print(v.info())

    # Score the current pose
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])

    root, ext = os.path.splitext(conf.out)
    if conf.score_only:
        remarks='VINA RESULT:    {:>.3f}      0.000      0.000'.format(energy[0])
        v.write_pose(root+'.pdbqt', remarks=remarks, overwrite=True)
    else:
        # Minimized locally the current pose
        energy_minimized = v.optimize()
        print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])

        if conf.local_only:
            v.write_pose(root+'.pdbqt', overwrite=True)
        else:
            # Dock the ligand
            v.dock(exhaustiveness=conf.exhaustiveness, n_poses=conf.num_modes)
            print("Vina Docking energies:\n", v.energies())
            v.write_poses(root+'.pdbqt', n_poses=conf.num_modes, overwrite=True)

    pdbqt_out = PDBQTMolecule.from_file(root+'.pdbqt', skip_typing=True)
    print(Chem.MolToMolBlock(pdbqt_out[0].export_rdkit_mol()))
    #Chem.MolToMolFile(pdbqt_out[0].export_rdkit_mol(), root +'_pose0.mol)
    for i, pose in enumerate(pdbqt_out):
        Chem.MolToMolFile(pose.export_rdkit_mol(), '{}_{:02}{}'.format(root, i, '.mol'))
        
def main():
    args = get_parser()
    print(args)
    vina_dock(args)

if __name__ == "__main__":
    main()
