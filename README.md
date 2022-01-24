# meeko-vina

python script easy to use Autodock Vina basic docking simualation

## Requirements

* python (>=3.6)
* numpy
* scipy
* rdkit
* meeko
* vina

## Install

Conda or Miniconda can install the dependencies:
```bash
conda install -c conda-forge numpy scipy rdkit
```

Installation (from PyPI)
```bash
pip install vina meeko
```

## Usage

### Basic usage

- Ligand input from file
```
python meeko-vina.py -l LIGAND -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y -cz CENTER_Z
```

- Ligand input from SMILES

```
python meeko-vina.py --input_smiles INPUT_SMILES -r RECEPTOR -o OUTPUT -cx CENTER_X -cy CENTER_Y -cz CENTER_Z
```

### Optional arguments

```
  -h, --help            show this help message and exit
  -l LIGAND, --ligand LIGAND
                        ligand (PDBQT, MOL, SDF, MOL2, PDB)
  -r RECEPTOR, --receptor RECEPTOR
                        rigid part of the receptor (PDBQT)
  -o OUT, --out OUT     output models (PDBQT), the default is chosen based on the ligand file name
  --input_smiles INPUT_SMILES
                        SMILES string (Need to put the atom you want to extend at the end of the string)
  --smi SMI             ligand (PDBQT, MOL, SDF, MOL2, PDB)
  -cx CENTER_X, --center_x CENTER_X
                        X coordinate of the center
  -cy CENTER_Y, --center_y CENTER_Y
                        Y coordinate of the center
  -cz CENTER_Z, --center_z CENTER_Z
                        Z coordinate of the center
  -sx SIZE_X, --size_x SIZE_X
                        size in the X dimension (Angstroms)
  -sy SIZE_Y, --size_y SIZE_Y
                        size in the Y dimension (Angstroms)
  -sz SIZE_Z, --size_z SIZE_Z
                        size in the Z dimension (Angstroms)
  -c CPU, --cpu CPU     the number of CPUs to use (the default is to try todetect the number of CPUs or, failing that, use 1)
  --scoring SCORING     force field name: vina(default), ad4, vinardo
  --seed SEED           explicit random seed
  --exhaustiveness EXHAUSTIVENESS
                        exhaustiveness of the global search(roughly proportional to time): 1+
  --num_modes NUM_MODES
                        maximum number of binding modes to generate
  --energy_range ENERGY_RANGE
                        maximum energy difference between the best bindingmode and the worst one displayed (kcal/mol)
  -d, --debug           debug mode
```

## License

This package is distributed under the MIT License.

## Contact

- Michio Katouda (katouda@rist.or.jp)
