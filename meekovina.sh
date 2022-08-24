#!/bin/bash

ligand=inputs/1iep_ligand.sdf
#ligand=inputs/1iep_ligand.mol
#ligand=inputs/1iep_ligand.pdbqt
ligand_smi=Oc1ccccc1
receptor=inputs/1iep_receptorH.pdbqt
out=1iep_ligand_out.pdbqt
center_x=15.190
center_y=53.903
center_z=16.917
size_x=20.0
size_y=20.0
size_z=20.0
cpu=8
exhaustiveness=8
num_modes=9
seed=1234

meekovina -l ${ligand} -r ${receptor} \
    --center_x ${center_x} --center_y ${center_y} --center_z ${center_z} \
    --size_x ${size_x} --size_y ${size_y} --size_z ${size_z} \
    --cpu ${cpu} --exhaustiveness ${exhaustiveness} --num_modes=${num_modes} \
    --seed ${seed} \
    --out ${out} \
    -d

#meekovina --input_smiles \"${ligand_smi}\" -r ${receptor} \
#    --center_x ${center_x} --center_y ${center_y} --center_z ${center_z} \
#    --size_x ${size_x} --size_y ${size_y} --size_z ${size_z} \
#    --cpu ${cpu} --exhaustiveness ${exhaustiveness} --num_modes=${num_modes} \
#    --seed ${seed} \
#    --out ${out} \
#    -d

exit 0
