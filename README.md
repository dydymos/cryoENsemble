# cryoENsemble

cryoENsemble [1] is a Bayesian reweighting method that refines conformational ensembles derived from molecular dynamics (MD) simulations to better match cryo-EM maps.

### Prerequisites:
Important: cryoENsemble requires an MD trajectory or structural ensemble that has already been fitted into the reference cryo-EM map. It does not perform this fitting automatically.

In our original publication, we used Situs for the fitting step: https://situs.biomachina.org/index.html

~/soft/Situs_3.2/bin/colores reference_map.mrc protmasses.pdb -res 8

Note: Situs does not support .xtc files. You’ll need to convert your trajectory into a series of .pdb files beforehand. Alternatively, other tools like ChimeraX may also be used for this purpose.

### Running cryoENsemble (standard mode)
python cryoENsemble.py --map reference_map.mrc --ref_pdb TF.pdb --ref_traj TF_100.xtc --threshold 0.005 --mask exp --resolution 4.1

Argument descriptions:\
--map — Experimental cryo-EM map in .mrc format\
--ref_pdb — Reference structure in .pdb format\
--ref_traj — MD trajectory (e.g., .xtc file)\
--threshold — Map threshold appropriate for your system\
--mask — Masking strategy:\
    exp: Uses only voxels from the experimental map\
    sim: Uses voxels from both experimental and simulated maps (see Suppl. Fig. 3)\
--resolution — Resolution of the map in Ångströms.\

### Running cryoENsemble (iterative mode)
cryoENsemble can also be run in iterative mode, which performs repeated reweighting and structure selection to reduce the ensemble to a minimal, best-fitting set.

python cryoENsemble.py --map reference_map.mrc --ref_pdb TF.pdb --ref_traj TF_100.xtc --threshold 0.005 --mask exp --resolution 4.1 --iterative

References:\
[1] Włodarski, T., Streit, J. O., Mitropoulou, A., Cabrita, L. D., Vendruscolo M., Christodoulou, J. Bayesian reweighting of biomolecular structural ensembles using heterogeneous cryo-EM maps with the cryoENsemble method. (2024) Sci. Rep., doi:10.1038/s41598-024-68468-7
