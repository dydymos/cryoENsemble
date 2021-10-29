import numpy as np
import MDAnalysis
from MDAnalysis.analysis import density


def pdb2map_array(PDBs,sigma):
    """
    Generates an array with calculated EM density maps for each structure
    """
    # prepare zero data
    data_array = []
    for ipdb in range(0,len(PDBs)):
        print(ipdb)
        data = np.zeros((nz,ny,nx), dtype=np.float32)
        u = MDAnalysis.Universe(PDBs[ipdb])
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S")
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX_))
            jj = int(round((apos[1]-em_origin[1])/VOX_))
            kk = int(round((apos[2]-em_origin[2])/VOX_))
            # get delta grid
            d=delta[atype]
            # get constant parameters
            invsig2 = INVSIG2[atype]
            pref =  PREFACT_[atype]
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX_ - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX_ - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX_ - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 / (sigma**2))
        data_array.append(data)
    return data_array



def pdb2map_avg(weights,sigma,PDBs):
    """
    Generates average EM maps based on structural ensemble and weights
    """
    # prepare zero data
    data = np.zeros((nz,ny,nx), dtype=np.float32)
    for ipdb in range(0,len(PDBs)):
        u = MDAnalysis.Universe(PDBs[ipdb])
        # get normalized weight
        w = weights[ipdb]
        # all-heavy selectors
        allheavy=u.select_atoms("type C O N S")
        # now cycle on all the atoms
        for atom in allheavy:
            # get atom type
            atype = atom.type
            # get atom position
            apos = atom.position
            # get indexes in the map
            ii = int(round((apos[0]-em_origin[0])/VOX_))
            jj = int(round((apos[1]-em_origin[1])/VOX_))
            kk = int(round((apos[2]-em_origin[2])/VOX_))
            # get delta grid
            d=delta[atype]
            # get constant parameters
            invsig2 = INVSIG2[atype]
            pref = w * PREFACT_[atype]
            # add contribution to map
            for k in range(max(0, kk-d), min(kk+d+1, nz)):
                distz = (em_origin[2] + float(k) * VOX_ - apos[2])**2
                for j in range(max(0, jj-d), min(jj+d+1, ny)):
                    disty = (em_origin[1] + float(j) * VOX_ - apos[1])**2
                    for i in range(max(0, ii-d), min(ii+d+1, nx)):
                        # get distance squared
                        dist = (em_origin[0] + float(i) * VOX_ - apos[0])**2 + disty + distz
                        # add contribution
                        data[k][j][i] += pref * np.exp(-0.5 * dist * invsig2 / (sigma**2))
    return data
