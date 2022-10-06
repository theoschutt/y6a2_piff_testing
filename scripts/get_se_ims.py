import fitsio
import numpy as np
import os
import tqdm

d = fitsio.read("/global/project/projectdirs/des/myamamot/pizza-slice/pizza-cutter-coadds-info.fits")

images_used = set()
for i in tqdm.trange(len(d)):
    fname = os.path.join(
        "/global/project/projectdirs/des/myamamot/pizza-slice",
        d["PATH"][i],
        d["FILENAME"][i] + d["COMPRESSION"][i],
    )
    if os.path.exists(fname):
        ii = fitsio.read(fname, ext="image_info")
        ei = fitsio.read(fname, ext="epochs_info")
        msk = ei["flags"] == 0
        uiids = np.array(list(set([iid for iid in ei["image_id"][msk]])))
        assert np.all(ii["image_id"][uiids] == uiids)
        images_used |= set(list(ii["image_path"][uiids]))

with open('y6a2_mdet_run1_se_ims_in_coadds.txt', 'w') as f:
    for impath in images_used:
        f.write(impath + '\n')
