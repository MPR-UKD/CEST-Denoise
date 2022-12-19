import math
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CEST import CEST
from Loader.src.dcm_transformer import dcm_to_nii
from Loader.src.loader import LoaderCEST, load_nii
from PCA import pca


def overlay_img(S0, mtr_asym):
    fig, ax = plt.subplots()
    plt.imshow(np.rot90(S0, 3), cmap='gray')
    plt.imshow(np.rot90(mtr_asym * 2, 3), alpha=0.7, cmap='jet', vmin=0, vmax=5)
    cbar = plt.colorbar()
    cbar.set_label('MTR$_{asym}$ [%]', fontsize=15)
    plt.axis('off')
    plt.close()
    return fig


def calc_mtr_asym(CestCurveS, x_calcentires, mask, d_shift, f_shift, hStep):
    (rows, colums, entires) = CestCurveS.shape
    entry_mtr = math.ceil(entires / 2)

    mtr_asym_curves = np.ones((rows, colums, entry_mtr), dtype=float) * np.nan
    mtr_asym_img = np.ones((rows, colums), dtype=float) * np.nan
    for i in range(rows):
            for j in range(colums):
                if mask[i, j] != 0:
                    mtr_asym_curves[i, j, :], mtr_asym_img[i, j] = \
                        calc_mtr_asym_pixel(entry_mtr, CestCurveS[i, j, :], d_shift, f_shift,
                                            x_calcentires, hStep)
    return mtr_asym_curves, mtr_asym_img


def calc_mtr_asym_pixel(entry_mtr, values, d_shift, f_shift, x_calcentires, hStep):
    mtra = np.flip(values[:entry_mtr]) - values[entry_mtr - 1:]
    mtra = mtra * 100

    range = (f_shift - d_shift / 2, f_shift + d_shift / 2)
    x_mrt_calcentries = x_calcentires[entry_mtr - 1:]
    vind1 = np.argmin(abs(x_mrt_calcentries - range[0]))
    vind2 = np.argmin(abs(x_mrt_calcentries - range[1]))

    asym = mtra[vind1: vind2 + 1]
    asym_value = np.sum(asym) * hStep
    return mtra, asym_value


if __name__ == '__main__':
    temp_dir = tempfile.TemporaryDirectory()

    FF_WASSR = Path(r'D:\Scibo\laufende_Kooperationen\DFG_Antrag_Anja_qMRI\in_vivo5\20_WASSR_96216')
    shutil.copy(FF_WASSR / "mask_D.nii.gz", Path(temp_dir.name) / 'mask.nii.gz')
    FF_WASSR = Path(r'D:\Scibo\laufende_Kooperationen\DFG_Antrag_Anja_qMRI\in_vivo5\18_N40PD100B1_09_95700')
    nii_file = Path(temp_dir.name) / 'image.nii.gz'
    dcm_to_nii(FF_WASSR, nii_file)
    loader = LoaderCEST(nii_file)
    wassr, mask, _ = loader.__getitem__(0)

    FF_CEST = Path(r'D:\Scibo\laufende_Kooperationen\DFG_Antrag_Anja_qMRI\in_vivo5\18_N40PD100B1_09_95700')
    dcm_to_nii(FF_CEST, nii_file)
    loader = LoaderCEST(nii_file)
    cest, _, _ = loader.__getitem__(0)

    S0 = load_nii(nii_file)[:, :, 0]

    hstep = 0.05
    #img = nlm_CEST(cest, 12, 8, True)
    cest = CEST.CEST(
            cest=cest,
            wassr=wassr,
            mask=mask[:, :, 0],
            cest_range=5,
            wassr_range=5,
            itp_step=hstep,
            max_wassr_offset=2)
    cest, ppm = cest.run()
    mtr_asym_curves, mtr_asym_img = calc_mtr_asym(cest, ppm, mask, d_shift=1, f_shift=1, hStep=hstep)
    fig = overlay_img(S0, mtr_asym_img)
    fig.savefig(fr'raw.png')

    #cest = nlm_CEST(cest, 12, 8, True)
    for crit in ['malinowski', 'nelson', 'median']:
        cest_pca = pca(cest.copy(), crit, mask[:, :, 0])
        mtr_asym_curves, mtr_asym_img = calc_mtr_asym(cest_pca, ppm, mask, d_shift=1, f_shift=1, hStep=hstep)
        fig = overlay_img(S0, mtr_asym_img)
        fig.savefig(fr'{crit}.png')

    cest_pca = pca(cest.copy(), 11, mask[:, :, 0])
    mtr_asym_curves, mtr_asym_img = calc_mtr_asym(cest_pca, ppm, mask, d_shift=1, f_shift=1, hStep=hstep)
    fig = overlay_img(S0, mtr_asym_img)
    fig.savefig(fr'pca_11.png')

