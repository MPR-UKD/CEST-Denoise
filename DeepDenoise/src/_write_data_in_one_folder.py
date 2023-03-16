from pathlib import Path
import shutil

if __name__ == "__main__":
    in_dir = Path(r"F:\CEST_Data")
    out_dir = Path(r"D:\Scibo\Promotion\__7_Denoise_CEST\Sim_data")

    count = 1
    for sim in in_dir.glob("*"):
        c = 1
        for img in sim.glob("*.nii"):
            shutil.copy(img, out_dir / f"{str(count).zfill(5)}.nii")
            if (sim / f"{c}").exists():
                shutil.copytree(sim / f"{c}", out_dir / f"{str(count).zfill(5)}")
            count += 1
            c += 1
