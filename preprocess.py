import os
import subprocess
import multiprocessing as mp
from pathlib import Path
import ants
import torchio as tio
import numpy as np
import config as cfg

# ============================================================
# Globals
# ============================================================

fixed = ants.image_read(cfg.template_path)

# ============================================================
# Stage 1: DICOM → NIfTI
# ============================================================

def dcm_to_nifti(dcm_dir):
    try:
        dcm_dir = Path(dcm_dir)
        out_dir = Path(cfg.input_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_nii = out_dir / f"{dcm_dir.name}.nii.gz"

        if out_nii.exists():
            print(f"[SKIP][DCM] {out_nii.name}")
            return out_nii.name

        print(f"[INFO][DCM] {dcm_dir.name}")
        cmd = [
            cfg.DCM2NIIX,
            "-z", "y",
            "-f", dcm_dir.name,
            "-o", str(out_dir),
            str(dcm_dir)
        ]
        subprocess.run(cmd, check=True)
        return out_nii.name

    except Exception as e:
        print(f"[ERROR][DCM] {dcm_dir}: {e}")


# ============================================================
# Stage 2: N4 Bias Correction
# ============================================================

def bias_correct(filename):
    try:
        in_path  = Path(cfg.input_folder) / filename
        out_name = filename.replace(".nii.gz", "_n4.nii.gz")
        out_path = Path(cfg.n4_folder) / out_name

        if out_path.exists():
            print(f"[SKIP][N4] {out_name}")
            return out_name

        print(f"[INFO][N4] {filename}")
        img = ants.image_read(str(in_path))
        corrected = ants.n4_bias_field_correction(img)
        ants.image_write(corrected, str(out_path))
        return out_name

    except Exception as e:
        print(f"[ERROR][N4] {filename}: {e}")


# ============================================================
# Stage 3: Registration
# ============================================================

def register(n4_name):
    try:
        if not n4_name:
            return

        in_path  = Path(cfg.n4_folder) / n4_name
        out_name = n4_name.replace("_n4.nii.gz", "_registered.nii.gz")
        out_path = Path(cfg.reg_folder) / out_name

        if out_path.exists():
            print(f"[SKIP][REG] {out_name}")
            return out_name

        print(f"[INFO][REG] {n4_name}")
        moving = ants.image_read(str(in_path))
        reg = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=cfg.reg_type
        )
        ants.image_write(reg["warpedmovout"], str(out_path))
        return out_name

    except Exception as e:
        print(f"[ERROR][REG] {n4_name}: {e}")


# ============================================================
# Stage 4: Deskulling
# ============================================================

def deskull(reg_name, gpu_id=0):
    try:
        if not reg_name:
            return

        in_path  = Path(cfg.reg_folder) / reg_name
        out_name = reg_name.replace("_registered.nii.gz", "_deskulled.nii.gz")
        out_path = Path(cfg.deskull_folder) / out_name

        if out_path.exists():
            print(f"[SKIP][BET] {out_name}")
            return out_name

        print(f"[INFO][BET] {reg_name} (GPU {gpu_id})")
        cmd = f'hd-bet -i "{in_path}" -o "{out_path}" -device cuda:{gpu_id}'
        subprocess.run(cmd, shell=True, check=True)
        return out_name

    except Exception as e:
        print(f"[ERROR][BET] {reg_name}: {e}")


# ============================================================
# Stage 5: NIfTI → NPY (TorchIO)
# ============================================================

def nifti_to_npy(nii_name):
    try:
        nii_path = Path(cfg.deskull_folder) / nii_name
        npy_path = Path(cfg.npy_folder) / nii_name.replace("_deskulled.nii.gz", ".npy")

        if npy_path.exists():
            print(f"[SKIP][NPY] {npy_path.name}")
            return npy_path.name

        transforms = tio.Compose([
            tio.Resample((1, 1, 1)),
            tio.CropOrPad((cfg.crop_size, cfg.crop_size, cfg.crop_size)),
            tio.Resize((cfg.img_size, cfg.img_size, cfg.img_size)),
            tio.ZNormalization()
        ])

        subject = tio.Subject(img=tio.ScalarImage(str(nii_path)))
        subject = transforms(subject)

        data = subject.img.data.squeeze(0).numpy()
        np.save(npy_path, data)

        print(f"[SAVE][NPY] {npy_path.name}")
        return npy_path.name

    except Exception as e:
        print(f"[ERROR][NPY] {nii_name}: {e}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Create folders
    for folder in [
        cfg.dcm_folder,
        cfg.input_folder,
        cfg.n4_folder,
        cfg.reg_folder,
        cfg.deskull_folder,
        cfg.npy_folder,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # ---- DICOM → NIfTI ----
    print("\n=== DICOM → NIfTI ===")
    dcm_dirs = [
        Path(cfg.dcm_folder) / d
        for d in os.listdir(cfg.dcm_folder)
        if (Path(cfg.dcm_folder) / d).is_dir()
    ]
    print(f"Total DICOM dirs: {len(dcm_dirs)}")

    with mp.Pool(4) as pool:
        nii_files = list(filter(None, pool.map(dcm_to_nifti, dcm_dirs)))
    print(f"\n[DONE] Total NIfTIs: {len(nii_files)}")

    # ---- N4 ----
    print("\n=== N4 Bias Correction ===")
    with mp.Pool(4) as pool:
        n4_files = list(filter(None, pool.map(bias_correct, nii_files)))
    print(f"\n[DONE] Total N4 Corrected: {len(n4_files)}")

    # ---- Registration ----
    print("\n=== Registration ===")
    with mp.Pool(4) as pool:
        reg_files = list(filter(None, pool.map(register, n4_files)))
    print(f"\n[DONE] Total Registered: {len(reg_files)}")

    # ---- Deskulling ----
    print("\n=== Deskulling ===")
    deskulled_files = []
    for f in reg_files:
        out = deskull(f, gpu_id=0)
        if out:
            deskulled_files.append(out)
    print(f"\n[DONE] Total Deskulled: {len(deskulled_files)}")

    # ---- TorchIO → NPY ----
    print("\n=== TorchIO → NPY ===")
    for f in deskulled_files:
        nifti_to_npy(f)

    print(f"\n[DONE] Total NPYS: {len(os.listdir(cfg.npy_folder))}")
