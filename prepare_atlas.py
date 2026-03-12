#! /usr/bin/env python

import nilearn


import os
import bids
import argparse
import torchio as tio
import numpy as np
import nibabel as nib
import torch 
import matplotlib.pyplot as plt
from rich.progress import track
from os.path import basename, splitext, join, exists
import logging
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import torch

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI to .npy with TorchIO preprocessing")
    parser.add_argument('--img_size', type=int, default=180, help='Image size for cropping/padding (default: 180)')
    parser.add_argument('--cohort', type=str, required=True, help='Cohort name (e.g., ukb, ppmi)')
    parser.add_argument('--crop_size', type=int, default=180, help='Crop size before resizing (default: 180)')
    parser.add_argument('--input_folder', type=str, default=None, help='Path to input .nii.gz files')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to save .npy files')
    parser.add_argument('--bids', default=False, action='store_true', help='Find Niifti-Files in bids compatible dataset')
    parser.add_argument('--subject', type=str, default=None, help='Query for subjectID, only used with bids')
    parser.add_argument('--session', type=str, default=None, help='Query for session, only used with bids')
    parser.add_argument('--space', type=str, default=None, help='Query for files in space, only used with bids')
    parser.add_argument('--label', type=str, default=None, help='Query for files with label, only used with bids')
    parser.add_argument('--suffix', type=str, default='T1w', help='Query for files with suffix, only used with bids, defaults to T1w')
    parser.add_argument('--extension', type=str, default='nii.gz', help='Query for files with fileextension, only used with bids, defaults to nii.gz')

    
    return parser.parse_args()

def transform_and_save_npy(nii_path, output_path, crop, norm):
    img = nib.load(nii_path)
    data = img.get_fdata()
    tensor_data = torch.tensor(data).unsqueeze(0)
    crop_data = crop(tensor_data)
    norm_data = norm(crop_data).squeeze(0)
    np.save(output_path, norm_data)

def process_bids_dir(bids_dir, query, npy_folder, transforms, derivatives=True):
    """
    Process files that reside in a BIDS-compatible dir. For the query, use one or more of the keys:
    subject, session, space, label, suffix, extension. 

    If no extension is given  it is set to 'nii.gz' files

    e.g. 
       { 
           "space": "mni", 
           "suffix": "T1w"
       }


    Parameters
    ---------
    bids_dir: str
        Root of the bids-dir
    query: dict
        query to find the files, e.g. `{"session": "001"}`
    derivatives: bool
        use derivatives in the bids-structure to find files, as well
    """

    log.info(f"Querying bids dataset in {bids_dir}")
    layout = bids.BIDSLayout(bids_dir, derivatives=derivatives)
    if not 'extension' in query.keys():
        query['extension'] = 'nii.gz'  
   
    log.info(f'Using query {query}')
    files = layout.get(**query, return_type='filename')
    
    log.debug(f'found {len(files)} files matching the query. ')

    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[current]}"),
            TimeRemainingColumn(),
        ) as progress:

        task = progress.add_task("Transform files", total=len(files), current="")

        for f in files:
            progress.update(task, advance=1, current=str(f), description=f"Processing {f}")

            out_path = join(npy_folder, splitext(basename(f))[0]+".npy")
            
            if not exists(out_path):
                transform_and_save_npy(f, out_path, transforms)
            else:
                log.warning(f'{out_path} already exists - skipping')



def transform_and_save_npy(nii_path, output_path, transforms):
    subject = tio.Subject(img=tio.ScalarImage(nii_path))
    subject = transforms(subject)
    data = subject.img.data.squeeze(0).numpy()  # Remove channel dimension
    np.save(output_path, data)

def process_nifti_files(root_dir, npy_folder, transforms):
    nii_files = [f for f in os.listdir(root_dir) if f.endswith('_deskulled.nii.gz')]
    for nii_file in nii_files:
        nii_path = os.path.join(root_dir, nii_file)
        npy_file = nii_file.replace('_deskulled.nii.gz', '') + '.npy'
        output_path = os.path.join(npy_folder, npy_file)

        if os.path.exists(output_path):
            print(f"Skipping {npy_file}, already exists.")
            continue

        transform_and_save_npy(nii_path, output_path, transforms)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    args = parse_args()

    input_folder = args.input_folder or f'/mnt/bulk-neptune/radhika/project/images/{args.cohort}/nifti_deskull/'
    output_folder = args.output_folder or f'/mnt/bulk-neptune/radhika/project/images/{args.cohort}/npy{args.img_size}/'
    os.makedirs(output_folder, exist_ok=True)

    # Full transform pipeline
    transforms = tio.Compose([
        tio.Resample((1, 1, 1)),  # Resample to 1mm isotropic
        tio.CropOrPad((args.crop_size, args.crop_size, args.crop_size)),  # Crop/Pad to 180³
        tio.Resize((args.img_size, args.img_size, args.img_size)),  # Downscale to 96³
        tio.ZNormalization()  # Normalize intensity
    ])

    # Process and save files
    if args.bids:
        query = {}
        bids_fields = ['subject', 'session', 'space', 'label', 'suffix', 'extension']
        for f in bids_fields:
            val = getattr(args, f, None)
            if val is not None:
                query[f] = val

        process_bids_dir(input_folder, query, npy_folder=output_folder, transforms=transforms )
    else:
        process_nifti_files(input_folder, output_folder, transforms )
    # Process
    # process_nifti_files(input_folder, output_folder, transforms)

    # Report
    npy_count = len([f for f in os.listdir(output_folder) if f.endswith('.npy')])
    print(f"Total .npy files in {output_folder}: {npy_count}")


