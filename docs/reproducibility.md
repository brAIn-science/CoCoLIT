# Reproducibility Guide: CoCoLIT

This document provides a step-by-step guide to reproduce the experiments and results presented in the paper *"CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis."* It is not intended for beginners; prior experience in medical image analysis, as well as practical programming and deep learning (e.g., using Python and PyTorch), is assumed.

## 1. Data Gathering

Request access to both the ADNI and A4 datasets through the [IDA portal](https://ida.loni.usc.edu/login.jsp). Download the MPRAGE MRIs and FBP PET scans from the portal. Since the data are not organized for deep learning tasks, please contact the corresponding authors with proof of access to obtain the exact image UIDs and dataset splits used in the study. For data preprocessing of both the MRI and FBP PET, refer to the Supplementary Material of the paper. As the preprocessing involves converting the PET scans to SUVR maps, we will refer to these maps as *SUVR* from this point forward.

## 2. Data Organization

Your data must be listed in a CSV file with the following mandatory fields:

* `split` – specifies whether the entry belongs to *train*, *valid*, or *test*
* `scan_path` – path to the T1w MRI
* `suvr_path` – path to the corresponding SUVR map
* `scan_latent_path` – path to the latent representation extracted from the T1w MRI (*leave empty for now*)
* `suvr_latent_path` – path to the latent representation extracted from the SUVR map (*leave empty for now*)

You may add additional fields as needed, but these five columns must be present in the CSV file.

## 3. Training the VAEs

To train the MRI and PET VAEs, use the training script located at `scripts/train_vae.py`. This script requires a single argument, `--config`, which should point to a configuration file. We have provided two sample configuration files in the `configs` directory that require minimal modification to get started: `mri_vae_training.yaml` and `pet_vae_training.yaml`.

Before running the training, open these YAML files and replace all `"<ADD HERE>"` placeholders with the appropriate values:

* `output_dir` – path to the folder where training outputs will be saved
* `wandb.entity` – your Weights and Biases (WandB) entity
* `wandb.project` – your WandB project name
* `dataloader.dataset_csv` – path to the CSV file you created earlier

Once these values are filled in, you can launch training with:

```bash
python scripts/train_vae.py --config configs/mri_vae_training.yaml
```

and

```bash
python scripts/train_vae.py --config configs/pet_vae_training.yaml
```

After both VAEs have been successfully trained, you'll need to generate latent representations for each MRI and SUVR scan and save them as `.npz` files. You can create a script similar to the example below:

```python
mri_vae = ...  # load trained MRI VAE
pet_vae = ...  # load trained PET VAE
mri_vae.eval()
pet_vae.eval()

with torch.inference_mode():
    # assuming MRI and PET batches have shape (B x 1 x H x W x D)
    for mri, pet in paired_data:
        batch_size = mri.shape[0]
        
        with autocast(device_type='cuda', dtype=torch.float32, enabled=True):
            mri_latent = mri_vae(mri)[1] 
            pet_latent = pet_vae(pet)[1]
        
        # save each latent using a unique ID in the filename
        for i in range(batch_size):
            # choose your own way to store these latents
            mri_z_path = "path/to/save/mri_latent_<ID>.npz"
            pet_z_path = "path/to/save/pet_latent_<ID>.npz"
            
            np.savez(mri_z_path, data=mri_latent[i].cpu().numpy())
            np.savez(pet_z_path, data=pet_latent[i].cpu().numpy())
```

**IMPORTANT:** After generating and saving the latent files, update your CSV file by filling in the `scan_latent_path` and `suvr_latent_path` columns with the correct file paths used in the script above.

## 4. Training the Latent Diffusion Model (LDM)

To train the Latent Diffusion Model (LDM), use the training script located at `scripts/train_ldm.py`. Like the VAE training script, it requires a `--config` argument pointing to a configuration file. A template configuration file, `ldm_training.yaml`, is provided in the `configs` directory and requires minimal changes.

Edit `ldm_training.yaml` and replace the following placeholders:

* `output_dir` – path to the folder where training outputs will be saved
* `wandb.entity` – your WandB entity
* `wandb.project` – your WandB project name
* `dataloader.dataset_csv` – path to the CSV file containing the latent paths
* `vae_checkpoints` – path to the trained PET VAE checkpoint (used to decode PET latents during evaluation)

Then run:

```bash
python scripts/train_ldm.py --config configs/ldm_training.yaml
```

## 5. Training the ControlNet

The final step is to train the ControlNet and fine-tune the decoder of the PET VAE. Use the script `scripts/train_m2p.py` along with the configuration file `controlnet_training.yaml`.

As before, pass the configuration file using the `--config` argument:

```bash
python scripts/train_m2p.py --config configs/controlnet_training.yaml
```

Edit `controlnet_training.yaml` to replace all `"<ADD HERE>"` placeholders with your specific values:

* `output_dir` – path to the folder where training outputs will be saved
* `wandb.entity` – your WandB entity
* `wandb.project` – your WandB project name
* `dataloader.dataset_csv` – path to the CSV file containing the latent paths
* `vae_checkpoints` – path to the trained PET VAE checkpoint
* `unet_checkpoints` – path to the trained LDM (UNet) checkpoint

Ensure your CSV contains valid entries for both `scan_latent_path` and `suvr_latent_path`, as ControlNet uses these latents during training.

Once configured, running the command above will begin the final training step.