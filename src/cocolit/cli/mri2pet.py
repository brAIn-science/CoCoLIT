import os
import argparse
import warnings
warnings.filterwarnings("ignore")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True, help='Path to the T1w MRI')
    parser.add_argument('--o', type=str, required=True, help='Where to store the predicted amyloid SUVR map')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--m',      type=int, default=1,  help='LAS hyperparameter')
    parser.add_argument('--ddim_n', type=int, default=50, help='Number of denoising steps in DDIM sampling')
    
    args = parser.parse_args()

    if not os.path.exists(args.i):
        print('Input not found.')
        exit(1)

    # Load libraries only when the input is validated.
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Console
    from huggingface_hub import hf_hub_download
    from monai.networks.schedulers.ddim import DDIMScheduler
    from monai import transforms
    from cocolit import networks
    from cocolit.utils import convert_to_suvr, save_suvr
    from cocolit.sampling import CondDistributionSampler
    from cocolit.defaults import (
        DEFAULT_ZSCORES_PARAMS,
        DEFAULT_DIFFSCHED_ARGS
    )
    
    citation_text = Text.from_markup(
        f"""
    [white]If you use this work for your research, please cite:[/white]

    [bold yellow]"CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis."[/bold yellow]
    [i][bold]Alec Sargood*, Lemuel Puglisi*[/bold], James Cole, Neil Oxtoby, Daniele Ravì, Daniel C. Alexander[/i]
    [bold]*Joint first authors[/bold].
    """,
        justify="left"
    )

    console = Console()
    console.print(
        Panel(
            citation_text,
            title="[bold #FF6AC1]CoCoLIT[/bold #FF6AC1]", # Using a hex color
            border_style="blue",
            padding=(1, 3)
        )
    )

    # Load the input MRI
    smri_mean, smri_std = DEFAULT_ZSCORES_PARAMS['smri_mean'], DEFAULT_ZSCORES_PARAMS['smri_std']
    
    loading_transforms = transforms.Compose([
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=1.5),
        transforms.DivisiblePad(k=8),
        transforms.NormalizeIntensity(subtrahend=smri_mean, divisor=smri_std)
    ])
    
    with console.status("> Pulling the models from huggingface...") as status:
        smri_vae_ckpt   = hf_hub_download(repo_id='lemuelpuglisi/CoCoLIT', filename='src.pth')
        suvr_vae_ckpt   = hf_hub_download(repo_id='lemuelpuglisi/CoCoLIT', filename='tgt.pth')
        diffusion_ckpt  = hf_hub_download(repo_id='lemuelpuglisi/CoCoLIT', filename='dif.pth')
        controlnet_ckpt = hf_hub_download(repo_id='lemuelpuglisi/CoCoLIT', filename='cnt.pth')
    
    with console.status("> Loading the models...") as status:
        smri_vae   = networks.create_maisi_vae(checkpoint=smri_vae_ckpt, device=args.device).eval()
        suvr_vae   = networks.create_maisi_vae(checkpoint=suvr_vae_ckpt, device=args.device).eval()
        diffusion  = networks.create_diffusion(checkpoint=diffusion_ckpt, device=args.device).eval()
        controlnet = networks.create_controlnet(checkpoint=controlnet_ckpt, device=args.device).eval()
    
    console.print("> Models loaded.")

    noise_scheduler = DDIMScheduler(**DEFAULT_DIFFSCHED_ARGS)
    noise_scheduler.set_timesteps(num_inference_steps=args.ddim_n, device=args.device)
    
    input_mri = loading_transforms(args.i)
    
    sampler = CondDistributionSampler(
        source_vae=smri_vae,
        target_vae=suvr_vae,
        udiffusion=diffusion,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
    )

    console.print(f"🧠 [bold] Predicting the amyloid-PET.")    
    suvr = sampler.sample(input_mri, device=args.device, las_m=args.m)
    suvr = convert_to_suvr(suvr)
    save_suvr(suvr, args.i, args.o)
    console.print(f"✅ [bold green]Predicted SUVR map saved in {args.o}")


    console.print(f"ℹ️ [bold cyan] CoCoLIT's threshold for amyloid positivity is SUVR > 1.326.")
    console.print(f"❌ [bold yellow]This tool is not intended for clinical or commercial use. Its output is for research purposes only and should not be used to inform clinical decisions.")