<div align="center">

# CoCoLIT (AAAI-26)

<a><img src='https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white' alt='PyTorch'></a>
<a><img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow?style=for-the-badge&logo=huggingface" alt="Hugging Face Model"></a>
<a><img src='https://img.shields.io/github/license/brAIn-science/CoCoLIT?style=for-the-badge' alt='PyTorch'></a>


<strong>CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis</strong><br>
<a href="https://scholar.google.com/citations?user=9kuYeWcAAAAJ&hl=it&oi=ao">Alec Sargood</a><sup>*</sup>, 
<a href="https://lemuelpuglisi.github.io/">Lemuel Puglisi</a><sup>*</sup>, 
<a href="https://profiles.ucl.ac.uk/32379-james-cole">James Cole</a>, 
<a href="https://neiloxtoby.com/science/">Neil Oxtoby</a>, 
<a href="https://daniravi.wixsite.com/researchblog">Daniele Ravì</a><sup>†</sup>, 
<a href="https://profiles.ucl.ac.uk/3589-daniel-alexander">Daniel C. Alexander</a><sup>†</sup><br>
<small>* <i>Joint first authors</i></small>,
<small>† <i>Joint senior authors</i></small>
</p>


</div>

![](docs/assets/preview.gif)

<div align="center">
  <a href="#installation" style="margin: 0 15px;">Installation</a> •
  <a href="#usage" style="margin: 0 15px;">Usage</a> •
  <a href="#training--reproducibility" style="margin: 0 15px;">Training & Reproducibility</a> •
  <a href="#disclaimer" style="margin: 0 15px;">Disclaimer</a> •
  <a href="#citing" style="margin: 0 15px;">Citing</a>
</div>

## Installation

This repository requires Python 3.10 and PyTorch 2.0 or later. To install the latest version, run:

```bash
pip install cocolit
```

## Usage

After installing the package, you can convert a T1-weighted MRI to a Florbetapir SUVR map by running:

```bash
mri2pet --i /path/to/t1.nii.gz --o /path/to/output.nii.gz
```

To replicate the results presented in the paper, include the `--m 64` flag.

## Training & Reproducibility

To reproduce the experiments reported in the paper, please follow the [reproducibility guide](./docs/reproducibility.md).

## Disclaimer

This software is not intended for clinical use. The code is not available for commercial applications. For commercial inquiries, please contact the corresponding authors.

## Citing

Arxiv Preprint:

```bib
@article{sargood2025cocolit,
  title={CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis},
  author={Sargood, Alec and Puglisi, Lemuel and Cole, James H and Oxtoby, Neil P and Rav{\`\i}, Daniele and Alexander, Daniel C},
  journal={arXiv preprint arXiv:2508.01292},
  year={2025}
}
```