import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, required=True, help='Path to the T1w MRI')
    parser.add_argument('--o', type=str, required=True, help='Where to store the predicted amyloid SUVR map')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--m',      type=int, default=1,  help='LAS hyperparameter')
    parser.add_argument('--ddim_n', type=int, default=50, help='Number of denoising steps in DDIM sampling')
    
    args = parser.parse_args()

    exit('Model weights will be made publicly available upon acceptance. Until then, the CLI app cannot be used.')