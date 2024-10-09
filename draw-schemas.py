import argparse
import numpy as np

from drawing import Draw
from pathlib import Path
from utils import IsValidFile, CreateFolder


def bivariate_gaussian(x, y, mu=(0, 0), sig=(1, 1)):
    return np.exp(-((x - mu[0])**2 / (2 * sig[0]**2) + (y - mu[1])**2 / (2 * sig[1]**2)))


def wrapped_bivariate_gaussian(x, y, mu=(0, 0), sig=(1, 1)):
    img = bivariate_gaussian(x, y, mu=mu, sig=sig)
    img += bivariate_gaussian(x, y - 2, mu=mu, sig=sig)
    img += bivariate_gaussian(x, y + 2, mu=mu, sig=sig)
    return img


def create_random_image():
    x = np.linspace(-1, 1, 14)
    y = np.linspace(-1, 1, 18)
    xx, yy = np.meshgrid(x, y)

    img = wrapped_bivariate_gaussian(xx, yy, mu=(-0.1, -0.8), sig=(0.1, 0.1))
    img += 0.3*wrapped_bivariate_gaussian(xx, yy, mu=(0.3, 0.15), sig=(0.2, 0.2))
    img += 0.7*wrapped_bivariate_gaussian(xx, yy, mu=(0.63, 0.47), sig=(0.15, 0.2))
    img += 0.5*wrapped_bivariate_gaussian(xx, yy, mu=(-0.7, 0.72), sig=(0.2, 0.1))
    
    img += 0.4 * np.random.rand(*img.shape)

    return img



def main(args=None) -> None:

    draw = Draw(output_dir=args.output, interactive=args.interactive)
    depos = create_random_image()
    # depos = depos.reshape((1, 18, 14, 1))
    draw.plot_individual_image(depos, 'example')

    draw.make_unrolling_plot(depos, 'unrolling')
    draw.make_unrolling_plot(depos, 'unrolling-video', make_animation=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Draw schemas"""
    )
    parser.add_argument(
        "--output", "-o",
        action=CreateFolder,
        type=Path,
        default="plots/",
        help="Path to directory where plots will be stored",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactively display plots as they are created",
        default=False,
    )
    parser.add_argument(
        "--file-format",
        type=str,
        default="png",
        help="File format of resulting plots",
    )
    main(parser.parse_args())
