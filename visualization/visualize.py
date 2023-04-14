import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import wandb
import jax
from torchvision.utils import make_grid

def setup_plot():
    mpl.rcParams['lines.linewidth'] = 1
    #mpl.rcParams['font.family'] = 'Microsoft Sans Serif'
    mpl.rcParams['font.family'] = 'Arial'

    
    #these don't work for some reason
    #mpl.rcParams['axes.titleweight'] = 'bold'
    #mpl.rcParams['axes.titlesize'] = '90'
    
    sns.set_theme(style="white", palette='pastel', font = 'Arial', font_scale=3)

    #sns.set_theme(style="white", palette='pastel', font = 'Microsoft Sans Serif', font_scale=1)
    #myFmt = mdates.DateFormatter('%b #Y')
    
    print("Plot settings applied")


def display_images(cfg, images, titles = [], rows = None, columns = 2, figsize= (7,7), pad=0.2):
    """
    Takes a list of images and plots them

    Takes the config, so we know how to plot the image in accordance with the dataset
    """
    
    if rows is None:
        rows = len(images)

    fig = plt.figure(figsize=figsize)

    for idx, img in enumerate(images):
        fig.add_subplot(rows, columns, idx+1) 
        if cfg.dataset.name == 'mnist':
            plt.imshow(img.reshape(28,28), cmap='gray')
        elif cfg.dataset.name == 'cifar10':
            plt.imshow((img).reshape(32,32,3)/2 + 0.5)
        plt.axis('off')
        if len(titles) == len(images):
            plt.title(titles[idx])
        else:
            plt.title(str(idx+1))
    plt.tight_layout(pad=pad) 

    if cfg.wandb.log.img:
        if wandb.run is None:
            run = wandb.init(entity=cfg.wandb.setup.entity, project=cfg.wandb.setup.project)

        wandb.log({f"plot {cfg.dataset.name}": fig})
    if cfg.visualization.visualize_img:
        plt.show()
    plt.close()

if __name__ == "__main__":
    from torchvision.datasets import MNIST, CIFAR10
    import multiprocessing as mp
    from data.dataload import NumpyLoader, FlattenAndCast
    DATA_PATH = './datasets'


    from utils.utils import get_hydra_config
    if False:
        cfg = get_hydra_config(overrides=['dataset=mnist', "visualization.visualize_img=true","wandb.log.img=false"])
        mnist_dataset = MNIST(DATA_PATH, download=True, transform=FlattenAndCast())
        training_generator = NumpyLoader(mnist_dataset, batch_size=10, num_workers=mp.cpu_count())
        data_samples = iter(training_generator)
        data_points, labels = next(data_samples)
        display_images(cfg, data_points, labels)

        cfg = get_hydra_config(overrides=['dataset=cifar10', "visualization.visualize_img=true","wandb.log.img=false"])
        cifar10_dataset = CIFAR10(DATA_PATH, download=True, transform=FlattenAndCast())
        training_generator = NumpyLoader(cifar10_dataset, batch_size=10, num_workers=mp.cpu_count())
        data_samples = iter(training_generator)
        data_points, labels = next(data_samples)
        print(jax.numpy.max(data_points), jax.numpy.min(data_points))
        display_images(cfg, (data_points/255 - 0.5) * 2, [cfg.dataset.classes[int(idx)] for idx in labels])

    cfg = get_hydra_config(overrides=['dataset=cifar10', "visualization.visualize_img=true","wandb.log.img=false"])
    from data.dataload import dataload
    training_generator, _test_generator = dataload(cfg)
    data_samples = iter(training_generator)
    data_points, labels = next(data_samples)
    print(make_grid(data_points).shape)
    data_points = data_points.numpy().astype(jax.numpy.float32)
    
    import numpy as np
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    #imshow()
    print(jax.numpy.max(data_points), jax.numpy.min(data_points))
    display_images(cfg, jax.numpy.transpose(data_points, (0, 2,3,1)) /255, [cfg.dataset.classes[int(idx)] for idx in labels])