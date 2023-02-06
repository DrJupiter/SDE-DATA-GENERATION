import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns

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


def display_images(cfg, images, titles, rows = None, columns = 2, figsize= (10,7), pad=1.08):
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
            plt.imshow((img/255).reshape(32,32,3))
        plt.axis('off')
        if len(titles) == len(images):
            plt.title(titles[idx])
        else:
            plt.title(str(idx+1))

    plt.tight_layout(pad=pad) 
    plt.show()
    plt.close()

if __name__ == "__main__":
    from torchvision.datasets import MNIST, CIFAR10
    import multiprocessing as mp
    from data.dataload import NumpyLoader, FlattenAndCast
    DATA_PATH = './datasets'


    from utils.utils import get_hydra_config
    
    cfg = get_hydra_config()
    mnist_dataset = MNIST(DATA_PATH, download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(mnist_dataset, batch_size=10, num_workers=mp.cpu_count())
    data_samples = iter(training_generator)
    data_points, labels = next(data_samples)
    display_images(cfg, data_points, labels)

    cfg = get_hydra_config(overrides=['dataset=cifar10'])
    cifar10_dataset = CIFAR10(DATA_PATH, download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(cifar10_dataset, batch_size=10, num_workers=mp.cpu_count())
    data_samples = iter(training_generator)
    data_points, labels = next(data_samples)
    display_images(cfg, data_points, [cfg.dataset.classes[int(idx)] for idx in labels])