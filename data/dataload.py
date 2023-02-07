import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST, CIFAR10
import multiprocessing as mp

# Stop loading it 90% VRAM
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax.numpy as jnp

def numpy_collate(batch):
  """
  Collation function for getting samples
  from `NumpyLoader`
  """
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  """
  The dataloader used for our image datasets
  """
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)
      
class FlattenAndCast(object):
  """
  Flattens an image and converts the datatype to be 
  jax numpy's float32 type
  """
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

# ASK PAUL, CAN THIS ALL BE DONE IN THE HYDRA CONF?
def dataload(cfg):
    """ 
    Returns the dataset specified in the config
    in the form of a pytorch dataloader
    """
    name = cfg.dataset.name
    if name == 'mnist':
        mnist_dataset = MNIST(cfg.dataset.path, download=True, transform=FlattenAndCast())
        training_generator = NumpyLoader(mnist_dataset, batch_size=cfg.training.batchsize, shuffle=cfg.training.shuffle, num_workers=mp.cpu_count())
        return training_generator

    elif name == 'cifar10':
        mnist_dataset = CIFAR10(cfg.dataset.path, download=True, transform=FlattenAndCast())

        training_generator = NumpyLoader(mnist_dataset, batch_size=cfg.training.batchsize, shuffle=cfg.training.shuffle, num_workers=mp.cpu_count())

        return training_generator
    
    # TODO: ASK PAUL how to do this in a hydra friendly way
    raise ValueError(f"The dataset with name {name} doesn't exist")



  

if __name__ == "__main__":
  # Define our dataset, using torch datasets
    mnist_dataset = MNIST('./tmp/mnist/', download=True, transform=FlattenAndCast())
    import multiprocessing as mp
    training_generator = NumpyLoader(mnist_dataset, batch_size=1, num_workers=mp.cpu_count())
    data_samples = iter(training_generator)
    data_point, label = next(data_samples)
    print(label)
    # print(jnp.dot(data_point.T, data_point).device()) 

    cifar10_dataset = CIFAR10('./tmp/cifar10/', download=True, transform=FlattenAndCast())
    training_generator = NumpyLoader(cifar10_dataset, batch_size=2, num_workers=mp.cpu_count())
    data_samples = iter(training_generator)
    data_point, label = next(data_samples)
    print(jnp.array(data_point[0]).device(), label)
    #item = cifar10_dataset.__getitem__(0)
    #print(type(item), len(item))
    #print(item[0], item[1])
    #print(jnp.array(item[0]).device())
    