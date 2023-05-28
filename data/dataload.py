import numpy as np
from torch.utils import data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import multiprocessing as mp

#from utils.text_embedding import get_label_embeddings
#from utils.utils import get_save_path_names
from utils import get_label_embeddings, get_save_path_names

import os

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
  # TODO: Work with data in torch tensor until we pass it to model. As this loader is made to work on those, and is slow and has problems if not done like this.
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

def dataload(cfg):
    """ 
    Returns the train and test dataset specified in the config
    in the form of a pytorch dataloader
    """
    name = cfg.dataset.name

    text_embedding_table = get_label_embeddings(cfg)
     
    if name == 'mnist':
        if cfg.dataset.padding > 0:
            print(f"Padding images with {cfg.dataset.padding}")
            transform = transforms.Compose([transforms.Pad(cfg.dataset.padding),FlattenAndCast()])
        else:
            transform = transforms.Compose([FlattenAndCast()])
        target_transform = transforms.Compose([transforms.Lambda(lambda x: (x, text_embedding_table[cfg.dataset.classes[int(x)]]))])
        mnist_dataset_train = MNIST(cfg.dataset.path, download=True, transform=transform, target_transform=target_transform)
        training_generator = NumpyLoader(mnist_dataset_train, batch_size=cfg.train_and_test.train.batch_size, shuffle=cfg.train_and_test.train.shuffle) # num_workers=mp.cpu_count()

        mnist_dataset_test = MNIST(cfg.dataset.path, train=False, download=True, transform=transform, target_transform=target_transform)
        test_generator = NumpyLoader(mnist_dataset_test, batch_size=cfg.train_and_test.test.batch_size, shuffle=cfg.train_and_test.test.shuffle) # num_workers=mp.cpu_count()

        return training_generator, test_generator 

    elif name == 'cifar10':
        transform = transforms.Compose([FlattenAndCast()])
        target_transform = transforms.Compose([transforms.Lambda(lambda x: (x, text_embedding_table[cfg.dataset.classes[int(x)]]))])

        cifar10_dataset_train = CIFAR10(cfg.dataset.path, download=True, transform=transform, target_transform=target_transform) 
        training_generator = NumpyLoader(cifar10_dataset_train, batch_size=cfg.train_and_test.train.batch_size, shuffle=cfg.train_and_test.train.shuffle) 

        cifar10_dataset_test = CIFAR10(cfg.dataset.path, train=False, download=True, transform=transform, target_transform=target_transform)
        test_generator = NumpyLoader(cifar10_dataset_test, batch_size=cfg.train_and_test.test.batch_size, shuffle=cfg.train_and_test.test.shuffle)

        return training_generator, test_generator
    
    # TODO: ASK PAUL how to do this in a hydra friendly way
    raise ValueError(f"The dataset with name {name} doesn't exist")

def get_all_test_data(cfg, dataset):
  file_name = get_save_path_names(cfg)["test_data"]
  name = os.path.join(cfg.parameter_loading.test_data_path, file_name)
  if cfg.parameter_loading.test_data:
    if os.path.isfile(name):

      with open(name, "rb") as f:
        file = np.load(f) 

        data = jnp.array(file["data"])
        labels = jnp.array(file["labels"])
        embeddings = jnp.array(file["embeddings"])
      print("Loaded test dataset, labels, embeddings")
      return data, labels, embeddings
    else:
      print(f"{name} not found, instead")
  
      print(f"Saving test data @ {name}")
      labels, embeddings = get_all_labels(cfg, dataset)
      data = get_all_data(cfg, dataset)
      with open(name, "wb") as f:
        np.savez_compressed(f, data=data, labels=labels, embeddings=embeddings)
        f.close() 
  return data, labels, embeddings  


def get_all_data(cfg, dataloader):
   gen = iter(dataloader)  
   x = []
   for s in gen:
      x.append(s[0])
   return jnp.vstack(x)

def get_all_labels(cfg, dataloader):
   gen = iter(dataloader)  
   label = []
   embedding = []

   for s in gen:
      label += list(s[1][0])
      embedding += list(s[1][1])
   return jnp.vstack(label), jnp.vstack(embedding)

def get_data_mean(cfg, dataloader):
   return jnp.mean(get_all_data(cfg, dataloader), axis=0)

if __name__ == "__main__":
  # Define our dataset, using torch datasets
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
#    mnist_dataset = MNIST('./tmp/mnist/', download=True, transform=FlattenAndCast())
#    import multiprocessing as mp
#    training_generator = NumpyLoader(mnist_dataset, batch_size=1, num_workers=mp.cpu_count())

    from utils import get_hydra_config
    cfg = get_hydra_config(overrides=['dataset=mnist', "visualization.visualize_img=true","wandb.log.img=false"])
    from visualization.visualize import display_images
    training_generator, test_generator = dataload(cfg)
    mean = get_data_mean(training_generator)
    print(jnp.max(mean))
    display_images(cfg, [mean], ["datamean"])
    import sys
    sys.exit(0)
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
    