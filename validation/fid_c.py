import torch
import numpy as np
import jax.numpy as jnp

from torchmetrics.image.fid import FrechetInceptionDistance
import tensorflow_hub 
import tensorflow_gan as tf_gan
import tensorflow as tf

def get_fid_model(cfg):
    """
        # TODO: Make sure the values for the images are within [0;255]
        samples = np.clip(samples * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
          (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # TODO: Pre compute stats on the dataset
    """
    if cfg.train_and_test.test.fid_model_type == "tensorflow":
        _DEFAULT_DTYPES = {
          "logits": tf.float32,
          "pool_3": tf.float32
        }
        inception_model = tensorflow_hub.load('https://tfhub.dev/tensorflow/tfgan/eval/inception/1')
        def _classifier_fn(images):
            output = inception_model(images)
            return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

        def compute_pool3_logit(images):
            # TODO: Make sure input has the range [0; 255]
            images = (tf.cast(images, tf.float32) - 127.5) / 127.5
            res = tf_gan.eval.run_classifier_fn(images, num_batches=1, classifier_fn=_classifier_fn, dtypes=_DEFAULT_DTYPES)
            return res["pool_3"], res["logits"]

        def compute_fid(generated_imgs, true_images):
            datashape = np.array(jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0]))
            datashape[0] = len(generated_imgs)
            datashape[-1] = -1

            if cfg.dataset.name == "mnist":
                generated_imgs = np.stack((np.clip(generated_imgs, 0, 255),) * 3, axis=-1).reshape(datashape).astype(np.uint8) # .transpose(0, -1, 1, 2)
                true_images = np.stack((true_images,) * 3, axis=-1).reshape(datashape).astype(np.uint8) # .transpose(0, -1, 1, 2)

            generated_imgs = tf.convert_to_tensor(generated_imgs)
            true_images = tf.convert_to_tensor(true_images)
            gen_pool_3, gen_logits = compute_pool3_logit(generated_imgs)
            true_pool_3, true_logits = compute_pool3_logit(true_images)
            #inception_score = None # TODO: Potentially do this later
            fid = tf_gan.eval.frechet_classifier_distance_from_activations(gen_pool_3, true_pool_3)
            return fid
        return compute_fid
        
    elif cfg.train_and_test.test.fid_model_type == "torch":
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = FrechetInceptionDistance(feature=cfg.train_and_test.test.fid_features)
        if DEVICE == "cuda:0":
            model.inception.cuda()

        def compute_fid(generated_imgs, real_images):
            datashape = np.array(jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0]))
            datashape[0] = len(generated_imgs)
            datashape[-1] = -1

            if cfg.dataset.name == "mnist":
                generated_imgs = np.stack((np.clip(generated_imgs, 0, 255),) * 3, axis=-1).reshape(datashape).astype(np.uint8).transpose(0, -1, 1, 2)
                real_images = np.stack((real_images,) * 3, axis=-1).reshape(datashape).astype(np.uint8).transpose(0, -1, 1, 2)
            generated_imgs = torch.from_numpy(generated_imgs)
            real_images = torch.from_numpy(real_images)
            model.update(real_images, real=True) 
            model.update(generated_imgs, real=False) 
            fid = model.compute()
            model.reset()
            return fid
        return compute_fid

    





