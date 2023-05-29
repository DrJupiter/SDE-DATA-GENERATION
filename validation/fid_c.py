import torch
import numpy as np
import jax.numpy as jnp

from torchmetrics.image.fid import FrechetInceptionDistance
import tensorflow_hub 
import tensorflow_gan as tf_gan
import tensorflow as tf

import utils.utility
import os

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

        def compute_statistic(images):

            split_factor = cfg.train_and_test.test.split_factor 
            images = tf.convert_to_tensor(images)
            images = tf.split(images, split_factor, axis=0)

            all_pool_3 = []  
            all_logits = []

            for img_batch in images:
                pool_3, logits = compute_pool3_logit(img_batch)
                all_pool_3.append(pool_3), all_logits.append(logits)

            all_pool_3 = tf.concat(all_pool_3, axis = 0)
            all_logits = tf.concat(all_logits, axis = 0)

            return all_pool_3, all_logits

        def compute_fid(generated_imgs, true_images, force_recompute=False):

            # Get the shape to reshape the data into
            datashape = np.array(jnp.array(cfg.dataset.shape)+jnp.array([0,cfg.dataset.padding*2,cfg.dataset.padding*2,0]))
            datashape[0] = len(generated_imgs)
            datashape[-1] = -1


            # Handle data format
            if cfg.dataset.name == "mnist":
                generated_imgs = np.stack((np.clip(generated_imgs, 0, 255),) * 3, axis=-1).reshape(datashape).astype(np.uint8) # .transpose(0, -1, 1, 2)
                true_images = np.stack((true_images,) * 3, axis=-1).reshape(datashape).astype(np.uint8) # .transpose(0, -1, 1, 2)

            gen_pool3, gen_logits = compute_statistic(generated_imgs) 

            # TODO: ADD ASSERT FOR SHAPES BEING THE SAME

            # Handle ground truth images
            if cfg.parameter_loading.test_statistics and (not force_recompute):
                file_name = utils.utility.get_save_path_names(cfg)["test_data_statistics"]
                name = os.path.join(cfg.parameter_loading.test_data_path, file_name)
                if os.path.isfile(name):
                    with open(name, "rb") as f:
                        stats = np.load(f)
                        true_pool3, true_logits = stats["pool3"], stats["logit"]
                        f.close()
                    print(f"Loaded saved test pool3 and logits @ {name}")
                    print(f"\t len(pool3), len(logits) = {len(true_pool3)}, {len(true_logits)}")
                    print(f"\t len(pool3 from score model) = {len(gen_pool3)}")
                else:
                    print(f"Unable to find {name}, creating and saving statistics instead")

                    true_pool3, true_logits = compute_statistic(true_images) 

                    with open(name, "wb") as f:
                        np.savez_compressed(f , pool3=true_pool3, logit=true_logits)
                    print(f"Saved statistics @ {name}") 
            else:
                    true_pool3, true_logits = compute_statistic(true_images) 
            
            #inception_score = None # TODO: Potentially do this later

            fid = tf_gan.eval.frechet_classifier_distance_from_activations(gen_pool3, true_pool3)
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

    





