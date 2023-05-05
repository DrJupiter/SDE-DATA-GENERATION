import torch
from PIL import Image
import open_clip
import jax.numpy as jnp
import numpy as np
import os
import pickle


def get_embedding_model_tokenizer(cfg):
    if cfg.text_embedding.name == "CLIP-ViT-H-14":
        model, _preprocess_train, _preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

        def text_embedding(tokens):
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_embeddings = model.encode_text(tokens).cpu().numpy() 
                
            return jnp.array(text_embeddings, dtype=jnp.float32)
        return text_embedding, tokenizer

    raise NotImplementedError(f"No text embedding available for {cfg.text_embedding.name}")

def get_label_embeddings(cfg):
    """
    
    """

    def assert_correct_keys(table, cfg):
        print("Asserting labels and keys match")
        table_keys = np.array(list(table.keys()))
        dataset_label_keys = np.array(cfg.dataset.classes)
        assert np.alltrue(table_keys.sort() == dataset_label_keys.sort()), f"Mismatch between\n Table keys: {table_keys}\n Dataset label keys: {dataset_label_keys}"
        
    name = os.path.join(cfg.text_embedding.path, f"{cfg.dataset.name}/{cfg.text_embedding.name}.pickle")

    if os.path.isdir(cfg.text_embedding.path):
        
        if os.path.isfile(name):
            print("Loading text embeddings")
            with open(name, "rb") as f:
                text_embeddings_table = pickle.load(f)
            assert_correct_keys(text_embeddings_table, cfg)
            return text_embeddings_table

    elif os.path.isfile(cfg.text_embedding.path):
        print("Loading text embeddings")
        with open(name, "rb") as f:
            text_embeddings_table = pickle.load(f)

        assert_correct_keys(text_embeddings_table, cfg)
        return text_embeddings_table


    
    print("Creating text embeddings and saving") 
    text_embedding_model, tokenizer = get_embedding_model_tokenizer(cfg)
    class_tokens = tokenizer(cfg.dataset.classes)
    text_embeddings = text_embedding_model(class_tokens)
    text_embeddings_table = dict(zip(cfg.dataset.classes, text_embeddings))
    with open(name, "wb") as f:
        pickle.dump(text_embeddings_table, f)

    del text_embedding_model 
    del tokenizer 
    
    assert_correct_keys(text_embeddings_table, cfg)

    return text_embeddings_table

if __name__ == "__main__":
    from utils import get_hydra_config   
    cfg = get_hydra_config()
    text_embeddings_table=get_label_embeddings(cfg)
    print(text_embeddings_table['zero'].shape)
    
    
