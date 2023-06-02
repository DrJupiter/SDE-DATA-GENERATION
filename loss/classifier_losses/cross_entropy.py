from utils.text_embedding import get_label_embeddings
import numpy as np


def get_cross_entropy(cfg):
    
    def cross_entropy(func, function_parameters, data, perturbed_data, time , z, text_embedding, key):

        model_out_put = func(data, None, None, function_parameters, key)
       
        return -np.mean(text_embedding * model_out_put)
    return cross_entropy


        
