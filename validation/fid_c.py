import torch

from torchmetrics.image.fid import FrechetInceptionDistance
def get_fid_model(cfg):
    #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DEVICE = "cpu"
    model = FrechetInceptionDistance(feature=cfg.train_and_test.test.fid_features)
    if DEVICE == "cuda:0":
        model.inception.cuda()

    def compute_fid(generated_imgs, real_images):
        model.update(real_images, real=True) 
        model.update(generated_imgs, real=False) 
        fid = model.compute()
        model.reset()
        return fid
    return compute_fid






