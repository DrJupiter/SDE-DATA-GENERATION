import torch
from validation.inception_model_help import inceptionv3


def FID_score(x1,x2):
    if type(x1) == str:
        path_to_imgs1 = x1
    else: 
        path_to_imgs1 = None
    if type(x2) == str:
        path_to_imgs2 = x2
    else: 
        path_to_imgs2 = None

    out1 = inceptionv3(x = x1, path_to_imgs = path_to_imgs1) # "./validation/imgs/real/*.jpg"
    out2 = inceptionv3(x = x2, path_to_imgs = path_to_imgs2) # "./validation/imgs/gen/*.jpg"

    mu1 = out1.mean(dim=0)
    sigma1 = out1.cov(correction=0)

    mu2 = out2.mean(dim=0)
    sigma2 = out2.cov(correction=0)

    print("mu dims",mu1.shape,mu2.shape)

    def gaussian_FID(mu1,mu2,sigma1,sigma2):
        m1mm2 = (mu1-mu2).unsqueeze(0)
        mu_dist = torch.mm(m1mm2,torch.transpose(m1mm2,0,1))
        print(mu_dist)

        sigma = sigma1+sigma2-2*(torch.sqrt(sigma1 @ sigma2))
        trace = torch.trace(sigma)
        return (mu_dist+trace)[0,0]

    return gaussian_FID(mu1,mu2,sigma1,sigma2)

if __name__ == "__main__":
    x1 = "./validation/imgs/real/*.jpg"
    x2 = "./validation/imgs/gen/*.jpg"

    # print(FID_score(x1,x2))
    from utils import get_hydra_config
    cfg = get_hydra_config()
    print(cfg)
    from data.dataload import dataload 
    train_dataset, test_dataset = dataload(cfg) 