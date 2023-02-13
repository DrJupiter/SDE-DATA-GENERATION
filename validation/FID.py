import torch
from inception_model_help import inceptionv3

out1 = inceptionv3(x = None,path_to_imgs ="./validation/imgs/real/*.jpg")
out2 = inceptionv3(x = None,path_to_imgs ="./validation/imgs/gen/*.jpg")

mu1 = out1.mean(dim=0)
sigma1 = out1.cov(correction=0)

mu2 = out2.mean(dim=0)
sigma2 = out2.cov(correction=0)

print("mu dims",mu1.shape,mu2.shape)

def gaussian_FID(mu1,mu2,sigma1,sigma2):
    m1mm2 = (mu1-mu2).unsqueeze(0)
    mu_dist = torch.mm(m1mm2,torch.transpose(m1mm2,0,1))

    sigma = sigma1+sigma2-2*(((sigma1**(0.5))*sigma2*(sigma1**(0.5)))**(0.5))
    trace = torch.trace(sigma)
    return (mu_dist+trace)[0,0]

print(gaussian_FID(mu1,mu2,sigma1,sigma2))