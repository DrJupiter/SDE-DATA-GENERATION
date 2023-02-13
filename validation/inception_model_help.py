
import torch
from torch import nn
import numpy as np
from PIL import Image
import glob
from torchvision import transforms

preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def inceptionv3(x = None,path_to_imgs = None, pre_transformed = False) -> torch.Tensor:
    """
    This function loads the inceptionv3 model from pytorch's library as fully pretrained.\\
    Additionally this model take image(s) converts them to correct size for this model (look into if other models dont need to reshape).\\
    Finally it passes the image(s) through the model and\\
        outputs: [N,2048] as desired.\\
        Input: if path_to_imgs = str, then images from the given path is used.\\
        Else input from x (images of shape [N,3,D1,D2]) where N and D1 and D2 can be whatever.\\
    """

    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.fc = Identity()
    model.eval()
    

    # for img in path_to_imgs:
    # input_image = Image.open(path_to_imgs)
    
    
    # start0 = perf_counter()
    if path_to_imgs == None:
        if pre_transformed == False:
            to_pil = transforms.ToPILImage()
            input_batch = []
            for xi in x:
                img = to_pil(xi)
                input_tensor = preprocess(img)
                input_batch.append(input_tensor.unsqueeze(0))
            input = torch.vstack(input_batch)
        elif pre_transformed == True:
            input = x
            print("as expected")

    else:
        input_batch = []
        images = glob.glob(path_to_imgs)
        for image in images:
            with open(image, 'rb') as file:
                img = Image.open(file)
                # img.show()
                input_tensor = preprocess(img)
                input_batch.append(input_tensor.unsqueeze(0))
        input = torch.vstack(input_batch)

    # print("tensor to img to tensor (if needed) time",perf_counter()-start0)

    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    # print(input.shape)

    # print("input shape",input.shape)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')

    # start2 = perf_counter()
    with torch.no_grad():
        output = model(input)
    # print("model time",perf_counter()-start2)

    return output

if __name__ == "__main__":
    import torchvision.datasets as datasets
    # from time import perf_counter
    
    cifar10_testset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=preprocess)
    # cifar10_testset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # extract only imgs not class
    x_test = torch.stack([cifar10_testset[i][0] for i in range(100)])

    # print(inceptionv3(path_to_imgs = "./validation/imgs/gen/*.jpg"))
    # start1 = perf_counter()
    print(inceptionv3(x = x_test, path_to_imgs = None,pre_transformed = False).shape)
    # print("time",perf_counter()-start1)