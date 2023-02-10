
import torch
from torch import nn
import numpy as np
from PIL import Image
import glob



def inceptionv3(path_to_imgs: str = "./validation/imgs/*.jpg") -> torch.Tensor:
    """
    This function loads the inceptionv3 model from pytorch's library as fully pretrained.\\
    Additionally this model take image(s) converts them to correct size for this model (look into if other models dont need to reshape).\\
    Finally it passes the image(s) through the model and\\
        outputs: [N,2048] as desired.\\
    OPS: stuff needs changing in the inception.py file for this to work. MAKE THIS NOT NEEDED.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

    from torchvision import transforms


    
    # for img in path_to_imgs:
    # input_image = Image.open(path_to_imgs)
    preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    input_batch = []
    images = glob.glob(path_to_imgs)
    for image in images:
        with open(image, 'rb') as file:
            img = Image.open(file)
            # img.show()
            input_tensor = preprocess(img)
            input_batch.append(input_tensor.unsqueeze(0))

    # input_tensor = preprocess(input_image)
    # input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input = torch.vstack(input_batch)
    print(input.shape)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input = input.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0].shape,output[0].flatten().sum())
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    # probabilities = torch.nn.functional.softmax(output[0], dim=1)
    # print(probabilities.shape,probabilities.flatten().sum())
    return output[0]

if __name__ == "__main__":
    print(inceptionv3(path_to_imgs = "./validation/imgs/*.jpg"))