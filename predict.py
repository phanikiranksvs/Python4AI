import argparse 
import torch 
import numpy as np
import json
import sys
import torchvision

from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
from collections import OrderedDict

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='mycheckpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # use a deafault filepath to a primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='cuda')
    return parser.parse_args()

def loadsavedmodel(savedmodel):
    loadmodel = torch.load(savedmodel)
    print(loadmodel['arch'])
    newmodel = getattr(torchvision.models, loadmodel['arch'])(pretrained=True)
    newmodel.classifier = loadmodel['classifier']
    newlearning_rate = loadmodel['learning_rate']
    newmodel.epochs = loadmodel['epochs']
    newmodel.optimizer = loadmodel['optimizer']
    newmodel.class_to_idx = loadmodel['class_to_idx']

    print(newmodel)
    return newmodel

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    print(type(image))
    height,width = image.size
    print(height)
    print(width)
    aspectratio=0
    if height > width:
        width = 256
        
    else:
        height = 256
        
    print(height)
    print(width)
    #image.resize(image.size)
    image.thumbnail([height,width])
    new_width = 224
    new_height = 224

    left = (256 - new_width)/2
    upper = (256 - new_height)/2
    right = (256 + new_width)/2
    lower = (256 + new_height)/2
    im_crop = image.crop((left, upper, right, lower))
    np_image = np.array(im_crop)
    
    # normalize
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image =np.transpose(np_image, (2, 0, 1))
    
    return np_image    
def predict(image_path, model, topk,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    predictimage = Image.open(image_path)
    
    np_image = process_image(predictimage)
    np_image = torch.from_numpy(np_image)
    model.zero_grad()
    np_image = np_image.unsqueeze_(0)   
    np_image = np_image.float()
    print(model)
    print(gpu)
    gpu= torch.device("cuda" if gpu == 'cuda' and torch.cuda.is_available() else "cpu")
    model = model.to(gpu)
    np_image = np_image.to(gpu)
    prediction = model.forward(np_image)
    probs, classes = torch.exp(prediction).topk(topk)
    print(probs)
    print(classes)
    return probs[0].tolist(), classes[0].add(1).tolist()

def load_names(njson):
    with open(njson, 'r') as f:
        names = json.load(f)
    return names
def main(): 
    args = parse()
    gpu = args.gpu
    print(args.checkpoint)
    model = loadsavedmodel(args.checkpoint)
    names = load_names(args.category_names)
    
    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k),gpu)
    labels = [names[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    
    i=0 # this prints out top k classes and probs as according to user 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1 # cycle through



if __name__ == "__main__":
    main()