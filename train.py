
import numpy as np
import torch
import matplotlib.pyplot as plt


from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image
from collections import OrderedDict
import time
import torchvision
import argparse






def parse():
    parser = argparse.ArgumentParser(description=' parameters for model training!')
    parser.add_argument('data_dir', help='data directory (required)')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch',  default='vgg11',help='models to use OPTIONS[vgg11,vgg19]')
    parser.add_argument('--learning_rate', default='0.001', help='learning rate')
    parser.add_argument('--hidden_units', help='number of hidden units')
    parser.add_argument('--epochs', default='1', help='epochs')
    parser.add_argument('--cpu',default='cuda', help='cpu')
    args = parser.parse_args()
    return args

def trainmymodel(mymodel,epochs,device,traindata,validationdata,mycriteria,myoptimizer):
    steps = 0
    print_every = 10
    
    print("start training")
    print(epochs)
    for epoch in range(epochs):
        running_loss = 0
        for image,labels in (traindata):
            steps +=1
            image,labels = image.to(device), labels.to(device)
            myoptimizer.zero_grad()
            output = mymodel.forward(image)
            loss = mycriteria(output,labels)
            loss.backward()
            myoptimizer.step()
            
            running_loss += loss.item()
            #print("running loss"+ str(running_loss))
            #print(steps)
            #print(print_every)
            if steps % print_every == 0:
                mymodel.eval()
                test_loss = 0
                accuracy = 0

                for image2,labels2 in (validationdata):
                    image2,labels2 = image.to(device), labels.to(device)
                    myoptimizer.zero_grad()

                    logps = mymodel.forward(image2)
                    loss = mycriteria(logps,labels2)

                    test_loss += loss.item()
                
                    # accuracy
                    ps = torch.exp(logps)
                    top_ps,top_class = ps.topk(1,dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                
                print(f"epoch {epoch+1}/{epoch}..."
                      f"train loss : {running_loss/print_every:.3f}.."
                      f"validation loss : {loss/ len(validationdata):.3f}.."
                      f"test accuracy : {accuracy/len(validationdata):.3f}")
                running_loss =0
    print("end training")            
def savemodel(mymodel,class_idx,epochs,lr,myoptimizer,myclassifier,arch):
    mymodel.class_to_idx = class_idx
    
    mycheckpoint = {'arch':arch,
                'input_size':25088,
                'output_size':102,
                'epochs':epochs,
                'learning_rate':lr,
                'class_to_idx':class_idx,
                'optimizer':myoptimizer.state_dict(),
                'classifier':myclassifier
               }
    

    torch.save(mycheckpoint, 'mycheckpoint.pth')
    
def main():
    print("parsing arguments")
    args = parse()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    training_data_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),                                               
                                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_data_tranforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                               transforms.ToTensor(),                                               
                                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testing_data_tranforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                               transforms.ToTensor(),                                               
                                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_datasets = [datasets.ImageFolder(train_dir,transform=training_data_transforms),
                      datasets.ImageFolder(valid_dir,transform=validation_data_tranforms),
                      datasets.ImageFolder(test_dir,transform=testing_data_tranforms)]


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64,  shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1],batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
    arch = args.arch
    print(arch)
    mymodel = getattr(models,arch )(pretrained=True)

    for params in mymodel.parameters():
        params.requires_grad =False

    myclassifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096,                                    bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(4096, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))]))
    mymodel.classifier = myclassifier
    mymodel
    print(mymodel)
    
    epochs = int(args.epochs)
    device = args.cpu
    
    mymodel.to(device)
    mycriteria = nn.NLLLoss()
    myoptimizer = optim.Adam(mymodel.classifier.parameters(),lr=float(args.learning_rate))
    trainmymodel(mymodel,epochs,device,dataloaders[0],dataloaders[1],mycriteria,myoptimizer)
    
    mymodel.class_to_idx = image_datasets[0].class_to_idx

    savemodel(mymodel,image_datasets[0].class_to_idx,args.epochs,args.learning_rate,myoptimizer,myclassifier,arch)
    

if __name__ == "__main__":
        print("calling main")
        main()