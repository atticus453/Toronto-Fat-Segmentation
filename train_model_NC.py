from load_data import loadDataGeneral
from model import UNet, AG_UNet, DSV_UNet, AG_DSV_UNet
import numpy as np
import pandas as pd
import os
import torch
from torch import optim, nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch.autograd import Function
import pytorchtrainer as ptt
from seed_everything import seed_everything


def dice_pytorch(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    #csv_path = '/path/to/dataset/idx-train.csv'
    csv_path = 'idx-val-train-NC.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    #path = csv_path[:csv_path.rfind('/')] + '/'
    path ='/home/jupyter-atticus453/Cardiac-CT-Image-10.7717/dataset/Epicardial_train_NC/'

    model_name = 'fat_model_NC.pth'
    f = "log_unet.csv"
    k_fold = 5
    epochs = 300
    log = []
    seed_everything(0)
    bz = 1
    
    # Select GPU. The GPU id to use, usually either "0" or "1";
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'

    #model = UNet(img_ch=1,init_f=32)
    #model = AG_UNet(img_ch=1,init_f=32)
    #model = DSV_UNet(img_ch=1,init_f=32)
    model = AG_DSV_UNet(img_ch=1,init_f=32)

    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count()>1:
        print("Let's use ", torch.cuda.device_count()," GPUs!")
        # model = nn.DataParallel(model)

        
        
    


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-8, momentum=0.9)

    df = pd.read_csv(csv_path)
    # Shuffle rows in dataframe. Random state is set for reproducibility.
    print(df)
    print(path)

    # Load training data
    append_coords = False
    X, y = loadDataGeneral(df, path, append_coords)
    print("finished loadData")

    ndata = X.shape[0]
    print(ndata)


    # Show dataset shape
    inp_shape = X[0].shape
    print('matrix size')
    print(inp_shape)

    
    # swap X, y to NCDHW
    #X = X.swapaxes(1,4)
    #y = y.swapaxes(1,4)
    X = X.swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)
    y = y.swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)


    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    # load dataset
    dataset = TensorDataset(x_tensor, y_tensor)

    #train_dataset, val_dataset = random_split(dataset, [180, 20])
    train_dataset = dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=bz)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=1)


    best_acc = 0.0
    best_loss = 100.0
    best_acc_e = 0
    best_loss_e = 0
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        model.train()
  
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        train_acc = 0.0
        val_acc = 0.0
        train_loss = 0.0
        val_loss = 0.0
        pred1 = 0
        
        #for i, data in enumerate(train_loader):
        for batch_idx, (data, targets) in loop:
            
            # get the inputs; data is a list of [inputs, labels]
            data, targets = data.to(device), targets.to(device)
            print(data.device, targets.device)

            #targets = targets.to('cuda:1')
            score = model(data)
            loss = criterion(score, targets)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # accuracy
            pred1 = score
            pred1 = (pred1 >0.5).float()
            train_acc += dice_pytorch(pred1,targets)

            # print statistics
            loop.set_description('Epoch {}/{}'.format(epoch + 1, epochs))
            
        #model.eval()

        # summary valve
        train_acc = train_acc / len(train_loader)
        #val_acc = val_acc / len(val_loader)
        train_loss = train_loss/len(train_loader)
        #val_loss = val_loss/len(val_loader)
        
        #log.append([epoch+1, train_acc, train_loss,val_acc,val_loss])
        log.append([epoch+1, train_acc, train_loss])
        #print('train acc = {:.5f}, train loss = {:.5f}, val_acc = {:.5f}, val loss = {:.5f}'.format(train_acc,train_loss,val_acc,val_loss))
        print('train acc = {:.5f}, train loss = {:.5f}'.format(train_acc,train_loss))

        if (train_acc > best_acc):
            torch.save(model,model_name[:-4]+'_best_acc.pth')
            best_acc = train_acc
            best_acc_e = epoch+1

            
    torch.save(model, model_name[:-4]+'_300.pth')
    df = pd.read_csv(f)
    df = df.append(pd.DataFrame(log, columns=df.columns))
    df.to_csv(f,index=False)
    print('best acc epoch = ',best_acc_e)

    print('Finished Training...')




