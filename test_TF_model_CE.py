from load_data_nrrd import loadDataGeneral
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd, os, math
import torch
from torch import optim, nn, Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset



#from scipy.misc import imresize
#from scipy import imresize
#from imageio import imresize
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure

def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    #return img[:, img.shape[1] / 2, ::-1].T
    return img[:, int(img.shape[1] / 2),::-1].T

img_size = 512

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    csv_path = 'idx-val-test-CE.csv'
    csv_result = 'result_unet_CE.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    #path = csv_path[:csv_path.rfind('/')] + '/'
    path ='Epicardial_test_CE/'
    
    df = pd.read_csv(csv_path)
    df_out =pd.read_csv(csv_result)
    #print(path)

    # Load test data
    append_coords = False
    X, y = loadDataGeneral(df, path, append_coords)

    n_test = X.shape[0]
    inpShape = X.shape[1:]

    # Select GPU. The GPU id to use, usually either "0" or "1";
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    X = X.swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)
    #y = y.swapaxes(1,4).swapaxes(2,3).swapaxes(3,4)

    print(X.shape)
    print(y.shape)
    
    x_tensor = torch.from_numpy(X).float()
    #y_tensor = torch.from_numpy(y).float()
    #dataset = TensorDataset(x_tensor, y_tensor)
    #test_loader = DataLoader(dataset=dataset, batch_size=1)
    
    # Load model

    model_name = 'fat_model_CE_best_acc.pth' # Model should be trained with the same `append_coords`
    #model = load_model(model_name)
    #model = TheModelClass(*args, **kwargs)
    
    model = torch.load(model_name)
    #model = torch.load_state_dict(torch.load(model_name))

    # Predict on test data
    #pred = model.predict(X, batch_size=1)[..., 1]
    #x_tensor = Tensor([x_tensor])
    

    predictions =[]
    model.eval()
     # Compute scores and visualize
    ious = np.zeros(n_test)
    dices = np.zeros(n_test)
    hds = np.zeros(n_test)
    result =[]   


    for i in range(n_test):
        with torch.no_grad():
            inputs = x_tensor[i].unsqueeze(0)
            #print('input shape')
            #print(inputs.shape)
            inputs = inputs.to('cuda:1')
            prediction = model(inputs)
         
            prediction = np.squeeze(prediction)
            prediction = prediction[1,...] #select only 1, remove 0
            pred = prediction.detach().cpu().numpy()
            pred = pred.swapaxes(0,2).swapaxes(0,1)


    
    #for i in range(n_test):
        gt = y[i, :, :, :, 1] > 0.5 # ground truth binary mask
        #pr = pred[i] > 0.5 # binary prediction
        pr = pred >0.5
        # Save 3D images with binary masks if needed
        org_data, header = nrrd.read(path + df.iloc[i]['pathmsk'])
        savefile = 'Predictions/'+df.iloc[i]['path'][:-11]+'TF-CE-pred.nrrd'
        #nrrd.write(savefile,pred.astype('uint8'),header)
        nrrd.write(savefile,pr.astype('uint8'),header)
        #nrrd.write(savefile,np.squeeze(pr.astype('uint8')),header)
        #if False:
        #    tImg = nib.load(path + df.ix[i].path)
        #    nib.save(nib.Nifti1Image(255 * pr.astype('float'), affine=tImg.get_affine()), df.ix[i].path+'-pred.nii.gz')
        #    nib.save(nib.Nifti1Image(255 * gt.astype('float'), affine=tImg.get_affine()), df.ix[i].path + '-gt.nii.gz')
        # Compute scores
        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        print (df.iloc[i]['path'], ious[i], dices[i])
        result.append([df.iloc[i]['path'],dices[i],ious[i]])

        

    print ('Mean IoU:')
    print (ious.mean()*100)

    print ('Mean Dice:')
    print (dices.mean()*100)

    df_out=df_out.append(pd.DataFrame(result, columns=df_out.columns))
    df_out.to_csv(csv_result, index=False)
    print('done..')
