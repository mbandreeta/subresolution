# -*- coding: utf-8 -*-
#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:


#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np;
import tqdm;
import matplotlib.pyplot as plt;
from skimage import filters
from skimage.measure import label,regionprops  


def getNiftiObject(filepath):
    import nibabel as nib;
    img = nib.load(filepath);
    header = img.header
    pixdim = header.get('pixdim');
    unit = pixdim[1]
    img_array = img.get_data();
    s = img_array.shape;
    if len(s)>3:
        img_array = img_array[:,:,:,0];
    return [img_array,unit];


def saveNiftiObject(data,unit,filepath):
    import nibabel as nib;
    header_ = nib.Nifti1Header();
    header_['pixdim'] = np.array([ 1. ,  unit,  1. ,  1. ,  1. ,  1. ,  1. ,  1. ], dtype=np.float32);
    nifti_img = nib.Nifti1Image(data, affine=None, header=header_);
    nib.save(nifti_img, filepath)


def histogram_normalization(vol,newMax,newMin):
    min_intensity = np.min(vol);
    max_intensity = np.max(vol);
    const = (newMax-newMin)/(max_intensity-min_intensity);
    nvol = (((vol-min_intensity)*const)+newMin);
    return nvol;




def rmvbckSlice(img,val,r_approx,tag):    
    binary = (img>(val))*1;
    labels = label(binary);
    props = regionprops(labels);
    l = np.argmax(np.bincount(labels.flat));
    bbox = props[l-1].bbox;
    [xmin,ymin,xmax,ymax] = bbox;
    i_0 = xmin+r_approx;
    j_0 = ymin+r_approx;
    mask = np.zeros(img.shape);
    [i,j] = np.where(mask==0);
    idx = ((i-int(i_0))**2 + (j-int(j_0))**2)<=((r_approx-3)**2);
    mask[i[idx],j[idx]]=1; 
    removed_bckground = mask*img; 
    mask = (1-mask)*tag 
    return (removed_bckground+mask);


def getRadiusBbox(img,val):
    binary = (img>(val))*1;
    labels = label(binary);
    props = regionprops(labels);
    l = np.argmax(np.bincount(labels.flat));
    [xmin,ymin,xmax,ymax] = props[l-1].bbox;    
    r_approx = np.floor((xmax-xmin)/2.0)-1;
    return [r_approx,[xmin,ymin,xmax,ymax]];       
  

                              
        
def blockSeg(vol,segmented,d1,d2,T_l_1,T_l_2,tag_1,tag_2):
    w = 1
    for i in tqdm.tqdm(range(2,vol.shape[0]-2)):
        for j in range(2,vol.shape[1]-2):
            for k in range(2,vol.shape[2]-2):
                if(segmented[i,j,k]==tag_1):
                    block = vol[i-w:i+w+1, j-w:j+w+1,k-w:k+w+1].flatten();
                    n = np.size(block);
                    t = np.sum(block>T_l_1);
                    maxdiff = np.max(np.diff(block));
                    if t == n and maxdiff<=d1:
                        segmented[i-w:i+w+1, j-w:j+w+1,k-w:k+w+1]=tag_1;
                if(segmented[i,j,k]==tag_2):
                    block = vol[i-w:i+w+1, j-w:j+w+1,k-w:k+w+1].flatten();
                    n = np.size(block);
                    t = np.sum(block<T_l_2);
                    maxdiff = np.max(np.diff(block));
                    if t == n and maxdiff<=d2:
                        segmented[i-w:i+w+1, j-w:j+w+1,k-w:k+w+1]=tag_2; 
    return segmented 
           

   
   
def filterVol(vol):
    import SimpleITK as itk;
    vol_filtered = itk.Bilateral(itk.GetImageFromArray(vol),2.0)
    return itk.GetArrayFromImage(vol_filtered);
 
def equalizeHistogramSlices(vol):
    from skimage.exposure import equalize_adapthist;
    vol_corrected = np.zeros(vol.shape);
    for z in tqdm.tqdm(range(vol.shape[2])):
        vol_corrected[:,:,z] = 255*equalize_adapthist(vol[:,:,z]/255)
    return vol_corrected;  
   
     
def filterSlices(vol):
    import SimpleITK as itk;
    vol_filtered = itk.Bilateral(itk.GetImageFromArray(vol),2.0)
    for z in range(vol.shape[2]):
        vol_filtered[:,:,z] = itk.GetArrayFromImage(itk.Bilateral(itk.GetImageFromArray(vol[:,:,z]),2.0));
    return vol_filtered;                      



def gaussianNoise(img,sigma,w): 
    from skimage import restoration
    def gauss2d(s,mu,sigma,A):
        g=np.zeros([s,s]);
        for i in range(s):
            for j in range(s):
                g[i,j] = A*np.exp(-((i-mu)**2+(j-mu)**2)/2/sigma**2)
        return g;
    
    A = w/(sigma*(np.sqrt(2*np.pi)));
    noise = gauss2d(2*sigma+1,sigma,sigma,A);
    plt.imshow(img,cmap="gray");
    plt.figure();
    plt.imshow(noise,cmap="gray");
    plt.figure();    
    deconvolved_RL = restoration.richardson_lucy(img, noise, iterations=30);
    plt.imshow(deconvolved_RL,cmap="gray");
    
      

def gaussn(x, *p):
    g = np.zeros_like(x)
    for idx in range(0,len(p),3):
        mu = p[idx];
        sigma = p[idx+1];
        A = p[idx+2];
        g = g +  (A*np.exp(-(x-mu)**2/(2.*sigma**2)));
    return g[0,:];         
                  
 
def gaussianMixture(img,components=4):
    from sklearn import mixture
    obs = img.flatten();
    obs = obs.reshape(-1,1);
    g = mixture.GaussianMixture(n_components=components);
    g.fit(obs);
    params = list();
    means = list();
    for i in range(components):
        means.append(g.means_[i][0]);
    for i in np.argsort(means):
        params.append(g.means_[i][0]);
        params.append(np.sqrt(g.covariances_[i][0,0]));
        params.append(g.weights_[i]/np.sqrt(2*np.pi*g.covariances_[i][0,0]));          
    plt.subplot(1,2,1);    
    yy,xx,_ =plt.hist(obs,bins=100)
    plt.subplot(1,2,2);
    xl = np.linspace(np.min(xx),np.max(xx),100).reshape(1,-1);
    for i in range(components):
        pg = params[i*3:(i+1)*3]
        plt.plot(xl[0,:],gaussn(xl,*pg))
    plt.hist(obs,bins=100,density=True)
    plt.plot(xl[0,:],gaussn(xl,*params),color="black")
    return params;
                            

if __name__ == "__main__":
    [vol,unit] = getNiftiObject("filename")

    val = filters.threshold_otsu(vol); # threshold to segment the background
    r_approx,bbox = getRadiusBbox(vol[:,:,0],val);    # gets an approximated radius of the sample 
    [xmin,ymin,xmax,ymax] = bbox;
    i_0 = int(xmin+r_approx);
    j_0 = int(ymin+r_approx);
    s = int(r_approx/2.0);
    params = gaussianMixture(vol[i_0-s:i_0+s,j_0-s:j_0+s,:],4); # gaussian mixture fitting of the histogram
    print(params[0],params[9]) 
    segmented = vol.copy();
    [i_s,j_s,k_s] = np.where(vol>=params[9]); # solid region
    segmented[i_s,j_s,k_s]=255;
    [i_p,j_p,k_p] = np.where(vol<=params[0]); # empty region
    segmented[i_p,j_p,k_p]=0;
    segmented = blockSeg(vol,segmented,3,5,params[9]-(4*params[10]),params[0]+(4*params[1]),255,0) # modified histereses     
    plt.figure();
    plt.imshow(vol[:,:,10], cmap="gray");
    plt.figure();
    plt.imshow(segmented[:,:,10], cmap="gray");
    plt.figure();
    plt.hist(vol[segmented==255],bins=50);
    plt.hist(vol[segmented==0],bins=50);
    plt.show();     

    vol_treated = np.zeros(vol.shape)
    for z in (range(segmented.shape[2])):
        vol_treated[:,:,z] = rmvbckSlice(segmented[:,:,z],val,r_approx,-1); #flag -1 for background
        
    vol_tot = np.sum(vol_treated>=0);  
    vol_poros = np.sum(vol_treated==0);
    print(vol_poros*100.0/vol_tot);  
    phi = 0;
    Esolid =  params[9];
    Evoid  =  params[0];
    vol_final = np.zeros(vol.shape)
    for i in range(vol_treated.shape[0]):
        for j in range(vol_treated.shape[1]):
            for k in range(vol_treated.shape[2]):
                I = vol_treated[i,j,k];
                if I >= 0:
                    if I == 0:
                        phi += 1.0;
                        vol_final[i,j,k] = 1;
                    elif (I < Esolid and I>=Evoid) :
                        phi_voxel = 1-((vol_treated[i,j,k] - Evoid)/(Esolid-Evoid)); # intermediate porosity
                        phi += phi_voxel; 
                        vol_final[i,j,k] =  phi_voxel;
                    else:
                        vol_final[i,j,k] = 0;
                else:
                    vol_final[i,j,k] = 0;
                         
    print((phi/vol_tot)*100);
    plt.figure();
    plt.imshow(vol[:,:,10], cmap="gray");
    plt.figure();
    plt.imshow(vol_final[:,:,10]);
    plt.show();
    saveNiftiObject(vol_final,unit,"filename")

#save .nhdr:
    # dp=vol_final.dtype;
    # import sys
    # native_byte = sys.byteorder
    # byteorder = native_byte+'-'+'endian';
    # if dp.byteorder == '<':
    #     byteorder='little-endian';
    # if dp.byteorder == '>':
    #     byteorder='big-endian'; 
    # if dp.byteorder == '|':
    #     byteorder='not-applicable';           
    #     
    # import nrrd as nr
    #     
    # nr.write('D:/Porosity/results.nhdr',vol_final,header={'endian':byteorder,'type':dp.name});
        
