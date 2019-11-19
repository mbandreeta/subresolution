import skimage
import tqdm
import numpy as np




def getDataTif():
	from tkinter.filedialog import askdirectory
	from tkinter import Tk
	import re
	import glob
	Tk().withdraw()
	filepath = askdirectory();
	filepath += '/*.tif';
	filenames = glob.glob(filepath);
	nfiles = len(filenames)-50;
	filenames = filenames[50:np.min([1550,nfiles])];
	try:
		dir_names = filenames[0].split("/");
		name = dir_names[-1].split("nm")[0];
		unit = [int(s) for s in re.findall(r'\d+', name)][-1]/1000;
	except:
		unit = 1
	try:
		img = ((np.array([skimage.io.imread(file) for file in filenames])).transpose(2,1,0))
	except:
		x,y = np.array(skimage.io.imread(filenames[0])).shape
		imgl=list()
		for file in filenames:
			t = np.array(skimage.io.imread(file));
			if x > t.shape[0]:
				x = t.shape[0];
			if y > t.shape[1]:
				y = t.shape[1];
		for file in filenames:
			imgl.append(t[:x,:y])
		img = np.array(imgl).transpose(2,1,0)
	return img,unit,name;


def getFilepath():
	from tkinter.filedialog import askopenfilename
	from tkinter import Tk
	import re
	Tk().withdraw()
	filepath = askopenfilename();
	extension = filepath.split(".")[-1];
	dir_names = filepath.split("/");
	filename = dir_names[-1].split("nm")[0];
	unit = [int(s) for s in re.findall(r'\d+', filename)][-1]/1000;
	filepath = filepath.replace(dir_names[-1],"*."+extension); 
	return filepath,unit,filename;

def openData():
	import glob
	filepath,unit,filename = getFilepath();
	img = ((np.array([skimage.io.imread(file) for file in sorted(glob.glob(filepath))])).transpose(2,1,0))
	img = img[:,:,50:np.min([1550,img.shape[2]-50])];
	return img,unit,filename;


def getNiftiObject():
	from tkinter.filedialog import askopenfilename
	from tkinter import Tk
	import re
	Tk().withdraw()
	filepath = askopenfilename();
	extension = filepath.split(".")[-1];
	dir_names = filepath.split("/");
	filename = dir_names[-1].split(".");
	import nibabel as nib;
	img = nib.load(filepath);
	header = img.header
	pixdim = header.get('pixdim');
	unit = pixdim[1]
	img_array = img.get_data();
	s = img_array.shape;
	if len(s)>3:
		img_array = img_array[:,:,:,0];
	return [img_array,unit,filename[0]];
	
	
def saveNiftiObject(data,unit,filepath):
	import nibabel as nib;
	print(unit);
	header_ = nib.Nifti1Header();
	header_['pixdim'] = np.array([ unit ,unit,unit,unit ,  1. ,  1. ,  1. ,  1. ], dtype=np.float32);
	nifti_img = nib.Nifti1Image(data, affine=None, header=header_);
	nib.save(nifti_img, filepath)


