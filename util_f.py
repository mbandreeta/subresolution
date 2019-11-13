import matplotlib.pyplot as plt
import skimage
import numpy as np
import tqdm
import SimpleITK as sitk

def resample_image(sitk_image, out_spacing=(2, 2, 2), is_label=False):
	
	original_spacing = sitk_image.GetSpacing()
	original_size = sitk_image.GetSize()

	out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
				int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
				int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

	resample = sitk.ResampleImageFilter()
	resample.SetOutputSpacing(out_spacing)
	resample.SetSize(out_size)
	resample.SetOutputDirection(sitk_image.GetDirection())
	resample.SetOutputOrigin(sitk_image.GetOrigin())
	resample.SetTransform(sitk.Transform())
	resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())

	if is_label:
		resample.SetInterpolator(sitk.sitkNearestNeighbor)
	else:
#		resample.SetInterpolator(sitk.sitkBSpline)
		resample.SetInterpolator(sitk.sitkLinear)

	return resample.Execute(sitk_image)


def histogram_normalization(img):
	img = (img/np.max(img));
	for z in tqdm.tqdm(range(img.shape[2])):
		# val = skimage.filters.threshold_otsu(img[:,:,z]);
		# obs = img[:,:,z].copy()
		# obs = obs[obs>val]
		# h,v = np.histogram(obs,100);
		# intensity = (v[1:]+v[:-1])/2.0;
		# cutoff = h[np.argmax(h)]*0.3;
		# p = np.where(h>cutoff);
		# min_intensity = intensity[p[0][0]];
		# max_intensity = intensity[p[0][-1]];	
		p2, p98 = np.percentile(img[:,:,z], (2, 98))
		img[:,:,z] = skimage.exposure.rescale_intensity(img[:,:,z], in_range=(p2, p98), out_range=(0,255));
	return img;

def createMask(filtered):
	from scipy.ndimage import label,binary_erosion
	from scipy.ndimage import distance_transform_edt as bwdist
	from skimage.filters import threshold_otsu
	val = threshold_otsu(filtered);
	mask = np.zeros(filtered.shape);
	for z in tqdm.tqdm(range(filtered.shape[2])):
		label_img = label(filtered[:,:,z]<=val)[0];
		mask[:,:,z] = binary_erosion(1-((label_img==label_img[0,0])*1));	
	dist_mask  = bwdist(mask[:,:,0]);
	radius = np.max(dist_mask);
	[x,y] = np.where(dist_mask==radius);
	x=int(x[0]);
	y=int(y[0]);
	z = int(filtered.shape[2]/2.0);
	radius = int(radius/2.0);
	obs = filtered[:,:,100:500].copy();
	mobs = mask[:,:,100:500].copy()
	mobs = mobs==1;
	obs = obs[mobs].flatten()
	return mask,obs;

def plot3d(image3d,precision,clr,opt):
	from mayavi import mlab;
	import numpy as np;
	sf=mlab.pipeline.scalar_field(image3d);
	if(opt!=0):
		figw = mlab.pipeline.iso_surface(sf,contours=[image3d.min()+precision*image3d.ptp()],color=clr,opacity=opt);
	else:
		figw = mlab.pipeline.iso_surface(sf,contours=[image3d.min()+precision*image3d.ptp()],color=clr);  


def plot_img_and_hist(img, bins=256):
	plt.figure();
	# Display cumulative distribution
	img_cdf, bins = skimage.exposure.cumulative_distribution(img, bins)
	plt.plot(bins, img_cdf, 'r')
	plt.show();
	return np.max(np.where(img_cdf<0.5))	

def showData(img,filtered,segmented):
	plt.show();
	plt.figure();
	plt.subplot(2,3,1);
	plt.imshow(img[:,:,100],cmap='gray');
	plt.subplot(2,3,2);
	plt.imshow(filtered[:,:,100],cmap='gray');
	plt.subplot(2,3,3);
	plt.imshow(segmented[:,:,100],cmap='gray');
	plt.subplot(2,3,4);
	plt.hist(img.flatten(),bins=100);
	plt.subplot(2,3,5);
	plt.hist(filtered.flatten(),bins=100);
	plt.subplot(2,3,6);
	plt.hist(segmented.flatten(),bins=100);	
	plt.show();