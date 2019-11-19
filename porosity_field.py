
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



from read_load import *
from porosity_funcs import *
from util_f import *
from deconvolve import *

if __name__ == "__main__":
	import sys
	porosity_estimated=0
	if len(sys.argv) == 2:
		porosity_estimated = float(sys.argv[1]);
	from scipy.ndimage import median_filter ,label
	img,unit,filename = openData();
	print(filename,unit)
	#unit = unit*1e-6;	
	print("Applying filter.");
	sitk_image = resample_image(sitk.GetImageFromArray(img))
	img = sitk.GetArrayFromImage(sitk_image);
	img = histogram_normalization(img)
	filtered = median_filter(img,2);
	print("Creating mask.");
	mask,obs = createMask(filtered);	
	print("Gaussian mixture");
	Evoid,Mvoid,Esolid,Msolid = gaussianMixture(obs,plotdata=True);
	t = skimage.filters.threshold_li(filtered);
	macro_porosity = filtered[mask==1];
	macro_porosity = np.sum(macro_porosity<t);	
	macro_porosity = 100.0*(np.sum(macro_porosity)/np.sum(mask));
	if macro_porosity > 0.1:
		Evoid = t;
	print(Esolid,Evoid)
	corrected = correctValues(int(Msolid),filtered);
	if(porosity_estimated>0):
		print("Interactive correction");
		porosity_matrix,Evoid = interactiveCorrection(Esolid,Msolid,Evoid,corrected,filtered,mask,porosity_estimated)
	else:
		print("Automatic map");
		segmented = correctValues(int(Msolid),filtered)#filtered.copy();
		segmented = ((segmented - Evoid)/(Esolid-Evoid));
		segmented[filtered>=Esolid]=1; # solid region
		segmentSolidPhase(segmented,filtered,Esolid-(4*Msolid))# empty region
		segmented[filtered<=Evoid]=0;
		porosity_matrix=(1-segmented)*mask;
	#saveNiftiObject(porosity_matrix,unit,"D:/POROSITY/pm.nii");
	print("Porosity:", 100.0*(np.sum(porosity_matrix)/np.sum(mask)) );
	macro_porosity = filtered[mask==1];
	macro_porosity = np.sum(macro_porosity<Evoid);
	print("Porosity:", 100.0*(np.sum(macro_porosity)/np.sum(mask)) );
	print("Downsampling image: 20%");
	sitk_image = resample_image(sitk.GetImageFromArray(porosity_matrix),out_spacing=(5.5555, 5.5555, 5.5555))
	porosity_matrix = sitk.GetArrayFromImage(sitk_image);
	sitk_image = resample_image(sitk.GetImageFromArray(mask*255),out_spacing=(5.5555, 5.5555, 5.5555),is_label=True)
	mask = (sitk.GetArrayFromImage(sitk_image)>0)*1;
	print("Porosity:", 100.0*(np.sum(porosity_matrix*mask)/np.sum(mask)) );
	print("h:", str(porosity_matrix.shape))	
	porosity_matrix=porosity_matrix*mask;
	map_radius,map_surface = radiusMap(porosity_matrix,mask,unit*5.5555)
	np.savetxt("map_radius.txt", map_radius.flatten('F'),fmt='%10.5f')	
	np.savetxt("map_surface.txt", map_surface.flatten('F'),fmt='%10.5f')	
	np.savetxt("porosity_matrix.txt", porosity_matrix.flatten('F'),fmt='%10.5f')	