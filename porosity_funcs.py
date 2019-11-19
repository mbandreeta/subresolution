import skimage
import tqdm
import numpy as np
from deconvolve import *

def gaussianMixture(img,components=4, iswormhole=False , plotdata=False):
	def gaussn(x, *p):
		g = np.zeros_like(x)
		for idx in range(0,len(p),3):
			mu = p[idx];
			sigma = p[idx+1];
			A = p[idx+2];
			g = g +  (A*np.exp(-(x-mu)**2/(2.*sigma**2)));
		return g[0,:]; 
	from sklearn import mixture
	obs = img.flatten();
	obs = obs.reshape(-1,1);
	N = np.arange(2, 7)
	models = [None for i in range(len(N))]
	for i in range(len(N)):
		models[i] = mixture.GaussianMixture(N[i]).fit(obs)
	AIC = [m.aic(obs) for m in models]	
	g = models[np.argmin(AIC)]

	# compute the AIC and the BIC

	# g = mixture.GaussianMixture(n_components=components);
	# g.fit(obs);
	params = list();
	means = list();
	for i in range(components):
		means.append(g.means_[i][0]);
	for i in np.argsort(means):
		params.append(g.means_[i][0]);
		params.append(np.sqrt(g.covariances_[i][0,0]));
		params.append(g.weights_[i]/np.sqrt(2*np.pi*g.covariances_[i][0,0]));		  
	if(plotdata):
		import matplotlib.pyplot as plt
		plt.subplot(1,2,1);	
		yy,xx,_ =plt.hist(obs,bins=100)
		plt.subplot(1,2,2);
		xl = np.linspace(np.min(xx),np.max(xx),100).reshape(1,-1);
		for i in range(components):
			pg = params[i*3:(i+1)*3]
			plt.plot(xl[0,:],gaussn(xl,*pg))
		plt.hist(obs,bins=100,density=True)
		plt.plot(xl[0,:],gaussn(xl,*params),color="black")
	n=len(params);
	if iswormhole:
		return [params[3],params[4],params[n-3],params[n-2]];
	else:
		return [params[0],params[1],params[n-3],params[n-2]];



def segmentSolidPhase(segmented,f,threshold):
	from scipy.ndimage import binary_dilation
	w=1;
	d = binary_dilation(segmented==1);
	d = d-((segmented==1)*1)
	positions = np.where(d>0);
	for idx in (range(positions[0].size)):
		[i,j,k] =  [positions[0][idx],positions[1][idx],positions[2][idx]];
		if f[i,j,k] >= threshold:
			segmented[i,j,k] = 1;



def segmentVoidPhase(segmented,f,threshold):
	from scipy.ndimage import binary_dilation
	w=1;
	d = binary_dilation(segmented==0);
	d = d-((segmented==0)*1)
	positions = np.where(d>0);
	for idx in (range(positions[0].size)):
		[i,j,k] =  [positions[0][idx],positions[1][idx],positions[2][idx]];
		if f[i,j,k] <= threshold:
			segmented[i,j,k] = 0;


def radiusMap(porosity_matrix,unit):
	from scipy.ndimage import median_filter 
	macropores = (porosity_matrix==1)*1;
	intermediate = porosity_matrix.copy();
	intermediate[macropores==1]=0;
	v = intermediate* (unit**3) *(3.0/(4.0*np.pi));
	r_eq = np.cbrt(v)
	surface = (r_eq**2)*np.pi ;
	from scipy.ndimage.morphology import distance_transform_edt as bwt
	dt = bwt(macropores)
	map_radius = r_eq+ (dt*unit);
	aux = (dt<np.sqrt(2))*dt * (unit**2);	
	surface_contact = aux*intermediate;
	map_surface = surface_contact+surface;
	return map_radius,map_surface;

def interactiveCorrection(Esolid,Msolid,Evoid,corrected,filtered,mask,porosity_estimated=0,tolerance=2):
	porosity=0;
	erro = np.abs(porosity_estimated-porosity)*100.0/porosity_estimated;
	step=10;
	segmented = corrected.copy();
	segmentSolidPhase(segmented,filtered,Esolid-(4*Msolid))# empty region;
	solid_region = (segmented==1)
	while(erro>tolerance):
		segmented = corrected.copy();#filtered.copy();
		segmented = ((segmented - Evoid)/(Esolid-Evoid));
		segmented[filtered>=Esolid]=1; # solid region
		segmented[solid_region]=1
		segmented[filtered<=Evoid]=0;
		porosity_matrix=(1-segmented)*mask;
		porosity =  100.0*(np.sum(porosity_matrix)/np.sum(mask));
		print(porosity_estimated,porosity,Evoid,erro)
		nerro = np.abs(porosity_estimated-porosity)*100.0/porosity_estimated;
		if(nerro<erro):
			erro = nerro;
		else:
			step = step/2;
			erro = nerro;
		if(porosity>porosity_estimated):
			Evoid=Evoid-step;
			if(Evoid<0):
				Evoid=0;
				step=10;
				Esolid = Esolid-step;
				segmented = corrected.copy();
				segmentSolidPhase(segmented,filtered,Esolid-(4*Msolid))# empty region;
				solid_region = (segmented==1)
		else:
			Evoid=Evoid+step;
	return porosity_matrix,Evoid;
