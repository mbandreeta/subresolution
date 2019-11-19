from scipy import signal
import numpy as np
import tqdm

def gaussn(x, *p):
	g = np.zeros_like(x)
	for idx in range(0,len(p),3):
		mu = p[idx];
		sigma = p[idx+1];
		A = p[idx+2];
		g = g +  (A*np.exp(-(x-mu)**2/(2.*sigma**2)));
	try:
		r = g[0,:];
	except:
		r = g;
	return r; 

def createMap(sigma,filtered):
	import skimage;
	obs = filtered[filtered<255];
	obs = obs[obs>0];
	y,x = skimage.exposure.histogram(obs);
	y = y/np.sum(y)
	psf = signal.gaussian(3*sigma,std=sigma);
	im_deconv = np.full(y.shape, 0.5)
	psf_mirror = psf[::-1]
	for _ in range(20000):
		relative_blur = y / signal.convolve(im_deconv, psf, 'same')
		im_deconv *= signal.convolve(relative_blur, psf_mirror, 'same')
	q,r = signal.deconvolve(y,psf);
	res = np.zeros(y.size);
	q[q<0]=0;
	res[0:q.size]=q[:]
	#plt.plot(res);plt.show();
	mu_map =list();
	for eta in (range(0,256)):
		params= list([eta,sigma,np.max(res)]);
		f_eta_mu = gaussn(x,*params)
		f_mu_eta = f_eta_mu*res/y;
		mu_real = np.argmax(f_mu_eta)
		mu_map.append(mu_real);
	return mu_map

def correctValues(sigma,filtered):
	mu_map = createMap(sigma,filtered)
	f = filtered.copy();
	nx,ny,nz = f.shape;
	for i in (range(nx)):
		for j in range(ny):
			for k in range(nz):
				f[i,j,k] = mu_map[int(f[i,j,k])];
	return f