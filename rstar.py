from pylab import *
import scipy
from scipy import interpolate,integrate
import glob
import emcee
import corner
import pickle
import seaborn as sns
from extinction import ccm89, apply
from bs4 import BeautifulSoup
import urllib
from astroquery.vizier import Vizier
import astropy.units as u
import astropy.coordinates as coord
import os 
from astropy.io import fits as pyfits

def download():
	if os.access('bt-settl-cifist',os.F_OK) == False:
		os.system('mkdir bt-settl-cifist')
	for i in range(1,447):
		os.system('wget http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-cifist&fid='+str(i)+'&format=ascii')



def download_cohelo():
	if os.access('coelho_seds',os.F_OK) == False:
		os.system('mkdir coelho_seds')

	os.system('wget http://specmodels.iag.usp.br/fits_search/compress/s_coelho14_sed.tgz')
	os.system('tar -zxvf s_coelho14_sed.tgz')
	os.system('mv s_coelho14_sed/*  coelho_seds')
	os.system('rm -r s_coelho14_sed s_coelho14_sed.tgz')
	

def trans_coords(ra,dec):
	cos = ra.split(':')
	rad = float(cos[0]) +  float(cos[1])/60. + float(cos[2])/3600.
	cos = dec.split(':')
	if float(cos[0])<0:
		pre = -1
	else:
		pre = 1

	decd = np.absolute(float(cos[0])) +  float(cos[1])/60. + float(cos[2])/3600.

	return rad*180./12.,pre*decd

def get_mags_2mass(RA,DEC):
	result = Vizier.query_region(coord.SkyCoord(ra=RA,dec=DEC,unit=(u.deg, u.deg),\
		frame="icrs"),width='10s',catalog='II/246')

	dct = {}
	if len(result)==0:
		dct['2mass_J']  = -1
		dct['2mass_H']  = -1
		dct['2mass_K']  = -1
		dct['e2mass_J'] = -1
		dct['e2mass_H'] = -1
		dct['e2mass_Ks'] = -1
	else:
		dat = result[0]
		if len(dat) > 1:
			dist = np.sqrt((dat['RAJ2000'] - RA)**2 + (dat['DEJ2000'] - DEC)**2)
			imin = np.argmin(dist)
			dat = dat[imin]
		dct['2mass_J']  = float(dat['Jmag'])
		dct['2mass_H']  = float(dat['Hmag'])
		dct['2mass_Ks']  = float(dat['Kmag'])
		dct['e2mass_J'] = float(dat['e_Jmag'])
		dct['e2mass_H'] = float(dat['e_Hmag'])
		dct['e2mass_Ks'] = float(dat['e_Kmag'])
	return dct

def get_mags_WISE(RA,DEC,models='bt-settl-cifist'):
	result = Vizier.query_region(coord.SkyCoord(ra=RA,dec=DEC,unit=(u.deg, u.deg),\
		frame="icrs"),width='10s',catalog='WISE')
	dct = {}
	if len(result) == 0:
		dct['W1']  = -1
		dct['W2']  = -1
		dct['W3']  = -1
		dct['W4']  = -1
		dct['eW1'] = -1
		dct['eW2'] = -1
		dct['eW3'] = -1
		dct['eW4'] = -1
	else:
		dat = result[0]
		#print dat
		if len(dat) > 1:
			dist = np.sqrt((dat['RAJ2000'] - RA)**2 + (dat['DEJ2000'] - DEC)**2)
			imin = np.argmin(dist)
			dat = dat[imin]
		dct['W1']  = float(dat['W1mag'])
		dct['W2']  = float(dat['W2mag'])
		dct['eW1'] = float(dat['e_W1mag'])
		dct['eW2'] = float(dat['e_W2mag'])

		if models == 'bt-settl-cifist':
			dct['W3']  = float(dat['W3mag'])
			dct['W4']  = float(dat['W4mag'])
			dct['eW3'] = float(dat['e_W3mag'])
			dct['eW4'] = float(dat['e_W4mag'])

	return dct

def get_mags_APASS(RA,DEC):
	query_url = 'https://www.aavso.org/cgi-bin/apass_download.pl?ra='+str(RA)+'&dec='+str(DEC)+'+%09&radius=0.01&outtype=0'
	html = urllib.urlopen(query_url).read()
	soup = BeautifulSoup(html, 'html.parser')
	dct =  str(soup).split('<tr>')
	if len(dct)>1:
		dct = dct[1:]
		dists = []
		for chunk in dct:
			#print chunk
			data = chunk.split('<td><font size="-1">')
			ra = float(data[1].split('<')[0])
			dec = float(data[3].split('<')[0])
			#print ra, dec
			dist = np.sqrt((RA - ra)**2 + (DEC - dec)**2)
			dists.append(dist)
		dists = np.array(dists)
		Im = np.argmin(dists)
		#print dists, Im
		chunk = dct[Im]

		mags = []
		smags =  chunk.split('<td><font size="-1">')
		i = 6
		while i < len(smags):
			mag = smags[i].split('<')[0]
			if not 'NA' in mag:
				mag = float(mag)
			else:
				mag = -1
			mags.append(mag)
			i+=2
		emags = []
		i = 7
		while i < len(smags):
			emag = smags[i].split('<')[0]
			if not 'NA' in emag:
				emag = float(emag)
			else:
				emag = -1
			emags.append(emag)
			i+=2
		
	dct = {}
	dct['Johnson_B'],dct['Johnson_V'],dct['SDSS_g'],dct['SDSS_r'],dct['SDSS_i'] = mags[1],mags[0],mags[2],mags[3],mags[4]
	dct['eJohnson_B'],dct['eJohnson_V'],dct['eSDSS_g'],dct['eSDSS_r'],dct['eSDSS_i'] = emags[1],emags[0],emags[2],emags[3],emags[4]

	return dct

def mag_to_flux(mag,emag,name):

	dct = {'Johnson_B':6.491e-9,
		   'Johnson_V':3.734e-9,
		   'SDSS_g':5.056e-9,
		   'SDSS_r':2.904e-9,
		   'SDSS_i':1.967e-9,
		   '2mass_J':3.129e-10,
		   '2mass_H':1.133e-10,
		   '2mass_Ks':4.283e-11,
		   'W1':8.081e-12,
		   'W2':2.397e-12,
		   'W3':7.112e-14,
		   'W4':4.507e-15,
		   }
	dct2 = {'Johnson_B':4297.17,
		   'Johnson_V':5340.00	,
		   'SDSS_g':4640.42,
		   'SDSS_r':6122.33,
		   'SDSS_i':7439.49,
		   '2mass_J':12350.00,
		   '2mass_H':16620.00,
		   '2mass_Ks':21590.00,
		   'W1':33526.00,
		   'W2':46028.00,
		   'W3':115608.00,
		   'W4':220883.00,
		   }

	flux  = dct[name] * 10**(-mag/2.5)
	flux2 = dct[name] * 10**(-(mag-emag)/2.5)
	flux3 = dct[name] * 10**(-(mag+emag)/2.5)

	f1 = flux2 - flux
	f2 = flux - flux3
	eflux = 0.5*(f1+f2)

	if mag == -1:
		flux = -1
	if emag == 0:
		eflux = -1
	return flux,eflux,dct2[name]

def test_it(RA,DEC,mname):
	dct = get_mags_2mass(RA,DEC)
	dct.update(get_mags_APASS(RA,DEC))
	dct.update(get_mags_WISE(RA,DEC))

	#dct.update(get_mags_WISE(242.573464954,-24.9905949453))
	#print dct
	#print mag_to_flux(dct[mname],dct['e'+mname],mname)

def plot(target,rstar,Av,query=True,correction=True,models='bt-settl-cifist'):
	# Set seaborn contexts:
	sns.set_context("talk")
	sns.set_style("ticks")

	# Fonts:
	# Arial font (pretty, not latex-like)
	#rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
	# Latex fonts, quick:
	#matplotlib.rcParams['mathtext.fontset'] = 'stix'
	#matplotlib.rcParams['font.family'] = 'STIXGeneral'
	# Latex fonts, slow (but accurate):
	rc('font', **{'family': 'Helvetica'})
	rc('text', usetex=True)
	matplotlib.rcParams.update({'font.size':20})
	plt.rc('legend', **{'fontsize':7})

	# Ticks to the outside:
	rcParams['axes.linewidth'] = 3.0
	rcParams['xtick.direction'] = 'out'
	rcParams['ytick.direction'] = 'out'

	global nsyn_wavs, nsyn_flxs, nsyn_widths,band
	global refw, fluxes

	f = open(target+'_pars.txt','r')
	lines = f.readlines()
	have_dist = False
	for line in lines:
		cos = line.split()
		if 'parallax' in cos[0]:
			parallax = float(cos[1])
			eparallax = float(cos[2])
		elif 'teff' in cos[0]:
			teff = float(cos[1])
			eteff = float(cos[2])
		elif 'logg' in cos[0]:
			logg = float(cos[1])
			elogg = float(cos[2])
		elif 'feh' in cos[0]:
			feh = float(cos[1])
			efeh = float(cos[2])
		elif 'ra' in cos[0]:
			ra = cos[1]
		elif 'dec' in cos[0]:
			dec = cos[1]
		elif 'dist' in cos[0]:
			dist = float(cos[1])
			edist = float(cos[2])
			have_dist = True

	print parallax, eparallax
	if correction:
		parallax += 0.029
		eparallax += 0.4*eparallax
	print parallax, eparallax
	global distance
	#print have_dist
	if not have_dist:
		#print parallax, eparallax
		parallax = parallax / 1000.
		eparallax = eparallax / 1000.
		distance = 3.0857e18 / parallax
	else:
		distance = 3.0857e16 * dist * 100.

	if query == False:

		f = open(target+'_fluxes.txt','r')
		lines = f.readlines()
		band,fluxes,efluxes,refw = [],[],[],[]
		for line in lines:
			cos = line.split()
			band.append(cos[0])
			fluxes.append(float(cos[2]))
			efluxes.append(float(cos[3]))
			refw.append(get_bpass_pos(cos[0]))
		band,fluxes,efluxes,refw = np.array(band),np.array(fluxes),np.array(efluxes),np.array(refw)	
		f.close()

	else:
		try:
			ra,dec = float(ra),float(dec)
		except:
			ra,dec = trans_coords(ra,dec)
		#print 'coords:',ra, dec
		dct = get_mags_2mass(ra,dec)
		dct.update(get_mags_APASS(ra,dec))
		dct.update(get_mags_WISE(ra,dec,models=models))
		print dct

		band,fluxes,efluxes,refw = [],[],[],[]
		for key in dct.keys():
			if key[0] != 'e':

				#print dct[key],dct['e'+key],key
				flux, eflux, refwt = mag_to_flux(dct[key],dct['e'+key],key)
				if flux != -1 and eflux != -1 and np.isnan(flux)==False and np.isnan(eflux)==False:
					fluxes.append(flux)
					efluxes.append(eflux)
					refw.append(refwt)
					band.append(key)
		band,fluxes,efluxes, refw = np.array(band),np.array(fluxes),np.array(efluxes),np.array(refw)

	"""
	f = open(target+'_pars.txt','r')
	lines = f.readlines()
	for line in lines:
		cos = line.split()
		if 'parallax' in cos[0]:
			parallax = float(cos[1])
			eparallax = float(cos[2])
		elif 'teff' in cos[0]:
			teff = float(cos[1])
			eteff = float(cos[2])
		elif 'logg' in cos[0]:
			logg = float(cos[1])
			elogg = float(cos[2])
		elif 'feh' in cos[0]:
			feh = float(cos[1])
			efeh = float(cos[2])
	parallax = parallax / 1000.
	eparallax = eparallax / 1000.
	distance = 3.0857e18 / parallax
	"""
	y =  fluxes * distance * distance

	Rscm = rstar * 6.95700e10

	y = y / (Rscm**2)

	w,f = get_sed(teff,logg,0.0,models='bt-settl-cifist')
	#Al = extinction.ccm89(w, Av, 3.1)
	#f = f*np.exp(-get_Al(Av, w)/2.5)

	f = apply(ccm89(w, Av, 3.1), f)

	figure()
	modsed = f*Rscm*Rscm/(distance*distance)
	loglog(w,modsed,color='g',alpha=0.4)
	loglog(refw,fluxes,'ro')
	for i in range(len(band)):
		cos = band[i].split('_')
		#print band[i]
		text(refw[i],3*fluxes[i],cos[-1],fontsize=18)
	xlim([1e3,3e5])

	ylim([1e-25,1e-10])
	xlabel(r'Wavelength [$\AA$]')
	ylabel(r'F [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]')
	plt.savefig(target+'_sed.pdf', bbox_inches='tight')

	#show()

def get_vals(vec):
	fvec   = np.sort(vec)

	fval  = np.median(fvec)
	nn = int(np.around(len(fvec)*0.15865))

	vali,valf = fval - fvec[nn],fvec[-nn] - fval
	return fval,vali,valf


def reduce_models():
	models = glob.glob('bt-settl-cifist/*txt')
	for model in models:
		d1 = np.loadtxt(model,skiprows=8)
		II = np.arange(0,d1.shape[0],100)
		plot(d1[:,0],d1[:,1])
		plot(d1[II,0],d1[II,1])
		show()
		#print gtfdxs

def Synthetic_Fluxes():
	models = glob.glob('bt-settl-cifist/*.dat.txt')
	db = np.array(['Johnson_U','Johnson_B','Johnson_V','Gunn_I','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','2mass_J','2mass_H','2mass_Ks','W1','W2','W3','W4'])
	for model in models:
		#print model
		d1 = np.loadtxt(model,skiprows=8)
		wavs,flxs = d1[:,0],d1[:,1]
		fname = model.replace('dat','bands')
		#print fname
		f = open(fname,'w')
		for tband in db:
			rw,rr = get_bpass_response(tband)
			rr = rr/np.max(rr)
			JJ    = np.where(rr>0.001)[0]
			rw,rr = rw[JJ],rr[JJ]

			II = np.where((wavs>rw[0]) & (wavs<rw[-1]))[0]
			tck = scipy.interpolate.splrep(rw,rr,k=1)
			rr  = scipy.interpolate.splev(wavs[II],tck)

			wt1 = wavs[II].copy()
			wt2 = np.hstack((wt1[0],wt1[:-1]))
			dws = wt1-wt2
			dws[0] = dws[1]
			FF  = np.sum(rr*flxs[II]*dws)/np.sum(rr*dws)#get_bpass_width(band)
			fout = tband + '\t' + str(FF) + '\n'
			#print fout[:-1]
			f.write(fout)
		f.close()
		#print '\n'


def syn_flux(wavs,flxs,db):
	#db = np.array(['Johnson_U','Johnson_B','Johnson_V','Gunn_I','SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z','2mass_J','2mass_H','2mass_Ks','W1','W2','W3','W4'])
	outs = []
	outsw = []
	#print wavs
	#print flxs
	#print db
	for tband in db:
		#print tband
		rw,rr = get_bpass_response(tband)
		rr = rr/np.max(rr)
		JJ    = np.where(rr>0.001)[0]
		rw,rr = rw[JJ],rr[JJ]
		rwo,rro = rw.copy(),rr.copy()
		II = np.where((wavs>rw[0]) & (wavs<rw[-1]))[0]
		#print wavs
		#print II, rw[0],rw[-1]
		tck = scipy.interpolate.splrep(rw,rr,k=1)
		rr  = scipy.interpolate.splev(wavs[II],tck)

		wt1 = wavs[II].copy()
		wt2 = np.hstack((wt1[0],wt1[:-1]))
		dws = wt1-wt2
		dws[0] = dws[1]
		FF  = np.sum(rr*flxs[II]*dws)/np.sum(rr*dws)#get_bpass_width(band)
		outsw.append(np.sum(rr*wavs[II]*dws)/np.sum(rr*dws))
		outs.append(FF)
	outs = np.array(outs)
	outsw = np.array(outsw)
	return outsw,outs



def get_bpass_pos(bpassname):
	bps = {'W1':33680.0,\
		   'W2':46180.0,\
		   'W3':120820.0,\
		   'W4':221940.0,\
		   'Johnson_U': 3750.0,\
		   'Johnson_B': 4300.0,\
		   'Johnson_V': 5300.0,\
		   'Gunn_I':    8600.0,\
		   '2mass_J':  12350.0,\
		   '2mass_H':  16620.0,\
		   '2mass_Ks': 21590.0,\
		   'SDSS_u':    3551.0,\
		   'SDSS_g':    4686.0,\
		   'SDSS_r':    6165.0,\
		   'SDSS_i':    7481.0,\
		   'SDSS_z':    8931.0\
		   }

	return bps[bpassname]

def get_bpass_width(bpassname):
	bps = {'W1': 6625.6,\
		   'W2': 10423.0,\
		   'W3': 55069.0,\
		   'W4': 41013.0,\
		   'Johnson_U':          600.0,\
		   'Johnson_B':  1100.0,\
		   'Johnson_V':          900.0,\
		   'Gunn_I':            1400.0,\
		   '2mass_J':           1620.0,\
		   '2mass_H':           2510.0,\
		   '2mass_Ks':          2620.0,\
		   'SDSS_u':    599.0,\
		   'SDSS_g':    1379.0,\
		   'SDSS_r':    1382.0,\
		   'SDSS_i':    1535.0,\
		   'SDSS_z':    1370.0\
		   }
	return bps[bpassname]

def get_bpass_response(bpassname):
	d = np.loadtxt('passbands/'+bpassname+'.txt')
	if bpassname == 'W1' or bpassname == 'W2' or bpassname == 'W3' or bpassname == 'W4' or '2mass' in bpassname: 
		return d[:,0]*10000,d[:,1]
	else:
		return d[:,0],d[:,1]

def get_Al(Av, waves,Rv=3.1):
	Al = np.zeros(len(waves))
	ass = np.zeros(len(waves))
	bs = np.zeros(len(waves))

	waves = waves/10000.

	x = 1./waves

	I = np.where((x>0.3)&(x<=1.1))[0]
	if len(I)>0:
		ass[I] = 0.574 * x[I] ** 1.61
		bs[I] = -0.527 * x[I] ** 1.61

	I = np.where((x>1.1) & (x<=3.3))[0]
	if len(I)>0:
		y = x[I] - 1.82
		ass[I] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 -0.77530*y**6 + 0.32999*y**7
		bs[I] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

	I = np.where((x>3.3) & (x<=8))[0]
	if len(I)>0:
		J = np.where(x[I]<5.9)[0]
		Fax = -0.04473 * (x[I] - 5.9)**2 - 0.009779 * (x[I] - 5.9)**3
		if len(J)>0:
			Fax[J] = 0.

		Fbx = 0.2130 * (x[I]-5.9)**2 + 0.1207 * (x[I] - 5.9)**3
		if len(J)>0:
			Fbx[J] = 0.

		ass[I] = Fax + 1.752 -0.316*x[I] -0.104 / ((x[I]-4.67)**2 + 0.341)
		bs[I] = Fbx - 3.090 + 1.825 * x[I] + 1.206 / ((x[I] - 4.62)**2 + 0.263)

	I = np.where(x>8)[0]
	if len(I) > 0:
		ass[I] = -1.073 -0.628*(x[I] - 8) + 0.137*(x[I]-8)**2 - 0.070*(x[I]-8)**3
		bs[I] = 13.670 + 4.257*(x[I]-8) - 0.420*(x[I] - 8)**2 +0.374*(x[I]-8)**3

	Al = Av * (ass + bs/Rv)
	return Al

def clean(wt,ft,w):
	i = 0
	while i< len(wt):
		if wt[i] != w[i]:
			#print i,wt[i],w[i]
			wt = np.delete(wt,i)
			ft = np.delete(ft,i)
		else:	
			i+=1
	return ft


def bilinear(t,g,t1,t2,g1,g2, feh=0., models='bt-settl-cifist'):
	
	#print t,g,t1,t2,g1,g2
	if models == 'bt-settl-cifist':
		st1 = str(int(t1/100.))
		st2 = str(int(t2/100.))
		sg1 = str(np.around(g1,1))
		sg2 = str(np.around(g2,1))

		if t1 == t2 and g1 == g2:
			d =  np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=20000)[:1200000]
			I = np.where((d[:,0]>100)&(d[:,0]<300000))[0]
			return d[I,0],d[I,1]

		elif t1 == t2:
			d1 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=20000)[:1200000]
			d2 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=20000)[:1200000]
			I1 = np.where((d1[:,0]>100)&(d1[:,0]<300000))[0]
			I2 = np.where((d2[:,0]>100)&(d2[:,0]<300000))[0]
			w1,f1 = d1[I1,0],d1[I1,1]
			w2,f2 = d2[I2,0],d2[I2,1]
			m = (f2 -f1)/(g2-g1)
			n = f2 - m*g2
			f = m*g + n
			return w1,f
		elif g1 == g2:
			d1 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=20000)[:1200000]
			d2 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=20000)[:1200000]
			I1 = np.where((d1[:,0]>100)&(d1[:,0]<300000))[0]
			I2 = np.where((d2[:,0]>100)&(d2[:,0]<300000))[0]
			w1,f1 = d1[I1,0],d1[I1,1]
			w2,f2 = d2[I2,0],d2[I2,1]
			m = (f2 -f1)/(t2-t1)
			n = f2 - m*t2
			f = m*t + n
			return w1,f

		else:
			#print 'bt-settl-cifist/lte0'+st1+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt'
			#print 'bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt'
			#print 'bt-settl-cifist/lte0'+st1+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.dat.txt'
			#print 'bt-settl-cifist/lte0'+st2+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.dat.txt'
			d11 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=500)[:1200000]
			d12 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=500)[:1200000]
			d21 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=500)[:1200000]
			d22 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.dat.txt',skiprows=500)[:1200000]
		
			#I11 = np.where((d11[:,0]>100)&(d11[:,0]<300000))[0]
			#I12 = np.where((d12[:,0]>100)&(d12[:,0]<300000))[0]
			#I21 = np.where((d21[:,0]>100)&(d21[:,0]<300000))[0]
			#I22 = np.where((d22[:,0]>100)&(d22[:,0]<300000))[0]
		
			w11,f11 = d11[:,0],d11[:,1]
			w12,f12 = d12[:,0],d12[:,1]
			w21,f21 = d21[:,0],d21[:,1]
			w22,f22 = d22[:,0],d22[:,1]

			if len(w11) <= len(w12) and len(w11) <= len(w21) and len(w11) <= len(w22):
				refw = w11
				#print 'w11'
			elif len(w12) <= len(w11) and len(w12) <= len(w21) and len(w12) <= len(w22):
				refw = w12
				#print 'w12'
			elif len(w21) <= len(w11) and len(w21) <= len(w12) and len(w21) <= len(w22):
				refw = w21
				#print 'w21'
			elif len(w22) <= len(w11) and len(w22) <= len(w11) and len(w22) <= len(w21):
				refw = w22
				#print 'w22'
		
			#print f11.shape,f12.shape,f21.shape,f22.shape
			f11 = clean(w11,f11,refw)
			f12 = clean(w12,f12,refw)
			f21 = clean(w21,f21,refw)
			f22 = clean(w22,f22,refw)
			#print f11.shape,f12.shape,f21.shape,f22.shape
			"""
			w12,f12 = d12[I12,0],d12[I12,1]
			I = np.where(np.isnan(f12))
			tck = interpolate.splrep(w12[:1000000],f12[:1000000],k=1)
			f12 = interpolate.splev(w11[:1000000],tck)
			print f12
			print gfds
			w21,f21 = d21[I21,0],d21[I21,1]
			tck = interpolate.splrep(w21,f21,k=1)
			f21 = interpolate.splev(w11,tck)
			w22,f22 = d22[I22,0],d22[I22,1]
			"""

			fxy1 = f11 * (t2-t)/(t2-t1) + f21 * (t-t1)/(t2-t1)
			fxy2 = f12 * (t2-t)/(t2-t1) + f22 * (t-t1)/(t2-t1)
			fxy  = fxy1 * (g2-g)/(g2-g1) + fxy2 * (g-g1)/(g2-g1)
			return w11, fxy
	elif models == 'coelho_seds':
		if feh<0:
			sign = 'm'
		else:
			sign = 'p'
		feh *= 10
		feh = int(np.absolute(feh))
				
		if feh<10:
			pre = sign+'0'+str(feh)
		else:
			pre = sign+str(feh)
		
		if t1 == t2 and g1 == g2:
			d = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			f = d[0].data
			w = np.arange(len(f))*d[0].header['CD1_1']+d[0].header['CRVAL1']
			return 10**w,f

		elif t1 == t2:
			d1 = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			d2 = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g2,1))+'_'+pre+'p00_sed.fits')
			f1,f2 = d1[0].data, d2[0].data
			w1 = np.arange(len(f1))*d1[0].header['CD1_1']+d1[0].header['CRVAL1']
			w2 = np.arange(len(f2))*d2[0].header['CD1_1']+d2[0].header['CRVAL1']
			m = (f2 -f1)/(g2-g1)
			n = f2 - m*g2
			f = m*g + n
			return 10**w1,f

		elif g1 == g2:
			d1 = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			d2 = pyfits.open('coelho_seds/t0'+str(int(t2))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			f1,f2 = d1[0].data, d2[0].data
			w1 = np.arange(len(f1))*d1[0].header['CD1_1']+d1[0].header['CRVAL1']
			w2 = np.arange(len(f2))*d2[0].header['CD1_1']+d2[0].header['CRVAL1']
			m = (f2 -f1)/(t2-t1)
			n = f2 - m*t2
			f = m*t + n
			return 10**w1,f
	
		else:
			d11 = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			d12 = pyfits.open('coelho_seds/t0'+str(int(t1))+'_g+'+str(np.around(g2,1))+'_'+pre+'p00_sed.fits')
			d21 = pyfits.open('coelho_seds/t0'+str(int(t2))+'_g+'+str(np.around(g1,1))+'_'+pre+'p00_sed.fits')
			d22 = pyfits.open('coelho_seds/t0'+str(int(t2))+'_g+'+str(np.around(g2,1))+'_'+pre+'p00_sed.fits')
			f11,f12 = d11[0].data, d12[0].data
			f21,f22 = d21[0].data, d22[0].data
			#print f11
			w11 = np.arange(len(f11))*d11[0].header['CD1_1']+d11[0].header['CRVAL1']
			fxy1 = f11 * (t2-t)/(t2-t1) + f21 * (t-t1)/(t2-t1)
			fxy2 = f12 * (t2-t)/(t2-t1) + f22 * (t-t1)/(t2-t1)
			fxy  = fxy1 * (g2-g)/(g2-g1) + fxy2 * (g-g1)/(g2-g1)
			w11 = 10**w11
			#plt.plot(w11,f11)
			#plt.plot(w11,f12)
			#plt.plot(w11,f21)
			#plt.plot(w11,f22)

			return w11,fxy


def bilinear_fluxes(t,g,t1,t2,g1,g2):
	#print t,g,t1,t2,g1,g2
	st1 = str(int(t1/100.))
	st2 = str(int(t2/100.))
	sg1 = str(np.around(g1,1))
	sg2 = str(np.around(g2,1))

	if t1 == t2 and g1 == g2:
		d =  np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		return d

	elif t1 == t2:
		f1 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		f2 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		m = (f2 -f1)/(g2-g1)
		n = f2 - m*g2
		f = m*g + n
		return f

	elif g1 == g2:
		f1 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		f2 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		m = (f2 -f1)/(t2-t1)
		n = f2 - m*t2
		f = m*t + n
		return f

	else:
		f11 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		f12 = np.loadtxt('bt-settl-cifist/lte0'+st1+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		f21 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg1+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])
		f22 = np.loadtxt('bt-settl-cifist/lte0'+st2+'.0-'+sg2+'-0.0a+0.0.BT-Settl.spec.7.bands.txt',usecols=[1])

		fxy1 = f11 * (t2-t)/(t2-t1) + f21 * (t-t1)/(t2-t1)
		fxy2 = f12 * (t2-t)/(t2-t1) + f22 * (t-t1)/(t2-t1)

		fxy  = fxy1 * (g2-g)/(g2-g1) + fxy2 * (g-g1)/(g2-g1)
		return fxy

def get_sed(teff,logg,feh,models='bt-settl-cifist'):
	if models == 'bt-settl-cifist':
		#print 'entering in get sed...'
		teffs = np.arange(4000,7050,100)
		loggs = np.arange(2.5,5.6,0.5)

	elif models == 'coelho_seds':
		teffs = np.arange(4000,7050,250)
		loggs = np.arange(2.0,5.1,0.5)	
		fehs = np.array([-1,-0.5,0.0,0.2])
		FEH = fehs[np.argmin(np.absolute(fehs-feh))]
	
	dift = teffs - teff
	I1 = np.where(dift<=0)[0][-1]
	I2 = np.where(dift>=0)[0][0]
	difg = loggs - logg
	J1 = np.where(difg<=0)[0][-1]
	J2 = np.where(difg>=0)[0][0]
	#print 'calling the interpolation...'
	wavs,flxs = bilinear(teff,logg,teffs[I1],teffs[I2],loggs[J1],loggs[J2],feh=feh,models=models)
	return wavs,flxs

def get_sed_fluxes(teff,logg,feh,models='bt-settl-cifist'):
	if models == 'bt-settl-cifist':
		print teff,logg,feh
		#print 'entering in get sed...'
		teffs = np.arange(4000,7050,100)
		loggs = np.arange(2.5,5.6,0.5)
		if teff >= 4000 and teff <=7000:
			dift = teffs - teff
			I1 = np.where(dift<=0)[0][-1]
			I2 = np.where(dift>=0)[0][0]
		elif teff <4000:
			I1 = 0
			I2 = 0
		else:
			I1 = -1
			I2 = -1

		difg = loggs - logg
		J1 = np.where(difg<=0)[0][-1]
		J2 = np.where(difg>=0)[0][0]

		#print 'calling the interpolation...'
		#wavs,flxs = bilinear(teff,logg,teffs[I1],teffs[I2],loggs[J1],loggs[J2])
		flxs = bilinear_fluxes(teff,logg,teffs[I1],teffs[I2],loggs[J1],loggs[J2])
		f = open('bt-settl-cifist/lte040.0-4.0-0.0a+0.0.BT-Settl.spec.7.bands.txt','r')
		lines = f.readlines()

		wavs,bands,widths = [],[],[]
		for line in lines:
			cos = line.split()
			tband = cos[0]
			bands.append(tband)
			wavs.append(get_bpass_pos(tband))
			widths.append(get_bpass_width(tband))
		wavs,bands,widths = np.array(wavs),np.array(bands),np.array(widths)

		return wavs,flxs,bands,widths


		#Al = get_Al(0.01,wavs)
		#print 'ended'

		flxs = flxs * (0.674*6.95700e10)**2
		loglog(wavs,flxs)
		#flxs2 = flxs*np.exp(-Al/2.5)
		flxs2 = apply(ccm89(wavs, Av, 3.1), flxs)

		loglog(wavs,flxs2,'k')


		#dw = np.array([2274.37,4280.00,4297.17,4640.42,5340.00,5394.29,5857.56,6122.33,7439.49,12350.00,16620.00,21590.00,33526.00,46028.00,115608.00])
		#df = np.array([1.778e-14,3.524e-13,4.214e-13,3.884e-13,4.364e-13,4.098e-13,3.365e-13,3.659e-13,2.683e-13,9.996e-14,4.594e-14,1.812e-14,3.637e-15,1.036e-15,2.868e-17])
		dw = np.array([4297.17,5394.29,12350.00,16620.00,21590.00,33526.00,46028.00,115608.00])
		df = np.array([4.214e-13,4.098e-13,9.996e-14,4.594e-14,1.812e-14,3.637e-15,1.036e-15,2.868e-17])
		db = np.array(['Johnson_B','Johnson_V','2mass_J','2mass_H','2mass_Ks','W1','W2','W3'])

		dw = np.array([4297.17,5394.29,12350.00,16620.00,21590.00,33526.00,46028.00,115608.00,220883])		
		db = np.array(['Johnson_B','Johnson_V','2mass_J','2mass_H','2mass_Ks','W1','W2','W3','W4'])
		df = np.array([7.977e-14,9.716e-14,7.261e-14,4.412e-14,1.868e-14,3.875e-15,1.063e-15,3.087e-17,3.426e-18])

		#print 'iterating...'
		exp_fluxes = []
		for i in range(len(db)):
			tband = db[i]
			rw,rr = get_bpass_response(tband)
			JJ = np.where(rr>0.001)[0]
			rw,rr = rw[JJ],rr[JJ]

			II = np.where((wavs>rw[0]) & (wavs<rw[-1]))[0]
			tck = scipy.interpolate.splrep(rw,rr,k=1)
			rr  = scipy.interpolate.splev(wavs[II],tck)

			wt1 = wavs[II].copy()
			wt2 = np.hstack((wt1[0],wt1[:-1]))
			dws = wt1-wt2
			dws[0] = dws[1]
			#print dws
			FF  = np.sum(rr*flxs2[II]*dws)/get_bpass_width(tband)
			exp_fluxes.append(FF)
		#print 'out'
		#print exp_fluxes
		exp_fluxes = np.array(exp_fluxes)
		loglog(dw,exp_fluxes,'go')

		D = 7.694021354007038 / 1000.
		D = 17.014529124468865/ 1000.
		D = 3.0857e18 * 1./D

		df = df* D**2
		loglog(dw,df,'ro')
		show()

def get_avge_ext(Av,tband):
	wavs,resp = get_bpass_response(tband)

	Al = get_Al(Av, wavs, Rv=3.1)
	wt1 = wavs.copy()
	wt2 = np.hstack((wt1[0],wt1[:-1]))
	dws = wt1-wt2
	dws[0] = dws[1]

	Al = np.sum(Al*resp*dws)/np.sum(resp*dws)

	return Al

def get_model(Rs,Av):
	#print Rs,Av
	Rscm = Rs * 6.95700e10
	flxs = apply(ccm89(nsyn_wavs, Av, 3.1), nsyn_flxs*Rscm*Rscm)
	#outsw,nsyn_flxs = syn_flux(mod_wav,flxs,band)
	#loglog(mod_wav,mod_flx*Rscm*Rscm)
	#loglog(mod_wav,flxs*Rscm*Rscm)
	#loglog(outsw,nsyn_flxs*Rscm*Rscm,'ro')
	#loglog(refw,fluxes*distance*distance,'bo')

	#show()

	#Al = []
	#for bd in band:
	#	Al.append(get_avge_ext(Av,bd))
	#Al = np.array(Al)
	return flxs#*Rscm*Rscm


def lnlike(theta, y, yerr):
	Rs, Av = theta
	model_fluxes = get_model(Rs,Av)
	inv_sigma2 = 1.0/(yerr**2)
	ret = -0.5*(np.sum(inv_sigma2*(y-model_fluxes)**2 - np.log(inv_sigma2)))
	if np.isnan(ret):
		return -np.inf
	else:
		return ret

def lnprior(theta):
	Rs, Av = theta
	if 0.1 < Rs < 100. and 0. < Av < 1.:
		return 0.0
	return -np.inf

def lnprob(theta, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, yerr)

def do_mcmc(target, query=False, avoid_plot=False, models='bt-settl-cifist',simple =True,correction=False,N=5000):
	#print '\n',target
	global nsyn_wavs, nsyn_flxs, nsyn_widths,band
	global refw, fluxes

	f = open(target+'_pars.txt','r')
	lines = f.readlines()
	have_dist = False
	for line in lines:
		cos = line.split()
		if 'parallax' in cos[0]:
			parallax = float(cos[1])
			eparallax = float(cos[2])
		elif 'teff' in cos[0]:
			teff = float(cos[1])
			eteff = float(cos[2])
		elif 'logg' in cos[0]:
			logg = float(cos[1])
			elogg = float(cos[2])
		elif 'feh' in cos[0]:
			feh = float(cos[1])
			efeh = float(cos[2])
		elif 'ra' in cos[0]:
			ra = cos[1]
		elif 'dec' in cos[0]:
			dec = cos[1]
		elif 'dist' in cos[0]:
			dist = float(cos[1])
			edist = float(cos[2])
			have_dist = True

	print parallax, eparallax
	if correction:
		parallax += 0.029
		eparallax += 0.4*eparallax
	print parallax, eparallax
	global distance
	#print have_dist
	if not have_dist:
		#print parallax, eparallax
		parallax = parallax / 1000.
		eparallax = eparallax / 1000.
		distance = 3.0857e18 / parallax
	else:
		distance = 3.0857e16 * dist * 100.
	#print distance
	if query == False:
		f = open(target+'_fluxes.txt','r')
		lines = f.readlines()
		band,fluxes,efluxes,refw = [],[],[],[]
		for line in lines:
			cos = line.split()
			band.append(cos[0])
			fluxes.append(float(cos[2]))
			efluxes.append(float(cos[3]))
			refw.append(get_bpass_pos(cos[0]))
		band,fluxes,efluxes,refw = np.array(band),np.array(fluxes),np.array(efluxes),np.array(refw)	
		f.close()
	else:
		try:
			ra,dec = float(ra),float(dec)
		except:
			ra,dec = trans_coords(ra,dec)
		#print 'coords:',ra, dec
		dct = get_mags_2mass(ra,dec)
		dct.update(get_mags_APASS(ra,dec))
		dct.update(get_mags_WISE(ra,dec,models=models))
		print dct

		band,fluxes,efluxes,refw = [],[],[],[]
		for key in dct.keys():
			if key[0] != 'e':

				#print dct[key],dct['e'+key],key
				flux, eflux, refwt = mag_to_flux(dct[key],dct['e'+key],key)
				if flux != -1 and eflux != -1 and np.isnan(flux)==False and np.isnan(eflux)==False:
					fluxes.append(flux)
					efluxes.append(eflux)
					refw.append(refwt)
					band.append(key)
		band,fluxes,efluxes, refw = np.array(band),np.array(fluxes),np.array(efluxes),np.array(refw)


	y = fluxes * distance * distance

	realiz = []
	#print parallax, eparallax
	for i in range(1000):
		errs1 = np.random.normal(len(fluxes))
		ftemp = fluxes + efluxes*errs1

		if not have_dist:
			ptemp = parallax + eparallax*np.random.normal()
			dtemp = 3.0857e18 / ptemp
		else:
			dtemp = distance + 3.0857e16*edist*np.random.normal()*100.
		ytemp = ftemp * dtemp * dtemp

		if i == 0:
			realiz = ytemp
		else:
			realiz = np.vstack((realiz,ytemp))

	yerr = np.sqrt(np.var(realiz,axis=0))
	#print y,yerr
	#errorbar(refw,y,yerr=yerr,fmt='ko')
	#yscale('log')
	#xscale('log')
	#show()

	#D = 17.014529124468865/ 1000.
	#D = 3.0857e18 * 1./D
	#Rs = 0.674*6.95700e10

	syn_wavs,syn_flxs,syn_bands,syn_widths =  get_sed_fluxes(teff,logg,feh,models='bt-settl-cifist')

	#global mod_wav, mod_flx
	#print 'getting SED...'
	#mod_wav, mod_flx = get_sed(teff,logg,feh,models=models)
	#print 'got SED...'
	#print get_model(0.93106886, 0.12345027)
	#print gfd
	#"""
	nsyn_wavs,nsyn_flxs,nsyn_widths = [],[],[]
	for i in range(len(fluxes)):
		bd = band[i]
		I = np.where(syn_bands == bd)[0]
		if len(I)>0:
			nsyn_wavs.append(syn_wavs[I[0]])
			nsyn_flxs.append(syn_flxs[I[0]])
			nsyn_widths.append(syn_widths[I[0]])
	nsyn_wavs,nsyn_flxs,nsyn_widths = np.array(nsyn_wavs),np.array(nsyn_flxs),np.array(nsyn_widths)
	#errorbar(nsyn_wavs,nsyn_flxs,xerr=nsyn_widths,fmt='ro')
	#show()
	#"""
	#print nsyn_wavs,nsyn_flxs,nsyn_widths
	#print vbhnj
	
	#print get_model(0.93106886, 0.12345027)
	#print gfd
	guess = [3.,0.5]
	ndim = len(y)
	nwalkers = 30
	pos = []
	while len(pos) < nwalkers:
		vala = guess[0] + 0.5*np.random.randn()
		valm = guess[1] + 0.01*np.random.randn()
		if vala>0 and vala < 100 and valm>0 and valm<1.:
			pos.append(np.array([vala,valm]))

	sampler = emcee.EnsembleSampler(nwalkers, 2, lnprob, args=(y, yerr))
	sampler.run_mcmc(pos, N)
	samples = sampler.chain[:, 50:, :].reshape((-1, 2))
	dicti = {'samples':samples}
	if not avoid_plot:
		fig = corner.corner(samples, labels=["$Rs$", "$Av$"])
		fig.savefig(target+'.'+str(teff)+'_'+str(logg)+".png")

	pickle.dump( dicti, open( target+'.'+str(teff)+'_'+str(logg)+"samples.pkl", 'w' ) )
	#"""
	dicti = pickle.load(open( target+'.'+str(teff)+'_'+str(logg)+"samples.pkl", 'r' ))
	samples = dicti['samples']

	frs   = np.sort(samples[:,0])
	fav   = np.sort(samples[:,1])

	RS,RS1,RS2 = get_vals(frs)
	AV,AV1,AV2 = get_vals(fav)

	#print 'Rs =', RS, '(',RS1, RS2,')'
	#print 'Av =', AV, '(',AV1, AV2,')'
	
	dct = {'name':target, 'rstar': RS, 'rstar_l': RS1, 'rstar_u': RS2, 'av':AV, 'av_l':AV1, \
		'av_u':AV2, 'logg':logg, 'logg_e':elogg, 'teff':teff, 'teff_e': eteff, }


	#swavs,sflxs = get_sed(teff,logg,feh,models='bt-settl-cifist')
	#plot(swavs,sflxs*Rs**2,'g')

	#ags = []
	#for bd in bands:
	#	ags.append(get_avge_ext(0.01,bd))
	#ags = np.array(ags)

	#plot(wavs,np.exp(-ags/1.086)*flxs*Rs**2,'ro')
	#errorbar(refw,fluxes*D**2,yerr=efluxes,fmt='ko')
	#xscale('log')
	#yscale('log')
	#show()
	#get_sed(4373,4.2584,0.0,models='bt-settl-cifist')
	return dct

def prop_teff(target, query=False, avoid_plot=False, models='bt-settl-cifist',simple =True,N=1000,correction=False):
	
	f = open(target +'_pars.txt','r')
	olines = f.readlines()
	f.close()
	for line in olines:
		cos = line.split()
		if cos[0] == 'teff':
			teff,eteff = float(cos[1]),float(cos[2])
	radii = []
	i=0
	while i < 100:
		f = open(target+'temp_pars.txt','w')
		for line in olines:
			cos = line.split()
			if cos[0] != 'teff':
				f.write(line)
			else:
				f.write('teff\t'+str(np.random.normal(teff,eteff))+'\t'+str(eteff)+'\n')
		f.close()
		dct = do_mcmc(target+'temp',query=query,avoid_plot=avoid_plot, models=models, simple= simple,N=N,correction=correction)
		radii.append(dct['rstar'])
		i+=1
	radii = np.array(radii)
	print np.mean(radii), np.sqrt(np.var(radii))
	dct = do_mcmc(target,query=query,avoid_plot=avoid_plot, models=models, simple= simple)
	print '\nRs = ',dct['rstar'],np.sqrt(dct['rstar_l']**2+np.sqrt(np.var(radii))**2),np.sqrt(dct['rstar_u']**2+np.sqrt(np.var(radii))**2)
	print 'Av = ',dct['av'],np.sqrt(dct['av_l']**2+np.sqrt(np.var(radii))**2),np.sqrt(dct['av_u']**2+np.sqrt(np.var(radii))**2)

	os.system('rm '+target+'temp*')
