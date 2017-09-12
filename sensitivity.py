import numpy as np
import scipy as sp
import math as m
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.optimize import brentq

import nuSQUIDSpy as nsq
import nuSQUIDSTools

units = nsq.Const()

datadir = '/data/user/jbarron'

def oscillationProbabilities(dcp=0.0,t12=0.588525,t13=0.147655,t23=0.72431,dm21=7.49e-05,dm31=0.002526,n=100):#best fit values from NuFit 2016
    units = nsq.Const()
    interactions = False
    E_min = 0.4*units.GeV
    E_max = 15.0*units.GeV
    E_nodes = n
    Erange = np.linspace(E_min,E_max,E_nodes)
    neutrino_flavors = 3
    nuSQ = nsq.nuSQUIDS(Erange*1.28,neutrino_flavors,nsq.NeutrinoType.neutrino,interactions)
    Einitial = np.ones(n).reshape(n,1)*np.array([0.,1.,0.]).reshape(1,3)
    nuSQ.Set_Body(nsq.Earth())
    nuSQ.Set_Track(nsq.Earth.Track(3114.692788*units.km))
    nuSQ.Set_rel_error(1.0e-15)
    nuSQ.Set_abs_error(1.0e-17)
    nuSQ.Set_CPPhase(0,1,dcp)
    nuSQ.Set_CPPhase(0,2,dcp)
    nuSQ.Set_CPPhase(1,2,dcp)
    nuSQ.Set_MixingAngle(0,1,t12)
    nuSQ.Set_MixingAngle(0,2,t13)
    nuSQ.Set_MixingAngle(1,2,t23)
    nuSQ.Set_SquareMassDifference(1,dm21)
    nuSQ.Set_SquareMassDifference(2,dm31)
    nuSQ.Set_initial_state(Einitial,nsq.Basis.flavor)
    nuSQ.EvolveState()
    return nuSQ

def BeamFlux():
    global datadir
    spectrum = np.genfromtxt(datadir+'/SIRENResources/Beam_Spectrum.txt')
    spectrum_interp = interpolate.interp1d(spectrum[:,0]*1.28,spectrum[:,1],kind='slinear')
    return spectrum_interp
    

def cdf(x,a=.4*1.28,b=15*1.28):
    return np.asarray([integrate.quad(BeamFlux(),a,y) for y in x])[:,0]

def inv_cdf(a=.4*1.28,b=15*1.28,n_bins=100):
    x_points = np.linspace(a,b,n_bins)
    inv = interpolate.interp1d(cdf(x_points,a=a,b=b),x_points,kind='cubic')
    return inv

def inverse_transform_sampling(n_bins=100,n_samples=1000,a=.4*1.28,b=15*1.28):
    r = np.random.rand(n_samples)*cdf([b])
    #inverse = inv_cdf(a,b,n_bins)
    return inverse(r)

def oscWeight(datadict,dcp=0.0,t12=0.588525,t13=0.147655,t23=0.72431,dm21=7.49e-05,dm31=0.002526):
    nodes=100
    nuSQ = oscillationProbabilities(dcp=dcp,t12 = t12,t13 = t13, t23 = t23, dm21 = dm21, dm31 = dm31)
    E_prob = np.array([nuSQ.EvalFlavor(0,x*units.GeV,0) for x in np.linspace(.4*1.28,15*1.28,400)])
    Mu_prob = np.array([nuSQ.EvalFlavor(1,x*units.GeV,0) for x in np.linspace(.4*1.28,15*1.28,400)])
    Tau_prob = np.array([nuSQ.EvalFlavor(2,x*units.GeV,0) for x in np.linspace(.4*1.28,15*1.28,400)])
    E_interp = interpolate.interp1d(np.linspace(.4*1.28,15*1.28,400),E_prob,kind='cubic')
    Mu_interp = interpolate.interp1d(np.linspace(.4*1.28,15*1.28,400),Mu_prob,kind='cubic')
    Tau_interp = interpolate.interp1d(np.linspace(.4*1.28,15*1.28,400),Tau_prob,kind='cubic')
    datadict['osc_weights'] = np.array([\
                                       E_interp(datadict['data']),\
                                       Mu_interp(datadict['data']),\
                                       Tau_interp(datadict['data'])]).T


def fourWeight(datadict):
    global datadir
    NuMu_CC_sigma = np.load(datadir+'/SIRENResources/NuMu_CC_CrossSection.npy')
    NuTau_CC_sigma = np.load(datadir+'/SIRENResources/NuTau_CC_CrossSection.npy')
    NuMu_CC_sigma_func = interpolate.interp1d(NuMu_CC_sigma[:,0],NuMu_CC_sigma[:,1],bounds_error=False,fill_value='extrapolate')
    NuE_CC_sigma_func = NuMu_CC_sigma_func
    NuTau_CC_sigma_func = interpolate.interp1d(NuTau_CC_sigma[:,0],NuTau_CC_sigma[:,1],bounds_error=False,fill_value=(0,NuTau_CC_sigma[:,1][-1]))
    NC_sigma_func = interpolate.interp1d(NuMu_CC_sigma[:,0],NuMu_CC_sigma[:,1]/2.7,bounds_error=False,fill_value='extrapolate')

    cm2m = 10**(-4)
    rho = 1029.
    rad = 135
    height = 180
    vol = np.pi*rad**2 * height
    mass = rho*vol
    m_prot = 1.6726 * 10**(-27)
    d = datadict['data']
    datadict['four_weights'] = np.array([\
                                        NuMu_CC_sigma_func(d),#*datadict['osc_weights'][:,0],
                                        NuE_CC_sigma_func(d),#*datadict['osc_weights'][:,1],
                                        NuTau_CC_sigma_func(d),#*datadict['osc_weights'][:,2],
                                        NC_sigma_func(d)]).T * cm2m * mass * 10**(-38) / m_prot
     

def detectorWeight(datadict):
    #Load in detection probability data to be interpolated. 
    global datadir
    NuE_DetectedProb = np.load(datadir+'/SIRENResources/CC_NuE_SANTAProbability_LowStats.npy')
    NuMu_DetectedProb = np.load(datadir+'/SIRENResources/CC_NuMu_SANTAProbability_LowStats.npy')
    NuTau_DetectedProb = np.load(datadir+'/SIRENResources/CC_NuTau_SANTAProbability_LowStats.npy')
    NC_DetectedProb = np.load(datadir+'/SIRENResources/NC_SANTAProbability_LowStats.npy')
    #Now get detection probability for each of these four channels, by evaluating the interpolated function from the arrays
    #loaded earlier. 
    datadict['detector_weights'] = np.array([interpolate.interp1d(NuE_DetectedProb[:,1],NuE_DetectedProb[:,0],kind='cubic',fill_value='extrapolate')(datadict['data']),\
                                             interpolate.interp1d(NuMu_DetectedProb[:,1],NuMu_DetectedProb[:,0],kind='cubic',fill_value='extrapolate')(datadict['data']),\
                                             interpolate.interp1d(NuTau_DetectedProb[:,1],NuTau_DetectedProb[:,0],kind='cubic',fill_value='extrapolate')(datadict['data']),\
                                             interpolate.interp1d(NC_DetectedProb[:,1],NC_DetectedProb[:,0],kind='cubic',fill_value='extrapolate')(datadict['data'])]).T

def preosc_weight(datadict):
    datadict['preosc_weights'] = datadict['four_weights']*datadict['detector_weights']*((1000./3114693)**2)*10*1.2
    #r^2 correction, and *10 for per 10^21 POT, * 1.2 for 2 years. (6 * 10^21 POT / 10 years estimated)
    
def weight(datadict):
    datadict['weights'] = np.array([datadict['preosc_weights'][:,0]*datadict['osc_weights'][:,0],\
                                   datadict['preosc_weights'][:,1]*datadict['osc_weights'][:,1],\
                                   datadict['preosc_weights'][:,2]*datadict['osc_weights'][:,2],\
                                   datadict['preosc_weights'][:,3]]).T
    
def change_weight(datadict,dcp,t12=0.588525,t13=0.147655,t23=0.72431,dm21=7.49e-05,dm31=0.002526):
    oscWeight(datadict,dcp,t12,t13,t23,dm21,dm31)
    weight(datadict)
    particleID(datadict)


    
def energyEstimator(datadict,mod = 1.0):
    from numpy.random import normal#Outputs 'reconstructed' energy according to some kind of distribution and bias, as [nue_cc,numu_cc,nutau_cc,all_nc]
    #If these become more complicated we'll load in data. For now we use linear regression best fits. 
    '''NuE_nHit = np.load('/data/user/jbarron/CC_NuE_nHit_LowStats.npy')
    NuMu_nHit = np.load('/data/user/jbarron/CC_NuMu_nHit_LowStats.npy')
    NuTau_nHit = np.load('/data/user/jbarron/CC_NuTau_nHit_LowStats.npy')
    NC_nHit = np.load('/data/user/jbarron/NC_nHit_LowStats.npy')
    E_mean = interpolate.interp1d(NuE_nHit[:,0],NuE_nHit[:,1],kind='cubic',fill_value='extrapolate')(datadict['data'])
    E_sig = interpolate.interp1d(NuE_nHit[:,0],NuE_nHit[:,2],kind='cubic',fill_value='extrapolate')(datadict['data'])
    Mu_mean = interpolate.interp1d(NuMu_nHit[:,0],NuMu_nHit[:,1],kind='cubic',fill_value='extrapolate')(datadict['data'])
    Mu_sig = interpolate.interp1d(NuMu_nHit[:,0],NuMu_nHit[:,2],kind='cubic',fill_value='extrapolate')(datadict['data'])
    Tau_mean = interpolate.interp1d(NuTau_nHit[:,0],NuTau_nHit[:,1],kind='cubic',fill_value='extrapolate')(datadict['data'])
    Tau_sig = interpolate.interp1d(NuTau_nHit[:,0],NuTau_nHit[:,2],kind='cubic',fill_value='extrapolate')(datadict['data'])
    NC_mean = interpolate.interp1d(NC_nHit[:,0],NC_nHit[:,1],kind='cubic',fill_value='extrapolate')(datadict['data'])
    NC_sig = interpolate.interp1d(NC_nHit[:,0],NC_nHit[:,2],kind='cubic',fill_value='extrapolate')(datadict['data'])'''
    
    '''E_mean = 13.48*datadict['data'] + 27.31
    #E_sig = 0.2*E_mean
    E_sig = (6.611*datadict['data'] + 6.03) * mod
    Mu_mean = 15.249*datadict['data'] + 27.30 
    #Mu_sig = 0.2*E_mean
    Mu_sig = (8.187*datadict['data'] + 4.47) * mod
    Tau_mean = 9.439*datadict['data'] + 25.46
    #Tau_sig = 0.2*Tau_mean
    Tau_sig = (5.1999*datadict['data'] + 5.61) * mod
    NC_mean = 7.183*datadict['data'] + 22.67
    #NC_sig = 0.2*NC_mean
    NC_sig = (4.102*datadict['data'] + 7.12) * mod'''
    
    E_mean = datadict['data']
    E_sig = res* E_mean
    Mu_mean = datadict['data']
    Mu_sig = res*Mu_mean
    Tau_mean = datadict['data']
    Tau_sig = res* Tau_mean
    NC_mean = datadict['data']
    NC_sig = res* NC_mean
    
    datadict['energyEstimator_%.2f'%(res)] = np.array([BoundedNormal(E_mean,E_sig),\
                                BoundedNormal(Mu_mean,Mu_sig),\
                                BoundedNormal(Tau_mean,Tau_sig),\
                                BoundedNormal(NC_mean,NC_sig)]).T

def BoundedNormal(mean,sig,minimum = 0):
    
    from numpy.random import normal
    nhits = []
    for i in range(len(mean)):
        result = minimum-1
        while result <= minimum:
            result = normal(loc=mean[i],scale=sig[i])
        nhits.append(result)
    return np.array(nhits)
    
    
def particleID(datadict):
    E_as_E = 0.8
    E_as_Mu = 0.2
    E_as_Tau = 0.0
    E_as_NC = 0.0
    Mu_as_E = 0.2
    Mu_as_Mu = 0.8
    Mu_as_Tau = 0.
    Mu_as_NC = 0.0
    Tau_as_E = 0.3
    Tau_as_Mu = 0.7
    Tau_as_Tau = 0.0
    Tau_as_NC = 0.0
    NC_as_E = 0.3
    NC_as_Mu = 0.7
    NC_as_Tau = 0.0
    NC_as_NC = 0.0
    
    w = datadict['weights']
    a = w[:,0]
    b = w[:,1]
    c = w[:,2]
    d = w[:,3]
    datadict['particleID'] = np.array([\
                                        E_as_E * a + Mu_as_E * b + Tau_as_E * c + NC_as_E * d,\
                                        E_as_Mu * a + Mu_as_Mu * b + Tau_as_Mu * c + NC_as_Mu * d,\
                                        E_as_Tau * a + Mu_as_Tau * b + Tau_as_Tau * c + NC_as_Tau * d,\
                                        E_as_NC * a + Mu_as_NC * b + Tau_as_NC * c + NC_as_NC * d]).T
    
    
        
def datadictgenerator(n):
    d = {}
    d['data'] = inverse_transform_sampling(n_samples=n)
    detectorWeight(d)
    energyEstimator(d)
    fourWeight(d)
    preosc_weight(d)
    return d

def make_obsandexp(n,scaling=1000,asimov=True):
    if asimov==True:
        expdict = datadictgenerator(n*scaling)
        obsdict = {}
        obsdict['data'] = expdict['data']
        obsdict['energyEstimator_%.2f'%(res)] = expdict['energyEstimator_%.2f'%(res)]
        obsdict['preosc_weights'] = expdict['preosc_weights']
    else:
        obsdict = datadictgenerator(n)
        expdict = datadictgenerator(n*scaling)
    return obsdict, expdict


def minimizerFunc(x,obs_datadict, exp_datadict,fix_dcp=None, flavor=0, nbins=15,asimov=True):
    if fix_dcp == None:
        dcp = x[0]
        flux_normalization = x[1]
        t13 = x[2]
        t23 = x[3]
        dm31 = x[4]
        
    else:
        dcp = fix_dcp
        flux_normalization = x[0]
        t13 = x[1]
        t23 = x[2]
        dm31 = x[3]
        
          
    a=0
    b=20
    bins=np.linspace(a,b,nbins)
    beam_max = max(BeamFlux()(np.linspace(5,8,100)))
    hist,bins2 = np.histogram(obs_datadict['data'],bins=np.linspace(0,20,101))
    width = bins2[1] - bins2[0]
    hist_max = max(hist)
    
    #Fit parameter means and uncertainties. 
    flux_mean = 1.0
    flux_std = 0.1
    t23_mean = (np.pi/180)*41.5
    t23_std = (np.pi/180)*1.3
    t13_mean = (np.pi/180)*8.46
    t13_std = (np.pi/180)*.15
    dm31_mean = .002526
    dm31_std = .000039
    
    fitparam_means = np.array([flux_mean,t13_mean,t23_mean,dm31_mean])
    fitparam_stds = np.array([flux_std,t13_std,t23_std,dm31_std])
    fitparam_values = np.array([flux_normalization,t13,t23,dm31])
    
    
    change_weight(exp_datadict,dcp=dcp,t13=t13,t23=t23,dm31=dm31)
    exp = exp_datadict['energyEstimator_%.2f'%(res)][:,flavor]
    exp_weights = exp_datadict['particleID'][:,flavor]*flux_normalization
    obs = obs_datadict['energyEstimator_%.2f'%(res)][:,flavor]

    obs_weights = obs_datadict['particleID'][:,flavor]
    obs_hist, _ = np.histogram(obs,weights=obs_weights*(float(beam_max)/hist_max)*width,bins=bins)
    exp_hist, _ = np.histogram(exp,weights=exp_weights*(float(beam_max)/hist_max)*width,bins=bins)
    
    
    return np.sum(np.power((obs_hist-exp_hist),2)/exp_hist) + chi_penalty(fitparam_values,fitparam_means,fitparam_stds) 

def chi_penalty(val,mean,std):
    return np.sum(np.power((val - mean),2)/np.power(std,2))
    
