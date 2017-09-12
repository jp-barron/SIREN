import sensitivity
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

res = 0.3
a = 0.4*1.28
b = 15 * 1.28
n_bins=100
inverse = inv_cdf(a,b,n_bins)
O, E = make_obsandexp(1000,scaling=1)

chi = []
#fitpoints = []

for i in np.linspace(0,2*np.pi,11):
    change_weight(O,i)
    #leftresult = minimize(minimizerFunc,np.array([np.pi/2.,1.01]),args=(O,E,None),method='L-BFGS-B',bounds = [(0,2*np.pi),(0,None)])
    #rightresult = minimize(minimizerFunc,np.array([3*np.pi/2,1.01]),args=(O,E,None),method='L-BFGS-B',bounds = [(0,2*np.pi),(0,None)])
    #if leftresult['fun'] < rightresult['fun']:
    #    fitresult = leftresult
    #else:
    #    fitresult = rightresult
        
    fixed_dcp_result = minimize(minimizerFunc,np.array([1.00,0.72431,0.147655,.002526]),args=(O,E,0),method='L-BFGS-B',bounds = [(0,None),(None,None),(None,None),(0,None)])
    fixed_dcp_result_2= minimize(minimizerFunc,np.array([1.00,0.72431,0.147655,.002526]),args=(O,E,0,1),method='L-BFGS-B',bounds = [(0,None),(None,None),(None,None),(0,None)])
    chi.append(abs(fixed_dcp_result['fun'] + fixed_dcp_result_2['fun']))
    #fitpoints.append(fixed_dcp_result['x'])
               # - fitresult['fun']))
    
    #d.append(minimizerFunc(0.0,1,O,E,0))#,minimizerFunc(np.pi,O,E,0)]))
    #e.append(minimizerFunc(np.pi,O,E,0))


fig = plt.figure(figsize=(8,8))
plt.plot(np.linspace(0,2*np.pi,11)[:8],np.sqrt(chi),label='$\delta_{CP} = 0$ hypothesis')
#plt.plot(np.linspace(0,2*np.pi,21),np.sqrt(chi2),label='$\delta_{CP} = \pi$ hypothesis')
#plt.plot(np.linspace(0,2*np.pi,10),1*np.ones(10),linestyle='dashed')
#plt.plot(np.linspace(0,2*np.pi,10),4*np.ones(10),linestyle='dashed')
#plt.plot(np.linspace(0,2*np.pi,10),9*np.ones(10),linestyle='dashed')
#plt.text(0,1.1,'$1 \sigma$')
#plt.text(0,2.1,'$2 \sigma$')
#plt.text(0,3.1,'$3 \sigma$')
plt.xticks(np.array([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]),('0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$'))
plt.title('Significance of rejecting $\delta_{CP} = 0$')
plt.xlabel('True $\delta_{CP}$')
plt.ylabel('$\sigma = \sqrt{\Delta \chi^2}$')
#plt.ylim(0,6)
plt.legend(loc='upper right')
plt.text(.6,0.1,'Reasonable PID, nhit energy reco,\nStatistical errors only,\nFitted params: $\delta_{CP}$, flux normalization (10% uncertainty), $\\theta_{23},\\theta_{13},\Delta m^2_{31}$,\n2 years runtime')
fig.savefig('/data/user/jbarron/SensitivityPlots/Sept11_2017_DeltaCPSignificancePlot_nhitreco_ReasonablePID_2years_dCPandfluxandoscparamsfit_staterronly_.1fluxuncertainty.pdf')

