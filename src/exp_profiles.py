"""
This code compares the ABL profile of the standard profiles with the CobraProbe
measurments. The experimental test was conducted in 2018 by Abiy Melaku at the 
BLWTL in Western university. 

This codes compares the measured profile with the standard profile provided by 
the BLWTL to pick the ones that will be used for aerodynamic modeling. 

This code compares only the mean velocity and streamwise turbulence intesity 
profiles. First, the profiles are compared and then the finally the cobraprobe
measurments are scaled in x,y,z direction so that the streamwise turbulence
intensity profiles match that of the standard profile. 

Thus, the factor is determined from streamwise component of the velocity and 
applied for the other two components, namely v and w.   


Everthing is done in model scale. 

"""
import matplotlib.pyplot as plt                                                 
import numpy as np
import CWE as cwe
from scipy import signal
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

exposures_z0 = [0.03, 0.30, 0.70]
exposure_names = ['Open', 'Suburb', 'Urban']

gradient_height = 1.4732

n_exposures = len(exposure_names)
cobra_u = []
cobra_v = []
cobra_w = []
cobra_z = []
std_prof = []
H = cwe.CAARC.H/400.0
#Read the time history data
for i in range (n_exposures):
    cobra_z.append(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[i] + '/location'))
    cobra_u.append(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[i] + '/U'))
    cobra_v.append(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[i] + '/V'))
    cobra_w.append(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[i] + '/W'))
    std_prof.append(np.loadtxt('../data/BLWTL/standard/' + exposure_names[i]))


#Mean velocity profile comparison 
plt, fig = cwe.setup_plot(plt, 22, 20, 18)
for i in range (n_exposures):
    ax = fig.add_subplot(1, 3, 1 + i)
    ax.tick_params(direction='in', size=8)
    ax.set_xlim([0,1.5])
    ax.set_ylim([0,3])   
    ax.set_xlabel(r'$U_{av}/U_{H}$') 
    ax.set_title(exposure_names[i], fontsize=24)

    std_prof_z =  std_prof[i][:,0]
    std_prof_Uav = std_prof[i][:,1]
    
    cobra_Uav = np.mean(cobra_u[i], axis=0)
    cobra_Uh = np.interp(H, cobra_z[i], cobra_Uav)
    cobra_Uav = cobra_Uav/cobra_Uh

    #scale standard profiles to building height from the gradient height 
    #applies only to mean velocity profile, TI need not to be scaled
    std_prof_Uh = np.interp(H, std_prof_z, std_prof_Uav)
    std_prof_Uav = std_prof_Uav/std_prof_Uh

    
    ax.plot(std_prof_Uav, std_prof_z/H,  'bo', 
            markersize=8, markerfacecolor='none',markeredgewidth=1.25)    
    ax.plot(cobra_Uav, cobra_z[i]/H,  'rs', 
            markersize=8, markerfacecolor='none',markeredgewidth=1.25)    
    if i == 0:    
        ax.set_ylabel('$z/H$')        
        ax.legend(['Standard','CobraProbe'], loc=0, framealpha=1.0)   
    if i !=0:
        ax.set_yticklabels([])
    plt.grid(linestyle='--')     

fig.set_size_inches(36/2.54, 18/2.54)
plt.tight_layout()
plt.savefig('../data/BLWTL/Plots/Uav_profiles.pdf')
plt.savefig('../data/BLWTL/Plots/Uav_profiles.png')
plt.show()


#Streamwise turbulence intensity profile comparison 
plt, fig = cwe.setup_plot(plt, 22, 20, 18)
for i in range (n_exposures):
    ax = fig.add_subplot(1, 3, 1 + i)
    ax.tick_params(direction='in', size=8)
    ax.set_xlim([0,0.5])
    ax.set_ylim([0,4])   
    ax.set_xlabel(r'$I_{u}(\%)$') 
    ax.set_title(exposure_names[i], fontsize=24)
    std_prof_z =  std_prof[i][:,0]
    std_prof_I = std_prof[i][:,2]/100.0    
    cobra_Uav = np.mean(cobra_u[i], axis=0)
    cobra_I = np.std(cobra_u[i], axis=0)/cobra_Uav

    ax.plot(std_prof_I, std_prof_z/H,  'bo', markersize=8, 
            markerfacecolor='none',markeredgewidth=1.25)    
    ax.plot(cobra_I, cobra_z[i]/H,  'rs', markersize=8, 
            markerfacecolor='none',markeredgewidth=1.25)    
    if i == 0:    
        ax.set_ylabel('$z/H$')        
        ax.legend(['Standard','CobraProbe', 'SplineFitted'], loc=0, framealpha=1.0)   
    if i !=0:
        ax.set_yticklabels([])
    plt.grid(linestyle='--')     

fig.set_size_inches(36/2.54, 18/2.54)
plt.tight_layout()
plt.savefig('../data/BLWTL/Plots/Iu_profiles.pdf')
plt.savefig('../data/BLWTL/Plots/Iu_profiles.png')
plt.show()




I_scale = np.zeros(n_exposures)

#Streamwise turbulence intensity profile comparison 

for i in range (n_exposures):
    std_prof_z =  std_prof[i][:,0]
    
    
    std_prof_I = std_prof[i][:,2]/100.0   
    
    cobra_Uav = np.mean(cobra_u[i], axis=0)
    cobra_I = np.std(cobra_u[i], axis=0)/cobra_Uav
        
    z_new = np.linspace(0.125*H, 1.5*H)    
    

    std_f = UnivariateSpline(std_prof_z, std_prof_I, k=5)

    I_scale[i] = (np.interp(H, z_new, std_f(z_new))
                  /np.interp(H, cobra_z[i], cobra_I))

    ax.plot(std_prof_I, std_prof_z/H,  'bo', markersize=8, 
            markerfacecolor='none',markeredgewidth=1.25)    
    ax.plot(cobra_I, cobra_z[i]/H,  'rs', markersize=8, 
            markerfacecolor='none',markeredgewidth=1.25)    
    ax.plot(std_f(z_new), z_new/H,  'k-',linewidth=2.0)    
    if i == 0:    
        ax.set_ylabel('$z/H$')        
        ax.legend(['Standard','CobraProbe', 'SplineFitted'], loc=0, framealpha=1.0)   
    if i !=0:
        ax.set_yticklabels([])
    plt.grid(linestyle='--')     

fig.set_size_inches(36/2.54, 18/2.54)
plt.tight_layout()
plt.savefig('../data/BLWTL/Plots/turbulence_intensity_profiles.pdf')
plt.savefig('../data/BLWTL/Plots/turbulence_intensity_profiles.png')
plt.show()