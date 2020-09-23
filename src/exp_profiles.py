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


Everthing is calculated in model scale. The standard profiles has three rows 
each representing the elevation, mean velocity normalized by the gradient 
velocity and the turbulence intensity in model scale. 

"""
import matplotlib.pyplot as plt                                                 
import numpy as np
import CWE as cwe
from scipy import signal
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import stats


exposures_z0 = [0.03, 0.30, 0.70]
exposure_names = ['Open', 'Suburb', 'Urban']
components = ['U', 'V', 'W']
gradient_height = 1.4732

n_exposures = len(exposure_names)
n_compnt = 3


cobra_shape = np.shape(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[0] + '/U'))
std_shape= np.shape(np.loadtxt('../data/BLWTL/standard/' + exposure_names[0]))


cobra_U = np.zeros((n_exposures, n_compnt, cobra_shape[1], cobra_shape[0]))
std_prof = np.zeros((n_exposures, std_shape[0], std_shape[1]))

cobra_z = np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[0] + '/location')

cobra_dt = 0.0008

H = cwe.CAARC.H/400.0


#Read the time history data
for i in range (n_exposures):
    for j in range(n_compnt):
        cobra_U[i,j,:,:] = np.transpose(np.loadtxt('../data/BLWTL/cobra_probes/' + exposure_names[i] + '/' + components[j]))
    std_prof[i,:,:] = np.loadtxt('../data/BLWTL/standard/' + exposure_names[i])



#The location of measurment for the BLWTL standard profiles has repeated 
#elevatoins thus, we need average turbulence quantities at the repeated 
#elevations and delete the repeated elevation.    

for i in range (n_exposures):
    z, indices, count = np.unique(std_prof[i,:,0], return_index=True, return_counts=True)
    del_rows = []
    #Take the average of the repeated elevations for the profiles
    for j in range(len(indices)):    
        mean = np.zeros(3)
        for k in range(count[j]):
            mean += std_prof[i,indices[j] + k,:]/count[j]
            if  k > 0:          
                del_rows.append(indices[j] + k)
        std_prof[i,indices[j],:] = mean

#delete the repeated rows and calculate the shape back     
std_prof = np.delete(std_prof, del_rows ,axis=1)
std_shape= np.shape(std_prof[0,:,:])


#Initialize arrays that will hold turbulence quantities
cobra_Uav = np.zeros((n_exposures, cobra_shape[1]))
cobra_I = np.zeros((n_exposures, n_compnt, cobra_shape[1]))
cobra_L = np.zeros((n_exposures, n_compnt, cobra_shape[1]))

#Mean velocity and trubulence intesity for the standard profiles
std_Uav = np.zeros((n_exposures, std_shape[0]))
std_Iu = np.zeros((n_exposures, std_shape[0]))
std_z = std_prof[0,:,0]

#The mean velocity at the building height calculated from cobraprobes
U_H = np.zeros(n_exposures)

#Calculate mean velocity, trubulence intensities and legth scales
#all the velocities are scaled to the building height and the turbulence 
#intensity is calculated in 0.0~1.0 not as a percentage. 
for i in range (n_exposures):
    cobra_Uav[i,:] = np.mean(cobra_U[i,0,:,:], axis=1)        
    U_H[i] = np.interp(H, cobra_z, cobra_Uav[i,:])
    std_Uav[i,:] = std_prof[i,:,1]
    std_U_H = np.interp(H, std_z, std_Uav[i,:])
    std_Uav[i,:] = U_H[i]*std_Uav[i,:]/std_U_H
    
    #Since the original data is in % devide by 100.0 
    std_Iu[i,:] = std_prof[i,:,2]/100.0 
    
    for j in range(n_compnt):
        cobra_I[i,j,:] = np.std(cobra_U[i,j,:,:], axis=1)/cobra_Uav[i,:]
        for k in range(len(cobra_z)):
            cobra_L[i,j,k] = cwe.calculate_length_scale(cobra_U[i,j,k,:], cobra_dt, cobra_Uav[i,k])


#Calculate the roof height turbulence intensity. The TI profile from the wind 
#tunnel is a bit messy around the roof height, thus necessary to fit a line 
#around the roof and extract a representative value from the fitted curve. 

#Streamwise turbulence intensity at the building height for each terrian.
Iu_H = np.zeros(n_exposures)

for i in range (n_exposures):    
    z_fit = np.linspace(0.25*H, 1.25*H, 25)
    Iu_fit = np.interp(z_fit, std_z, std_Iu[i,:])
    func = UnivariateSpline(z_fit, Iu_fit)    
    Iu_fit = func(z_fit)
    Iu_H[i] = np.interp(H, z_fit, Iu_fit)

#In the standard profiles from the BLWTL, only streamwise turbulence intensity
#is given, therefore, the turbulence intensities in the other directions are 
#are calculated using the cobra probe measurments. 
#
#The procedure involves first calculating the ratios of Iv/Iu and Iw/Iu from 
#the cobra probe measurment and then scale the standard Iu profile by the 
#coresponding ratio to get the Iv and Iw based on the standard profile  

std_I = np.zeros((n_exposures, n_compnt, std_shape[0]))
for i in range (n_exposures):        
    Iu = np.interp(std_z, cobra_z, cobra_I[i,0,:])
    for j in range(n_compnt):
        I = np.interp(std_z, cobra_z, cobra_I[i,j,:])
        std_I[i,j,:] = std_Iu[i,:]*(I/Iu)  
        
#The final profiles to be used in the CFD
#The turbulence length scale profile from the cobra probe measurment is 
#very dispersed. A spline is used to fit the measured data for cfd use
fit_z = np.linspace(0.0125, 2.5, 200) + 0.0072
fit_L = np.zeros((n_exposures, n_compnt, len(fit_z)))
fit_Uav = np.zeros((n_exposures, len(fit_z)))
fit_I = np.zeros((n_exposures, n_compnt, len(fit_z)))

for i in range (n_exposures):        
    spl = UnivariateSpline( std_z, std_Uav[i,:], k=5,  ext=3, s=0.5)
    fit_Uav[i,:] = spl(fit_z)
    for j in range(n_compnt):
        fit_I[i,j,:] = np.interp(fit_z, std_z, std_I[i,j,:])        
        spl = UnivariateSpline(cobra_z, cobra_L[i,j,:], k=3,  ext=0)
        fit_L[i,j,:] = spl(fit_z)
        #Linearly regress the values above 2H because of limitted data
        x = cobra_z[10:]; y = cobra_L[i,j,10:]
        slope, intercept,_,_,_ = stats.linregress(x, y)
        for k in range(len(fit_z)):
            if fit_z[k] > 2.0*H:
                fit_L[i,j,k] = slope*fit_z[k] + intercept

       
#Mean velocity profile comparison 
plt, fig = cwe.setup_plot(plt, 22, 20, 18)
for i in range (n_exposures):
    ax = fig.add_subplot(1, 3, 1 + i)
    ax.tick_params(direction='in', size=8)
    ax.set_xlim([0,1.5])
    ax.set_ylim([0,5.5])   
    ax.set_xlabel(r'$U_{av}/U_{H}$') 
    ax.set_title(exposure_names[i], fontsize=24)
    
    ax.plot(fit_Uav[i,:]/U_H[i], fit_z/H,  'k-', 
            markersize=8, markerfacecolor='none',markeredgewidth=1.5) 
    
    ax.plot(std_Uav[i,:]/U_H[i], std_z/H,  'bo', 
            markersize=8, markerfacecolor='none',markeredgewidth=1.5)    
    ax.plot(cobra_Uav[i,:]/U_H[i], cobra_z/H,  'rs', 
            markersize=8, markerfacecolor='none',markeredgewidth=1.5)    
    if i == 0:    
        ax.set_ylabel('$z/H$')        
        ax.legend(['Fitted', 'Standard','CobraProbe'], loc=0, framealpha=1.0)   
    if i !=0:
        ax.set_yticklabels([])
    plt.grid(linestyle='--')     

fig.set_size_inches(36/2.54, 20/2.54)
plt.tight_layout()
plt.savefig('../data/BLWTL/Plots/Uav_profiles.pdf')
plt.savefig('../data/BLWTL/Plots/Uav_profiles.png')
plt.show()


#Plot the turbulence intensity profile comparison 
plot_name = ['I_u','I_v','I_w']
for j in range(n_compnt):
    plt, fig = cwe.setup_plot(plt, 22, 20, 18)
    for i in range (n_exposures):
        ax = fig.add_subplot(1, 3, 1 + i)
        ax.tick_params(direction='in', size=8)
        ax.set_xlim([0,0.5])
        ax.set_ylim([0,5.5])   
        ax.set_xlabel('$' + plot_name[j] + '$') 
        ax.set_title(exposure_names[i], fontsize=24)
        
        ax.plot(fit_I[i,j,:], fit_z/H,  'k-', markersize=8, 
                markerfacecolor='none',markeredgewidth=1.5)  
        
        ax.plot(std_I[i,j,:], std_z/H,  'bo', markersize=8, 
                markerfacecolor='none',markeredgewidth=1.5)    
        ax.plot(cobra_I[i,j,:], cobra_z/H,  'rs', markersize=8, 
                markerfacecolor='none',markeredgewidth=1.5)    
        if i == 0:    
            ax.set_ylabel('$z/H$')        
            ax.legend(['Fitted', 'Standard','CobraProbe'], loc=0, framealpha=1.0)   
        if i !=0:
            ax.set_yticklabels([])
        plt.grid(linestyle='--')     
    
    fig.set_size_inches(36/2.54, 20/2.54)
    plt.tight_layout()
    plt.savefig('../data/BLWTL/Plots/' + plot_name[j] + '_profiles.png')
    plt.savefig('../data/BLWTL/Plots/' + plot_name[j] + '_profiles.pdf')
    plt.show()


#Plot the integral legth scale of turbulence profile comparison 
plot_name = ['L_u','L_v','L_w']
for j in range(n_compnt):
    plt, fig = cwe.setup_plot(plt, 22, 20, 18)
    for i in range (n_exposures):
        ax = fig.add_subplot(1, 3, 1 + i)
        ax.tick_params(direction='in', size=8)
        ax.set_xlim([0,7.0])
        ax.set_ylim([0,5.5])   
        ax.set_xlabel('$' + plot_name[j] + '/H$') 
        ax.set_title(exposure_names[i], fontsize=24)
        
        ax.plot(fit_L[i,j,:]/H, fit_z/H,  'k-', markersize=8, 
                markerfacecolor='none',markeredgewidth=1.5)    
        ax.plot(cobra_L[i,j,:]/H, cobra_z/H,  'rs', markersize=8, 
                markerfacecolor='none',markeredgewidth=1.5)    
        
        if i == 0:    
            ax.set_ylabel('$z/H$')     
        ax.legend(['Fitted','CobraProbe'], loc=0, framealpha=1.0)   

        if i !=0:
            ax.set_yticklabels([])
        plt.grid(linestyle='--')     
    
    fig.set_size_inches(36/2.54, 20/2.54)
    plt.tight_layout()
    plt.savefig('../data/BLWTL/Plots/' + plot_name[j] + '_profiles.png')
    plt.savefig('../data/BLWTL/Plots/' + plot_name[j] + '_profiles.pdf')
    plt.show()

#Print the turbulence intensity at roof height for each exposure condition
print("Roof-height stream-wise mean velocity and turbulence intensities(%):")
print("\tTerrain\tUh(m/s)\tIu(%)")
for i in range (n_exposures):
    print(("\t" + exposure_names[i] + '\t%0.2f\t%0.2f')%(U_H[i], 100*Iu_H[i]))


#Write the ABL file based on the fitted data that holdes all the profile 
# in the following form:
# z(m)  Uav(m/s)    Iu  Iv  Iw  Lu(m)   Lv(m)   Lw(m)
for i in range (n_exposures): 
    file_name = '../data/BLWTL/cfd_abl/' + exposure_names[i] + '_abl' 
    cwe.write_abl_profile(fit_z, fit_Uav[i,:], fit_I[i,0,:], fit_I[i,1,:], 
                          fit_I[i,2,:], fit_L[i,0,:], fit_L[i,1,:], 
                          fit_L[i,2,:], file_name)

