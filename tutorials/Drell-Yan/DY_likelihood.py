from timeit import default_timer as timer

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import integrate, special, stats

########################################
## Data

def import_hdf5(file):
    f = h5py.File(filename, 'r')
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])
    return np.array(data)

filename = "Replicas.h5"
binsmass = np.array([300, 330, 365, 410, 460, 520, 590, 680, 790, 910, 1070, 1260, 1500,
                     1800, 2160, 2620, 3200, 3930, 13000])
nbins = len(binsmass)-1
Replicas_Ik = import_hdf5(filename)
_, ncoefficients, nreplicas = np.shape(Replicas_Ik)
ndims = nbins*ncoefficients
central_replica = 12
SV_replicas = range(0, 25)
pdf_replicas = range(25, 55)
alphaS_replicas = [55, 56]
indexmap=np.array([[i,j] for i in range(nbins) for j in range(ncoefficients)])
inverseindexmap = np.reshape(np.arange(ndims),(nbins,ncoefficients))
Replicas_alpha = Replicas_Ik.reshape((nbins*ncoefficients, nreplicas))

########################################
########### Central values #############

X_0_Ik = Replicas_Ik[:, :, central_replica]
X_0_alpha = Replicas_alpha[:, central_replica]

########################################
############ PDF uncertainty ###########

X_pdfreplicas_Ik = Replicas_Ik[:, :, pdf_replicas]
X_pdfreplicas_alpha = Replicas_alpha[:, pdf_replicas]
#X_pdfreplicas_alpha_norm = (X_pdfreplicas_alpha.T/X_0_alpha).T
#X_pdf_mu_alpha_norm = np.mean(X_pdfreplicas_alpha_norm, axis=1)
#X_Sigma_pdf_alphabeta_norm = np.cov(X_pdfreplicas_alpha_norm)
#X_Sigma_pdf_alphabeta_norm_eigenval, X_Sigma_pdf_alphabeta_norm_eigenvec = np.linalg.eig(X_Sigma_pdf_alphabeta_norm)
#X_Sigma_pdf_alphabeta_norm_eigenvec = X_Sigma_pdf_alphabeta_norm_eigenvec.T
#Sigma_pdf_alphabeta_norm = np.array([[np.log(X_Sigma_pdf_alphabeta_norm[i, j]/X_pdf_mu_alpha_norm[i]/X_pdf_mu_alpha_norm[j]+1)
#                                      for i in range(len(X_pdf_mu_alpha_norm))] for j in range(len(X_pdf_mu_alpha_norm))])
#Sigma_pdf_alphabeta_norm_eigenval, Sigma_pdf_alphabeta_norm_eigenvec = np.linalg.eig(Sigma_pdf_alphabeta_norm)
#Sigma_pdf_alphabeta_norm_eigenvec = Sigma_pdf_alphabeta_norm_eigenvec.T
nnuis_pdf = 30
#nuis_pdf_sign=np.array([-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,-1])
#Sigma_pdf_alphabeta_norm_eigenval_reduced = Sigma_pdf_alphabeta_norm_eigenval[:nnuis_pdf]
#Sigma_pdf_alphabeta_norm_eigenvec_reduced = (nuis_pdf_sign*Sigma_pdf_alphabeta_norm_eigenvec[:nnuis_pdf].T).T
SM_indices = inverseindexmap[:, 0]
c1_indices = inverseindexmap[:, 1]
c2_indices = inverseindexmap[:, 2]
c3_indices = inverseindexmap[:, 3]
c4_indices = inverseindexmap[:, 4]
c5_indices = inverseindexmap[:, 5]

def X_alpha(delta_pdf):
    if len(delta_pdf) != nnuis_pdf:
        raise Exception("Incorrect number of pdf nuisance parameters.")
    delta_pdf = np.array(delta_pdf)
    tmp = np.array([(X_pdfreplicas_alpha[:, i]-X_0_alpha)/X_0_alpha*delta_pdf[i] for i in range(len(delta_pdf))])
    return X_0_alpha*np.exp(np.sum(tmp,axis=0))

def logprior_pdf(delta_pdf):
    if len(delta_pdf) != nnuis_pdf:
        raise Exception("Incorrect number of pdf nuisance parameters.")
    return -np.log((2*np.pi)**(len(delta_pdf)/2))-1/2*np.sum(delta_pdf**2)

########################################
###### Scale Variation uncertainty #####

X_SV_replicas_Ik=Replicas_Ik[:,:,SV_replicas]
X_SV_min_Ik = np.min(X_SV_replicas_Ik, axis=2)
X_SV_max_Ik = np.max(X_SV_replicas_Ik, axis=2)
X_SV_mu_Ik = np.mean(X_SV_replicas_Ik, axis=2)
sigma_SM_SV_min_I = X_SV_min_Ik[:, 0]
sigma_SM_SV_max_I = X_SV_max_Ik[:, 0]
sigma_SM_SV_mean_I = X_SV_mu_Ik[:, 0]
sigma_SM_0_I = X_0_Ik[:, 0]

# Gaussian
sigma_SM_SV_var_I = np.max([np.abs(sigma_SM_SV_max_I-sigma_SM_0_I),np.abs(sigma_SM_SV_min_I-sigma_SM_0_I)],axis=0)**2
k_SV_I = np.sqrt(sigma_SM_SV_var_I)/sigma_SM_SV_mean_I
nnuis_SV = nbins

def Factor_SV_gaussian_I(delta_SV):
    if len(delta_SV) != nnuis_SV:
        raise Exception("Incorrect number of SV nuisance parameters.")
    return (1+k_SV_I*delta_SV)

def logprior_SV_gaussian(delta_SV):
    if len(delta_SV) != nnuis_SV:
        raise Exception("Incorrect number of SV nuisance parameters.")
    return -np.log((2*np.pi)**(len(delta_SV)/2))-1/2*np.sum(delta_SV**2)

# Uniform (smoothed)
sigma_SV_min_I = sigma_SM_0_I-np.sqrt(3*sigma_SM_SV_var_I)
sigma_SV_max_I = sigma_SM_0_I+np.sqrt(3*sigma_SM_SV_var_I)
eta_SV_I = (sigma_SV_max_I-sigma_SV_min_I)/sigma_SM_0_I
phi_SV_I = (sigma_SV_min_I)/sigma_SM_0_I

def Factor_SV_Uniform_I(delta_SV):
    if len(delta_SV) != nnuis_SV:
        raise Exception("Incorrect number of SV nuisance parameters.")
    return ((1+delta_SV)/2*eta_SV_I+phi_SV_I)

def smooth_uniform(x,min,max,smooth_factor):
    if max<=min:
        raise("Max should be greater than min.")
    return 1/2/(max-min)*(special.erf(1/np.sqrt(2)/smooth_factor*(max-x))-special.erf(1/np.sqrt(2)/smooth_factor*(min-x)))

min_SV, max_SV = [-1, 1]
smooth_factor_SV = 1/5.
def logprior_SV_uniform(delta_SV):
    if len(delta_SV) != nnuis_SV:
        raise Exception("Incorrect number of SV nuisance parameters.")
    return np.sum(np.log(smooth_uniform(delta_SV,min_SV,max_SV,smooth_factor_SV)))

########################################
########### AlphaS uncertainty #########

X_aplhaS_replicas_Ik=Replicas_Ik[:,:,alphaS_replicas]
X_alphaS_0_Ik = X_0_Ik
X_alphaS_min_Ik = np.min(X_aplhaS_replicas_Ik, axis=2)
X_alphaS_max_Ik = np.max(X_aplhaS_replicas_Ik, axis=2)
X_alphaS_mu_Ik = np.mean(X_aplhaS_replicas_Ik, axis=2)
X_alphaS_0_Ik = X_0_Ik
sigma_SM_alphaS_min_I = X_alphaS_min_Ik[:, 0]
sigma_SM_alphaS_max_I = X_alphaS_max_Ik[:, 0]
sigma_SM_alphaS_mu_I = X_alphaS_mu_Ik[:, 0]
sigma_SM_0_I = X_0_Ik[:, 0]
var_alphaS_I = np.max([np.abs(sigma_SM_alphaS_max_I-sigma_SM_0_I),np.abs(sigma_SM_alphaS_min_I-sigma_SM_0_I)],axis=0)**2
k_alphaS_I = np.sqrt(var_alphaS_I)/sigma_SM_0_I
nnuis_alphaS = 1

def Factor_alphaS_I(delta_alphaS):
    try:
        len(delta_pdf)
        raise Exception("pdf nuisance parameters should be a number and not a list or array.")
    except:
        pass
    return (1+k_alphaS_I*delta_alphaS)

def logprior_alphaS(delta_alphaS):
    try:
        len(delta_pdf)
        raise Exception("pdf nuisance parameters should be a number and not a list or array.")
    except:
        pass
    return -np.log(np.sqrt(2*np.pi))-1/2*delta_alphaS**2

########################################
####### Experimental systematics #######

nnuis_Syst = nbins
k_Syst_I = np.full(nnuis_Syst,0.02)

def Factor_Syst_I(delta_Syst):
    if len(delta_Syst) != nnuis_Syst:
        raise Exception("Incorrect number of experimental systematics nuisance parameters.")
    return (1+k_Syst_I*delta_Syst)

def logprior_Syst(delta_Syst):
    if len(delta_Syst) != nnuis_Syst:
        raise Exception(
            "Incorrect number of experimental systematics nuisance parameters.")
    return -np.log((2*np.pi)**(len(delta_Syst)/2))-1/2*np.sum(delta_Syst**2)

########################################
######### Observed/Expected XS #########

npoi = 2
N_obs_I = [18415, 14073, 11113, 7442, 5133, 3414, 2374, 1370, 785, 486, 263, 123, 53, 26, 3, 2, 1, 0]

def sigma_exp_I(pars):
    if len(pars) != 2+nnuis_alphaS+nnuis_SV+nnuis_pdf+nnuis_Syst:
        raise Exception("Incorrect number of parameters.")
    cW = -1*pars[0]*((0.65**2)/(80.385**2))/1000
    cY = -1*pars[1]*((0.35**2)/(80.385**2))/1000
    delta_alphaS = pars[npoi]
    delta_SV = pars[npoi+1:npoi+nnuis_SV+1]
    delta_pdf = pars[npoi+nnuis_SV+1:npoi+nnuis_SV+nnuis_pdf+1]
    delta_Syst = pars[npoi+nnuis_SV+nnuis_pdf+1:npoi+nnuis_SV+nnuis_pdf+nnuis_Syst+1]
    sigma_SM_I = X_alpha(delta_pdf)[SM_indices]
    c1_I = X_alpha(delta_pdf)[c1_indices]
    c2_I = X_alpha(delta_pdf)[c2_indices]
    c3_I = X_alpha(delta_pdf)[c3_indices]
    c4_I = X_alpha(delta_pdf)[c4_indices]
    c5_I = X_alpha(delta_pdf)[c5_indices]
    factor_pdf=(1+
                (c1_I**2+c2_I**2)*cW**2+
                (c3_I**2+c4_I**2+c5_I**2)*cY**2+
                (2*c1_I*c3_I+2*c2_I*c4_I)*cW*cY+
                2*c1_I*cW+
                2*c3_I*cY)
    res = sigma_SM_I*factor_pdf*Factor_alphaS_I(delta_alphaS)*Factor_SV_Uniform_I(delta_SV)*Factor_Syst_I(delta_Syst)
    return res

def N_exp_I(pars):
    """
    Expected number of events with 100/fb.
    """
    return 100*1000*sigma_exp_I(pars)

########################################
################ Logpdf ################

def loglik_DY(pars, obs):
    """
    pars = poi and nuis parameters
    obs = np.array with observed XS in each bin (shape: (bin,))
    """
    exp = N_exp_I(pars)
    logfact = np.array(list(map(lambda x: np.math.lgamma(x+1), obs)))
    res = -1*logfact+obs*np.log(exp)-exp
    res = np.sum(res)
    if np.isnan(res):
        return -np.inf
    return res

def logprior_nuis_DY(pars):
    """
    pars = poi and nuis parameters
    obs = np.array with observed XS in each bin (shape: (bin,))
    obs_cov = np.array with XS covariance matrix (shape: (bin,bin))
    """
    delta_alphaS = pars[npoi]
    delta_SV = pars[npoi+1:npoi+nnuis_SV+1]
    delta_pdf = pars[npoi+nnuis_SV+1:npoi+nnuis_SV+nnuis_pdf+1]
    delta_Syst = pars[npoi+nnuis_SV+nnuis_pdf+1:npoi+nnuis_SV+nnuis_pdf+nnuis_Syst+1]
    logprior = logprior_alphaS(delta_alphaS) + logprior_SV_uniform(delta_SV)+logprior_pdf(delta_pdf)+logprior_Syst(delta_Syst)
    return logprior

min_WY, max_WY = [-5., 5.]
smooth_factor_WY = 1/5.
def logprior_poi_DY(pars):    
    WY = pars[0:2]
    return np.sum(np.log(smooth_uniform(WY, min_WY, max_WY, smooth_factor_WY)))

def logpdf_DY(pars, obs):
    return loglik_DY(pars, obs)+logprior_nuis_DY(pars)+logprior_poi_DY(pars)

########################################
########### Logpdf parameters ##########

name = "DY_likelihood"
logpdf = logpdf_DY
logpdf_args = [N_obs_I]
logpdf_kwargs = None
pars_central = np.zeros(nnuis_pdf+nnuis_SV+nnuis_alphaS+nnuis_Syst+npoi)
pars_pos_poi = [0,1]
pars_pos_nuis = range(2, len(pars_central))
pars_labels_poi = [r"$W$", r"$Y$"]
pars_labels_nuis_alphaS = [r"$\delta^{\alpha_{S}}$"]
pars_labels_nuis_SV = [r"$\delta^{\rm SV}_{%s}$"%str(i) for i in range(1,nnuis_SV+1)]
pars_labels_nuis_pdf = [r"$\delta^{\rm pdf}_{%s}$"%str(i) for i in range(1,nnuis_pdf+1)]
pars_labels_nuis_Syst = [r"$\delta^{\rm sys}_{%s}$"%str(i) for i in range(1,nnuis_Syst+1)]
pars_labels = pars_labels_poi+pars_labels_nuis_alphaS+pars_labels_nuis_SV+pars_labels_nuis_pdf+pars_labels_nuis_Syst
pars_bounds_poi = np.array([[-5,5],[-5,5]])
pars_bounds_nuis_alphaS = np.array([[-5, 5]])
pars_bounds_nuis_SV = np.array([[-5, 5] for i in range(nnuis_SV)])
pars_bounds_nuis_pdf = np.array([[-5, 5] for i in range(nnuis_pdf)])
pars_bounds_nuis_Syst = np.array([[-5, 5] for i in range(nnuis_Syst)])
pars_bounds = np.concatenate((pars_bounds_poi, pars_bounds_nuis_alphaS, pars_bounds_nuis_SV, pars_bounds_nuis_pdf,pars_bounds_nuis_Syst))
output_folder = "Likelihood"
