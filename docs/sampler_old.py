__all__ = ["Sampler"]

from os import path
import importlib
import sys
from timeit import default_timer as timer
from datetime import datetime
import sys
from scipy.optimize import minimize
import builtins
import matplotlib.pyplot as plt
try:
    from jupyterthemes import jtplot
except:
    print("No module named 'jupyterthemes'. Continuing without.\nIf you wish to customize jupyter notebooks please install 'jupyterthemes'.")
import numpy as np
import psutil
from multiprocessing import Pool

import emcee
from . import utils
from .data import Data

ShowPrints = True
def print(*args, **kwargs):
    global ShowPrints
    if type(ShowPrints) is bool:
        if ShowPrints:
            return builtins.print(*args, **kwargs)
    if type(ShowPrints) is int:
        if ShowPrints != 0:
            return builtins.print(*args, **kwargs)

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")

class Sampler(object):
    """
    .. _sampler_class:
    This class contains the ``Sampler`` object, which allows to perform Markov Chain Monte Carlo
    (MCMC) using the ``emcee`` package, which implements ensemble sampling MCMC. On top of performing
    MCMC the ``Sampler`` object contains several methods to check convergence, and export ``Data`` objects
    that can be used to train and test the DNNLikelihood.
    """
    def __init__(self,
                 name=None,
                 logpdf_input_file=None,
                 logpdf=None,
                 logpdf_args=None,
                 pars_pos_poi=None,
                 pars_pos_nuis=None,
                 pars_init_vec=None,
                 pars_labels=None,
                 nwalkers=None,
                 nsteps=None,
                 output_folder=None,
                 new_sampler=True,
                 moves_str=None,
                 parallel_CPU=False,
                 vectorize=False,
                 seed=1
                 ):
        self.logpdf_input_file = logpdf_input_file
        if self.logpdf_input_file is None:
            if name is None:
                self.name = "sampler_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                self.name = utils.check_add_suffix(name, "_sampler")
            self.logpdf = logpdf
            self.logpdf_args = logpdf_args
            self.pars_pos_poi = np.array(pars_pos_poi)
            self.pars_pos_nuis = np.array(pars_pos_nuis)
            self.pars_init_vec = pars_init_vec
            self.pars_labels = pars_labels
            if nwalkers is None:
                self.nwalkers = len(self.pars_init_vec)
            else:
                self.nwalkers = nwalkers
        else:
            print("logpdf input file has been specified. Arguments logpdf, logpdf_args, pars_pos_poi, pars_pos_nuis, pars_init_vec, pars_labels, nwalkers, and chaisn_name will be ignored and set from file.")
            in_folder,in_file = path.split(self.logpdf_input_file)
            in_file = in_file.replace(".py","")
            sys.path.append(in_folder)
            lik = importlib.import_module(in_file)
            self.logpdf = lik.logpdf
            self.logpdf_args = lik.logpdf_args
            self.pars_pos_poi = lik.pars_pos_poi
            self.pars_pos_nuis = lik.pars_pos_nuis
            self.pars_init_vec = lik.pars_init_vec
            self.pars_labels = lik.pars_labels
            self.nwalkers = len(lik.pars_init_vec)
            self.name = lik.name
        self.ndim = len(self.pars_init_vec[0])
        self.nsteps = nsteps
        if output_folder is None:
                output_folder = ""
        self.output_folder = path.abspath(output_folder)
        self.output_base_filename = path.join(self.output_folder, name)
        self.backend_filename = self.output_base_filename+"_backend.h5"
        self.data_sample_filename = self.output_base_filename+"_data.h5"
        self.figure_base_filename = self.output_base_filename+"_figure"
        self.new_sampler = new_sampler
        self.moves_str = moves_str
        if moves_str is None:
            self.moves = [(emcee.moves.StretchMove(), 1), (emcee.moves.GaussianMove(
                0.0005, mode="random", factor=None), 0)]
            print("No moves_str parameter has been specified. moves has been set to the default StretchMove() of emcee")
        else:
            self.moves = eval(moves_str)
        self.parallel_CPU = parallel_CPU
        self.vectorize = vectorize
        if self.vectorize:
            self.parallel_CPU = False
            print("Since vectorize=True the parameter parallel_CPU has been set to False.")
        self.seed = seed
        self.backend = None
        self.sampler = None
        self.__check_define_pars_labels()
        if self.new_sampler:
            self.backend_filename = utils.check_rename_file(
                self.backend_filename)
        else:
            if not path.exists(self.backend_filename):
                print("The new_sampler flag was set to false but the backend file", self.backend_filename, "does not exists.\nPlease change filename if you meant to import an existing backend.\nContinuing with new_sampler=True.")
                self.new_sampler = True
            else:
                print("Loading existing sampler from backend file",self.backend_filename)
                self.load_sampler()
                self.nsteps = nsteps
                self.check_params_backend()
        if self.pars_init_vec is None:
            print("To perform sampling you need to specify initialization for the parameters (pars_init_vec).")

    #def __set_param(self, par_name, par_val):
    #    if par_val is not None:
    #        setattr(self, par_name, par_val)
    #    #return par_val

    #def __set_param(self, par_name, par_val):
    #    if par_val is None:
    #        par_val = eval("self."+par_name)
    #    else:
    #        setattr(self, par_name, par_val)
    #    return par_val

    def __check_define_pars_labels(self):
        if self.pars_labels is None:
            self.pars_labels = []
            i_poi = 1
            i_nuis = 1
            for i in range(len(self.pars_pos_poi)+len(self.pars_pos_nuis)):
                if i in self.pars_pos_poi:
                    self.pars_labels.append(r"$\theta_{%d}$" % i_poi)
                    i_poi = i_poi+1
                else:
                    self.pars_labels.append(r"$\nu_{%d}$" % i_nuis)
                    i_nuis = i_nuis+1

    def check_params_backend(self):
        global ShowPrints
        ShowPrints = True
        nwalkers_from_backend, ndim_from_backend = self.backend.shape
        nsteps_from_backend = self.backend.iteration
        if nwalkers_from_backend != self.nwalkers:
            print("Specified number of walkers (nwalkers) is inconsitent with loaded backend. nwalkers has been set to",nwalkers_from_backend, ".")
            self.nwalkers = nwalkers_from_backend
        if ndim_from_backend != self.ndim:
            print("Specified number of dimensions (ndim) is inconsitent with loaded backend. ndim has been set to",ndim_from_backend, ".")
            print("Please check pars_pos_poi, pars_pos_nuis and pars_labels, which cannot be inferred from backend.")
####### We can think of a sidecar of backend where we also write other parameters of our object. In other words, extend the emcee backend
####### to save our parameters too
            self.ndim = ndim_from_backend
        if nsteps_from_backend != self.nsteps:
            print("Specified number of steps nsteps is inconsitent with loaded backend. nsteps has been set to",nsteps_from_backend, ".")
            self.nsteps = nsteps_from_backend

    def set_steps_to_run(self,verbose=True):
        global ShowPrints
        ShowPrints = verbose
        try:
            nsteps_current = self.backend.iteration
        except:
            nsteps_current = 0
        if self.nsteps <= nsteps_current:
            print("Please increase nsteps to run for more steps")
            nsteps_to_run = 0
        else:
            nsteps_to_run = self.nsteps-nsteps_current
        return nsteps_to_run

    def generate_sampler(self, progress=True, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        # Initializes backend (either connecting to existing one or generating new one)
        # Initilized p0 (chains initial state)
        if self.new_sampler:
            print("Initialize backend in file", self.backend_filename)
            self.backend = emcee.backends.HDFBackend(self.backend_filename, name=self.name)
            self.backend.reset(self.nwalkers, self.ndim)
            p0 = self.pars_init_vec
        else:
            if self.backend is None:
                try:
                    print("Initialize backend in file", self.backend_filename)
                    self.backend = emcee.backends.HDFBackend(
                        self.backend_filename, name=self.name)
                    #print(self.backend.iteration)
                    #print(self.nsteps)
                    ShowPrints = verbose
                except FileNotFoundError:
                    print("Backend file does not exist. Please either change the filename or run with new_sampler=True.")
            p0 = self.backend.get_last_sample()
        print("Initial number of steps: {0}".format(self.backend.iteration))

        # Defines sampler and runs the chains
        start = timer()
        nsteps_to_run = self.set_steps_to_run(verbose=verbose)
        #print(nsteps_to_run)
        if nsteps_to_run == 0:
            progress = False
        if self.parallel_CPU:
            n_processes = psutil.cpu_count(logical=False)
            #if __name__ == "__main__":
            if progress:
                print("Running", n_processes, "parallel processes.")
            with Pool(n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logpdf, moves=self.moves, pool=pool, backend=self.backend, args=self.logpdf_args)
                self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logpdf, moves=self.moves,
                                                 args=self.logpdf_args, backend=self.backend, vectorize=self.vectorize)
            self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        end = timer()
        print("Done in", end-start, "seconds.")
        print("Final number of steps: {0}.".format(self.backend.iteration))

    def load_sampler(self, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        print("Notice: when loading sampler from backend, all parameters of the sampler but the 'logpdf', its args 'logpdf_args', and 'moves' are set by the backend. All other parameters are set consistently with the ``sampler class attributes.\nPay attention that they are consistent with the parameters used to produce the sampler saved in backend.")
        self.new_sampler = False
        self.nsteps = 0
        start = timer()
        self.generate_sampler(progress=False,verbose=False)
        ShowPrints = verbose
        self.nsteps = self.sampler.iteration
        end = timer()
        print("Sampler for chains", self.name, "from backend file",
              self.backend_filename, "loaded in", end-start, "seconds.")
        print("Available number of steps: {0}.".format(self.backend.iteration))

    def get_data_sample(self, nsamples="all", test_fraction=1, burnin=0, thin=1, dtype='float64', save=False):
        print("Notice: When requiring an unbiased data sample please check that the required burnin is compatible with MCMC convergence.")
        start = timer()
        if nsamples is "all":
            allsamples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
            logpdf_values = self.sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
        else:
            if nsamples > (self.nsteps-burnin)*self.nwalkers/thin:
                print("Less samples than available are requested. Returning all available samples:",
                  str((self.nsteps-burnin)*self.nwalkers/thin),"\nYou may try to reduce burnin and/or thin to get more samples.")
                allsamples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)
                logpdf_values = self.sampler.get_log_prob(discard=burnin, thin=thin, flat=True)
            else:
                burnin = self.nsteps-nsamples*thin/self.nwalkers
                allsamples=self.sampler.get_chain(discard = burnin, thin = thin, flat = True)
                logpdf_values=self.sampler.get_log_prob(discard = burnin, thin = thin, flat = True)
        if len(np.unique(logpdf_values, axis=0, return_index=False)) < len(logpdf_values):
            print("There are non-unique samples")
        if np.count_nonzero(np.isfinite(logpdf_values)) < len(logpdf_values):
            print("There are non-numeric logpdf values.")
        end = timer()
        print(len(allsamples), "unique samples generated in", end-start, "s.")
        data_sample_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ds = Data_sample(data_X=allsamples,
                         data_Y=logpdf_values,
                         dtype=dtype,
                         pars_pos_poi=self.pars_pos_poi,
                         pars_pos_nuis=self.pars_pos_nuis,
                         pars_labels=self.pars_labels,
                         test_fraction=test_fraction,
                         name=self.name+"_"+data_sample_timestamp,
                         data_sample_input_filename=None,
                         data_sample_output_filename=self.data_sample_filename,
                         load_on_RAM=False)
        if save:
            ds.save_samples()
        return ds
    
    ##### Functions from the emcee documentation (with some modifications) #####

    def autocorr_func_1d(self,x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError( "invalid dimensions for 1D autocorrelation function")
        if len(np.unique(x)) == 1:
            print("Chain does not change in "+str(len(x))+" steps. Autocorrelation for this chain may return nan.")
        n = utils.next_power_of_two(len(x))
        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2*n)
        acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
        acf /= 4*n
        # Optionally normalize
        if norm:
            acf /= acf[0]
        return acf

    # Automated windowing procedure following Sokal (1989)
    def auto_window(self,taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010(self, y, c=5.0):
        f = self.autocorr_func_1d(np.mean(y, axis=0))
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_new(self,y, c=5.0):
        f = np.zeros(y.shape[1])
        counter=0
        for yy in y:
            fp = self.autocorr_func_1d(yy)
            if np.isnan(np.sum(fp)):
                print("Chain",counter,"returned nan. Values changed to 0 to proceed.")
                fp = np.full(len(fp),0)
            f += fp
            counter += 1
        f /= len(y)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_ml(self, y, thin=1, c=5.0, bound=5.0):
        from celerite import terms, GP
        # Compute the initial estimate of tau using the standard method
        init = self.autocorr_new(y, c=c)
        z = y[:, ::thin]
        N = z.shape[1]

        # Build the GP model
        tau = max(1.0, init / thin)
        kernel = terms.RealTerm(
            np.log(0.9 * np.var(z)), 
            -np.log(tau), 
            bounds=[(-bound, bound), (-np.log(N), 0.0)]
        )
        kernel += terms.RealTerm(
            np.log(0.1 * np.var(z)),
            -np.log(0.5 * tau),
            bounds=[(-bound, bound), (-np.log(N), 0.0)],
        )
        gp = GP(kernel, mean=np.mean(z))
        gp.compute(np.arange(z.shape[1]))

        # Define the objective
        def nll(p):
            # Update the GP model
            gp.set_parameter_vector(p)

            # Loop over the chains and compute likelihoods
            v, g = zip(*(gp.grad_log_likelihood(z0, quiet=True) for z0 in z))

            # Combine the datasets
            return -np.sum(v), -np.sum(g, axis=0)

        # Optimize the model
        p0 = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(nll, p0, jac=True, bounds=bounds)
        gp.set_parameter_vector(soln.x)

        # Compute the maximum likelihood tau
        a, c = kernel.coefficients[:2]
        tau = thin * 2 * np.sum(a / c) / np.sum(a)
        return tau

    def gelman_rubin(self, pars=0, steps="all"):
        res = []
        pars = np.array([pars]).flatten()
        for par in pars:
            if steps is "all":
                chain = self.sampler.get_chain()[:, :, par]
                si2 = np.var(chain, axis=0, ddof=1)
                W = np.mean(si2, axis=0)
                ximean = np.mean(chain, axis=0)
                xmean = np.mean(ximean, axis=0)
                n = chain.shape[0]
                m = chain.shape[1]
                B = n / (m - 1) * np.sum((ximean - xmean)**2, axis=0)
                sigmahat2 = (n - 1) / n * W + 1 / n * B
                # Exact
                Vhat = sigmahat2+B/m/n
                varVhat = ((n-1)/n)**2 * 1/m * np.var(si2, axis=0)+((m+1)/(m*n))**2 * 2/(m-1) * B**2 + 2*(
                    (m+1)*(n-1)/(m*(n**2)))*n/m * (np.cov(si2, ximean**2)[0, 1]-2*xmean*np.cov(si2, ximean)[0, 1])
                df = (2*Vhat**2) / varVhat
                Rh = np.sqrt((Vhat / W)*(df+3)/(df+1))  # correct Brooks-Gelman df
                res.append([par, n, Rh, Vhat, W])
            else:
                steps = np.array([steps]).flatten()
                for step in steps:
                    chain = self.sampler.get_chain()[:step, :, par]
                    si2 = np.var(chain, axis=0, ddof=1)
                    W = np.mean(si2, axis=0)
                    ximean = np.mean(chain, axis=0)
                    xmean = np.mean(ximean, axis=0)
                    n = chain.shape[0]
                    m = chain.shape[1]
                    B = n / (m - 1) * np.sum((ximean - xmean)**2, axis=0)
                    sigmahat2 = (n - 1) / n * W + 1 / n * B
                    # Exact
                    Vhat = sigmahat2+B/m/n
                    varVhat = ((n-1)/n)**2 * 1/m * np.var(si2, axis=0)+((m+1)/(m*n))**2 * 2/(m-1) * B**2 + 2*((m+1)*(n-1)/(m*(n**2)))*n/m *(np.cov(si2,ximean**2)[0,1]-2*xmean*np.cov(si2,ximean)[0,1])
                    df = (2*Vhat**2) / varVhat
                    Rh = np.sqrt((Vhat / W)*(df+3)/(df+1)) #correct Brooks-Gelman df
                    res.append([par, n, Rh, Vhat, W])
        return np.array(res)

    def plot_gelman_rubin(self, pars=0, npoints=5, labels=None, filename=None, save=False, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        try:
            jtplot.reset()
        except:
            pass
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        pars = np.array([pars]).flatten()
        if labels is None:
            labels = self.pars_labels
        if filename is None:
            filename = self.figure_base_filename
        for par in pars:
            idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps))))])
            idx = np.unique(idx[idx <= self.nsteps])
            idx = utils.get_spaced_elements(idx, numElems=npoints+1)
            idx = idx[1:]
            gr = self.gelman_rubin(par, steps=idx)
            plt.plot(gr[:,1], gr[:,2], '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\hat{R}_{c}(%s)$" % (labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            if save:
                figure_filename = filename+"_GR_Rc_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            plt.plot(gr[:, 1], np.sqrt(gr[:, 3]), '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{\hat{V}}(%s)$"% (labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            if save:
                figure_filename = filename+"_GR_sqrtVhat_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            plt.plot(gr[:, 1], np.sqrt(gr[:, 4]), '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\sqrt{W}(%s)$"% (labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            if save:
                figure_filename = filename+"_GR_sqrtW_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()
            

    def plot_dist(self, pars=0, labels=None, filename=None, save=False, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        try:
            jtplot.reset()
        except:
            pass
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        pars = np.array([pars]).flatten()
        if labels is None:
            labels = self.pars_labels
        if filename is None:
            filename = self.figure_base_filename
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            counts, bins = np.histogram(chain.flatten(), 100)
            integral = counts.sum()
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.step(bins[:-1], counts/integral, where='post')
            plt.xlabel(r"$%s$" % (labels[par].replace('$', '')))
            plt.ylabel(r"$p(%s)$" % (labels[par].replace('$', '')))
            plt.tight_layout()
            if save:
                figure_filename = filename+"_distr_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()

    def plot_autocorr(self, pars=0, labels=None, filename=None, save=False, verbose=True, methods=["G&W 2010", "DFM 2017", "DFM 2017: ML"]):
        global ShowPrints
        ShowPrints = verbose
        try:
            jtplot.reset()
        except:
            pass
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        pars = np.array([pars]).flatten()
        if labels is None:
            labels = self.pars_labels
        if filename is None:
            filename = self.figure_base_filename
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            # Compute the largest number of duplicated at the beginning of chains
            n_dupl = []
            for c in chain:
                n_dupl.append(utils.check_repeated_elements_at_start(c))
            n_start = max(n_dupl)+10
            if n_start > 100:
                print("There is at least one chain starting with", str(
                        n_start-10), "duplicate steps. Autocorrelation will be computer starting at", str(n_start), "steps.")
            else:
                n_start = 100
            N = np.exp(np.linspace(np.log(n_start), np.log(chain.shape[1]), 10)).astype(int)
            # GW10 method
            if "G&W 2010" in methods:
                gw2010 = np.empty(len(N))
            # New method
            if "DFM 2017" in methods:
                new = np.empty(len(N))
            # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                new = np.empty(len(N))
                ml = np.empty(len(N))
                ml[:] = np.nan

            for i, n in enumerate(N):
                # GW10 method
                if "G&W 2010" in methods:
                    gw2010[i] = self.autocorr_gw2010(chain[:, :n])
                # New method
                if "DFM 2017" in methods or "DFM 2017: ML" in methods:
                    new[i] = self.autocorr_new(chain[:, :n])
                # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                succeed = None
                bound = 5.0
                while succeed is None:
                    try:
                        for i, n in enumerate(N[1:-1]):
                            k = i + 1
                            thin = max(1, int(0.05 * new[k]))
                            ml[k] = self.autocorr_ml(chain[:, :n], thin=thin, bound=bound)
                        succeed = True
                        if bound > 5.0:
                            print("Succeeded with bounds (",str(-(bound)), ",", str(bound), ").")
                    except:
                        print("Bounds (", str(-(bound)), ",", str(bound), ") delivered non-finite log-prior. Increasing bound to (",
                              str(-(bound+5)), ",", str(bound+5), ") and retrying.")
                        bound = bound+5
            # Plot the comparisons
            plt.plot(N, N / 50.0, "--k", label=r"$\tau = S/50$")
            #plt.plot(N, N / 100.0, "--k", label=r"$\tau = S/100$")
            # GW10 method
            if "G&W 2010" in methods:
                plt.loglog(N, gw2010, "o-", label=r"G\&W 2010")
            # New method
            if "DFM 2017" in methods:
                plt.loglog(N, new, "o-", label="DFM 2017")
            # Approx method (Maximum Likelihood)
            if "DFM 2017: ML" in methods:
                plt.loglog(N, ml, "o-", label="DFM 2017: ML")
            ylim = plt.gca().get_ylim()
            plt.ylim(ylim)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\tau_{%s}$ estimates" % (labels[par].replace('$', '')))
            plt.legend()
            plt.tight_layout()
            if save:
                figure_filename = filename+"_autocorr_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()

    def plot_chains(self, pars=0, n_chains=100, labels=None, filename=None, save=False, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        try:
            jtplot.reset()
        except:
            pass
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        pars = np.array([pars]).flatten()
        if labels is None:
            labels = self.pars_labels
        if filename is None:
            filename = self.figure_base_filename
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("n_chains larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.")
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par]
            idx = np.sort([(i)*(10**j) for i in range(1, 11)
                           for j in range(int(np.ceil(np.log10(self.nsteps))))])
            idx = np.unique(idx[idx < len(chain)])
            plt.plot(idx,chain[idx][:,rnd_chains], '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$%s$" %(labels[par].replace('$', '')))
            plt.xscale('log')
            plt.tight_layout()
            if save:
                figure_filename = filename+"_chains_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()

    def plot_chains_logprob(self, n_chains=100, filename=None, save=False, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        try:
            jtplot.reset()
        except:
            pass
        try:
            plt.style.use(mplstyle_path)
        except:
            pass
        if filename is None:
            filename = self.figure_base_filename
        if n_chains > self.nwalkers:
            n_chains = np.min([n_chains, self.nwalkers])
            print("n_chains larger than the available number of walkers. Plotting all",self.nwalkers,"available chains.")
        rnd_chains = np.sort(np.random.choice(np.arange(
            self.nwalkers), n_chains, replace=False))
        chain_lp = self.sampler.get_log_prob()
        idx = np.sort([(i)*(10**j) for i in range(1, 11)
                       for j in range(int(np.ceil(np.log10(self.nsteps))))])
        idx = np.unique(idx[idx < len(chain_lp)])
        plt.plot(idx, -chain_lp[:, rnd_chains][idx], '-', alpha=0.8)
        plt.xlabel("number of steps, $S$")
        plt.ylabel(r"-logpdf")
        plt.xscale('log')
        plt.tight_layout()
        if save:
            figure_filename = filename+"_chains_logpdf.pdf"
            figure_filename = utils.check_rename_file(figure_filename)
            plt.savefig(figure_filename)
            print('Saved figure', figure_filename+'.')
        if verbose:
            plt.show()
        plt.close()
