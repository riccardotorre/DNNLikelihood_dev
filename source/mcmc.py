__all__ = ["MCMC"]

import builtins
ShowPrints = True

def print(*args, **kwargs):
    global ShowPrints
    if ShowPrints:
        return builtins.print(*args, **kwargs)
    else:
        return None  
        
# print("This should print")
#ShowPrints = False
#print("This should not print")
#ShowPrints = True

import os
from timeit import default_timer as timer
from datetime import datetime
import sys

from jupyterthemes import jtplot
import matplotlib.pyplot as plt
import numpy as np
import pickle
import psutil
from multiprocessing import Pool

import emcee
from .data_sample import Data_sample
from . import utility

class MCMC(object):
    """Class defining MCMC sampling based on the emcee3 sampler.
    Parameters
    ----------
    logprob_fn : callable
        A function that takes a vector in the parameter space as input
        and returns the natural logarithm of the posterior probability 
        (up to an additive constant) for that position.
        It assumes that all parameters ar input as a single vector of
        length 'n_pars_phys+n_pars_nuis' with the first n_pars_phys being
        the physical parameters and the next n_pars_nuis the nuisance
        parameters.
        It may have additional arguments logprob_fn_args.
        In case vectorize=True, emcee accepts a function which returns
        a list of lenght nwalkers.
        See also emcee documentation (https://emcee.readthedocs.io/en/stable/user/sampler/)

    logprob_fn_args (optional) : array_like[args]
        List (or array) of additional arguments of the logprob_fn function.
        In case a DNN 'model' is provided, this list is automatically
        parsed as
        [model, scalerX, scalerY, nwalkers, logprob_threshold]
        See also emcee documentation (https://emcee.readthedocs.io/en/stable/user/sampler/)

    biased (optional) : bool
        Choose between an unbiased sampling and a sampling biased
        towards maxima of the logprob profiled over the nuisance 
        parameters computed for random points of the physical parameters
        (generated according to their distribution).
        If moves parameters is not given 'biased' acts by fixing the 
        moves parameter either to
        moves = emcee.moves.StretchMove()
        for unbiased sampling ('biased=False') or to
        emcee.moves.GaussianMove(0.0005, mode="random", factor=None)
        for biased sampling ('biased=True'). Independently from the moves parameters, 
        'biased' also determines walkers initialization. For 'biased=False'
        the walkers initialization is chosen as random points of all the parameters 
        (generated according to their distribution), while for 'biased=True' as points
        corresponding to maxima of the logprob function profiled over the nuisance 
        parameters computed for random points of the physical parameters 
        (generated according to their distribution).
        
    n_pars_phys : int
        Number of physical parameters. It should be the first n_pars_phys
        parameters in the 'n_pars_phys+n_pars_nuis' dimensional 
        parameters vector.
        
    n_pars_nuis : int
        Number of nuisance parameters. It should be the last n_pars_nuis
        parameters in the 'n_pars_phys+n_pars_nuis' dimensional 
        parameters vector.

    pars_distrib : array_like[args]
       List (or array) of distributions from the scipy.stats module
       indicating the distribution of the parameters in the same order
       as in the parameters vector.
       Ex.
       pars_distrib = [stats.uniform(..),stats.norm(...),...]

    nwalkers : inf
       Number of walkers for the emcee sampler.
       See also emcee documentation (https://emcee.readthedocs.io/en/stable/user/sampler/)

    nsteps : inf
       Number of steps for the emcee sampler.
       See also emcee documentation (https://emcee.readthedocs.io/en/stable/user/sampler/)

    basefilename : str
       Base name of the file containing the backend and the sampling.
       From this two attributes are constructes:
       backend_filename = os.path.splitext(basefilename)[0]+"_backend.h5"
       data_sample_filename = os.path.splitext(basefilename)[0]+"_data_sample.pickle"
       For backend_filename see also emcee documentation 
       (https://emcee.readthedocs.io/en/stable/user/sampler/)

    chains_name : str
       Name of the chains in the file backend_filename.
       See also emcee documentation (https://emcee.readthedocs.io/en/stable/user/sampler/)

    new_sampler (defaule=True): bool
       Whether to generate a new sampler or start with an existing sampler
       loaded from backend_filename.

    moves (defaule= StretchMove()): iterable (ndim,)
       This can be a single move object, a list of moves, or a “weighted” list of moves
       Ex.
       [(emcee.moves.StretchMove(), 0.1), ...]
       When running, the sampler will randomly select a move from this list 
       (optionally with weights) for each proposal.
       
    model : keras.model
       This is an optional argument of logprob_fn to compute the logprobability
       using a pre-trained DNNLikelihood
    
    scalerX, scalerY : StandardScaler objects (from sklearn.preprocessing)
       This are optional arguments of logprob_fn to compute the logprobability
       using a pre-trained DNNLikelihood. They scale X and Y data properly to make inference
       with the same scale as training and scale them back to deliver result.
       If no scaler has been applied to data this can be set to
       Ex.
       scalerX = StandardScaler(with_mean=False, with_std=False)
       scalerX.fit(X_train)
    
    logprob_threshold : float
       This are optional arguments of logprob_fn to compute the logprobability
       using a pre-trained DNNLikelihood. It is the minimum value of logprob
       used to train the given DNNLikelihood model.
    
    vectorize : bool
       If True, log_prob_fn is expected to accept a list of parameter vectors
       and to output a list of logprobs.
    
    Attributes
    ----------
    ndim : int
       Number of dimensions given by 'n_pars_phys + n_pars_nuis'
    
    backend_filename : str
       Name of the file containing the emcee backend. It is set from
       basefilename as os.path.splitext(basefilename)[0]+"_backend.h5"
       
    data_sample_filename : str
       Name of the file containing the emcee backend. It is determined from
       basefilename as os.path.splitext(basefilename)[0]+"_data_sample.pickle"
    
    Methods
    ----------
    
    """

    def __init__(self,
                 logprob_fn=None,
                 logprob_fn_args=None,
                 biased=False,
                 n_pars_phys=None,
                 n_pars_nuis=None,
                 labels=None,
                 pars_distrib=None,
                 nwalkers=None,
                 nsteps=None,
                 basefilename=None,
                 chains_name=None,
                 new_sampler=True,
                 moves=None,
                 parallel_CPU=False,
                 model=None,
                 scalerX=None,
                 scalerY=None,
                 logprob_threshold=None,
                 vectorize=False,
                 #backend = None,
                 #sampler = None
                 ):
        self.logprob_fn = logprob_fn
        self.biased = biased
        self.n_pars_phys = n_pars_phys
        self.n_pars_nuis = n_pars_nuis
        self.ndim = self.n_pars_phys + self.n_pars_nuis
        self.labels = labels
        self.pars_distrib = pars_distrib
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.basefilename = os.path.splitext(basefilename)[0]
        self.backend_filename = basefilename+"_backend.h5"
        self.data_sample_filename = basefilename+"_data_sample.pickle"
        self.chains_name = chains_name
        self.new_sampler = new_sampler
        self.moves = moves
        self.parallel_CPU = parallel_CPU
        self.model = model
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.logprob_threshold = logprob_threshold
        self.vectorize = vectorize
        self.backend = None
        self.sampler = None
        if self.model != None:
            self.logprob_fn_args = [
                self.model, self.scalerX, self.scalerY, self.nwalkers, self.logprob_threshold]
        else:
            self.logprob_fn_args = None
        if self.vectorize:
            self.parallel_CPU = False
        if moves is None:
            if self.biased:
                self.moves = [(emcee.moves.StretchMove(), 0), (emcee.moves.GaussianMove(
                    0.0005, mode="random", factor=None), 1)]
            else:
                self.moves = [(emcee.moves.StretchMove(), 1), (emcee.moves.GaussianMove(
                    0.0005, mode="random", factor=None), 0)]
        if labels is None:
            self.labels = [r"$\theta_{%d}$" % i for i in range(
                n_pars_phys)]+[r"$\nu_{%d}$" % i for i in range(n_pars_nuis)]
        if self.new_sampler:
            if os.path.exists(self.backend_filename):
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                print("Backend file", self.backend_filename, "already exists.")
                self.backend_filename = os.path.splitext(
                    self.backend_filename)[0]+"_"+now+".h5"
                print("In order not to overwrite previous results the backend filename will be changed to",
                      self.backend_filename)     
        else:
            if not os.path.exists(self.backend_filename):
                print("The new_sampler flag was set to false but the backend file", self.backend_filename, "does not exists.\nPlease change filename if you meant to import an existing backend.\nContinuing with new_sampler=True.")
                self.new_sampler = True
            else:
                print("Loading existing sampler from backend file",self.backend_filename)
                self.load_sampler()

    def __set_param__(self, par_name, par_val):
        if par_val is None:
            par_val = eval("self."+par_name)
            print("No parameter"+par_val+"specified. Its value has been set to",
                  par_val, ".")
        else:
            setattr(self, par_name,par_val)

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
            print("Please check n_pars_phys, n_pars_nuis and labels, which cannot be inferred from backend.")
####### We can think of a sidecar of backend where we also write other parameters of our object. In other words, extend the emcee backend
####### to save our parameters too
            self.ndim = ndim_from_backend
        if nsteps_from_backend != self.nsteps:
            print("Specified number of steps (nsteps) is inconsitent with loaded backend. nsteps has been set to",nsteps_from_backend, ".")
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
            progress = False
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
            self.backend = emcee.backends.HDFBackend(self.backend_filename, name=self.chains_name)
            self.backend.reset(self.nwalkers, self.ndim)
            start = timer()
            if self.biased:
                print("Computing maximum of profiled likelihood for random valued of the physical parameters to initialize chains.\nThis may take a while depending on the complexity of the likelihood function.")
                p0_pars_phys = np.transpose(
                    np.array([i.rvs(nwalkers) for i in self.pars_distrib[0:self.n_pars_phys]]))
                p0 = list(map(lambda p: np.concatenate((np.array(p), minimize(lambda delta: - self.logprob_fn(
                    np.concatenate((np.array(p), delta))), np.full(self.n_pars_nuis, 0), method="Powell")["x"])), p0_pars_phys))
            else:
                print("Initialize chains randomly according to parameters distributions")
                p0 = np.transpose(
                    np.array([i.rvs(self.nwalkers) for i in self.pars_distrib])).tolist()
            end = timer()
            print(str(self.nwalkers), "chains initialized in", end-start, "s.")
        else:
            if self.backend is None:
                try:
                    print("Initialize backend in file", self.backend_filename)
                    self.backend = emcee.backends.HDFBackend(
                        self.backend_filename, name=self.chains_name)
                    self.check_params_backend()
                    ShowPrints = verbose
                except FileNotFoundError:
                    print("Backend file does not exist. Please either change the filename or run with new_sampler=True.")
            p0 = self.backend.get_last_sample()
        print("Initial number of steps: {0}".format(self.backend.iteration))

        # Defines sampler and runs the chains
        start = timer()
        nsteps_to_run = self.set_steps_to_run(verbose=verbose)
        if nsteps_to_run == 0:
            progress = False
        if self.parallel_CPU:
            n_processes = psutil.cpu_count(logical=False)
            #if __name__ == "__main__":
            if progress:
                print("Running", n_processes, "parallel processes.")
            with Pool(n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob_fn, moves=self.moves, pool=pool, backend=self.backend, args=self.logprob_fn_args)
                self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob_fn, moves=self.moves,
                                                 args=self.logprob_fn_args, backend=self.backend, vectorize=self.vectorize)
            self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        end = timer()
        print("Done in", end-start, "seconds.")
        print("Final number of steps: {0}.".format(self.backend.iteration))

    def load_sampler(self, backend_filename=None, chains_name=None, verbose=True):
        global ShowPrints
        ShowPrints = verbose
        print("Notice: when loading sampler from backend, all parameters of the sampler but the 'logprob_fn', its args 'logprob_fn_args', and 'moves' are set by the backend. All other parameters are set consistently with the MCMC class attributes.\nPay attention that they are consistent with the parameters used to produce the sampler saved in backend.")
        self.__set_param__("backend_filename", backend_filename)
        self.__set_param__("chains_name", chains_name)
        self.new_sampler = False
        self.nsteps = 0
        start = timer()
        self.generate_sampler(progress=False,verbose=False)
        ShowPrints = verbose
        self.nsteps = self.sampler.iteration
        end = timer()
        print("Sampler for chains", self.chains_name, "from backend file",
              self.backend_filename, "loaded in", end-start, "seconds.")
        print("Available number of steps: {0}.".format(self.backend.iteration))

    def get_data_sample(self, nsamples="all", burnin=0, save=True, data_sample_filename=None):
        if save:
            self.__set_param__("data_sample_filename", data_sample_filename)
        else:
            if data_sample_filename is not None:
                data_sample_filename = None
                print("Save flag set to False. Sample file name has been set to None.")
        print("Notice: When requiring an unbiased data sample please check that the required burnin is compatible with MCMC convergence.")
        start = timer()
        print("Computing availanle number of samples")
        logprobs = self.sampler.get_log_prob()
        logprobsflat = logprobs.flatten()
        logprobsflatminusburnin = logprobs[burnin:, :].flatten()
        logprobs2 = logprobsflat[logprobsflat > -np.inf]
        logprobs2minusburnin = logprobsflatminusburnin[logprobsflatminusburnin > -np.inf]
        logprobs3 = np.unique(logprobs2, return_index=False)
        logprobs3minusburnin = np.unique(
            logprobs2minusburnin, return_index=False)
        print("There are", len(logprobsflat), "total samples, of which", len(
            logprobs2), "have finite logprob, of which", len(logprobs3), "are unique.")
        maxsamples = len(logprobs3)
        maxsamplesminusburnin = len(logprobs3minusburnin)
        if nsamples is "all":
            nsamples = maxsamplesminusburnin
        elif maxsamplesminusburnin < nsamples:
            print("Total available samples (with required burnin) is", maxsamplesminusburnin,
                  ", which is smaller than the required number of samples.\n")
            if maxsamples < nsamples:
                  print("Total available samples (witout burnin) is", maxsamples,
                        "and is also smaller than the required number of samples.\nPlease request less samples or run generate_sampler for more steps.\n")
            else:
                  print("Total available samples (witout burnin) is", maxsamples,
                        "which is larger than the required number of samples.\nPlease require less samples, reduce burnin or run generate_sampler for more steps.\n")
            return None
        print("Generating unique samples.")
        chains = self.sampler.get_chain()
        allsamples_tmp = chains[burnin:, :, :].reshape(
            [(self.nsteps-burnin)*self.nwalkers, self.ndim])
        logprob_values_tmp = logprobs[burnin:, :].reshape(
            (self.nsteps-burnin)*self.nwalkers)
        allsamples_tmp2 = np.transpose(np.append(np.transpose(
            allsamples_tmp), np.array([logprob_values_tmp]), axis=0))
        allsamples_tmp2 = allsamples_tmp2[allsamples_tmp2[:, -1] > -np.inf]
        allsamples_unique = np.unique(
            allsamples_tmp2, axis=0, return_index=False)
        indices = np.arange(len(allsamples_unique))
        rnd_indices = np.random.choice(indices, size=nsamples, replace=False)
        allsamples = allsamples_unique[rnd_indices]
        logprob_values = allsamples[:, -1]
        allsamples = allsamples[:, :-1]
        end = timer()
        print(len(allsamples), "unique samples generated in", end-start, "s.")
        if save:
            utility.save_samples(allsamples, logprob_values,data_sample_filename, chains_name)
            self.data_sample_filename = data_sample_filename
        return Data_sample(data_X=allsamples,
                           data_logprob=logprob_values,
                           name=self.chains_name,
                           data_sample_filename=data_sample_filename,
                           import_from_file=False,
                           npoints=len(allsamples),
                           shuffle=False)

    ##### Functions from the emcee documentation #####

    def next_pow_two(self,n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def autocorr_func_1d(self,x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError(
                "invalid dimensions for 1D autocorrelation function")
        n = self.next_pow_two(len(x))

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
        for yy in y:
            f += self.autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def plot_dist_and_autocorr(self, pars=0, labels=None, filename=None, save=True, verbose=False):
        jtplot.reset()
        try:
            plt.style.use("matplotlib.mplstyle")
        except:
            pass
        pars = np.array([pars]).flatten()
        self.__set_param__("labels", labels)
        if filename is None:
            filename = self.basefilename+"_figure"
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            counts, bins = np.histogram(chain.flatten(), 100)
            integral = counts.sum()
            plt.grid(linestyle="--", dashes=(5, 5))
            plt.step(bins[:-1], counts/integral, where='post')
            plt.xlabel(r"$%s$" % (labels[par].replace('$', '')))
            plt.ylabel(r"$p(%s)$" % (labels[par].replace('$', '')))
            plt.tight_layout()
            if save:
                distr_filename = filename+"_distr_"+str(par)+".pdf"
                distr_filename = utility.check_rename_file(distr_filename)
                plt.savefig(distr_filename)
                print('Saved figure', distr_filename+'.')
            if verbose:
                plt.show()
            #    print(r"%s"%(folder + "/" + modname + '_results_' + figname + ".pdf" +
            #      " created and saved."))
            plt.close()

            N = np.exp(np.linspace(np.log(100), np.log(
                chain.shape[1]), 10)).astype(int)
            gw2010 = np.empty(len(N))
            new = np.empty(len(N))
            for i, n in enumerate(N):
                gw2010[i] = self.autocorr_gw2010(chain[:, :n])
                new[i] = self.autocorr_new(chain[:, :n])

            # Plot the comparisons
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.grid(linestyle="--", dashes=(5, 5))
            plt.plot(N, N / 50.0, "--k", label=r"$\tau = S/50$")
            plt.loglog(N, gw2010, "o-", label=r"G\&W 2010")
            plt.loglog(N, new, "o-", label="new")
            ylim = plt.gca().get_ylim()
            plt.ylim(ylim)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\tau_{%s}$ estimates" %
                       (labels[par].replace('$', '')))
            plt.legend()
            plt.tight_layout()
            if save:
                autocorr_filename = filename+"_autocorr_"+str(par)+".pdf"
                autocorr_filename = utility.check_rename_file(autocorr_filename)
                plt.savefig(autocorr_filename)
                if verbose:
                    print('Saved figure', autocorr_filename+'.')
            if verbose:
                plt.show()
            plt.close()
