__all__ = ["Sampler"]

from os import path
import importlib
from copy import copy
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
    This class contains the ``Sampler`` object, which allows to perform Markov Chain Monte Carlo
    (MCMC) using the |emcee_link| package (ensemble sampling MCMC). See ref. :cite:`ForemanMackey:2012ig` for
    details about |emcee_link|. On top of performing
    MCMC the ``Sampler`` object contains several methods to check convergence, and export ``Data`` objects
    that can be used to train and test the DNNLikelihood.
    The object can be instantiated both passing a ``Likelihood`` object or a ``likelihood_script_file`` created 
    with the ``Likelihood.generate_likelihood_script_file`` method.

.. |emcee_link| raw:: html
    
    <a href="https://emcee.rhttps://emcee.readthedocs.io/en/stable/"  target="_blank"> emcee</a>
    """
    def __init__(self,
                 new_sampler=True,
                 likelihood_script_file=None,
                 likelihood=None,
                 nsteps=None,
                 moves_str=None,
                 parallel_CPU=True,
                 vectorize=False
                 ):
        """
        Instantiates the ``Sampler`` object. 
        
        There are two different ways of instantiating the ``Sampler`` object, depending on the input arguments
        :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` 
        and :attr:`Sampler.likelihood <DNNLikelihood.Sampler.likelihood>`.

        1. when :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` is specified,
        the method loads the file as a module and assigns all ``Sampler`` attributes from it. See 
        :meth:`Likelihood.generate_likelihood_script_file <DNNLikelihood.Likelihood.generate_likelihood_script_file>`
        and :attr:`Likelihood.likelihood_script_file <DNNLikelihood.Likelihood.likelihood_script_file>` for details
        about the script.

        2. when :attr:`Sampler.likelihood_script_file <DNNLikelihood.Sampler.likelihood_script_file>` is not specified,
        a likelihood object should be passed through the :attr:`Sampler.likelihood <DNNLikelihood.Sampler.likelihood>` argument. 
        The object is copied, the copy is used to save a script file (with suffix "_from_sampler") in the output folder
        and then the script is used as in 1.

        The flag ``new_sampler`` determines how the ``Sampler`` should work in case it finds and existing backend file
        with the same name. If ``new_sampler=True``, a new backend file is created appending a time stamp to the name, 
        while if ``new_sampler=False`` the backend is set to the existing file, whose content is read to check consistency
        with ``nwalkers``, ``ndims``, and ``nsteps``.

        The argument ``moves_str`` is evaluated and assigned to :attr:`Sampler.moves <DNNLikelihood.Sampler.moves>`.

        - **Arguments**

            See Class arguments.
        """
        if likelihood_script_file is None and likelihood is None:
            raise Exception("At least one of the arguments 'likelihood_script_file' and 'likelihood' need to be specified.")
        if likelihood_script_file is not None and likelihood is not None:
            print("When both the arguments 'likelihood_script_file' and 'likelihood' are specified, the latter is ignored.")
        self.new_sampler = new_sampler
        self.likelihood_script_file = likelihood_script_file
        if self.likelihood_script_file is None:
            tmp_likelihood = copy(likelihood)
            self.likelihood_script_file = tmp_likelihood.likelihood_script_file.replace(".py", "_from_sampler.py")
            tmp_likelihood.likelihood_script_file = self.likelihood_script_file
            tmp_likelihood.generate_likelihood_script_file()
        else:
            self.likelihood_script_file = path.abspath(self.likelihood_script_file)
        in_folder, in_file = path.split(self.likelihood_script_file)
        in_file = in_file.replace(".py","")
        sys.path.insert(0, in_folder)
        lik = importlib.import_module(in_file)
        self.name = lik.name.replace("likelihood","sampler")
        self.logpdf = lik.logpdf
        self.logpdf_args = lik.logpdf_args
        self.pars_pos_poi = lik.pars_pos_poi
        self.pars_pos_nuis = lik.pars_pos_nuis
        self.pars_init_vec = lik.pars_init_vec
        self.pars_labels = lik.pars_labels
        self.generic_pars_labels = utils.define_generic_pars_labels(self.pars_pos_poi, self.pars_pos_nuis)
        self.output_folder = lik.output_folder
        self.nwalkers = len(lik.pars_init_vec)
        self.ndims = len(self.pars_init_vec[0])
        self.nsteps = nsteps
        self.output_base_filename = path.join(self.output_folder, self.name)
        self.backend_filename = self.output_base_filename+"_backend.h5"
        self.data_filename = self.output_base_filename+"_data.h5"
        self.figure_base_filename = self.output_base_filename+"_figure"
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
        self.backend = None
        self.sampler = None
        #self.__check_define_pars_labels()
        if self.new_sampler:
            self.backend_filename = utils.check_rename_file(self.backend_filename)
        else:
            if not path.exists(self.backend_filename):
                print("The new_sampler flag was set to false but the backend file", self.backend_filename, "does not exists.\nPlease change filename if you meant to import an existing backend.\nContinuing with new_sampler=True.")
                self.new_sampler = True
            else:
                print("Loading existing sampler from backend file",self.backend_filename)
                self.__load_sampler()
                self.nsteps = nsteps
                self.__check_params_backend()

    def __check_params_backend(self):
        """
        Checks consistency between the parameters ``nwalkers``, ``ndims``, and ``nsteps`` assigned in the 
        :meth:``Sampler.__init__ <DNNLikelihood.Sampler.__init__> and the corresponding ones in the existing backend.
        If ``nwalkers`` or ``ndims`` are found to be inconsistent an exception is raise. If ``nsteps`` is found to 
        be inconsistent it is set to the number of available steps in the backend.
        """
        global ShowPrints
        ShowPrints = True
        nwalkers_from_backend, ndims_from_backend = self.backend.shape
        nsteps_from_backend = self.backend.iteration
        if nwalkers_from_backend != self.nwalkers:
            raise Exception("Number of walkers (nwalkers) determined from the input likelihood is inconsitent with the loaded backend. Please check inputs.")
        if ndims_from_backend != self.ndims:
            raise Exception("Number of steps (nsteps)  determined from the input likelihood is inconsitent with loaded backend. Please check inputs.")
        if nsteps_from_backend > self.nsteps:
            print("Specified number of steps nsteps is inconsitent with loaded backend. nsteps has been set to",nsteps_from_backend, ".")
            self.nsteps = nsteps_from_backend

    def __load_sampler(self, verbose=True):
        """
        Loads an existig backend when :attr:``Sampler.new_sampler <DNNLikelihood.Sampler.new_sampler>` is set to ``False``.
        In order to reconstruct the state of :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>` and 
        :attr:``Sampler.sampler <DNNLikelihood.Sampler.sampler>` consistently, the method calls the method
        :meth:``Sampler.sampler <DNNLikelihood.Sampler.run_sampler>` with zero steps. After loading the sampler, the value
        of :attr:``Sampler.nsteps <DNNLikelihood.Sampler.nsteps>` is set to the number of steps already available in
        the backend.

        - **Arguments**

            - **verbose**

                Verbose mode.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
        global ShowPrints
        ShowPrints = verbose
        print("Notice: when loading sampler from backend, all parameters of the sampler but the 'logpdf', its args 'logpdf_args', and 'moves' are set by the backend. All other parameters are set consistently with the ``sampler class attributes.\nPay attention that they are consistent with the parameters used to produce the sampler saved in backend.")
        self.nsteps = 0
        start = timer()
        self.run_sampler(progress=False, verbose=False)
        ShowPrints = verbose
        self.nsteps = self.sampler.iteration
        end = timer()
        print("Sampler for chains", self.name, "from backend file",
              self.backend_filename, "loaded in", end-start, "seconds.")
        print("Available number of steps: {0}.".format(self.backend.iteration))

    def __set_steps_to_run(self,verbose=True):
        """
        Based on the number of steps already available in the current :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>`,
        it sets the remaining number of steps to run to reach :attr:``Sampler.nsteps <DNNLikelihood.Sampler.nsteps>`. If the
        value of the latter is less or equal to the number of available steps, a warning message asking to increase the value 
        of ``nsteps`` is printed.

        - **Arguments**

            - **verbose**

                Verbose mode.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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

    def __set_pars_labels(self, pars_labels):
        """
        Returns the ``pars_labels`` choice based on the ``pars_labels`` input.

        - **Arguments**

            - **pars_labels**
            
                Could be either one of the keyword strings ``"original"`` and ``"generic"`` or a list of labels
                strings with the length of the parameters array. If ``pars_labels="original"`` or ``pars_labels="generic"``
                the function returns :attr:`Sampler.pars_labels <DNNLikelihood.Sampler.pars_labels>`
                and :attr:`Sampler.generic_pars_labels <DNNLikelihood.Sampler.generic_pars_labels>`, respectively,
                while if ``pars_labels`` is a list, the function just returns the input.

                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``
        """
        if pars_labels is "original":
            return self.pars_labels
        elif pars_labels is "generic":
            return self.generic_pars_labels
        else:
            return pars_labels

    def run_sampler(self, progress=True, verbose=True):
        """
        Constructs the attributed :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>` and 
        :attr:``Sampler.sampler <DNNLikelihood.Sampler.sampler>` and calls the ``sampler.run_mcmc`` function to
        run the sampler. Depending on the value of :attr:``Sampler.parallel_cpu <DNNLikelihood.Sampler.parallel_cpu>`
        the sampler is run on a single core (if ``False``) or in parallel using the ``Multiprocessing.Pool`` method
        (if ``True``). When running in parallel, the number of processes is set to the number of available (physical)
        cpu cores using the ``psutil`` package by ``n_processes = psutil.cpu_count(logical=False)``.
        See the documentation of the |emcee_link| and |multiprocessing_link| packages for more details on parallel
        sampling.

        If running a new sampler, the initial value of walkers is set to 
        :attr:``Sampler.pars_init <DNNLikelihood.Sampler.pars_init>`, otherwise it is set to the state of the walkers
        in the last step available in :attr:``Sampler.backend <DNNLikelihood.Sampler.backend>`.

        A progress bar of the sampling is shown by default.

        - **Arguments**

            - **verbose**

                Verbose mode.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``

.. |multiprocessing_link| raw:: html
    
    <a href="https://docs.python.org/3/library/multiprocessing.html"  target="_blank"> multiprocessing</a>

.. |psutil_link| raw:: html
    
    <a href="https://pypi.org/project/psutil/"  target="_blank"> psutil</a>

        """
        global ShowPrints
        ShowPrints = verbose
        # Initializes backend (either connecting to existing one or generating new one)
        # Initilized p0 (chains initial state)
        if self.new_sampler:
            print("Initialize backend in file", self.backend_filename)
            self.backend = emcee.backends.HDFBackend(self.backend_filename, name=self.name)
            self.backend.reset(self.nwalkers, self.ndims)
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
        nsteps_to_run = self.__set_steps_to_run(verbose=verbose)
        #print(nsteps_to_run)
        if nsteps_to_run == 0:
            progress = False
        if self.parallel_CPU:
            n_processes = psutil.cpu_count(logical=False)
            #if __name__ == "__main__":
            if progress:
                print("Running", n_processes, "parallel processes.")
            with Pool(n_processes) as pool:
                self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves, pool=pool, backend=self.backend, args=self.logpdf_args)
                self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndims, self.logpdf, moves=self.moves,args=self.logpdf_args, backend=self.backend, vectorize=self.vectorize)
            self.sampler.run_mcmc(p0, nsteps_to_run, progress=progress)
        end = timer()
        print("Done in", end-start, "seconds.")
        print("Final number of steps: {0}.".format(self.backend.iteration))

    def get_data_object(self, nsamples="all", burnin=0, thin=1, dtype='float64', test_fraction=1,  save=False):
        """
        Returns a :class:`DNNLikelihood.Data` object with ``nsamples`` samples by taking chains and logpdf values, discarding ``burnin`` steps,
        thinning by ``thin`` and converting to dtype ``dtype``. When ``nsamples="All"`` all samples available for the 
        given choice of ``burnin`` and ``thin`` are included to the :class:`DNNLikelihood.Data` object, otherwise only the first
        ``nsamples`` are included. If ``nsamples`` is more than the available number all the available samples are included
        and a warning message is printed.
        Before including samples in the :class:`DNNLikelihood.Data` object the method checks if there are duplicate samples
        (which would suggest a larger value of ``thin``) and non finite values of logpdf (e.g. ``np.nan`` or ``np.inf``)
        and print a warning in any of these cases.

        The method also allows one to pass to the :class:`DNNLikelihood.Data` a value for ``test_fraction``, which already
        splits data into ``train`` (sample from which training and valudation data are extracted) and ``test`` (sample only
        used for final test) sets. See :attr:`Data.test_fraction <DNNLikelihood.Data.test_fraction>` for more details.

        Finally, based on the value of ``save``, the generated :class:`DNNLikelihood.Data` object

        - **Arguments**

            - **verbose**

                Verbose mode.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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
        ds = Data(data_X=allsamples,
                         data_Y=logpdf_values,
                         dtype=dtype,
                         pars_pos_poi=self.pars_pos_poi,
                         pars_pos_nuis=self.pars_pos_nuis,
                         pars_labels=self.pars_labels,
                         test_fraction=test_fraction,
                         name=self.name+"_"+data_sample_timestamp,
                         data_sample_input_filename=None,
                         data_sample_output_filename=self.data_filename,
                         load_on_RAM=False)
        if save:
            ds.save_samples()
        return ds
    
    ##### Functions from the emcee documentation (with some modifications) #####

    def autocorr_func_1d(self,x, norm=True):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        
.. |emcee_tutorial_autocorr_link| raw:: html
    
    <a href="https://emcee.readthedocs.io/en/stable/tutorials/autocorr/"  target="_blank"> emcee autocorrelation tutorial</a>
        """
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
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Following the suggestion from Goodman & Weare (2010)
    def autocorr_gw2010(self, y, c=5.0):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
        f = self.autocorr_func_1d(np.mean(y, axis=0))
        taus = 2.0*np.cumsum(f)-1.0
        window = self.auto_window(taus, c)
        return taus[window]

    def autocorr_new(self,y, c=5.0):
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
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
        """
        Function from the |emcee_tutorial_autocorr_link|.
        See the link for documentation.
        """
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

    def gelman_rubin(self, pars=0, nsteps="all"):
        """
        Given a parameter (or list of parameters) ``pars`` and a number of ``nsteps``, the method computes 
        the Gelman-Rubin :cite:`Gelman:1992zz` ratio and related quantities for monitoring convergence.
        The formula for :math:`R_{c}` implements the correction due to :cite:`Brooks_1998` and is implemented here as
        
        .. math::
            R_{c} = \\sqrt{\\frac{\\hat{d}+3}{\\hat{d}+1}\\frac{\\hat{V}}{W}}.

        See the original papers for the notation.
        
        In order to be able to monitor not only the :math:`R_{c}` ratio, but also the values of :math:`\\hat{V}` 
        and :math:`W` independently, the method also computes these quantities. Usually a reasonable convergence 
        condition is :math:`R_{c}<1.1`, together with stability of both :math:`\hat{V}` and :math:`W` :cite:`Brooks_1998`.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: 0

            - **nsteps**

                If ``"all"`` then all nsteps available in current ``backend`` are included. Otherwise an integer
                number of steps or a list of integers to monitor for different steps numbers can be input.

                    - **type**: ``str`` or ``int`` or ``list``
                    - **allowed str**: ``all``
                    - **shape of list**: ``[ ]``
                    - **default**: ``all``
        
        - **Returns**

            An array constructed concatenating lists of the type ``[par, nsteps, Rc, Vhat, W]`` for each parameter
            and each choice of nsteps.

                    - **type**: ``numpy.ndarray``
                    - **shape**: ``(len(pars)*len(nsteps),5)``
        """
        res = []
        pars = np.array([pars]).flatten()
        for par in pars:
            if nsteps is "all":
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
                Rc = np.sqrt((Vhat / W)*(df+3)/(df+1))  # correct Brooks-Gelman df
                res.append([par, n, Rc, Vhat, W])
            else:
                nsteps = np.array([nsteps]).flatten()
                for step in nsteps:
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
                    Rc = np.sqrt((Vhat / W)*(df+3)/(df+1)) #correct Brooks-Gelman df
                    res.append([par, n, Rc, Vhat, W])
        return np.array(res)

    def plot_gelman_rubin(self, pars=0, npoints=5, pars_labels="original", save=True, verbose=True):
        """
        Produces plots of the evolution with the number of steps of the convergence metrics :math:`R_{c}`, 
        :math:`\\sqrt{\\hat{V}}`, and :math:`\\sqrt{W}` computed by the method 
        :meth:`Sampler.gelman_rubin <DNNLikelihood.Sampler.gelman_rubin>` for parameter (or list of parameters) ``pars``. 
        The plots are produced by computing the quantities in ``npoints`` equally spaced (in base-10 log scale) points  
        between one and the total number of available steps. 

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **npoints**

                Number of points in which the convergence metrics are computed to produce the plot.
                The points are taken equally spaced in base-10 log scale.

                    - **type**: ``int``
                    - **default**: ``5``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **save**
            
                If ``save=True`` the figures are saved as :attr:`Sampler.figure_base_filename <DNNLikelihood.Sampler.figure_base_filename>` 
                with the different suffixes ``"_GR_Rc_"+str(par)``, ``"_GR_sqrtVhat_"+str(par)``, and ``"_GR_sqrtW_"+str(par)``
                for the different quantities and parameters.
                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_base_filename
        for par in pars:
            idx = np.sort([(i)*(10**j) for i in range(1, 11) for j in range(int(np.ceil(np.log10(self.nsteps))))])
            idx = np.unique(idx[idx <= self.nsteps])
            idx = utils.get_spaced_elements(idx, numElems=npoints+1)
            idx = idx[1:]
            gr = self.gelman_rubin(par, steps=idx)
            plt.plot(gr[:,1], gr[:,2], '-', alpha=0.8)
            plt.xlabel("number of steps, $S$")
            plt.ylabel(r"$\hat{R}_{c}(%s)$" % (pars_labels[par].replace('$', '')))
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
            plt.ylabel(r"$\sqrt{\hat{V}}(%s)$"% (pars_labels[par].replace('$', '')))
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
            plt.ylabel(r"$\sqrt{W}(%s)$"% (pars_labels[par].replace('$', '')))
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
            
    def plot_dist(self, pars=0, pars_labels="original", save=False, verbose=True):
        """
        Plots the 1D distribution of parameter (or list of parameters) ``pars``.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**
            
                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **save**
            
                If ``save=True`` the figures are saved as :attr:`Sampler.figure_base_filename <DNNLikelihood.Sampler.figure_base_filename>` 
                with the different suffixes ``"_distr_"+str(par)`` for the different parameters.
                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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
        pars_labels = self.__set_pars_labels(pars_labels)
        filename = self.figure_base_filename
        for par in pars:
            chain = self.sampler.get_chain()[:, :, par].T
            counts, bins = np.histogram(chain.flatten(), 100)
            integral = counts.sum()
            #plt.grid(linestyle="--", dashes=(5, 5))
            plt.step(bins[:-1], counts/integral, where='post')
            plt.xlabel(r"$%s$" % (pars_labels[par].replace('$', '')))
            plt.ylabel(r"$p(%s)$" % (pars_labels[par].replace('$', '')))
            plt.tight_layout()
            if save:
                figure_filename = filename+"_distr_"+str(par)+".pdf"
                figure_filename = utils.check_rename_file(figure_filename)
                plt.savefig(figure_filename)
                print('Saved figure', figure_filename+'.')
            if verbose:
                plt.show()
            plt.close()

    def plot_autocorr(self, pars=0, pars_labels="original", methods=["G&W 2010", "DFM 2017", "DFM 2017: ML"], save=False, verbose=True):
        """
        Plots the autocorrelation time estimate evolution with the number of steps for parameter (or list of parameters) ``pars``.
        Three different methods are used to estimate the autocorrelation time: "G&W 2010", "DFM 2017", and "DFM 2017: ML", described in details
        in the |emcee_tutorial_autocorr_link|. The function accepts a list of methods and by default it makes the plot including all available
        methods. Notice that to use the method "DFM 2017: ML" based on fitting an autoregressive model, the |celerite_link| package needs to be
        installed.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **pars_labels**

                Argument that is passed to the :meth:`Sampler.__set_pars_labels <DNNLikelihood.Sampler._Likelihood__set_pars_labels>`
                method to set the parameters labels to be used in the plots.
                    - **type**: ``list`` or ``str``
                    - **shape of list**: ``[ ]``
                    - **accepted strings**: ``"original"``, ``"generic"``

            - **methods**

                List of methods to estimate the autocorrelation time. The three availanle methods are "G&W 2010", "DFM 2017", and "DFM 2017: ML". 
                One curve for each method will be produced.
                    - **type**: ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``["G&W 2010", "DFM 2017", "DFM 2017: ML"]``

            - **save**

                If ``save=True`` the figures are saved as :attr:`Sampler.figure_base_filename <DNNLikelihood.Sampler.figure_base_filename>`
                with the different suffixes ``"_autocorr_"+str(par)`` for the different parameters.
                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``

.. |celerite_link| raw:: html
    
    <a href="https://celerite.readthedocs.io/en/stable/"  target="_blank"> celerite</a>
        """
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
        pars_labels = self.__set_pars_labels(pars_labels)
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
            plt.ylabel(r"$\tau_{%s}$ estimates" % (pars_labels[par].replace('$', '')))
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

    def plot_chains(self, pars=0, n_chains=100, pars_labels="original", save=False, verbose=True):
        """
        Plots the evolution of chains (walkers) with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:``Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **pars**

                Could be a single integer or a list of parameters 
                for which the convergence metrics are computed.

                    - **type**: ``int`` or ``list``
                    - **shape of list**: ``[ ]``
                    - **default**: ``0``

            - **n_chains**
            
                The number of chains to 
                add to the plot.
                    - **type**: ``int``
                    - **default**: ``100``

            - **save**
            
                If ``save=True`` the  figures are saved as :attr:`Sampler.figure_base_filename <DNNLikelihood.Sampler.figure_base_filename>` 
                with the different suffixes ``"_chains_"+str(par)`` for the different parameters.
                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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
        pars_labels = self.__set_pars_labels(pars_labels)
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
            plt.ylabel(r"$%s$" %(pars_labels[par].replace('$', '')))
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

    def plot_chains_logprob(self, n_chains=100, save=False, verbose=True):
        """
        Plots the evolution of minus the logpdf values with the number of steps for ``n_chains`` randomly selected chains among the 
        :attr:``Sampler.nwalkers <DNNLikelihood.Sampler.nwalkers>`` walkers. If ``n_chains`` is larger than the available number
        of walkers, the plot is done for all walkers.

        - **Arguments**

            - **n_chains**
            
                The number of chains to 
                add to the plot.
                    - **type**: ``int``
                    - **default**: ``100``

            - **save**
            
                If ``save=True`` the figure is saved as :attr:`Sampler.figure_base_filename <DNNLikelihood.Sampler.figure_base_filename>` 
                with the suffix ``"_chains_logpdf``.
                    - **type**: ``bool``
                    - **default**: ``True``

            - **verbose**

                Verbose mode. The plots are shown in the interactive console calling ``plt.show()`` only if ``verbose=True``.
                See :ref:`notes on verbose implementation <verbose_implementation>`.

                    - **type**: ``bool``
                    - **default**: ``True``
        """
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