import sys
sys.path.append('../../source')
import DNNLikelihood

lik = DNNLikelihood.Likelihood(name=None,
	likelihood_input_file='C:/Users/Admin/Dropbox/Work/09_Resources/Git/GitHub/DNNLikelihood/DNNLikelihood_dev/tutorials/toy/outfiles/toy_likelihood.json', 
verbose = 1)

name = lik.name
def logpdf(x_pars,*args):
	if args is None:
		return lik.logpdf_fn(x_pars)
	else:
		return lik.logpdf_fn(x_pars,*args)
logpdf_args = lik.logpdf_args
pars_pos_poi = lik.pars_pos_poi
pars_pos_nuis = lik.pars_pos_nuis
pars_init_vec = lik.X_prof_logpdf_max
pars_labels = lik.pars_labels
pars_bounds = lik.pars_bounds
output_folder = lik.output_folder