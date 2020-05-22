import DNNLikelihood

lik = DNNLikelihood.Lik(name=None,
	input_file='C:/Users/Admin/Dropbox/Work/09_Resources/Git/GitHub/DNNLikelihood/DNNLikelihood_dev/tutorials/toy/toy/likelihood/toy_likelihood.h5', 
verbose = 1)

name = lik.name
def logpdf(x_pars,*args,**kwargs):
	return lik.logpdf_fn(x_pars,*args,**kwargs)
logpdf_args = lik.logpdf.args
logpdf_kwargs = lik.logpdf.kwargs
pars_pos_poi = lik.pars_pos_poi
pars_pos_nuis = lik.pars_pos_nuis
pars_central = lik.pars_central
pars_init_vec = lik.logpdf_profiled_max['X']
pars_labels = lik.pars_labels
pars_bounds = lik.pars_bounds
ndims = lik.ndims
output_folder = lik.output_folder