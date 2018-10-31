# Machine-learned patterns in economic development

This repository contains the code and instructions for reproducing the results of the paper "Machine-learned patterns suggest that diversification drives economic development" by Charles D. Brummitt, Andrés Gómez-Liévano, Ricardo Hausmann, and Matthew H. Bonds.

## Reproducing the results

The results of this paper were generated mostly with Python 3.5.3 and, in a few minor parts, in R.

Clone this repository to your local machine. For the bulk of the results, create a virtual environment using your favorite tool for Python (e.g., `virtualenv` or Anaconda), and then install the requirements in `requirements.txt`. For example, with Anaconda:

	# Clone the repo:
	git clone <repo-URL>

	# Create a virtual environment:
	conda create --name id_pat_econ_dev python=3.5.3

	# Install pip inside that virtual environment
	conda install -n id_pat_econ_dev pip

	# Activate the virtual environment and install the requirements
	source activate id_pat_econ_dev
	pip install -r requirements.txt

### Jupyter notebook for creating (most of) the figures

The Jupyter notebook

	Create_figures.ipynb

in the `notebooks` folder uses the scripts in the `scripts` folder to create most of the figures in the paper and to run the experiments. The fitted GAM model is stored in the folder entitled `notebooks/results`.

The notebook
	
	robustness_checks_phi_0.ipynb

conducts the two robustness checks described in the SI, in which the score on the first principal component is substituted with total export value per capita or with diversification (defined as the number of products with revealed comparative advantage greater than one).


### QQ plot 

The folder

	notebook/use mgcv and R to determine how to transform the target

contains a single CSV file called

	Rpop__data_target__pca_2__target_is_difference_True.csv

that is created by the Jupyter notebook `Create_figures.ipynb`. This CSV file contains the preprocessed data, and it is imported into R for analysis with the package [`mgcv`](https://cran.r-project.org/web/packages/mgcv/index.html).


The R Markdown file

	Determine how to transform the target.Rmd

creates the three GAMs on the preprocessed data with and without the target transformed by square root. It uses the package [`mgcv`](ttps://cran.r-project.org/web/packages/mgcv/index.html) to compute quantile-quantile plots and plots of the residuals to determine how far the target is from normally distributed.


### Regressions with other indicators

Figures SI-7 through SI-10 are made by the R script `notebooks/scriptsR/PriSDA_correlations_with_PC0.R`.
