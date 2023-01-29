# Sharper Rates and Flexible Framework for Nonconvex SGD with Client and Data Sampling

## Quick Start
### 1. Install [Singularity](https://sylabs.io/guides/3.5/user-guide/introduction.html) (optional)
If you don't want to install Singularity, make sure that you have all dependecies from Singularity.def (python3, numpy, pytorch, etc.)

a. Pull an image 
````
singularity pull library://k3nfalt/default/python_ml:sha256.efcd1fc038228cb7eb0f6f1942dfbaa439cd95d6463015b83ceb2dbaad9e1e98
````
b. Open a shell console of the image
````
singularity shell --nv ~/python_ml_sha256.efcd1fc038228cb7eb0f6f1942dfbaa439cd95d6463015b83ceb2dbaad9e1e98.sif
````
### 2. Prepare scripts for experiments
````
PYTHONPATH=./code python3 ./code/distributed_optimization_library/experiments/page_ab/config_quadratic.py 
--experiments_name EXPERIMENT_NAME --num_nodes_list 1000 
--theretical_step_size --step_size_range -8 10 --number_of_iterations 10000 --cpus_per_task 1 
--noise_lambdas 0.0 0.1 0.5 1.0 10.0 --dim 10 --samplings 'original_page' 'uniform_with_replacement' 'importance' 
--strongly_convex_constant 0.001 --generate_type worst_case --batch_size 1 10 25 50 100 500 1000 
--dumps_path SOME_PATH --dataset_path PATH_TO_FOLDER_WITH_DATASET
````

### 3. Execute scripts
````
sh SOME_PATH/EXPERIMENT_NAME/singularity_*.sh
````
### 4. Plot results
````
python3 code/distributed_optimization_library/experiments/plots/page_ab/quad_prog_plot.py 
--dumps_paths SOME_PATH/EXPERIMENT_NAME
--output_path SOME_OUTPUT_PATH --filter_sampling importance original_page --filter_noise_lambda 0.1 --batch_experiment
````

One can find all other scripts [here](code/distributed_optimization_library/experiments/plots/page_ab/scripts.txt) that generate experiments from the paper.
