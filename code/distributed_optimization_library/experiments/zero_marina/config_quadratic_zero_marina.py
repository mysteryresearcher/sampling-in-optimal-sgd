import itertools
import tempfile
import math
import os
import sys
import yaml
import copy
import numpy as np
import argparse
import json

from distributed_optimization_library.function import QuadraticFunction, TridiagonalQuadraticFunction, generate_random_vector

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob{experiments_name}
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time={time}:30:00
#SBATCH --mem={gb_per_task}G
#SBATCH --constraint=intel
#SBATCH --cpus-per-task={cpus_per_task}

cd {folder_with_project}
module load singularity
singularity exec ~/rsync_watch/python_ml_sha256.efcd1fc038228cb7eb0f6f1942dfbaa439cd95d6463015b83ceb2dbaad9e1e98.sif sh {singularity_script}
'''

singularity_template = 'MKL_NUM_THREADS=1 PYTHONPATH=./code python3 code/distributed_optimization_library/experiments/experiments.py --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "quadratic",
  "algorithm_master_params": {"gamma": 1.0},
  "compressor_params": {},
  "calculate_norm_of_gradients": True,
  "calculate_gradient_estimator_error": True,
  "calculate_function": True,
}

def generate_task(num_nodes, dim, noise_lambda, seed, strongly_convex_constant, correct_b):
    print("#" * 100)
    print("Num nodes: {}, Noise lambda: {}".format(num_nodes, noise_lambda))
    functions = TridiagonalQuadraticFunction.create_worst_case_functions(
        num_nodes, dim, seed=seed, noise_lambda=noise_lambda, strongly_convex_constant=strongly_convex_constant)
    if correct_b is not None:
        for function in functions:
            function._b += correct_b
    min_eigenvalue = TridiagonalQuadraticFunction.min_eigenvalue_functions(functions)
    assert min_eigenvalue > -1e-4, min_eigenvalue
    print("Min eigenvalue: {}".format(min_eigenvalue))
    function_stats = {}
    liptschitz_gradient_constant = TridiagonalQuadraticFunction.liptschitz_gradient_constant_functions(functions)
    print("Liptschitz gradient constant: {}".format(liptschitz_gradient_constant))
    analytical_solution = TridiagonalQuadraticFunction.analytical_solution_functions(functions)
    print(analytical_solution)
    print("Norm solution: {}".format(np.linalg.norm(analytical_solution)))
    min_value = TridiagonalQuadraticFunction.value_functions(functions, analytical_solution)
    print("Min value: {}".format(min_value))
    function_stats['liptschitz_gradient_constant'] = float(liptschitz_gradient_constant)
    function_stats['min_eigenvalue'] = float(min_eigenvalue)
    function_stats['noise_lambda'] = float(noise_lambda)
    function_stats['dim'] = int(dim)
    function_stats['num_nodes'] = int(num_nodes)
    function_stats['min_value'] = float(min_value)
    return functions, function_stats


def dump_task(folder, functions, function_stats, dim, generator, type_stochastic):
    os.mkdir(folder)
    TridiagonalQuadraticFunction.dump_functions(functions, os.path.join(folder, "functions"))
    with open(os.path.join(folder, 'function_stats.json'), 'w') as fd:
        json.dump(function_stats, fd)
    if type_stochastic == 'stochastic_matrix':
        point = np.ones((function_stats['dim'],), dtype=np.float32)
    else:
        point = np.zeros((function_stats['dim'],), dtype=np.float32)
        point[0] += np.sqrt(function_stats['dim'])
    np.save(os.path.join(folder, 'point'), point)


def generate_yaml(dumps_path, experiments_name, num_nodes_list, compressor_to_step_size_range, number_of_seeds,
                  number_of_iterations, algorithm_names, func, parallel, cpus_per_task,
                  time, number_of_processes, compressors,
                  calculate_smoothness_variance,
                  scale_initial_point, reg_paramterer, ef21_init_with_gradients,
                  point_initializer, batch_size, homogeneous, number_of_coordinates,
                  quality_check_rate, logistic_regression_nonconvex,
                  gb_per_task, oracle, mega_batch_size, no_initial_mega_batch_size,
                  no_initial_mega_batch_size_zero, dim, noise, type_stochastic,
                  correct_b, strongly_convex_constant, save_rate):
    folder_with_project = os.getcwd()
    assert len(num_nodes_list) == 1
    omega = dim / number_of_coordinates - 1
    print(omega, mega_batch_size, omega * math.sqrt(mega_batch_size), omega ** 2)
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    script_to_execute = ''
    generator = np.random.default_rng(seed=42)
    folder = os.path.join(tmp_dir, "problem_0")
    functions, function_stats = generate_task(num_nodes=num_nodes_list[0], dim=dim, noise_lambda=0.5, seed=generator,
                                              strongly_convex_constant=strongly_convex_constant, correct_b=correct_b)
    dump_task(folder, functions, function_stats, dim, generator, type_stochastic)
    print(compressor_to_step_size_range)
    index = 0
    for _, (compressor_name, algorithm_name, num_nodes, seed_number) in enumerate(
        itertools.product(compressors, algorithm_names, num_nodes_list, range(number_of_seeds))):
        step_size_range = compressor_to_step_size_range[compressor_name]
        for gamma_multiply in [2**i for i in range(step_size_range[0], step_size_range[1])]:
            yaml_prepared = copy.deepcopy(yaml_template)
            yaml_prepared['dim'] = dim
            yaml_prepared['noise'] = noise
            yaml_prepared['type'] = type_stochastic
            yaml_prepared['dump_path'] = folder
            yaml_prepared['num_nodes'] = num_nodes
            yaml_prepared['algorithm_name'] = algorithm_name
            yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
            if oracle == 'minibatch':
                yaml_prepared['algorithm_master_params']['batch_size'] = batch_size
            elif oracle == 'stochastic':
                if algorithm_name == 'marina_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = mega_batch_size
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                elif algorithm_name == 'zero_marina_stochastic':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)))
                elif algorithm_name == 'zero_marina_stochastic_overestimated':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega ** 2))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_name'] = 'zero_marina_stochastic'
                    yaml_prepared['version'] = 'overestimated'
                elif algorithm_name == 'zero_marina_stochastic_underestimated_batch':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                    yaml_prepared['algorithm_name'] = 'zero_marina_stochastic'
                    yaml_prepared['version'] = 'underestimated_batch'
                elif algorithm_name == 'zero_marina_stochastic_underestimated_momentum':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega + 1))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_name'] = 'zero_marina_stochastic'
                    yaml_prepared['version'] = 'underestimated_momentum'
                elif algorithm_name == 'zero_marina_stochastic_underestimated_all':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega + 1))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                    yaml_prepared['algorithm_name'] = 'zero_marina_stochastic'
                    yaml_prepared['version'] = 'underestimated_all'
                elif algorithm_name == 'zero_marina_sync_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = mega_batch_size
                    yaml_prepared['algorithm_master_params']['prob_sync'] = min(1 / float(mega_batch_size), 1 / (omega + 1))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                else:
                    assert False
                if no_initial_mega_batch_size:
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = 1
                if no_initial_mega_batch_size_zero:
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = 0
                yaml_prepared['batch_size'] = batch_size
            yaml_prepared['save_rate'] = save_rate
            yaml_prepared['number_of_iterations'] = number_of_iterations
            yaml_prepared['shuffle'] = True
            yaml_prepared['ignore_remainder'] = True
            yaml_prepared['homogeneous'] = homogeneous
            yaml_prepared['shared_memory_size'] = 100000
            yaml_prepared['parallel'] = parallel
            yaml_prepared['number_of_processes'] = number_of_processes
            yaml_prepared['compressor_name'] = compressor_name
            yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
            yaml_prepared['function'] = func
            if func in ['logistic_regression', 'stochastic_logistic_regression'] and logistic_regression_nonconvex > 0.0:
                yaml_prepared['function_parameters'] = {'nonconvex_regularizer': True,
                                                        'reg_paramterer': logistic_regression_nonconvex}
            yaml_prepared['seed'] = seed_number
            yaml_prepared['quality_check_rate'] = quality_check_rate
            if scale_initial_point is not None:
                assert scale_initial_point > 0.0
                yaml_prepared['scale_initial_point'] = scale_initial_point
            if point_initializer is not None:
                yaml_prepared['point_initializer'] = point_initializer
            if reg_paramterer is not None and reg_paramterer > 0.0:
                yaml_prepared['reg_paramterer'] = reg_paramterer
            
            config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
            with open(config_name, 'w') as fd:
                yaml.dump([yaml_prepared], fd)
            singularity_script = singularity_template.format(experiments_name=experiments_name,
                                                            config_name=config_name,
                                                            dumps_path=dumps_path)
            singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
            open(singularity_script_name, 'w').write(singularity_script)
            ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                            singularity_script=singularity_script_name,
                                            experiments_name=experiments_name,
                                            cpus_per_task=cpus_per_task,
                                            time=time,
                                            gb_per_task=gb_per_task)
            script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
            open(script_name, 'w').write(ibex_str)
            script_to_execute += 'sbatch {}\n'.format(script_name)
            index += 1
    final_script_name = os.path.join(tmp_dir, 'execute.sh')
    open(final_script_name, 'w').write(script_to_execute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--number_of_seeds', type=int, default=1)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--algorithm_names', required=True, nargs='+')
    parser.add_argument('--function', default="logistic_regression")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--gb_per_task', default=30, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--compressors', nargs='+', default=['rand_k'])
    parser.add_argument('--calculate_smoothness_variance', action='store_true')
    parser.add_argument('--point_initializer')
    parser.add_argument('--scale_initial_point', type=float)
    parser.add_argument('--reg_paramterer', type=float)
    parser.add_argument('--ef21_init_with_gradients', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--mega_batch_size', default=128, type=int)
    parser.add_argument('--number_of_coordinates', default=10, type=int)
    parser.add_argument('--homogeneous', action='store_true')
    parser.add_argument('--logistic_regression_nonconvex', default=0.0, type=float)
    parser.add_argument('--quality_check_rate', default=1, type=int)
    parser.add_argument('--oracle', default='gradient')
    parser.add_argument('--no_initial_mega_batch_size', action='store_true')
    parser.add_argument('--no_initial_mega_batch_size_zero', action='store_true')
    parser.add_argument('--dim', required=True, type=int)
    parser.add_argument('--noise', required=True, type=float)
    parser.add_argument('--type_stochastic', default='stochastic')
    parser.add_argument('--strongly_convex_constant', default=0.00001, type=float)
    parser.add_argument('--correct_b', default=None, type=float)
    parser.add_argument('--save_rate', default=1000, type=int)
    args = parser.parse_args()
    step_size_ranges = []
    for index in range(len(args.step_size_range) // 2):
        step_size_ranges.append([args.step_size_range[2 * index],
                                 args.step_size_range[2 * index + 1]])
    compressor_to_step_size_range = {}
    if len(step_size_ranges) == 1:
        for compressor in args.compressors:
            compressor_to_step_size_range[compressor] = step_size_ranges[0]
    else:
        assert len(step_size_ranges) == len(args.compressors)
        for compressor, step_size_range in zip(args.compressors, step_size_ranges):
            compressor_to_step_size_range[compressor] = step_size_range
    generate_yaml(args.dumps_path, args.experiments_name, args.num_nodes_list, compressor_to_step_size_range,
                  args.number_of_seeds, args.number_of_iterations, args.algorithm_names,
                  args.function, args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes, args.compressors,
                  args.calculate_smoothness_variance,
                  args.scale_initial_point, args.reg_paramterer,
                  args.ef21_init_with_gradients, args.point_initializer,
                  args.batch_size, args.homogeneous, args.number_of_coordinates,
                  args.quality_check_rate, args.logistic_regression_nonconvex,
                  args.gb_per_task, args.oracle, args.mega_batch_size,
                  args.no_initial_mega_batch_size, args.no_initial_mega_batch_size_zero,
                  args.dim, args.noise, args.type_stochastic, args.correct_b,
                  args.strongly_convex_constant, args.save_rate)
