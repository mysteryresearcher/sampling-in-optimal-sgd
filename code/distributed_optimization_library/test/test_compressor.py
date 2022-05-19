import itertools
import pytest
import torch

import numpy as np

from distributed_optimization_library.compressor import get_compressors, FactoryCompressor


@pytest.mark.parametrize("compressor_name", ['permutation', 'coordinate_sampling', 'permutation_fixed_blocks'])
def test_permutation_compressor(compressor_name):
    seed = np.random.default_rng(seed=42)
    check_dims = np.array(range(1, 100))
    check_total_number_of_nodes = np.array(range(1, 10))
    for dim, total_number_of_nodes in itertools.product(check_dims, check_total_number_of_nodes):
        if dim < total_number_of_nodes:
            continue
        compressors = get_compressors(compressor_name=compressor_name, 
                                      params={'total_number_of_nodes': total_number_of_nodes}, 
                                      total_number_of_nodes=total_number_of_nodes, 
                                      seed=seed)
        vector = np.random.randint(low=1, high=50, size=(dim,))
        compressed_vectors = [compressor.compress(vector).decompress() for compressor in compressors]
        
        total_non_zero = 0
        total_count_nonzero = 0
        for node_number, (compressed_vector, compressor) in enumerate(zip(compressed_vectors, compressors)):
            count_nonzero = np.count_nonzero(compressed_vector)
            total_count_nonzero += count_nonzero
            if 'permutation' in compressor_name:
                assert count_nonzero > 0
            if dim % total_number_of_nodes == 0:
                assert (compressor.omega() + 1) == total_number_of_nodes
            if 'permutation' in compressor_name:
                if node_number != total_number_of_nodes - 1:
                    total_non_zero += count_nonzero
                else:
                    assert count_nonzero == dim - total_non_zero
        assert total_count_nonzero == dim
        for compressor in compressors:
            np.testing.assert_almost_equal(compressor.num_nonzero_components(), 
                                           total_count_nonzero / total_number_of_nodes)
        
        for i in range(len(compressed_vectors) - 1):
            for j in range(i + 1, len(compressed_vectors)):
                assert np.dot(compressed_vectors[i], compressed_vectors[j]) == 0


@pytest.mark.parametrize("compressor_name", ['permutation', 'coordinate_sampling', 'permutation_fixed_blocks'])
def test_permutation_compressor_unbiasedness(compressor_name):
    seed = 42
    dim = 9
    total_number_of_nodes = 5
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'total_number_of_nodes': total_number_of_nodes}, 
                                  total_number_of_nodes=total_number_of_nodes, 
                                  seed=seed)
    vector = np.random.randn(dim).astype(np.float32)
    number_of_samples = 30000
    expected_vectors = [0] * total_number_of_nodes
    for _ in range(number_of_samples):
        for node_number, compressor in enumerate(compressors):
            compressed_vector = compressor.compress(vector).decompress()
            expected_vectors[node_number] += compressed_vector / number_of_samples
    for node_number in range(total_number_of_nodes):
        np.testing.assert_array_almost_equal(expected_vectors[node_number], vector, decimal=1)


@pytest.mark.parametrize("compressor_name", ['permutation', 'coordinate_sampling', 'permutation_fixed_blocks'])
def test_permutation_compressor_restores_vector(compressor_name):
    seed = 42
    dim = 113
    total_number_of_nodes = 24
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'total_number_of_nodes': total_number_of_nodes}, 
                                  total_number_of_nodes=total_number_of_nodes, 
                                  seed=seed)
    vector = np.random.randn(dim).astype(np.float32)
    compressed_vectors = [compressor.compress(vector).decompress() for compressor in compressors]
    np.testing.assert_array_almost_equal(sum(compressed_vectors) / total_number_of_nodes, vector)


@pytest.mark.parametrize("compressor_name", ['rand_k', 'rand_k_torch'])
def test_rand_k_compressor_unbiasedness(compressor_name):
    seed = 42
    number_of_coordinates = 5
    dim = 19
    compressor = FactoryCompressor.get(compressor_name)(number_of_coordinates, seed)
    vector = np.random.randn(dim).astype(np.float32)
    if compressor_name == 'rand_k_torch':
        vector = torch.tensor(vector)
    number_of_samples = 100000
    expected_vector = 0
    for _ in range(number_of_samples):
        compressed_vector = compressor.compress(vector)
        assert np.count_nonzero(compressed_vector.decompress()) == compressor.num_nonzero_components()
        expected_vector += compressed_vector.decompress() / number_of_samples
    np.testing.assert_array_almost_equal(expected_vector, vector, decimal=1)


def test_rand_k_compressors_independent():
    seed = 42
    number_of_coordinates = 5
    dim = 19
    compressor_name = 'rand_k'
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'number_of_coordinates': number_of_coordinates}, 
                                  total_number_of_nodes=2,
                                  seed=seed)
    vector = np.random.randn(dim).astype(np.float32)
    lhs = compressors[0].compress(vector).decompress()
    rhs = compressors[1].compress(vector).decompress()
    err = np.max(np.abs(lhs - rhs))
    assert err > 0.1


def test_rand_k_copy():
    seed = 42
    number_of_coordinates = 5
    dim = 19
    compressor_name = 'rand_k'
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'number_of_coordinates': number_of_coordinates}, 
                                  total_number_of_nodes=1,
                                  seed=seed)
    compressor = compressors[0]
    vector = np.random.randn(dim).astype(np.float32)
    copy_compressor = compressor.copy()
    copy_copy_compressor = copy_compressor.copy()
    compressed_vector = compressor.compress(vector).decompress()
    copy_compressed_vector = copy_compressor.compress(vector).decompress()
    copy_copy_compressed_vector = copy_copy_compressor.compress(vector).decompress()
    np.testing.assert_array_almost_equal(compressed_vector, copy_compressed_vector)
    np.testing.assert_array_almost_equal(compressed_vector, copy_copy_compressed_vector)
    
    compressed_vector = compressor.compress(vector).decompress()
    copy_compressed_vector = copy_compressor.compress(vector).decompress()
    np.testing.assert_array_almost_equal(compressed_vector, copy_compressed_vector)


@pytest.mark.parametrize("compressor_name", ['unbiased_top_k'])
def test_unbiased_top_k_compressor_unbiasedness(compressor_name):
    seed = 42
    dim = 19
    compressor = FactoryCompressor.get(compressor_name)(seed)
    vector = np.random.randn(dim).astype(np.float32)
    number_of_samples = 100000
    expected_vector = 0
    expected_norm = 0
    for _ in range(number_of_samples):
        compressed_vector = compressor.compress(vector)
        expected_vector += compressed_vector.decompress() / number_of_samples
        expected_norm += np.linalg.norm(compressed_vector.decompress()) ** 2 / number_of_samples
    np.testing.assert_array_almost_equal(expected_vector, vector, decimal=1)
    np.testing.assert_array_almost_equal(expected_norm / 100, np.linalg.norm(vector, ord=1) ** 2 / 100, decimal=1)


def test_top_k_compressor():
    number_of_coordinates = 3
    compressor = FactoryCompressor.get("top_k")(number_of_coordinates)
    vector = np.array([1, 3, 5, 12, 5, 3, -11, 9])
    compressed_vector = compressor.compress(vector).decompress()
    count_nonzero = np.count_nonzero(compressed_vector)
    assert count_nonzero == compressor.num_nonzero_components()
    assert count_nonzero == number_of_coordinates
    assert (compressed_vector == np.array([0, 0, 0, 12, 0, 0, -11, 9])).all()


def test_nodes_permutation():
    compressor_name = 'nodes_permutation'
    seed = 42
    dim = 7
    total_number_of_nodes = 28
    generator = np.random.default_rng(seed)
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'total_number_of_nodes': total_number_of_nodes}, 
                                  total_number_of_nodes=total_number_of_nodes, 
                                  seed=seed)
    vector = generator.normal(size=(dim,)).astype(np.float32)
    number_of_samples = 20000
    expected_vectors = [0] * total_number_of_nodes
    for _ in range(number_of_samples):
        estimated_vector = 0
        for node_number, compressor in enumerate(compressors):
            compressed_vector = compressor.compress(vector).decompress()
            expected_vectors[node_number] += compressed_vector / number_of_samples
            assert np.count_nonzero(compressed_vector) == 1
            estimated_vector += compressed_vector
        np.testing.assert_array_almost_equal(estimated_vector / total_number_of_nodes, vector)
    for node_number in range(total_number_of_nodes):
        np.testing.assert_array_almost_equal(expected_vectors[node_number], vector, decimal=1)


def test_group_permutation_compressor():
    seed = np.random.default_rng(seed=42)
    dim = 120
    compressor_name = 'group_permutation'
    total_number_of_nodes = 10
    nodes_indices_splits = [0, 3, 7, 10]
    compressors = get_compressors(compressor_name=compressor_name, 
                                  params={'nodes_indices_splits': nodes_indices_splits}, 
                                  total_number_of_nodes=total_number_of_nodes, 
                                  seed=seed)
    vector = np.random.randn(dim)
    compressed_vectors = [compressor.compress(vector).decompress() for compressor in compressors]
    np.testing.assert_array_almost_equal(sum(compressed_vectors[:3]) / 3, vector)
    np.testing.assert_array_almost_equal(sum(compressed_vectors[3:7]) / 4, vector)
    np.testing.assert_array_almost_equal(sum(compressed_vectors[7:]) / 3, vector)
    np.testing.assert_array_almost_equal(sum(compressed_vectors) / len(compressed_vectors), vector)
    
    groups = [range(0, 3), range(3, 7), range(7, 10)]
    for group in groups:
        for l, r in zip(group[:-1], group[1:]):
            np.dot(compressed_vectors[l], compressed_vectors[r]) == 0
    
    number_of_samples = 30000
    expected_vectors = [0] * total_number_of_nodes
    for _ in range(number_of_samples):
        for node_number, compressor in enumerate(compressors):
            compressed_vector = compressor.compress(vector).decompress()
            expected_vectors[node_number] += compressed_vector / number_of_samples
    for node_number in range(total_number_of_nodes):
        np.testing.assert_array_almost_equal(expected_vectors[node_number], vector, decimal=1)
