import numpy as np
import torch

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.signature import Signature


NUMBER_OF_BITES_IN_BYTE = 8
NUMBER_OF_BITES_IN_FLOAT32 = 32


def _generate_seed(generator):
    return generator.integers(10e9)


class FactoryCompressor(Factory):
    pass

class BaseCompressor(object):
    BIASED = None
    INDEPENDENT = None
    RANDOM = None
    
    def compress(self, vector):
        raise NotImplementedError()

    def num_nonzero_components(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
    
    @classmethod
    def biased(cls):
        assert cls.BIASED is not None
        return cls.BIASED
    
    @classmethod
    def independent(cls):
        assert cls.INDEPENDENT is not None
        return cls.INDEPENDENT
    
    @classmethod
    def random(cls):
        assert cls.RANDOM is not None
        return cls.RANDOM
    
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        assert cls.independent()
        generator = np.random.default_rng(seed)
        signatures = []
        for _ in range(total_number_of_nodes):
            if cls.random():
                unique_seed = _generate_seed(generator)
                signatures.append(Signature(cls, seed=unique_seed, **params))
            else:
                signatures.append(Signature(cls, **params))
        return signatures


class SameSeedCompressor(BaseCompressor):
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        assert not cls.independent()
        assert cls.random()
        
        generator = np.random.default_rng(seed)
        seed = _generate_seed(generator)
        assert params['total_number_of_nodes'] == total_number_of_nodes
    
        signatures = []
        for node_index in range(total_number_of_nodes):
            signatures.append(Signature(cls, node_index, seed=seed, **params))
        return signatures


class UnbiasedBaseCompressor(BaseCompressor):
    BIASED = False
    def omega(self):
        raise NotImplementedError()


class BiasedBaseCompressor(BaseCompressor):
    BIASED = True
    def alpha(self):
        raise NotImplementedError()


class CompressedVector(object):
    def __init__(self, indices, values, dim):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        assert len(indices) == len(values)
        self._indices = indices
        self._values = values
        self._dim = dim
        
    def decompress(self):
        decompressed_array = np.zeros((self._dim,), dtype=self._values.dtype)
        decompressed_array[self._indices] = self._values
        return decompressed_array
        
    def size_in_memory(self):
        #  Omitted self._indices
        return len(self._values) * self._values.itemsize * NUMBER_OF_BITES_IN_BYTE


class CompressedTorchVector(object):
    def __init__(self, indices, values, dim):
        assert torch.is_tensor(indices)
        assert torch.is_tensor(values) and values.ndim == 1 and \
            values.dtype == torch.float32
        
        assert len(indices) == len(values)
        self._indices = indices
        self._values = values
        self._dim = dim
        
    def decompress(self):
        decompressed_array = torch.zeros((self._dim,), dtype=self._values.dtype,
                                         device=self._values.device)
        decompressed_array[self._indices] = self._values
        return decompressed_array
        
    def size_in_memory(self):
        #  Omitted self._indices
        return len(self._values) * NUMBER_OF_BITES_IN_FLOAT32


class BasePermutationCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        assert node_number < total_number_of_nodes
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
        self._dim = dim
        self._generator = np.random.default_rng(seed=seed)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        nodes_random_permutation = self._generator.permutation(
            self._total_number_of_nodes)
        block_number = nodes_random_permutation[self._node_number]
        start_dim, end_dim = self._end_start_dim(self._dim, block_number)
        random_permutation = self._get_coordiante_permutation(dim)
        correction_bias = self._total_number_of_nodes
        indices = random_permutation[start_dim:end_dim]
        values = vector[indices] * correction_bias
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def _get_coordiante_permutation(self, dim):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        assert self._dim is not None
        return self._dim / float(self._total_number_of_nodes)
    
    def omega(self):
        assert self._dim is not None
        return self._total_number_of_nodes - 1
    
    def _end_start_dim(self, dim, block_number):
        assert dim >= self._total_number_of_nodes
        block_size = dim // self._total_number_of_nodes
        residual = dim - block_size * self._total_number_of_nodes
        if block_number < residual:
            start_dim = block_number * (block_size + 1)
            end_dim = min(start_dim + block_size + 1, dim)
        else:
            start_dim = residual * (block_size + 1) + (block_number - residual) * block_size
            end_dim = min(start_dim + block_size, dim)
        return start_dim, end_dim


@FactoryCompressor.register("permutation")
class PermutationCompressor(BasePermutationCompressor):
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        super(PermutationCompressor, self).__init__(
            node_number, total_number_of_nodes, seed, dim)
    
    def _get_coordiante_permutation(self, dim):
        return self._generator.permutation(dim)


@FactoryCompressor.register("permutation_fixed_blocks")
class PermutationFixedBlocksCompressor(BasePermutationCompressor):
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        super(PermutationFixedBlocksCompressor, self).__init__(
            node_number, total_number_of_nodes, seed, dim)
    
    def _get_coordiante_permutation(self, dim):
        return np.arange(dim)


@FactoryCompressor.register("group_permutation")
class GroupPermutationCompressor(PermutationCompressor):
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        nodes_indices_splits = params['nodes_indices_splits']
        assert nodes_indices_splits[0] == 0
        assert nodes_indices_splits[-1] == total_number_of_nodes
        generator = np.random.default_rng(seed=seed)
        signatures = []
        for start, end in zip(nodes_indices_splits[:-1], nodes_indices_splits[1:]):
            total_number_of_nodes_group = end - start
            seed = _generate_seed(generator)
            for node_index in range(total_number_of_nodes_group):
                signatures.append(Signature(cls, node_index, 
                                            total_number_of_nodes=total_number_of_nodes_group, 
                                            seed=seed,
                                            dim=params.get('dim', None)))
        return signatures


@FactoryCompressor.register("nodes_permutation")
class NodesPermutationCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
        self._dim = dim
        self._generator = np.random.default_rng(seed=seed)
        self._indices = None

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._dim <= self._total_number_of_nodes and \
            self._total_number_of_nodes % self._dim == 0
        num_coordinates_per_node = self._total_number_of_nodes / self._dim
        if self._indices is None:
            self._indices = np.arange(self._dim)
            self._indices = np.repeat(self._indices, num_coordinates_per_node)
        random_permutation = self._generator.permutation(self._indices)
        indices = random_permutation[self._node_number:self._node_number + 1]
        values = vector[indices] * self._dim
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector

    def num_nonzero_components(self):
        return 1
    
    def omega(self):
        return self._dim - 1


@FactoryCompressor.register("identity_unbiased")
class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, dim=None):
        self._dim = dim
    
    def compress(self, vector):
        dim = vector.shape[0]
        compressed_vector = CompressedVector(np.arange(dim), np.copy(vector), dim)
        return compressed_vector
    
    def omega(self):
        return 0
    
    def num_nonzero_components(self):
        return self._dim


@FactoryCompressor.register("identity_biased")
class IdentityBiasedCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def compress(self, vector):
        dim = vector.shape[0]
        compressed_vector = CompressedVector(np.arange(dim), np.copy(vector), dim)
        return compressed_vector


@FactoryCompressor.register("coordinate_sampling")
class CoordinateSamplingCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        self._generator = np.random.default_rng(seed=seed)
        self._dim = dim
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
    
    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        nodes_assignment = self._generator.integers(
            low=0, high=self._total_number_of_nodes, size=(self._dim,))
        mask = nodes_assignment == self._node_number
        sequence = np.arange(self._dim)
        indices = sequence[mask]
        values = vector[mask] * self._total_number_of_nodes
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector

    def num_nonzero_components(self):
        return float(self._dim) / self._total_number_of_nodes

    def omega(self):
        return self._total_number_of_nodes - 1


def _torch_generator(seed, is_cuda):
    device = 'cpu' if not is_cuda else 'cuda'
    generator_numpy = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(
        int(_generate_seed(generator_numpy)))
    return generator


class BaseRandKCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    def __init__(self, number_of_coordinates, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim

    def compress(self, vector):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        return self._number_of_coordinates

    def omega(self):
        assert self._dim is not None
        return float(self._dim) / self._number_of_coordinates - 1


@FactoryCompressor.register("rand_k")
class RandKCompressor(BaseRandKCompressor):
    def __init__(self, number_of_coordinates, seed, dim=None):
        super(RandKCompressor, self).__init__(number_of_coordinates, dim)
        self._generator = np.random.default_rng(seed)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        indices = self._generator.choice(dim, self._number_of_coordinates, replace = False)
        values = vector[indices] * float(dim / self._number_of_coordinates)
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def copy(self):
        seed = np.random.default_rng()
        seed.bit_generator.state = self._generator.bit_generator.state
        return RandKCompressor(
            number_of_coordinates=self._number_of_coordinates, 
            seed=seed,
            dim=self._dim)


@FactoryCompressor.register("rand_k_torch")
class RandKTorchCompressor(BaseRandKCompressor):
    def __init__(self, number_of_coordinates, seed, dim=None, is_cuda=False):
        super(RandKTorchCompressor, self).__init__(number_of_coordinates, dim)
        self._generator = _torch_generator(seed, is_cuda)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        indices = torch.randperm(
            dim, generator=self._generator, device=vector.device)[:self._number_of_coordinates]
        values = vector[indices] * float(dim / self._number_of_coordinates)
        compressed_vector = CompressedTorchVector(indices, values, dim)
        return compressed_vector


@FactoryCompressor.register("unbiased_top_k")
class UnbiasedTopKCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    def __init__(self, seed, dim=None):
        super(UnbiasedTopKCompressor, self).__init__()
        self._generator = np.random.default_rng(seed)
        self._dim = dim

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        vector_abs = np.abs(vector)
        l1_norm = np.sum(vector_abs)
        probs = vector_abs / l1_norm
        indices = np.where(self._generator.random(dim) <= probs)[0]
        values = l1_norm * (np.sign(vector[indices]))
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def num_nonzero_components(self):
        return 1

    def omega(self):
        return float(self._dim) - 1


@FactoryCompressor.register("top_k")
class TopKCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, number_of_coordinates, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates <= self._dim
        abs_vector = np.abs(vector)
        indices = abs_vector.argsort()[dim - self._number_of_coordinates:]
        values = vector[indices]
        return CompressedVector(indices, values, dim)

    def num_nonzero_components(self):
        return self._number_of_coordinates
    
    def alpha(self):
        return float(self._number_of_coordinates) / self._dim


def get_compressor_signatures(compressor_name, params, total_number_of_nodes, seed):
    return FactoryCompressor.get(compressor_name).get_compressor_signatures(
        params, total_number_of_nodes, seed)


def get_compressors(*args, **kwargs):
    signatures = get_compressor_signatures(*args, **kwargs)
    return [signature.create_instance() for signature in signatures]
