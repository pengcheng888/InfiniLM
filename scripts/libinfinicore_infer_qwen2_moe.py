import ctypes
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p,c_bool, POINTER
import os
import torch
from libinfinicore_infer import DataType, DeviceType



from libinfinicore_infer_qwen3_moe import MoEMetaCStruct

from libinfinicore_infer_qwen3_moe import AttentionCStruct, MLPCStruct,SparseMLPCStruct,DecoderLayerCStruct, WeightsCStruct




class ModelCSruct(ctypes.Structure):
    pass


class KVCacheCStruct(ctypes.Structure):
    pass





def __open_library__():
    lib_path = os.path.join(
        os.environ.get("INFINI_ROOT"), "lib", "libinfinicore_infer.so"
    )
    
    lib = ctypes.CDLL(lib_path)
    lib.Qwen2MoEcreateModel.restype = POINTER(ModelCSruct)
    lib.Qwen2MoEcreateModel.argtypes = [
        POINTER(MoEMetaCStruct),  # JiugeMeta const *
        POINTER(WeightsCStruct),  # JiugeWeights const *
        DeviceType,  # DeviceType
        c_int,  # int ndev
        POINTER(c_int),  # int const *dev_ids
    ]
    lib.Qwen2MoEdestroyModel.argtypes = [POINTER(ModelCSruct)]
    lib.Qwen2MoEcreateKVCache.argtypes = [POINTER(ModelCSruct)]
    lib.Qwen2MoEcreateKVCache.restype = POINTER(KVCacheCStruct)
    lib.Qwen2MoEdropKVCache.argtypes = [POINTER(ModelCSruct), POINTER(KVCacheCStruct)]
    lib.Qwen2MoEinferBatch.restype = None
    lib.Qwen2MoEinferBatch.argtypes = [
        POINTER(ModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        POINTER(c_float),  # float temperature
        POINTER(c_uint),  # unsigned int topk
        POINTER(c_float),  # float topp
        POINTER(c_uint),  # unsigned int *output
    ]
    lib.Qwen2MoEforwardBatch.restype = None
    lib.Qwen2MoEforwardBatch.argtypes = [
        POINTER(ModelCSruct),  # struct JiugeModel const *
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *req_lens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCacheCStruct)),  # struct KVCache **kv_caches
        c_void_p,  # void *logits
    ]

    return lib


LIB = __open_library__()

create_model = LIB.Qwen2MoEcreateModel
destroy_model = LIB.Qwen2MoEdestroyModel
create_kv_cache = LIB.Qwen2MoEcreateKVCache
drop_kv_cache = LIB.Qwen2MoEdropKVCache
infer_batch = LIB.Qwen2MoEinferBatch
forward_batch = LIB.Qwen2MoEforwardBatch
