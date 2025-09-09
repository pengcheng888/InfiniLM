from .base import (
    BaseModel,
    DataType,
    DeviceType,
    register_model,
    ModelWeightsCStruct,
    KVCacheCStruct,
    MambaCacheCStruct,
)
from ctypes import (
    c_size_t,
    c_uint,
    c_int,
    c_float,
    c_void_p,
    POINTER,
    Structure,
    c_char,
    c_char_p,
    c_bool
)


class Qwen2MetaCStruct(Structure):
    _fields_ = [
        # common
        ("dtype", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("epsilon", c_float),
        ("end_token", c_uint),
        # mha
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("theta", c_float),
    ]


class Qwen2ModelCStruct(Structure):
    pass


@register_model
class Qwen2Model(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register Qwen2 model functions with the library"""
        lib.createQwen2Weights.restype = POINTER(ModelWeightsCStruct)
        lib.createQwen2Weights.argtypes = [
            POINTER(Qwen2MetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createQwen2Model.restype = POINTER(Qwen2ModelCStruct)
        lib.createQwen2Model.argtypes = [
            POINTER(Qwen2MetaCStruct),
            POINTER(ModelWeightsCStruct),
        ]

        lib.destroyQwen2Model.argtypes = [POINTER(Qwen2ModelCStruct)]

        lib.createKVCache.argtypes = [
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
        ]
        lib.createKVCache.restype = POINTER(KVCacheCStruct)

        lib.dropKVCache.argtypes = [POINTER(KVCacheCStruct)]

        lib.createMambaCache.argtypes = [
            c_size_t,
            DataType,
            DeviceType,
            POINTER(c_int),
            c_size_t,
        ]
        lib.createMambaCache.restype = POINTER(MambaCacheCStruct)

        lib.dropMambaCache.argtypes = [POINTER(MambaCacheCStruct)]

        lib.inferBatchQwen2.argtypes = [
            POINTER(Qwen2ModelCStruct),
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            c_uint,
            POINTER(c_uint),
            POINTER(POINTER(KVCacheCStruct)),
            POINTER(POINTER(MambaCacheCStruct)),
            POINTER(c_float),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_uint),
        ]

        lib.loadModelWeight.argtypes = [
            POINTER(ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]

    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createQwen2Weights(meta, device_type, ndev, dev_ids)

    def create_model(self, meta, weights):
        return self.lib.createQwen2Model(meta, weights)

    def destroy_model(self, model):
        self.lib.destroyQwen2Model(model)

    def create_kv_cache(
        self, nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
    ):
        return self.lib.createKVCache(
            nlayer, max_len, nkvh, dk, dv, dtype, device, dev_ids, ndev
        )

    def drop_kv_cache(self, kv_cache):
        self.lib.dropKVCache(kv_cache)

    def create_mamba_cache(self, nlayer, dtype, device, dev_ids, ndev):
        return self.lib.createMambaCache(nlayer, dtype, device, dev_ids, ndev)

    def drop_mamba_cache(self, mamba_cache):
        self.lib.dropMambaCache(mamba_cache)

    def load_weight(self, weights, name, data):
        self.lib.loadModelWeight(weights, name.encode("utf-8"), data)

    def infer_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        mamba_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchQwen2(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            mamba_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        self.lib.forwardBatchQwen2(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )
