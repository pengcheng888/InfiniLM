#include "inference_context.hpp"
#include "../tensor.hpp"
#include "../utils.hpp"

InferenceContext::InferenceContext(DeviceResource *rsrc, CacheManager *cache_manager, infinirtStream_t stream)
    : rsrc(rsrc), cache_manager(cache_manager), stream(stream) {}

void InferenceContext::ensure_workspace(size_t required_size) {
    if (required_size > current_workspace_size || !workspace_storage) {
        workspace_storage = Storage::createFromPool(required_size, rsrc->memory_pool);
        current_workspace_size = required_size;
    }
}

void InferenceContext::add(std::shared_ptr<Tensor> c,
                           std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopAddDescriptor_t desc;
    if (!cache_manager->getAddDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateAddDescriptor(rsrc->handle, &desc, c->desc(), a->desc(), b->desc()));
        cache_manager->putAddDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetAddWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopAdd(
        desc, workspace, workspace_size,
        c->data(), a->data(), b->data(), stream));
}

void InferenceContext::mul(std::shared_ptr<Tensor> c,
                           std::shared_ptr<Tensor> a,
                           std::shared_ptr<Tensor> b) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopMulDescriptor_t desc;
    if (!cache_manager->getMulDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateMulDescriptor(rsrc->handle, &desc, c->desc(), a->desc(), b->desc()));
        cache_manager->putMulDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetMulWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopMul(
        desc, workspace, workspace_size,
        c->data(), a->data(), b->data(), stream));
}

void InferenceContext::sigmoid(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x) {
    size_t key = CacheManager::createDescriptorKey(y, x);

    infiniopSigmoidDescriptor_t desc;
    if (!cache_manager->getSigmoidDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateSigmoidDescriptor(rsrc->handle, &desc, y->desc(), x->desc()));
        cache_manager->putSigmoidDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSigmoidWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopSigmoid(
        desc, workspace, workspace_size,
        y->data(), x->data(), stream));
}

void InferenceContext::rmsnorm(std::shared_ptr<Tensor> y,
                               std::shared_ptr<Tensor> x,
                               std::shared_ptr<Tensor> w,
                               float epsilon) {
    size_t key = CacheManager::createDescriptorKey(y, x, w);

    infiniopRMSNormDescriptor_t desc;
    if (!cache_manager->getRMSNormDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRMSNormDescriptor(
            rsrc->handle, &desc, y->desc(), x->desc(), w->desc(), epsilon));
        cache_manager->putRMSNormDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRMSNorm(
        desc, workspace, workspace_size,
        y->data(), x->data(), w->data(), stream));
}

void InferenceContext::gemm(std::shared_ptr<Tensor> c,
                            std::shared_ptr<Tensor> a,
                            std::shared_ptr<Tensor> b,
                            float alpha, float beta) {
    size_t key = CacheManager::createDescriptorKey(c, a, b);

    infiniopGemmDescriptor_t desc;
    if (!cache_manager->getGemmDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateGemmDescriptor(rsrc->handle, &desc, c->desc(), a->desc(), b->desc()));
        cache_manager->putGemmDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopGemm(
        desc, workspace, workspace_size,
        c->data(), a->data(), b->data(), alpha, beta, stream));
}

void InferenceContext::rearrange(std::shared_ptr<Tensor> dst,
                                 std::shared_ptr<Tensor> src) {
    size_t key = CacheManager::createDescriptorKey(dst, src);

    infiniopRearrangeDescriptor_t desc;
    if (!cache_manager->getRearrangeDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRearrangeDescriptor(rsrc->handle, &desc, dst->desc(), src->desc()));
        cache_manager->putRearrangeDescriptor(key, desc);
    }

    RUN_INFINI(infiniopRearrange(
        desc,
        dst->data(),
        src->data(),
        stream));
}

void InferenceContext::rope(std::shared_ptr<Tensor> q,
                            std::shared_ptr<Tensor> k,
                            std::shared_ptr<Tensor> pos,
                            std::shared_ptr<Tensor> sin,
                            std::shared_ptr<Tensor> cos) {
    size_t key = CacheManager::createDescriptorKey(q, k, pos, sin, cos);

    infiniopRoPEDescriptor_t desc;
    if (!cache_manager->getRoPEDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRoPEDescriptor(
            rsrc->handle, &desc, q->desc(), k->desc(),
            pos->desc(), sin->desc(), cos->desc()));
        cache_manager->putRoPEDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRoPE(
        desc, workspace, workspace_size,
        q->data(), k->data(), pos->data(),
        sin->data(), cos->data(), stream));
}

void InferenceContext::causalSoftmax(std::shared_ptr<Tensor> y,
                                     std::shared_ptr<Tensor> x) {
    size_t key = CacheManager::createDescriptorKey(y, x);

    infiniopCausalSoftmaxDescriptor_t desc;
    if (!cache_manager->getCausalSoftmaxDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateCausalSoftmaxDescriptor(
            rsrc->handle, &desc, y->desc(), x->desc()));
        cache_manager->putCausalSoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetCausalSoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopCausalSoftmax(desc, workspace, workspace_size,
                                     y->data(), x->data(), stream));
}

void InferenceContext::swiglu(std::shared_ptr<Tensor> out,
                              std::shared_ptr<Tensor> up,
                              std::shared_ptr<Tensor> gate) {
    size_t key = CacheManager::createDescriptorKey(out, up, gate);

    infiniopSwiGLUDescriptor_t desc;
    if (!cache_manager->getSwiGLUDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateSwiGLUDescriptor(
            rsrc->handle, &desc, out->desc(), up->desc(), gate->desc()));
        cache_manager->putSwiGLUDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopSwiGLU(desc, workspace, workspace_size,
                              out->data(), up->data(), gate->data(), stream));
}

void InferenceContext::randomSample(std::shared_ptr<Tensor> out,
                                    std::shared_ptr<Tensor> prob,
                                    float random_val, float top_p, uint32_t top_k, float temperature) {
    size_t key = CacheManager::createDescriptorKey(out, prob);

    infiniopRandomSampleDescriptor_t desc;
    if (!cache_manager->getRandomSampleDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateRandomSampleDescriptor(
            rsrc->handle, &desc, out->desc(), prob->desc()));
        cache_manager->putRandomSampleDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopRandomSample(
        desc, workspace, workspace_size,
        out->data(), prob->data(),
        random_val, top_p, top_k, temperature,
        stream));
}

void InferenceContext::linear(std::shared_ptr<Tensor> c,
                              std::shared_ptr<Tensor> a,
                              std::shared_ptr<Tensor> b,
                              float alpha, float beta,
                              std::shared_ptr<Tensor> residual,
                              std::shared_ptr<Tensor> bias) {
    bool residual_flag = residual != nullptr;

    if (bias && !residual) {
        int ndim_diff = c->ndim() - 1;
        ASSERT_EQ(bias->ndim(), 1);
        ASSERT_EQ(bias->shape()[0], c->shape()[ndim_diff]);
        std::vector<ptrdiff_t> strides(ndim_diff, 0);
        strides.push_back(bias->strides()[0]);
        rearrange(c, bias->view_as(c->shape(), strides));
        residual = c;
    }

    if (residual) {
        if (residual->data() == c->data()) {
            if (beta == 0.0) {
                gemm(c, a, b, alpha, 1.0);
            } else {
                auto c_copy = Tensor::buffer(c->dtype(), c->shape(), rsrc->memory_pool);
                c_copy->copyFrom(c, rsrc->handle, stream);
                gemm(c, a, b, alpha, beta);
                add(c, c, c_copy);
            }
        } else {
            gemm(c, a, b, alpha, beta);
            add(c, c, residual);
        }
    } else {
        gemm(c, a, b, alpha, beta);
    }

    if (bias && residual_flag) {
        int ndim_diff = c->ndim() - 1;
        ASSERT_EQ(bias->ndim(), 1);
        ASSERT_EQ(bias->shape()[0], c->shape()[ndim_diff]);
        std::vector<ptrdiff_t> strides(ndim_diff, 0);
        strides.push_back(bias->strides()[0]);
        add(c, c, bias->view_as(c->shape(), strides));
    }
}

void InferenceContext::topksoftmax(std::shared_ptr<Tensor> values,  // F32
                                   std::shared_ptr<Tensor> indices, // I32
                                   std::shared_ptr<Tensor> x,
                                   size_t topk,
                                   bool norm_topk_prob) {

    size_t key = CacheManager::createDescriptorKey(values, indices, x);

    infiniopTopksoftmaxDescriptor_t desc;
    if (!cache_manager->getTopksoftmaxDescriptor(key, desc)) {
        RUN_INFINI(infiniopCreateTopksoftmaxDescriptor(rsrc->handle, &desc, x->desc()));
        cache_manager->putTopksoftmaxDescriptor(key, desc);
    }

    size_t workspace_size = 0;
    RUN_INFINI(infiniopGetTopksoftmaxWorkspaceSize(desc, &workspace_size));
    ensure_workspace(workspace_size);
    void *workspace = workspace_storage->memory();

    RUN_INFINI(infiniopTopksoftmax(
        desc, workspace, workspace_size,
        values->data(), indices->data(), x->data(), topk, norm_topk_prob, stream));
}