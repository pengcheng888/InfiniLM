#ifndef _QWEN2MOE_WEIGHT_HPP_
#define _QWEN2MOE_WEIGHT_HPP_

#include "../qwen_weight.hpp"
#include "infinicore_infer/models/qwen2moe.h"
#include <cmath>

//
// 存储 cpu 地址
//
namespace Qwen2MoE {

using AttentionCStruct = Qwen::AttentionCStruct;
using SparseMLPCStruct = Qwen::SparseMLPCStruct;

using DecoderLayerCStruct = Qwen::DecoderLayerCStruct<AttentionCStruct, SparseMLPCStruct>;

struct Weights {
    size_t _nlayer{0};                     //  ("_nlayer", c_size_t)
    infiniDtype_t _dt_norm;                //  ("_dt_norm", DataType)
    infiniDtype_t _dt_mat;                 //  ("_dt_mat", DataType),
    int _transpose_linear_weights{false};  //  ("_transpose_linear_weights", c_int),
    void *_embed_tokens_weight{nullptr};   //  ("_embed_tokens_weight", c_void_p),
    void *_norm_weight{nullptr};           //  ("_norm_weight", c_void_p),
    void *_lm_head_weight{nullptr};        //  ("_lm_head_weight", c_void_p)
    DecoderLayerCStruct *_layers{nullptr}; //  ("_layers", POINTER(DecoderLayerCStruct))

    void print_info() const {
        printf("\n DecoderLayerCStruct:\n");
        printf("\t nlayer : %ld\n", _nlayer);
        printf("\t transpose_linear_weights : %d\n", _transpose_linear_weights);
        printf("\t embed_tokens_weight : %p\n", _embed_tokens_weight);
        printf("\t norm_weight : %p\n", _norm_weight);
        printf("\t lm_head_weight : %p\n", _lm_head_weight);
        if (_layers) {
            _layers[0].print_info();
        }
    }
};
}; // namespace Qwen2MoE

//
// 存储 gpu 地址
//
namespace Qwen2MoE {

using SharedMLPTensor = Qwen::SharedMLPTensor<Meta, Weights>;

using RouterMLPTensor = Qwen::RouterMLPTensor<Meta, Weights>;

using SparseMLPTensor = Qwen::SparseMLPTensor<SharedMLPTensor, RouterMLPTensor, Meta, Weights>;

using AttentionTensor = Qwen::AttentionTensor<Meta, Weights>;

using DecoderLayerTensor = Qwen::DecoderLayerTensor<AttentionTensor, SparseMLPTensor, Meta, Weights>;

struct WeightsTensor {
    size_t nlayer{0};
    std::shared_ptr<Tensor> sin_table;
    std::shared_ptr<Tensor> cos_table;
    std::shared_ptr<Tensor> w_in_embd;
    std::shared_ptr<Tensor> w_out_norm;
    std::shared_ptr<Tensor> w_out_embd;
    std::vector<std::shared_ptr<DecoderLayerTensor>> layers;

public:
    WeightsTensor(Meta const *meta, Weights const *w, size_t idev, size_t ndev) {
        this->nlayer = meta->nlayer;

        size_t d = meta->d;
        size_t dh = meta->dh;
        float theta = meta->theta;
        size_t dvoc = meta->dvoc;
        size_t dctx = meta->dctx;

        infiniDtype_t dt_logits = meta->dt_logits;
        infiniDtype_t dt_norm = w->_dt_norm;

        int transpose_linear_weights = w->_transpose_linear_weights;

        void *embed_tokens_weight_ptr = w->_embed_tokens_weight;
        void *lm_head_weight_ptr = w->_lm_head_weight;
        void *norm_weight_ptr = w->_norm_weight;

        this->sin_table = Qwen::getSinTable(dh, theta, dctx, dt_logits);
        this->cos_table = Qwen::getCosTable(dh, theta, dctx, dt_logits);
        this->w_in_embd = Qwen::getInEmbd(d, dvoc, dt_logits, embed_tokens_weight_ptr);
        this->w_out_embd = Qwen::getOutEmbd(d, dvoc, dt_logits, transpose_linear_weights, lm_head_weight_ptr);
        this->w_out_norm = Qwen::getNorm(d, dt_norm, norm_weight_ptr);

        this->layers.reserve(this->nlayer);
        for (size_t ilayer = 0; ilayer < this->nlayer; ++ilayer) {

            this->layers.push_back(std::make_shared<DecoderLayerTensor>(meta, w, ilayer, idev, ndev));
        }
    }

    void print_info() const {
        printf(" \n ");
        printf("Qwen2MoE::WeightsTensor  nlayer: %ld \n ", nlayer);
        printf("\t\t sin_table :: %p\t%s \n", sin_table.get(), sin_table->info().c_str());
        printf("\t\t cos_table :: %p\t%s \n ", cos_table.get(), cos_table->info().c_str());
        printf("\t\t w_in_embd :: %p\t%s \n ", w_in_embd.get(), w_in_embd->info().c_str());
        printf("\t\t w_out_norm :: %p\t%s \n ", w_out_norm.get(), w_out_norm->info().c_str());
        printf("\t\t w_out_embd :: %p\t%s \n", w_out_embd.get(), w_out_embd->info().c_str());
        for (auto &layer : layers) {
            layer->print_info();
            break;
        }
    }
};

}; // namespace Qwen2MoE
#endif
