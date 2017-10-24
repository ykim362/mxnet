/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* \file mkldnn_batch_norm-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKLDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_MKL_MKLDNN_BATCH_NORM_INL_H_

#include <mkldnn_types.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "mkldnn_base-inl.h"
#include "mkl_util-inl.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../batch_norm-inl.h"

namespace mxnet {
namespace op {


template<typename xpu, typename DType>
class MKLDNNBatchNormOp : public Operator, public MKLDNNLayer<DType> {
 public:
  static int s_id_gen;
  int m_id;
  explicit MKLDNNBatchNormOp(BatchNormParam param) : MKLDNNLayer<DType>()
    , fwd_top_data(NULL), fwd_bottom_data(NULL)
    , fwd_inference_pd(NULL), fwd_training_pd(NULL)
    , bwd_top_diff(NULL), bwd_bottom_diff(NULL), bwd_scaleshift_pd(NULL) {
    this->param_ = param;
    m_id = s_id_gen++;
  }
  virtual ~MKLDNNBatchNormOp() {
  }
  static std::string getName() {
    std::string name = "MKLDNNBatchNormOp_";
    // name = name + std::to_string(m_id);
    return name;
  }

 private:
  /**
   * This method returns a data layout format by query what a MKLDNN convolution
   * operator would use for a specific platform. For example, on BDW, we expect
   * this to be a 8 width layout, while for SKL or KNL, it should be 16 wide.
   * @param ctx
   * @param default_data_type
   * @return
   */
  memory::format GetPlatformLayoutFormat(const OpContext &ctx,
                                   const memory::data_type default_data_type) {
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    auto propagation =
        (ctx.is_train) ? prop_kind::forward_training : prop_kind::forward_scoring;
    memory::dims convolutionStrides{1, 1};
    memory::dims padding{1, 1};
    memory::format memory_format_any = memory::format::any;
    memory::desc init_bottom_md({128, 3, 28, 28}, default_data_type, memory_format_any);
    memory::desc init_top_md({128, 16, 28, 28}, default_data_type, memory_format_any);
    memory::desc init_weights_md({16, 3, 3, 3}, default_data_type, memory_format_any);
    std::shared_ptr<convolution_forward::desc> convFwd_desc;
    convFwd_desc.reset(
        new convolution_forward::desc(propagation, algorithm::convolution_direct
            , init_bottom_md, init_weights_md, init_top_md
            , convolutionStrides, padding, padding, padding_kind::zero));
    std::shared_ptr<convolution_forward::primitive_desc> convFwd_pd;
    convFwd_pd.reset(new convolution_forward::primitive_desc(*convFwd_desc, cpu_engine));

    return static_cast<memory::format>(convFwd_pd->dst_primitive_desc().desc().data.format);
  }

  void LayerSetUp(const OpContext &ctx,
                  const mshadow::Tensor<xpu, 4, DType> &data,
                  const mshadow::Tensor<xpu, 4, DType> &out) {
    eps_ = param_.eps;
    num_ = data.shape_[0];
    channels_ = data.shape_[1];
    height_ = data.shape_[2];
    width_ = data.shape_[3];
    int32_t n = this->num_;
    int32_t ic = this->channels_;
    int32_t ih = this->height_;
    int32_t iw = this->width_;
    memory::data_type default_data_type = memory::data_type::f32;
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    // Set up usr memory descriptor
    // by default we expect usr input format is nchw
    fwd_usr_input_md.reset(new memory::desc({{n, ic, ih, iw}},
                                            default_data_type,
                                            memory::format::nchw));
    fwd_usr_mpd.reset(new memory::primitive_desc(*fwd_usr_input_md,
                                                 cpu_engine));


    // on BDW or SKL, where we expect AVX2 (nChw8c) and AVX512 (nChw16c) optimized layouts, they
    // require the channel to be multiples of 8.
    if (ic % 8 == 0) {
      // TODO lfeng: this is a workaround to query architecture specific layout
      // for best performance
      memory::format platform_format = GetPlatformLayoutFormat(ctx,
                                                               default_data_type);
      fwd_prv_input_md.reset(new memory::desc({{n, ic, ih, iw}},
                                              default_data_type,
                                              platform_format));
      fwd_prv_mpd.reset(new memory::primitive_desc(*fwd_prv_input_md,
                                                   cpu_engine));
    } else {
      fwd_prv_input_md = nullptr;
      fwd_prv_mpd = nullptr;
    }
  }
  void initFwd(const std::vector<TBlob> &in_data) {
    void *bottom_data =
        const_cast<DType *>(mkl_prv_data<DType>(in_data[batchnorm::kData]));
    // ---- Initialize memory descriptors -------------
    std::shared_ptr<memory::primitive_desc> usr_mpd(NULL);
    if (bottom_data != NULL) {
      std::shared_ptr<MKLDNNData<DType> > mem_descr =
          get_mkldnn_prv_descriptor<DType>(in_data[batchnorm::kData]);
      CHECK(mem_descr != NULL);
      fwd_bottom_data = mem_descr;
      input_md.reset(new memory::desc(mem_descr->prv_memory_pd()->desc()));
      usr_mpd = mem_descr->usr_memory_pd();
      prv_mpd = mem_descr->prv_memory_pd();
    } else {
      if (fwd_prv_input_md != nullptr)
        input_md = fwd_prv_input_md;
      else
        input_md = fwd_usr_input_md;
      usr_mpd = fwd_usr_mpd;
      prv_mpd = fwd_prv_mpd;
      fwd_bottom_data.reset(new MKLDNNData<DType>(usr_mpd, prv_mpd));
      fwd_bottom_data->name = "fwd_bottom_data   @ " + this->getName();
    }
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    // ---- Initialize BatchNorm primitive descriptor -------------
    batch_normalization_forward::desc BatchNormFwdInference_desc(prop_kind::forward_scoring,
      *input_md, eps_, mkldnn_use_global_stats | mkldnn_use_scaleshift);
    batch_normalization_forward::desc BatchNormFwdTraining_desc(prop_kind::forward_training,
      *input_md, eps_, mkldnn_use_scaleshift);

    fwd_inference_pd.reset(
      new batch_normalization_forward::primitive_desc(BatchNormFwdInference_desc, cpu_engine));
    fwd_training_pd.reset(
      new batch_normalization_forward::primitive_desc(BatchNormFwdTraining_desc, cpu_engine));

    fwd_top_data.reset(new MKLDNNData<DType>(usr_mpd, prv_mpd));
    fwd_top_data->name = "fwd_top_data   @ " + this->getName();

    weight_memory.reset(new memory(fwd_inference_pd->weights_primitive_desc()));
  }

 public:
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(aux_states.size(), 2);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3);
      CHECK_EQ(req.size(), 3);
    } else {
      CHECK_GE(out_data.size(), 1);
      CHECK_GE(req.size(), 1);
      CHECK_EQ(req[batchnorm::kOut], kWriteTo);
    }
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data;
    Tensor<xpu, 4, DType> out;
    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[batchnorm::kData].shape_[0],
                               in_data[batchnorm::kData].shape_[1],
                               1,
                               1);
      data = mkl_experimental_direct_get_with_shape<xpu,
                                                    4,
                                                    DType>(in_data[batchnorm::kData],
                                                           dshape,
                                                           s);
      out = mkl_experimental_direct_get_with_shape<xpu,
                                                   4,
                                                   DType>(out_data[batchnorm::kOut],
                                                          dshape,
                                                          s);
    } else {
      data =
          mkl_experimental_direct_get<xpu, 4, DType>(in_data[batchnorm::kData],
                                                     s);
      out =
          mkl_experimental_direct_get<xpu, 4, DType>(out_data[batchnorm::kOut],
                                                     s);
    }
    Tensor<xpu, 1, DType>
        slope = in_data[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType>
        bias = in_data[batchnorm::kBeta].get<xpu, 1, DType>(s);
    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    if (param_.fix_gamma)
      slope = 1.f;

    int32_t ic = this->channels_;
    if (fwd_inference_pd == NULL) {
      LayerSetUp(ctx, data, out);
      initFwd(in_data);
    }

    // Setup weight
    DType * scaleShift_buf =
        reinterpret_cast<DType *>(weight_memory->get_data_handle());
    // use_weight_bias_
    for (int i = 0; i < channels_; i++) {
      scaleShift_buf[i] = (slope.dptr_)[i];
    }
    for (int i = 0; i < channels_; i++) {
      scaleShift_buf[channels_ + i] = (bias.dptr_)[i];
    }

    if (!init_mkldnn_) {
      init_mkldnn_ = true;
      fwd_input_primitive = fwd_bottom_data->get_converted_prv(data.dptr_,
                                                               false,
                                                               in_data[batchnorm::kData]);
      fwd_output_memory = fwd_top_data->create_output_memory(out.dptr_,
                                                             out_data[batchnorm::kOut],
                                                             fwd_top_data);
      if (ctx.is_train && !param_.use_global_stats) {
        Tensor<xpu, 1, DType>
            mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
        Tensor<xpu, 1, DType>
            var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
        CHECK(req[batchnorm::kMean] == kNullOp
                  || req[batchnorm::kMean] == kWriteTo);
        CHECK(req[batchnorm::kVar] == kNullOp
                  || req[batchnorm::kVar] == kWriteTo);
        mean_memory.reset(new memory(fwd_training_pd->mean_primitive_desc(),
                                     mean.dptr_));
        var_memory.reset(new memory(fwd_training_pd->variance_primitive_desc(),
                                    var.dptr_));
      } else {
        Tensor<xpu, 1, DType> moving_mean =
            aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
        Tensor<xpu, 1, DType> moving_var =
            aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);
        mean_memory.reset(new memory(fwd_inference_pd->mean_primitive_desc(),
                                     moving_mean.dptr_));
        var_memory.reset(new memory(fwd_inference_pd->variance_primitive_desc(),
                                    moving_var.dptr_));
      }
      // ---- Create BatchNorm --------------------
      if (ctx.is_train) {
        BatchNormFwd.reset(new batch_normalization_forward(*fwd_training_pd,
                                                           *fwd_input_primitive,
                                                           *weight_memory,
                                                           *fwd_output_memory,
                                                           *mean_memory,
                                                           *var_memory));
      } else {
        // TODO lfeng: potential bug here, primitive::at() must be called explicitly
        // here for mean_memory and var_memory, otherwise we get a segfault for
        // inference.
        BatchNormFwd.reset(new batch_normalization_forward(*fwd_inference_pd,
                                                           *fwd_input_primitive,
                                                           primitive::at(*mean_memory),
                                                           primitive::at(*var_memory),
                                                           *weight_memory,
                                                           *fwd_output_memory));
      }
    } else {
      fwd_bottom_data->sync_converted_prv(data.dptr_,
                                          false,
                                          in_data[batchnorm::kData]);
      fwd_top_data->sync_output_memory(out_data[batchnorm::kOut], fwd_top_data);
    }
    BatchNormFwd.submit();
  }
  void InitBatchNormBwd(const std::vector<TBlob> &out_grad) {
    int32_t n = this->num_;
    int32_t w = this->width_;
    int32_t h = this->height_;
    int32_t c = this->channels_;

    unsigned flags = use_scale_shift;
    if (param_.use_global_stats)
      flags |= use_global_stats;
    void * top_diff_data =
      const_cast<DType*>(mkl_prv_data<DType>(out_grad[batchnorm::kOut]));
    bool top_diff_is_prv = (top_diff_data != NULL);

    mkldnn::engine cpu_engine = CpuEngine::Instance().get_engine();
    memory::data_type mpcsn = memory::data_type::f32;
    // ---- Initialize memory descriptors -------------

    std::shared_ptr<memory::desc> top_diff_md, top_data_md;
    std::shared_ptr<memory::primitive_desc> usr_diff_mpd(NULL), prv_diff_mpd(NULL);
    std::shared_ptr<memory::desc> default_md;
    default_md.reset(new memory::desc({ { n, c, h, w } }, mpcsn, memory::format::nchw));
    if (top_diff_is_prv) {
      std::shared_ptr<MKLDNNMemoryDescriptor<DType> > mem_descr
        = get_mkldnn_prv_descriptor<DType>(out_grad[batchnorm::kOut]);
      usr_diff_mpd = mem_descr->usr_memory_pd();
      prv_diff_mpd = mem_descr->prv_memory_pd();
    } else {
      if (prv_mpd != NULL) prv_diff_mpd = prv_mpd;
      usr_diff_mpd.reset(new memory::primitive_desc(*default_md, cpu_engine));
    }

    if (prv_diff_mpd != NULL)
      top_diff_md.reset(new memory::desc(prv_diff_mpd->desc()));
    else 
      top_diff_md.reset(new memory::desc(*default_md));
    batch_normalization_backward::desc BatchNormBwd_desc(prop_kind::backward, *top_diff_md,
      fwd_output_memory->get_primitive_desc().desc(), eps_, mkldnn_use_scaleshift);
    bwd_scaleshift_pd.reset(
      new batch_normalization_backward::primitive_desc(BatchNormBwd_desc, cpu_engine,
        *fwd_training_pd));

    diff_weight_memory.reset(
      new memory(bwd_scaleshift_pd->diff_weights_primitive_desc()));

    bwd_bottom_diff.reset(new MKLDNNData<DType>(usr_diff_mpd, prv_diff_mpd));
    bwd_bottom_diff->name = "bwd_bottom_diff   @ " + this->getName();
    bwd_top_diff.reset(new MKLDNNData<DType>(usr_diff_mpd, prv_diff_mpd));
    bwd_top_diff->name = "bwd_top_diff   @ " + this->getName();
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, DType> data, grad, grad_in;

    if (in_data[batchnorm::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[batchnorm::kOut].shape_[0],
        out_grad[batchnorm::kOut].shape_[1], 1, 1);
      data = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_data[batchnorm::kData], dshape, s);
      grad = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        out_grad[batchnorm::kOut], dshape, s);
      grad_in = mkl_experimental_direct_get_with_shape<xpu, 4, DType>(
        in_grad[batchnorm::kData], dshape, s);
    } else {
      data = mkl_experimental_direct_get<xpu, 4, DType>(in_data[batchnorm::kData], s);
      grad = mkl_experimental_direct_get<xpu, 4, DType>(out_grad[batchnorm::kOut], s);
      grad_in = mkl_experimental_direct_get<xpu, 4, DType>(in_grad[batchnorm::kData], s);
    }

    Tensor<xpu, 1, DType> gslope = in_grad[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> gbias = in_grad[batchnorm::kBeta].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> slope = in_data[batchnorm::kGamma].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> mean = out_data[batchnorm::kMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> var = out_data[batchnorm::kVar].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_mean = aux_states[batchnorm::kMovingMean].get<xpu, 1, DType>(s);
    Tensor<xpu, 1, DType> moving_var = aux_states[batchnorm::kMovingVar].get<xpu, 1, DType>(s);

    if (param_.fix_gamma)
      slope = 1.f;


    DType * mean_dptr = NULL;
    DType * var_dptr = NULL;
    if (ctx.is_train && !param_.use_global_stats) {
      int size = mean.size(0);  // Tensor<xpu, 1, DType>
      float * moving_mean_ptr = reinterpret_cast<float*>(moving_mean.dptr_);
      float * mean_ptr = reinterpret_cast<float*>(mean.dptr_);
      float * moving_var_ptr = reinterpret_cast<float*>(moving_var.dptr_);
      float * var_ptr = reinterpret_cast<float*>(var.dptr_);
      float minus_mom = (1 - param_.momentum);
      for (int i = 0; i < size; i++) {
        moving_mean_ptr[i] = moving_mean_ptr[i] * param_.momentum
          + mean_ptr[i] * minus_mom;
      }
      for (int i = 0; i < size; i++) {
        moving_var_ptr[i] = moving_var_ptr[i] * param_.momentum
          + var_ptr[i] * minus_mom;
      }
      mean_dptr = mean.dptr_;
      var_dptr = var.dptr_;
    } else {
      mean_dptr = moving_mean.dptr_;
      var_dptr = moving_var.dptr_;
    }

    if (bwd_scaleshift_pd == NULL) {
      InitBatchNormBwd(out_grad);
      bmean_memory.reset(new memory(bwd_scaleshift_pd->mean_primitive_desc(), mean_dptr));
      bvar_memory.reset(new memory(bwd_scaleshift_pd->variance_primitive_desc(), var_dptr));
      bwd_diff_src_memory = bwd_bottom_diff->create_output_memory(grad_in.dptr_,
        in_grad[batchnorm::kData], bwd_bottom_diff);

      bwd_diff_dst_memory = bwd_top_diff->get_converted_prv(grad.dptr_,
        false, out_grad[batchnorm::kOut]);

      BatchNormBwd.reset(new batch_normalization_backward(*bwd_scaleshift_pd,
        *fwd_input_primitive, *bmean_memory, *bvar_memory,
        *bwd_diff_dst_memory,
        *weight_memory, *bwd_diff_src_memory, *diff_weight_memory));
    } else {
      bwd_top_diff->sync_converted_prv(grad.dptr_,
        false, out_grad[batchnorm::kOut]);
      bwd_bottom_diff->sync_output_memory(
        in_grad[batchnorm::kData], bwd_bottom_diff);
    }
    BatchNormBwd.submit();
    DType * scaleShiftDiff_buf = reinterpret_cast<DType*>(diff_weight_memory->get_data_handle());
    if (!param_.fix_gamma) {
      // Store ScaleShift blobs
      DType* diff_scale = gslope.dptr_;
      for (int i = 0; i < channels_; i++) {
        diff_scale[i] = scaleShiftDiff_buf[i];
      }
    } else {
      int gslope_size = gslope.size(0);
      float * gslope_ptr = reinterpret_cast<float*>(gslope.dptr_);
      for (int i = 0; i < gslope_size; i++) {
        *gslope_ptr++ = 0.0f;
      }
    }
    DType* diff_shift = gbias.dptr_;
    for (int i = 0; i < channels_; i++) {
      diff_shift[i] = scaleShiftDiff_buf[channels_ + i];
    }
  }

 private:
  BatchNormParam param_;
  bool init_mkldnn_ = false;
  std::shared_ptr<memory::desc> fwd_usr_input_md;
  std::shared_ptr<memory::primitive_desc> fwd_usr_mpd;
  std::shared_ptr<memory::desc> fwd_prv_input_md;
  std::shared_ptr<memory::primitive_desc> fwd_prv_mpd, prv_mpd;

  // Forward
  std::shared_ptr<MKLDNNData<DType> > fwd_top_data, fwd_bottom_data;
  std::shared_ptr<batch_normalization_forward::primitive_desc> fwd_inference_pd;
  std::shared_ptr<batch_normalization_forward::primitive_desc> fwd_training_pd;
  MKLDNNPrimitive<DType> BatchNormFwd;

  // Backward
  MKLDNNPrimitive<DType> BatchNormBwd;
  std::shared_ptr<MKLDNNData<DType> > bwd_top_diff;
  std::shared_ptr<MKLDNNData<DType> > bwd_bottom_diff;
  std::shared_ptr<memory::desc> input_md, scaleshift_md;
  std::shared_ptr<batch_normalization_backward::primitive_desc> bwd_scaleshift_pd;
  std::shared_ptr<memory> weight_memory;
  std::shared_ptr<memory> diff_weight_memory;
  std::shared_ptr<memory> fwd_output_memory;
  // common
  int32_t num_, width_, height_, channels_;
  DType eps_;
  std::shared_ptr<memory> mean_memory, var_memory;
  std::shared_ptr<memory> fwd_input_primitive;
    std::shared_ptr<memory> bmean_memory, bvar_memory;
    std::shared_ptr<memory> bwd_input_primitive;
    std::shared_ptr<memory> bwd_diff_dst_memory;
    std::shared_ptr<memory> bwd_diff_src_memory;
};  // class MKLDNNBatchNormOp
  template<> int MKLDNNBatchNormOp<cpu, float>::s_id_gen = 1;
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKLDNN_BATCH_NORM_INL_H_
