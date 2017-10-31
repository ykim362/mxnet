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
* \file mkldnn_rnn-inl.h
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/
#pragma once

#include <string>
#include <vector>
#include "mkl_util-inl.h"
#include "mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template <typename xpu, typename DType>
class MKLDNNRnnOp : public Operator, public MKLDNNLayer<DType> {
 public:
  explicit MKLDNNRnnOp(RNNParam p) : init_mkldnn_(false) {
    param_ = p;
    param_.lstm_q_ = (param_.mode == rnn_enum::kLstm);
  }

  ~MKLDNNRnnOp() {}
  std::string getName() { return "MKLDNNRnnOp"; }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using mshadow::Stream;
    using mshadow::Tensor;
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs) out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    // get input + output tensors
    Tensor<xpu, 3, DType> x =
        mkl_experimental_direct_get<xpu, 3, DType>(in_data[rnn_enum::kData], s);
    Tensor<xpu, 1, DType> w = mkl_experimental_direct_get<xpu, 1, DType>(
        in_data[rnn_enum::kParams], s);
    Tensor<xpu, 3, DType> hx = mkl_experimental_direct_get<xpu, 3, DType>(
        in_data[rnn_enum::kState], s);
    Tensor<xpu, 3, DType> y =
        mkl_experimental_direct_get<xpu, 3, DType>(out_data[rnn_enum::kOut], s);

    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);

    if (!init_mkldnn_) {
      LayerSetup(ctx, in_data, out_data);
    }
    // standard inputs/outputs
    x_p_f = x_f->get_converted_prv(x.dptr_, false, in_data[rnn_enum::kData]);
    w_p_f =
        w_f->get_converted_prv(w.dptr_, false, in_data[rnn_enum::kParams]);
    hx_p_f =
        hx_f->get_converted_prv(hx.dptr_, false, in_data[rnn_enum::kState]);
    y_m_f = y_f->create_output_memory(y.dptr_, out_data[rnn_enum::kOut], y_f);

    // extra inputs/outputs
    DType *hy_ptr = nullptr;
    if (param_.state_outputs) {
      hy_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                   out_data[rnn_enum::kStateOut], s)
                   .dptr_;
      assert(hy_ptr != nullptr);
      hy_m_f = hy_f->create_output_memory(hy_ptr,
                                          out_data[rnn_enum::kStateOut],
                                          hx_f);
    }

    DType *cx_ptr = nullptr;
    if (param_.lstm_q_) {
      cx_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                   in_data[rnn_enum::kStateCell], s)
                   .dptr_;
      assert(cx_ptr != nullptr);
      cx_p_f = hx_f->get_converted_prv(cx_ptr,
                                       false,
                                       in_data[rnn_enum::kStateCell]);
    }

    DType *cy_ptr = nullptr;
    if (param_.lstm_q_ && param_.state_outputs) {
      cy_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                   out_data[rnn_enum::kStateCellOut], s)
                   .dptr_;
      assert(cy_ptr != nullptr);
      cy_m_f = cy_f->create_output_memory(cy_ptr,
                                          out_data[rnn_enum::kStateCellOut],
                                          hx_f);
    }

    // lstm & state_outputs
    if (param_.lstm_q_ && param_.state_outputs) {
      rnnFwd.reset(new rnn_forward(*rnnFwd_pd,
                                   primitive::at(*x_p_f),
                                   primitive::at(*hx_p_f),
                                   primitive::at(*cx_p_f),
                                   primitive::at(*w_p_f),
                                   *y_m_f,
                                   *hy_m_f,
                                   *cy_m_f,
                                   *workspace_m_f));
    }
    // lstm
    else if (param_.lstm_q_) {
      rnnFwd.reset(new rnn_forward(*rnnFwd_pd,
                                   primitive::at(*x_p_f),
                                   primitive::at(*hx_p_f),
                                   primitive::at(*cx_p_f),
                                   primitive::at(*w_p_f),
                                   *y_m_f,
                                   *workspace_m_f));
    }
    // state_outputs
    else if (param_.state_outputs) {
      rnnFwd.reset(new rnn_forward(*rnnFwd_pd,
                                   primitive::at(*x_p_f),
                                   primitive::at(*hx_p_f),
                                   primitive::at(*w_p_f),
                                   *y_m_f,
                                   *hy_m_f,
                                   *workspace_m_f));
    }
    // standard
    else {
      rnnFwd.reset(new rnn_forward(*rnnFwd_pd,
                                   primitive::at(*x_p_f),
                                   primitive::at(*hx_p_f),
                                   primitive::at(*w_p_f),
                                   *y_m_f,
                                   *workspace_m_f));
    }

    rnnFwd.submit();
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs) out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);
    CHECK_EQ(in_grad.size(), in_expected);
    CHECK_EQ(out_grad.size(), out_expected);
    CHECK_EQ(req.size(), in_expected);
    CHECK_NE(req[rnn_enum::kData], kAddTo) << "AddTo is not supported for data";
    CHECK_NE(req[rnn_enum::kState], kAddTo)
        << "AddTo is not supported for state";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // get input + output tensors
    Tensor<xpu, 3, DType> x =
        mkl_experimental_direct_get<xpu, 3, DType>(in_data[rnn_enum::kData], s);
    Tensor<xpu, 3, DType> dx =
        mkl_experimental_direct_get<xpu, 3, DType>(in_grad[rnn_enum::kData], s);
    Tensor<xpu, 1, DType> w = mkl_experimental_direct_get<xpu, 1, DType>(
        in_data[rnn_enum::kParams], s);
    Tensor<xpu, 1, DType> dw = mkl_experimental_direct_get<xpu, 1, DType>(
        in_grad[rnn_enum::kParams], s);
    Tensor<xpu, 3, DType> hx = mkl_experimental_direct_get<xpu, 3, DType>(
        in_data[rnn_enum::kState], s);
    Tensor<xpu, 3, DType> dhx = mkl_experimental_direct_get<xpu, 3, DType>(
        in_grad[rnn_enum::kState], s);
    Tensor<xpu, 3, DType> y =
        mkl_experimental_direct_get<xpu, 3, DType>(out_data[rnn_enum::kOut], s);
    Tensor<xpu, 3, DType> dy =
        mkl_experimental_direct_get<xpu, 3, DType>(out_grad[rnn_enum::kOut], s);
    if (req[rnn_enum::kParams] != kAddTo) {
      dw = mshadow::expr::ScalarExp<DType>(0.0f);
    }
    CHECK_EQ(x.CheckContiguous(), true);
    CHECK_EQ(w.CheckContiguous(), true);
    CHECK_EQ(dw.CheckContiguous(), true);
    CHECK_EQ(hx.CheckContiguous(), true);
    CHECK_EQ(dhx.CheckContiguous(), true);
    CHECK_EQ(y.CheckContiguous(), true);
    CHECK_EQ(dy.CheckContiguous(), true);

    // only need kStateOut grad output_states is true
    DType *dhy_ptr = nullptr;
    if (param_.state_outputs)
      dhy_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                    out_grad[rnn_enum::kStateOut], s)
                    .dptr_;

    // Deal with lstm
    DType *dcx_ptr = nullptr;
    DType *dcy_ptr = nullptr;
    DType *cx_ptr = nullptr;

    if (param_.mode == rnn_enum::kLstm) {
      CHECK_NE(req[rnn_enum::kStateCell], kAddTo)
          << "AddTo is not supported for state cell";
      cx_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                   in_data[rnn_enum::kStateCell], s)
                   .dptr_;
      dcx_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                    in_grad[rnn_enum::kStateCell], s)
                    .dptr_;
    }
    if ((param_.mode == rnn_enum::kLstm) && param_.state_outputs)
      dcy_ptr = mkl_experimental_direct_get<xpu, 3, DType>(
                    out_grad[rnn_enum::kStateCellOut], s)
                    .dptr_;

    if (!init_mkldnn_) {
      LayerSetup(ctx, in_data, out_data);

      // standard inputs/outputs
      x_p_b = x_f->get_converted_prv(x.dptr_, false, in_data[rnn_enum::kData]);
      w_p_b =
          w_f->get_converted_prv(w.dptr_, false, in_data[rnn_enum::kParams]);
      hx_p_b =
          hx_f->get_converted_prv(hx.dptr_, false, in_data[rnn_enum::kState]);
      dy_p_b =
          y_f->get_converted_prv(dy.dptr_, false, out_grad[rnn_enum::kOut]);
      dhy_p_b = hx_f->get_converted_prv(dhy_ptr, false,
                                        out_grad[rnn_enum::kStateOut]);
      cx_p_b =
          hx_f->get_converted_prv(cx_ptr, false, in_data[rnn_enum::kStateCell]);
      dcy_p_b = hx_f->get_converted_prv(dcy_ptr, false,
                                        out_grad[rnn_enum::kStateCellOut]);
      dx_m_b =
          x_f->create_output_memory(dx.dptr_, in_grad[rnn_enum::kData], x_f);
      dw_m_b =
          w_f->create_output_memory(dw.dptr_, in_grad[rnn_enum::kParams], w_f);
      dhx_m_b = hx_f->create_output_memory(dhx.dptr_, in_grad[rnn_enum::kState],
                                           hx_f);
      dcx_m_b = hx_f->create_output_memory(dcx_ptr,
                                           in_grad[rnn_enum::kStateCell], hx_f);
      std::shared_ptr<memory> workspace;
      auto workspace_primitive_desc = rnnBwd_pd->workspace_primitive_desc();
      workspace.reset(new memory(workspace_primitive_desc));
      // lstm & state_outputs
      if (param_.lstm_q_ && param_.state_outputs) {
        rnnBwd.reset(
            new rnn_backward(*rnnBwd_pd, primitive::at(*x_p_b),
                             primitive::at(*hx_p_b), primitive::at(*cx_p_b),
                             primitive::at(*dy_p_b), primitive::at(*dhy_p_b),
                             primitive::at(*dcy_p_b), primitive::at(*w_p_b),
                             *workspace, *dx_m_b, *dhx_m_b, *dcx_m_b, *dw_m_b));
      }
      // lstm
      else if (param_.lstm_q_) {
        rnnBwd.reset(
            new rnn_backward(*rnnBwd_pd, primitive::at(*x_p_b),
                             primitive::at(*hx_p_b), primitive::at(*cx_p_b),
                             primitive::at(*dy_p_b), primitive::at(*w_p_b),
                             *workspace, *dx_m_b, *dhx_m_b, *dcx_m_b, *dw_m_b));
      }
      // state_outputs
      else if (param_.state_outputs) {
        rnnBwd.reset(new rnn_backward(
            *rnnBwd_pd, primitive::at(*x_p_b), primitive::at(*hx_p_b),
            primitive::at(*dy_p_b), primitive::at(*dhy_p_b),
            primitive::at(*w_p_b), *workspace, *dx_m_b, *dhx_m_b, *dw_m_b));
      }
      // standard
      else {
        rnnBwd.reset(new rnn_backward(
            *rnnBwd_pd, primitive::at(*x_p_b), primitive::at(*hx_p_b),
            primitive::at(*dy_p_b), primitive::at(*w_p_b), *workspace, *dx_m_b,
            *dhx_m_b, *dw_m_b));
      }

    } else {
      // sync inputs/ouputs
    }
    rnnBwd.submit();
  }

 private:
  inline void LayerSetup(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<TBlob> &out_data) {
    if (init_mkldnn_) return;
    init_mkldnn_ = true;
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    size_t in_expected = param_.lstm_q_ ? 4 : 3;
    size_t out_expected = param_.lstm_q_ ? 3 : 2;
    if (!param_.state_outputs) out_expected = 1;

    CHECK_EQ(in_data.size(), in_expected);
    CHECK_EQ(out_data.size(), out_expected);

    Tensor<xpu, 3, DType> x =
        mkl_experimental_direct_get<xpu, 3, DType>(in_data[rnn_enum::kData], s);
    param_.seq_length_ = x.shape_[0];
    param_.batch_size_ = x.shape_[1];
    param_.input_size_ = x.shape_[2];

    algorithm algo;
    switch (param_.mode) {
      case rnn_enum::kRnnRelu:
        algo = algorithm::rnn_relu;
        break;
      case rnn_enum::kRnnTanh:
        algo = algorithm::rnn_tanh;
        break;
      case rnn_enum::kLstm:
        algo = algorithm::rnn_lstm;
        break;
      default:
        LOG(FATAL) << "Not implmented";
    }
    auto dir = param_.bidirectional ? direction::rnn_bidirectional
                                    : direction::rnn_unidirectional;
    auto dirn = param_.bidirectional ? 2 : 1;
    auto fkind =
        ctx.is_train ? prop_kind::forward_training : prop_kind::forward_scoring;
    auto state_outputs = param_.state_outputs ? 1 : 0;

    memory::data_type mpcsn = memory::data_type::f32;
    memory::format mfmt = memory::format::rnx;
    mkldnn::engine cpu_engine = mxnet::CpuEngine::Instance().get_engine();

    // memory descriptors for x,hx,y,weights
    memory::desc x_desc(
        {param_.seq_length_, param_.batch_size_, param_.input_size_}, mpcsn,
        mfmt);
    memory::desc hx_desc(
        {param_.num_layers, param_.batch_size_, param_.state_size}, mpcsn,
        mfmt);
    memory::desc y_desc(
        {param_.seq_length_, param_.batch_size_, dirn * param_.state_size},
        mpcsn, mfmt);
    // calculate weights size
    auto w_m = param_.lstm_q_ ? 4 : 1;
    auto w1_size =
        param_.state_size * (param_.state_size + param_.input_size_ + 2) * w_m;
    auto wx_size =
        param_.state_size * (param_.state_size + param_.state_size + 2) * w_m;
    auto total_w = param_.num_layers == 1
                       ? dirn * w1_size
                       : dirn * (w1_size + (param_.num_layers - 1) * wx_size);
    memory::desc w_desc({total_w}, mpcsn, memory::format::x);

    // create RNN primitive descriptors for fwd and bwd
    auto rnn_fwd_desc =
        rnn_forward::desc(fkind, algo, dir, input_mode::rnn_linear_input,
                          (size_t)param_.state_size, (size_t)param_.num_layers,
                          (int)param_.seq_length_, (int)state_outputs, x_desc,
                          hx_desc, y_desc, w_desc);
    rnnFwd_pd.reset(new rnn_forward::primitive_desc(rnn_fwd_desc, cpu_engine));

    auto rnn_bwd_desc = rnn_backward::desc(
        prop_kind::backward, algo, dir, input_mode::rnn_linear_input,
        param_.state_size, param_.num_layers, param_.seq_length_, state_outputs,
        x_desc, hx_desc, y_desc, w_desc);
    rnnBwd_pd.reset(
        new rnn_backward::primitive_desc(rnn_bwd_desc, cpu_engine, *rnnFwd_pd));

    // Create MKLDNNData for inputs and outputs
    std::shared_ptr<memory::primitive_desc> prv_mpd;
    std::shared_ptr<memory::primitive_desc> x_mpd(
        new memory::primitive_desc(x_desc, cpu_engine));
    x_f.reset(new MKLDNNData<DType>(x_mpd, prv_mpd));
    std::shared_ptr<memory::primitive_desc> hx_mpd(
        new memory::primitive_desc(hx_desc, cpu_engine));
    hx_f.reset(new MKLDNNData<DType>(hx_mpd, prv_mpd));
    hy_f.reset(new MKLDNNData<DType>(hx_mpd, prv_mpd));
    cy_f.reset(new MKLDNNData<DType>(hx_mpd, prv_mpd));
    std::shared_ptr<memory::primitive_desc> y_mpd(
        new memory::primitive_desc(y_desc, cpu_engine));
    y_f.reset(new MKLDNNData<DType>(y_mpd, prv_mpd));
    std::shared_ptr<memory::primitive_desc> w_mpd(
        new memory::primitive_desc(w_desc, cpu_engine));
    w_f.reset(new MKLDNNData<DType>(w_mpd, prv_mpd));

    if (ctx.is_train) {
      auto workspace_primitive_desc = rnnFwd_pd->workspace_primitive_desc();
      workspace_m_f.reset(new memory(workspace_primitive_desc));
    }
  }

  bool init_mkldnn_;
  RNNParam param_;

  using MKLDNNDataPtr = std::shared_ptr<MKLDNNData<DType>>;
  using PrimitivePtr = std::shared_ptr<primitive>;
  using MemoryPtr = std::shared_ptr<memory>;

  // Forward vars
  MKLDNNPrimitive<DType> rnnFwd;
  std::shared_ptr<rnn_forward::primitive_desc> rnnFwd_pd;

  // inputs
  MKLDNNDataPtr x_f;
  PrimitivePtr x_p_f;
  MKLDNNDataPtr hx_f;
  PrimitivePtr hx_p_f;
  MKLDNNDataPtr w_f;
  PrimitivePtr w_p_f;
  /// lstm
  MKLDNNDataPtr cx_f;
  /// lstm
  PrimitivePtr cx_p_f;

  // outputs
  MKLDNNDataPtr y_f;
  MemoryPtr y_m_f;
  /// state_outputs
  MKLDNNDataPtr hy_f;
  /// state_outputs
  MemoryPtr hy_m_f;
  /// lstm & state_outputs
  MKLDNNDataPtr cy_f;
  /// lstm & state_outputs
  MemoryPtr cy_m_f;

  /// used for training only
  MemoryPtr workspace_m_f;

  // Backward vars
  MKLDNNPrimitive<DType> rnnBwd;
  std::shared_ptr<rnn_backward::primitive_desc> rnnBwd_pd;
  MKLDNNDataPtr x_b;
  MKLDNNDataPtr hx_b;
  MKLDNNDataPtr cx_b;
  MKLDNNDataPtr dy_b;
  MKLDNNDataPtr dhy_b;
  MKLDNNDataPtr dcy_b;
  MKLDNNDataPtr w_b;
  MKLDNNDataPtr workspace_b;
  MKLDNNDataPtr dx_b;
  MKLDNNDataPtr dhx_b;
  MKLDNNDataPtr dcx_b;
  MKLDNNDataPtr dw_b;
  PrimitivePtr x_p_b;
  PrimitivePtr hx_p_b;
  PrimitivePtr cx_p_b;
  PrimitivePtr dy_p_b;
  PrimitivePtr dhy_p_b;
  PrimitivePtr dcy_p_b;
  PrimitivePtr w_p_b;
  PrimitivePtr workspace_p_b;
  MemoryPtr dx_m_b;
  MemoryPtr dhx_m_b;
  MemoryPtr dcx_m_b;
  MemoryPtr dw_m_b;

};  // class MKLDNNRnnOp
}  // namespace op
}  // namespace mxnet
