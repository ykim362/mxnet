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
* \file mkl_memory.cc
* \brief
* \author lingyan.guo@intel.com
*         zhenlin.luo@intel.com
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_MKL_MKL_MEMORY_H_
#define MXNET_OPERATOR_MKL_MKL_MEMORY_H_

#include <string>
#include <vector>
#include <memory>


namespace mxnet {
// Base class
struct PrvMemDescr {
  virtual void convert_from_prv(void* cpu_ptr) = 0;
  virtual void convert_to_prv(void* cpu_ptr) = 0;
  virtual void convert_from_other(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual void* prv_ptr(bool allocate_when_uninit = true) = 0;
  // returns true for matching layouts
  virtual bool layout_compare(std::shared_ptr<PrvMemDescr> other) = 0;
  virtual size_t prv_count() = 0;
  virtual size_t prv_size() = 0;
  // This might help using prv_ptr_ by different accelerators/engines
  enum PrvDescrType {
    PRV_DESCR_MKL2017,
    PRV_DESCR_MKLDNN
  };
  virtual PrvDescrType get_descr_type() = 0;
};

#if MKL_EXPERIMENTAL == 1 || MXNET_USE_MKLDNN == 1
// Currently HEAD_AT_PRV do not free CPU data
enum SyncedHead {
  ///  most recent data is in the cpu/usr buffer
  HEAD_AT_CPU,
  ///  most recent data is in the prv buffer
  HEAD_AT_PRV,
};
struct MKLMemHolder {

  /**
   * This flag indicates where the most recent (or valid) data is. We
   * essentially have two pointers, one inside TBlob: dptr_, one inside
   * MKLDNNMemoryDescriptorBase: _internal_ptr. Depending how we read/write to
   * these buffers, we need to update this flag to point to the valid buffer.
   */
  SyncedHead head_;
  /**
   * prv_descriptor_ provides access to the underlying prv data buffer and its
   * layout descriptors. We keep track of both usr/cpu and prv data descriptor.
   */
  std::shared_ptr<PrvMemDescr> prv_descriptor_;
  bool  b_disable_prv_2_cpu;
  bool  b_eager_mode;
  void disable_prv_2_cpu(bool flag) {
    b_disable_prv_2_cpu = flag;
  }
  void set_eager_mode(bool eager_mode) {
    b_eager_mode = eager_mode;
  }
  void set_prv_descriptor(std::shared_ptr<PrvMemDescr> descriptor, bool same_data = false) {
    if (descriptor != nullptr)
      head_ = HEAD_AT_PRV;
    prv_descriptor_ = descriptor;
  }
  std::shared_ptr<PrvMemDescr> get_prv_descriptor() {
    return  prv_descriptor_;
  }
  bool head_at_cpu() {
    return (head_ == HEAD_AT_CPU);
  }
  bool head_at_prv() {
    return (head_ == HEAD_AT_PRV);
  }
  void* prv_data(bool allocate_when_uninit = true) {
    if (head_ != HEAD_AT_PRV) {
      return NULL;
    }
    if (prv_descriptor_ == NULL) {
      LOG(FATAL) << " prv_descriptor_  is NULL";
    }
    CHECK(prv_descriptor_.get());
    return prv_descriptor_->prv_ptr(allocate_when_uninit);
  }

  int prv_count() {
    if (head_ != HEAD_AT_PRV) {
      return 0;
    }
    if (prv_descriptor_ == NULL) {
      LOG(FATAL) << " prv_descriptor_  is NULL";
    }
    CHECK(prv_descriptor_.get());
    return prv_descriptor_->prv_count();
  }
  static std::shared_ptr<MKLMemHolder> create() {
    return std::make_shared<MKLMemHolder>();
  }

  /**
   * Verify we have a valid prv descriptor and if prv has the latest data
   * convert to cpu and update head_ to point to HEAD_AT_CPU. This is
   * intended to convert data to CPU and use/modify the cpu dptr_ buffer.
   * After this operation, to access prv data, one needs to explicitly
   * update/convert from cpu to prv.
   * data.
   * @param dptr_ cpu pointer to data buffer where prv data will be converted
   * to.
   * @param convert 
   */
  void  check_and_prv_to_cpu(void *dptr_, bool convert = true) {
//    std::cout << __FUNCTION__ << " " << __LINE__ << std::endl;
    if (!b_disable_prv_2_cpu && head_ == HEAD_AT_PRV) {
      CHECK(prv_descriptor_ != nullptr);
      if (convert)
        prv_descriptor_->convert_from_prv(dptr_);
      // Because operator use CPU & maybe change it, change to CPU Flag
      head_ = HEAD_AT_CPU;
    }
    if (b_disable_prv_2_cpu) {
      b_disable_prv_2_cpu = false;
    }
  }
  MKLMemHolder() :
    head_(HEAD_AT_CPU), prv_descriptor_(nullptr),
    b_disable_prv_2_cpu(false), b_eager_mode(false) {}
};

#else
struct MKLMemHolder {
 public:
  virtual std::shared_ptr<PrvMemDescr> get_prv_descriptor() = 0;
};
#endif

}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_MKL_MEMORY_H_
