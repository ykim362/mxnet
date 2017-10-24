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
* \file mkldnn_memory.cc
* \brief
* \author young.jin.kim@intel.com
*         ashok.emani@intel.com
*         deepthi.karkada@intel.com
*         louis.feng@intel.com
*         adam.d.straw@intel.com
*
*******************************************************************************/


#include <mxnet/base.h>
#if MXNET_USE_MKLDNN == 1
#include <mkl_memory.h>
#include "mkldnn_memory-inl.h"

namespace mxnet {

template <typename DType>
MKLDNNMemoryDescriptorBase<DType>::MKLDNNMemoryDescriptorBase(
        std::shared_ptr<memory::primitive_desc> usr_memory_pd
        , std::shared_ptr<memory::primitive_desc> prv_memory_pd)
                                    : name("MKLDNNMemoryDescriptorBase"),
                                    _prv_memory(NULL), _internal_ptr(NULL), _internal_size(0),
                                    _usr_memory(NULL), _dbg_cpu_ptr(NULL) {
    _usr_memory_pd_not_null = false;
    _prv_memory_pd_not_null = false;
    set_usr_memory_pd(usr_memory_pd);
    set_prv_memory_pd(prv_memory_pd);
}

template <typename DType>
void MKLDNNMemoryDescriptorBase<DType>::check_usr_with_prv_descriptors() {
    CHECK(_usr_memory_pd);
    CHECK(_prv_memory_pd);
    int32_t ndims = _usr_memory_pd->desc().data.ndims;
    CHECK_EQ(ndims, _prv_memory_pd->desc().data.ndims)
            << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions number";
    for (int32_t dim = 0; dim < ndims; ++dim) {
        CHECK_EQ(_usr_memory_pd->desc().data.dims[dim]
                , _prv_memory_pd->desc().data.dims[dim])
                << "MKLDNNMemoryDescriptorBase: Usr and Prv memory must have same dimensions";
    }
}



////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Implementation of MKLDNNMemoryDescriptor
//
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DType>
 MKLDNNMemoryDescriptor<DType>::MKLDNNMemoryDescriptor(
                        std::shared_ptr<memory::primitive_desc> usr_memory_pd
                        , std::shared_ptr<memory::primitive_desc> prv_memory_pd)
        : MKLDNNMemoryDescriptorBase<DType>(usr_memory_pd, prv_memory_pd) {
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::create_reorder_to_prv(void* cpu_ptr) {
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);

    if (this->_usr_memory == NULL)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if (this->_reorder_usr2prv.aprimitive == NULL)
        this->_reorder_usr2prv.reset(new reorder(*this->_usr_memory, *this->get_prv_memory()));
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::convert_to_prv(void* cpu_ptr) {
    CHECK(cpu_ptr);
    if (this->_dbg_cpu_ptr == NULL)
      this->_dbg_cpu_ptr = cpu_ptr;
    create_reorder_to_prv(cpu_ptr);
    // MKL_DLOG(INFO) << "convert usr => priv @" << this->name;
    this->_reorder_usr2prv.submit();;
}
template <typename DType>
void MKLDNNMemoryDescriptor<DType>::convert_from_other(std::shared_ptr<PrvMemDescr> other) {
  CHECK(NULL);  // Not implementation
}
template <typename DType>
void MKLDNNMemoryDescriptor<DType>::create_reorder_from_prv(void* cpu_ptr) {
    CHECK(cpu_ptr);
    CHECK(this->_usr_memory_pd);
    CHECK(this->_prv_memory_pd);

    if (this->_usr_memory == NULL)
        this->_usr_memory.reset(new memory(*this->_usr_memory_pd, cpu_ptr));
    if (this->_reorder_prv2usr.aprimitive == NULL) {
        this->_reorder_prv2usr.reset(new reorder(*this->_prv_memory, *this->_usr_memory));
    }
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::convert_from_prv(void* cpu_ptr) {
    CHECK(cpu_ptr);
    if (this->_dbg_cpu_ptr == NULL)
      this->_dbg_cpu_ptr = cpu_ptr;
    create_reorder_from_prv(cpu_ptr);
    // MKL_DLOG(INFO) << "convert priv => usr @" << this->name;
    this->_reorder_prv2usr.submit();
    // on_to_cpu();
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::convert_from_extprv(std::shared_ptr<memory> extprv_memory) {
    MKLDNNPrimitive<DType> reorder_extprv2prv;
    reorder_extprv2prv.reset(new reorder(*extprv_memory, *this->get_prv_memory()));
    // MKL_DLOG(INFO) << "convert extprv => priv @" << this->name;
    reorder_extprv2prv.submit();;
}


template <typename DType>
bool MKLDNNMemoryDescriptor<DType>::on_to_cpu() {
    if (StreamHolder::Instance().current_stream() != NULL
      && StreamHolder::Instance().current_stream()->ready()) {
        StreamHolder::Instance().current_stream()->wait();
    }
    return true;
}

template <typename DType>
bool MKLDNNMemoryDescriptorBase<DType>::layout_compare(std::shared_ptr<PrvMemDescr> other) {
    CHECK_EQ(other->get_descr_type(),
              PrvMemDescr::PRV_DESCR_MKLDNN);
    std::shared_ptr<MKLDNNMemoryDescriptorBase<DType> > other_descr =
        std::static_pointer_cast<MKLDNNMemoryDescriptorBase<DType> >(other);
    return (*other_descr->prv_memory_pd() == *this->prv_memory_pd());
}


template <typename DType>
std::shared_ptr<memory> MKLDNNMemoryDescriptor<DType>::get_converted_prv(DType* cpu_data,
                                            bool set_prv_ptr, const TBlob &tblob) {
  std::shared_ptr<MKLMemHolder> blob = tblob.Mkl_mem_;
  if (this->conversion_needed()) {
    // have private format
    const DType* prv_ptr = reinterpret_cast<DType*>(blob->prv_data());
    if (prv_ptr == NULL) {
      this->convert_to_prv(cpu_data);
      if (set_prv_ptr) {
        blob->set_prv_descriptor(this->get_shared_ptr(), true);
      }
      return this->get_prv_memory(true);
    } else {
      std::shared_ptr<MKLDNNData<DType> > blob_prv_mkldnn_mem_descr
        = get_mkldnn_prv_descriptor<DType>(blob);
      if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() != *this->prv_memory_pd()) {
        // prv in blob and in this descrptor may have different layouts
        this->convert_from_extprv(blob_prv_mkldnn_mem_descr->get_prv_memory(true));
        if (set_prv_ptr) {
          blob->set_prv_descriptor(this->get_shared_ptr(), true);
        }
        return this->get_prv_memory(true);
      } else if (blob_prv_mkldnn_mem_descr.get() != this) {
        // MKL_DLOG(INFO) << "layout OK ";
      }
      // Need:    CHECK(blob_prv_mkldnn_mem_descr->mkldnn_primitive());
      return blob_prv_mkldnn_mem_descr->get_prv_memory(true);
    }
  } else {
    const DType* prv_ptr = reinterpret_cast<DType*>(blob->prv_data());
    if (prv_ptr != NULL) {
      std::shared_ptr<MKLDNNData<DType> > blob_prv_mkldnn_mem_descr
        = get_mkldnn_prv_descriptor<DType>(blob);
      blob_prv_mkldnn_mem_descr->convert_from_prv(cpu_data);
    }
  }
  std::shared_ptr<mkldnn::memory> pres;
  memory * input_memory = new memory(*this->usr_memory_pd(), const_cast<DType*>(cpu_data));
  pres.reset(input_memory);
  return pres;
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::sync_converted_prv(DType* cpu_data,
                                            bool set_prv_ptr, const TBlob &tblob) {
  std::shared_ptr<MKLMemHolder> blob = tblob.Mkl_mem_;
  if (this->conversion_needed()) {
    // have private format
    const DType* prv_ptr = reinterpret_cast<DType*>(blob->prv_data());
    if (prv_ptr == NULL) {
      this->convert_to_prv(const_cast<DType*>(cpu_data));
      if (set_prv_ptr) {
        blob->set_prv_descriptor(this->get_shared_ptr(), true);
      }
    } else {
      std::shared_ptr<MKLDNNData<DType> > blob_prv_mkldnn_mem_descr
        = get_mkldnn_prv_descriptor<DType>(blob);
      if (*blob_prv_mkldnn_mem_descr->prv_memory_pd() != *this->prv_memory_pd()) {
        // prv in blob and in this descrptor may have different layouts
        this->convert_from_extprv(blob_prv_mkldnn_mem_descr->get_prv_memory(true));
        if (set_prv_ptr) {
          blob->set_prv_descriptor(this->get_shared_ptr(), true);
        }
      } else if (blob_prv_mkldnn_mem_descr.get() != this) {
        // MKL_DLOG(INFO) << "layout OK ";
      }
      // Need:    CHECK(blob_prv_mkldnn_mem_descr->mkldnn_primitive());
    }
  } else {
    const DType* prv_ptr = reinterpret_cast<DType*>(blob->prv_data());
    if (prv_ptr != NULL) {
      std::shared_ptr<MKLDNNData<DType> > blob_prv_mkldnn_mem_descr
        = get_mkldnn_prv_descriptor<DType>(blob);
      blob_prv_mkldnn_mem_descr->convert_from_prv(cpu_data);
    }
  }
}


template <typename DType>
std::shared_ptr<memory> MKLDNNMemoryDescriptor<DType>::create_output_memory(
    DType* cpu_data, const TBlob &blob,
    std::shared_ptr<MKLDNNMemoryDescriptor<DType> > thisData, bool in_place) {
    std::shared_ptr<memory> omem;
    if (this->conversion_needed()) {
      if (in_place) {
        std::shared_ptr<MKLDNNData<DType> > blob_omem = get_mkldnn_prv_descriptor<DType>(blob);
        omem = blob_omem->get_prv_memory();
      } else {
        omem = this->get_prv_memory();
        blob.Mkl_mem_->set_prv_descriptor(thisData);
      }
    } else {
      blob.Mkl_mem_->check_and_prv_to_cpu(cpu_data, false);
      omem.reset(new memory(*this->usr_memory_pd(), cpu_data));
    }
    return omem;
}

template <typename DType>
void MKLDNNMemoryDescriptor<DType>::sync_output_memory(const TBlob &blob,
    std::shared_ptr<MKLDNNMemoryDescriptor<DType> > thisData, bool in_place) {
    if (this->conversion_needed()) {
      if (!in_place) {
        blob.Mkl_mem_->set_prv_descriptor(thisData);
      }
    } else {
      blob.Mkl_mem_->check_and_prv_to_cpu(nullptr, false);
    }
}



template <typename DType>
std::shared_ptr<MKLDNNData<DType> > get_mkldnn_prv_descriptor(
    std::shared_ptr<MKLMemHolder> blob) {
    std::shared_ptr<PrvMemDescr> blob_prv_mem_descriptor =
        blob->get_prv_descriptor();
    if (blob_prv_mem_descriptor == nullptr)
      return nullptr;
    CHECK_EQ(blob_prv_mem_descriptor->get_descr_type(), PrvMemDescr::PRV_DESCR_MKLDNN);
    std::shared_ptr<MKLDNNData<DType> > blob_prv_mkldnn_mem_descr =
        std::static_pointer_cast<MKLDNNData<DType> >(blob_prv_mem_descriptor);
    CHECK(blob_prv_mkldnn_mem_descr != NULL);
    return blob_prv_mkldnn_mem_descr;
}

template class MKLDNNMemoryDescriptor<double>;
template class MKLDNNMemoryDescriptor<float>;
template class MKLDNNMemoryDescriptor<uint8_t>;
template class MKLDNNMemoryDescriptor<int8_t>;
template class MKLDNNMemoryDescriptor<int32_t>;

template class MKLDNNMemoryDescriptorBase<float>;
template class MKLDNNMemoryDescriptorBase<double>;
template class MKLDNNMemoryDescriptorBase<uint8_t>;
template class MKLDNNMemoryDescriptorBase<int8_t>;
template class MKLDNNMemoryDescriptorBase<int32_t>;

}  // namespace mxnet
#endif  // #ifdef MKLDNN_SUPPORTED
