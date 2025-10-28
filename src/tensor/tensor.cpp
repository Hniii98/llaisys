#include "tensor.hpp"

#include "../utils.hpp"
#include "../ops/rearrange/op.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size(); 
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        // may need custom deleter for storage, so can't use make_shared to build.
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    int ndim_ = static_cast<int>(this->ndim());

    if (ndim_ == 0) return true;  // empty 
    
    ptrdiff_t expected_stride = 1;
    for (int i = ndim_ - 1; i >= 0; --i) {  
        // expect stride[k+1] = stride[k] * shape[k]
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= _meta.shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // check dimension
    size_t ndim_ = this->ndim();
    size_t permute_dim = order.size();
    CHECK_ARGUMENT(ndim_ == permute_dim, "Mismatch dimension!");


    // reconstruct meta information
    std::vector<ptrdiff_t> strides_(ndim_);
    std::vector<size_t> shape_(ndim_);
    for (size_t i = 0; i < ndim_; ++i) {
        shape_[i] = this->shape()[order[i]];
        strides_[i] = this->strides()[order[i]];
    }

    TensorMeta meta{this->dtype(), shape_, strides_}; 

    return std::shared_ptr<Tensor>(new Tensor(meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // check tensor
    CHECK_ARGUMENT(this->isContiguous(), "Tensor should be contiguous.");
    size_t expected_numel = this->numel();
    size_t view_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), 
        std::multiplies<size_t>());
    CHECK_SAME(EXCEPTION_SHAPE_MISMATCH, expected_numel, view_numel); // check 

    // reconstruct meta information
    int view_ndim = static_cast<int>(shape.size()); 
    std::vector<ptrdiff_t> strides(view_ndim);
    size_t stride = 1;
    for (int i = view_ndim - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    TensorMeta meta{this->dtype(), shape, strides}; // keep same data type

    return std::shared_ptr<Tensor>(new Tensor(meta, _storage)); // using same storage
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Dimension out of range");
    //  slice indices in [start, end)
    CHECK_ARGUMENT(start <= end && end < this->shape()[dim] + 1, 
        "Slice indices out of range");

    std::vector<size_t> shape_ = this->shape();
    shape_[dim] = end - start;

    // calculate memory offset
    size_t offset_ = start * this->strides()[dim] * this->elementSize();


    TensorMeta meta{this->dtype(), shape_, this->strides()};


    return std::shared_ptr<Tensor>(new Tensor(meta, _storage, offset_));
}

void Tensor::load(const void *src_) {
    // get runtime device type
    size_t total_size = this->numel() * this->elementSize();
    core::context().runtime().api()->memcpy_sync(this->data(), src_, total_size, 
        LLAISYS_MEMCPY_H2D); 
   
}

tensor_t Tensor::contiguous() const {
    if(isContiguous()) {
        // should transfer const Tensor* to Tensor* first.
        // return a shared_ptr point to same meta and storage.
        return std::const_pointer_cast<Tensor>(shared_from_this());
    }

    tensor_t new_tensor = create(_meta.shape, 
                                 _meta.dtype, 
                                 deviceType(),
                                 deviceId());

    llaisys::ops::rearrange(new_tensor, 
                            std::const_pointer_cast<Tensor>(shared_from_this()));
    
    return new_tensor;
        
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
