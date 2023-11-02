#include <memory>
#include <stdexcept>

template <typename T>
class PinnedAllocator : public std::allocator<T> {
public:
    using value_type = T;

    PinnedAllocator() = default;
    
    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        auto status = cudaMallocHost((void **)&ptr, sizeof(T) * n);
        if (status != cudaSuccess || ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    void deallocate(T* ptr, std::size_t) noexcept {
        cudaFreeHost(ptr);
    }
};
