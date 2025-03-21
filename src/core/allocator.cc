#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size); // 对齐后要分配的大小。

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t offset = 0;
        std::cout<<"alloc: "<<size<<std::endl;
        this->used += size;
        for(auto& pair: freed_blocks) {
            std::cout<<"block: "<<pair.first<<"  size: "<<pair.second<<std::endl;
            if(pair.second >= size) {
                pair.second = 0;
                offset = pair.first;
                return offset;
            }

        }
        
        freed_blocks[this->peak] = 0;
        offset = this->peak;
        this->peak += size;

        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        std::cout<<"free "<< addr << "  size : " << size<<std::endl;
        // freed_blocks[addr] = size;
        auto end = freed_blocks.rbegin();
        
        if(addr == end->first) {
            std::cout<<"free last one"<<std::endl;
            freed_blocks.erase(addr);
            peak -= size;
            used -= size;
        }else {
            freed_blocks[addr] = size;
            used -= size;
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
