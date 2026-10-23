#ifndef INPLACE_RSQRT_FP16_H
#define INPLACE_RSQRT_FP16_H

#include "kernel_operator.h"

namespace local_ops {

// Checked flattening of contiguous ND [N4,N3,N2,N1]. No elementwise loop.
// limit is an element count, not a byte count.
__aicore__ inline bool Numel4D(uint32_t n4, uint32_t n3,
    uint32_t n2, uint32_t n1, uint32_t limit, uint32_t &count)
{
    count = 0;
    if (n4 == 0 || n3 == 0 || n2 == 0 || n1 == 0) {
        return true;
    }
    if (n4 > limit / n3) return false;
    uint32_t value = n4 * n3;
    if (value > limit / n2) return false;
    value *= n2;
    if (value > limit / n1) return false;
    count = value * n1;
    return true;
}

// self: FP16 UB tensor at VECIN/VECCALC/VECOUT, starting at a 32-byte
// aligned address. Reserve round_up(count * sizeof(half), 32) bytes.
// All valid input values must be finite and strictly positive.
// Caller must make self ready before entry (e.g. MTE2_V after GM->UB DMA).
// Returns false, without modifying self, for an invalid size/capacity.
// Empty shapes are a successful no-op. No extra UB or output tensor is used.
__aicore__ inline bool InplaceRsqrt(const AscendC::LocalTensor<half> &self,
    uint32_t n4, uint32_t n3, uint32_t n2, uint32_t n1)
{
    const uint32_t size = self.GetSize();
    const uint32_t limit = size < 0x7fffffffu ? size : 0x7fffffffu;
    uint32_t count = 0;
    if (!Numel4D(n4, n3, n2, n1, limit, count)) {
        return false;
    }
    if (count == 0) return true;

    // Complete source/destination overlap, with no shifted overlap between
    // vector repeats. calCount processes only valid elements, including tails.
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Rsqrt<half>(self, self, static_cast<int32_t>(count));
    AscendC::PipeBarrier<PIPE_V>();
    return true;
}

} // namespace local_ops

#endif // INPLACE_RSQRT_FP16_H
