#ifndef INPLACE_RSQRT_API_H
#define INPLACE_RSQRT_API_H
#include <cstdint>
#include "acl/acl.h"
// Synchronous C ABI for an initialized ACL context/current device.
// deviceSelf owns at least N4*N3*N2*N1*2 bytes of contiguous positive FP16 data.
// Input storage is overwritten; there is no output allocation or workspace.
// Empty shapes are no-ops and permit deviceSelf=nullptr.
// The caller owns the device allocation and stream; no resources are retained.
extern "C" aclError InplaceRsqrtFp16(void *deviceSelf,
    uint64_t n4, uint64_t n3, uint64_t n2, uint64_t n1, aclrtStream stream);
#endif
