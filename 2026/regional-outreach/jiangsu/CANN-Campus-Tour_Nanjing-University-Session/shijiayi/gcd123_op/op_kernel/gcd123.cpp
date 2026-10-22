// Copyright (C) 2026.
// Gcd123 Kernel：out = gcd(self, other)，逐元素，支持 ND 4 维 broadcast。
//
// 计算语义：
//  - GCD 是针对整数的运算。float16 输入先截断取整（向负无穷），对绝对值做欧几里得
//    辗转相除求最大公约数，结果再转回 float16。
//  - 若输入本身是整数值（如 6.0、8.0），结果即为整数 GCD（如 2.0）。
//
// 策略（兼顾正确性与访存效率）：
//  - 输出按“行”组织：一行 = 固定的 (N4, N3, N2)，沿 N1 连续。
//  - 每行内按 chunkLen 分块，保证 UB 占用可控；GM<->UB 用块级 DataCopy。
//  - 当某输入的内维 N1 为 1（broadcast）时，该行的值只取一次并复用。
//  - 逐元素 GCD 用标量(Scalar)循环在 UB 上完成。
//
#include "kernel_operator.h"
#include "gcd123_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t EVT_IN = 0;  // MTE2(GM->UB) 完成 -> Scalar(S) 读
constexpr uint32_t EVT_OUT = 1; // Scalar(S) 写完成 -> MTE3(UB->GM) 读
constexpr uint32_t DEFAULT_CHUNK = 1024;
} // namespace

class KernelGcd123 {
public:
    __aicore__ inline KernelGcd123() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out, const Gcd123TilingData *t) {
        selfN4_ = t->selfN4; selfN3_ = t->selfN3; selfN2_ = t->selfN2; selfN1_ = t->selfN1;
        otherN4_ = t->otherN4; otherN3_ = t->otherN3; otherN2_ = t->otherN2; otherN1_ = t->otherN1;
        outN4_ = t->outN4; outN3_ = t->outN3; outN2_ = t->outN2; outN1_ = t->outN1;
        totalSelf_ = t->totalSelf; totalOther_ = t->totalOther; totalOut_ = t->totalOut;
        totalLines_ = t->totalLines;
        chunkLen_ = (t->chunkLen == 0) ? DEFAULT_CHUNK : t->chunkLen;

        selfGm_.SetGlobalBuffer((__gm__ half *)self, totalSelf_);
        otherGm_.SetGlobalBuffer((__gm__ half *)other, totalOther_);
        outGm_.SetGlobalBuffer((__gm__ half *)out, totalOut_);

        pipe_.InitBuffer(selfQueue_, 1, chunkLen_ * sizeof(half));
        pipe_.InitBuffer(otherQueue_, 1, chunkLen_ * sizeof(half));
        pipe_.InitBuffer(outQueue_, 1, chunkLen_ * sizeof(half));
    }

    __aicore__ inline void Process() {
        selfLocal_ = selfQueue_.AllocTensor<half>();
        otherLocal_ = otherQueue_.AllocTensor<half>();
        outLocal_ = outQueue_.AllocTensor<half>();

        // 将 totalLines 行尽量均匀地划分到各个 AI Core（含余数）。
        const uint32_t coreNum = GetBlockNum();
        const uint32_t coreIdx = GetBlockIdx();
        const uint32_t linesPerCore = totalLines_ / coreNum;
        const uint32_t remainder = totalLines_ % coreNum;
        const uint32_t startLine = coreIdx * linesPerCore + (coreIdx < remainder ? coreIdx : remainder);
        const uint32_t coreLines = linesPerCore + (coreIdx < remainder ? 1 : 0);
        for (uint32_t line = startLine; line < startLine + coreLines; ++line) {
            ProcessLine(line);
        }

        selfQueue_.FreeTensor(selfLocal_);
        otherQueue_.FreeTensor(otherLocal_);
        outQueue_.FreeTensor(outLocal_);
    }

private:
    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

    __aicore__ inline int32_t AbsI32(int32_t v) { return v < 0 ? -v : v; }

    // 欧几里得辗转相除，输入先取绝对值。
    __aicore__ inline int32_t GcdI32(int32_t a, int32_t b) {
        a = AbsI32(a);
        b = AbsI32(b);
        while (b != 0) {
            int32_t r = a % b;
            a = b;
            b = r;
        }
        return a;
    }

    // float16 -> int32：先转 float，再按 CAST_FLOOR 截断取整。
    __aicore__ inline int32_t HalfToI32(half h) {
        float f = HalfToFloat(h);
        return AscendC::Cast<float, int32_t, AscendC::RoundMode::CAST_FLOOR>(f);
    }

    // int32 -> float16：先转 float，再用标量 Cast 转 half。
    __aicore__ inline half I32ToHalf(int32_t v) {
        float f = static_cast<float>(v);
        return AscendC::Cast<float, half, AscendC::RoundMode::CAST_ODD>(f);
    }

    // IEEE-754 float16 -> float 的位级转换（不依赖 half 的隐式转换运算符，
    // 以保证跨 CANN 版本可移植）。
    __aicore__ inline float HalfToFloat(half h) {
        uint32_t hb = *reinterpret_cast<const uint16_t *>(&h);
        const uint32_t sign = (hb & 0x8000u) << 16;
        const uint32_t exp = (hb >> 10) & 0x1Fu;
        uint32_t mant = hb & 0x3FFu;
        uint32_t fbits = 0;
        if (exp == 0) {
            if (mant == 0) {
                fbits = sign; // +/-0
            } else {
                uint32_t e = 122; // 127 - 15 + 10（半规整后的偏差基准）
                while ((mant & 0x200u) == 0) { mant <<= 1; --e; }
                mant &= 0x1FFu;
                fbits = sign | (e << 23) | (mant << 14);
            }
        } else if (exp == 0x1Fu) {
            fbits = sign | 0x7F800000u | (mant << 13); // inf / nan
        } else {
            const uint32_t e = exp - 15 + 127;
            fbits = sign | (e << 23) | (mant << 13);
        }
        union FBits { float f; uint32_t u; };
        FBits c;
        c.u = fbits;
        return c.f;
    }

    __aicore__ inline void ProcessLine(uint32_t line) {
        // 行号 line -> (d3, d2, d1)，对应 (N4, N3, N2)；N1 为行内连续维。
        const uint32_t d1 = line % outN2_;
        const uint32_t d2 = (line / outN2_) % outN3_;
        const uint32_t d3 = line / (outN2_ * outN3_);

        const uint32_t outLineStart =
            d3 * (outN3_ * outN2_ * outN1_) + d2 * (outN2_ * outN1_) + d1 * outN1_;

        // self 行起点：broadcast 维（size=1）取 index 0
        const uint32_t aD3 = (selfN4_ == 1) ? 0 : d3;
        const uint32_t aD2 = (selfN3_ == 1) ? 0 : d2;
        const uint32_t aD1 = (selfN2_ == 1) ? 0 : d1;
        const uint32_t selfLineStart =
            aD3 * (selfN3_ * selfN2_ * selfN1_) + aD2 * (selfN2_ * selfN1_) + aD1 * selfN1_;

        const uint32_t bD3 = (otherN4_ == 1) ? 0 : d3;
        const uint32_t bD2 = (otherN3_ == 1) ? 0 : d2;
        const uint32_t bD1 = (otherN2_ == 1) ? 0 : d1;
        const uint32_t otherLineStart =
            bD3 * (otherN3_ * otherN2_ * otherN1_) + bD2 * (otherN2_ * otherN1_) + bD1 * otherN1_;

        // 内维 N1 被 broadcast（size=1）时，整行只取一次该值。
        if (selfN1_ == 1) {
            DataCopy(selfLocal_, selfGm_[selfLineStart], 1);
        }
        if (otherN1_ == 1) {
            DataCopy(otherLocal_, otherGm_[otherLineStart], 1);
        }

        for (uint32_t cStart = 0; cStart < outN1_; cStart += chunkLen_) {
            const uint32_t cLen = MinU32(chunkLen_, outN1_ - cStart);
            if (selfN1_ != 1) {
                DataCopy(selfLocal_, selfGm_[selfLineStart + cStart], cLen);
            }
            if (otherN1_ != 1) {
                DataCopy(otherLocal_, otherGm_[otherLineStart + cStart], cLen);
            }
            // GM->UB 完成后再让标量读取 UB。
            SetFlag<HardEvent::MTE2_S>(EVT_IN);
            WaitFlag<HardEvent::MTE2_S>(EVT_IN);

            for (uint32_t j = 0; j < cLen; ++j) {
                const half sv = (selfN1_ == 1) ? selfLocal_(0) : selfLocal_(j);
                const half ov = (otherN1_ == 1) ? otherLocal_(0) : otherLocal_(j);
                outLocal_(j) = I32ToHalf(GcdI32(HalfToI32(sv), HalfToI32(ov)));
            }

            // 标量写完成后再让 UB->GM 读取 UB。
            SetFlag<HardEvent::S_MTE3>(EVT_OUT);
            WaitFlag<HardEvent::S_MTE3>(EVT_OUT);
            DataCopy(outGm_[outLineStart + cStart], outLocal_, cLen);
        }
    }

private:
    uint32_t selfN4_ = 0, selfN3_ = 0, selfN2_ = 0, selfN1_ = 0;
    uint32_t otherN4_ = 0, otherN3_ = 0, otherN2_ = 0, otherN1_ = 0;
    uint32_t outN4_ = 0, outN3_ = 0, outN2_ = 0, outN1_ = 0;
    uint32_t totalSelf_ = 0, totalOther_ = 0, totalOut_ = 0, totalLines_ = 0;
    uint32_t chunkLen_ = DEFAULT_CHUNK;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> selfQueue_;
    TQue<QuePosition::VECIN, 1> otherQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<half> selfGm_;
    GlobalTensor<half> otherGm_;
    GlobalTensor<half> outGm_;
    LocalTensor<half> selfLocal_;
    LocalTensor<half> otherLocal_;
    LocalTensor<half> outLocal_;
};

extern "C" __global__ __aicore__ void gcd123(GM_ADDR self, GM_ADDR other, GM_ADDR out, GM_ADDR workspace,
                                             GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(Gcd123TilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelGcd123 op;
    op.Init(self, other, out, &tilingData);
    op.Process();
}
