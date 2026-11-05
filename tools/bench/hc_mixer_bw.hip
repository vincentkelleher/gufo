// Bandwidth probe for the Flash-Next hyper-connection mixer projections.
//
// HcMixQ8FusedKernel spends 12.8% of a decode token in two Q8_0 projections
// per layer. This probe measures the same bytes for both shapes at several
// block counts, rotating the weight base through copies whose total exceeds
// the 32 MiB MALL so the figure is DRAM and not L2.
//
// Findings on gfx1151 (ROCm 10.1; 228.6 GB/s measured DRAM read peak from
// gfx1151_peak):
//   down 10240->320, one row per wave, 40 blocks    198 GB/s
//   down, 80 / 160 blocks                           200 / 198 GB/s
//   down, prefetch 4 instead of 2                   200 GB/s
//   up 320->10240, 16 rows per wave (production)    167 GB/s
//   up, 8 / 4 / 2 rows per wave                     191 / 188 / 188 GB/s
//
// The down projection is already at 87% of the DRAM peak and more blocks do
// not help, so its 40-work-item phase is not occupancy limited. The up
// projection's 16-row waves cost about 13% against 8-row waves, which is
// 0.3% of a token.
//
// Build: nix develop -c tools/bench/build.sh hc_mixer_bw
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define HIP_OK(e)                                            \
  do {                                                       \
    hipError_t r_ = (e);                                     \
    if (r_ != hipSuccess) {                                  \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(r_));                   \
      std::exit(1);                                          \
    }                                                        \
  } while (0)

namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;
// Rotate through copies that together exceed the 32 MiB MALL.
constexpr int kCopies = 32;

struct Q8_0Block {
  __half d;
  signed char qs[32];
};  // 34 bytes, never 16-byte aligned
struct Q8_1Block {
  __half2 ds;
  signed char qs[32];
};

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  return v;
}

// ---- down projection: 320 rows x 320 blocks (10240 wide) ----
constexpr int kDownRows = 320;
constexpr int kDownBlocks = 320;

template<int kPrefetch>
__global__ void Down(const Q8_0Block* __restrict__ w,
                     const Q8_1Block* __restrict__ x, float* __restrict__ out,
                     int rows_per_block) {
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
  const int row = blockIdx.x * rows_per_block + wave;
  if (row >= kDownRows)
    return;
  const int part = (lane & 3) * 2;
  float sum = 0.0f;
  for (int first = lane / 4; first < kDownBlocks; first += 8 * kPrefetch) {
    int v[kPrefetch][2], u[kPrefetch][2];
    __half dw[kPrefetch], dx[kPrefetch];
#pragma unroll
    for (int j = 0; j < kPrefetch; ++j) {
      const int kb = first + 8 * j;
      const Q8_0Block& wb = w[row * kDownBlocks + kb];
      dw[j] = wb.d;
      __builtin_memcpy(v[j], wb.qs + 4 * part, sizeof(v[j]));
      const Q8_1Block& xb = x[kb];
      dx[j] = __low2half(xb.ds);
      __builtin_memcpy(u[j], xb.qs + 4 * part, sizeof(u[j]));
    }
#pragma unroll
    for (int j = 0; j < kPrefetch; ++j) {
      int dot = __builtin_amdgcn_sudot4(true, v[j][0], true, u[j][0], 0, false);
      dot = __builtin_amdgcn_sudot4(true, v[j][1], true, u[j][1], dot, false);
      const float sc = __fmul_rn(__half2float(dw[j]), __half2float(dx[j]));
      sum = __fadd_rn(sum, __fmul_rn(sc, static_cast<float>(dot)));
    }
  }
  sum = WaveSum(sum);
  if (lane == 0)
    out[row] = sum;
}

// ---- up projection: 10240 rows x 10 blocks (320 wide) ----
constexpr int kUpRows = 10240;
constexpr int kUpBlocks = 10;

template<int kRowsPerWave>
__global__ void Up(const Q8_0Block* __restrict__ w,
                   const Q8_1Block* __restrict__ x, float* __restrict__ out,
                   int rows_per_block) {
  const int lane = threadIdx.x % 32;
  const int wave = threadIdx.x / 32;
  const int base = (blockIdx.x * rows_per_block + wave) * kRowsPerWave;
  const int part = (lane & 3) * 2;
  const int kb0 = lane / 4;
  const bool second = lane < 8;
  const int kb1 = second ? kb0 + 8 : kb0;
  int u0[2], u1[2];
  __builtin_memcpy(u0, x[kb0].qs + 4 * part, sizeof(u0));
  __builtin_memcpy(u1, x[kb1].qs + 4 * part, sizeof(u1));
  const float dx0 = __half2float(__low2half(x[kb0].ds));
  const float dx1 = __half2float(__low2half(x[kb1].ds));
  for (int r = 0; r < kRowsPerWave; ++r) {
    const int row = base + r;
    if (row >= kUpRows)
      return;
    const Q8_0Block* rw = w + static_cast<long>(row) * kUpBlocks;
    int v0[2];
    __builtin_memcpy(v0, rw[kb0].qs + 4 * part, sizeof(v0));
    int d0 = __builtin_amdgcn_sudot4(true, v0[0], true, u0[0], 0, false);
    d0 = __builtin_amdgcn_sudot4(true, v0[1], true, u0[1], d0, false);
    float sum = __fmul_rn(__fmul_rn(__half2float(rw[kb0].d), dx0),
                          static_cast<float>(d0));
    if (second) {
      int v1[2];
      __builtin_memcpy(v1, rw[kb1].qs + 4 * part, sizeof(v1));
      int d1 = __builtin_amdgcn_sudot4(true, v1[0], true, u1[0], 0, false);
      d1 = __builtin_amdgcn_sudot4(true, v1[1], true, u1[1], d1, false);
      sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(__half2float(rw[kb1].d), dx1),
                                     static_cast<float>(d1)));
    }
    sum = WaveSum(sum);
    if (lane == 0)
      out[row] = sum;
  }
}

float Time(hipStream_t s, int iters, auto&& launch) {
  for (int i = 0; i < 5; ++i)
    launch();
  HIP_OK(hipStreamSynchronize(s));
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a));
  HIP_OK(hipEventCreate(&b));
  HIP_OK(hipEventRecord(a, s));
  for (int i = 0; i < iters; ++i)
    launch();
  HIP_OK(hipEventRecord(b, s));
  HIP_OK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_OK(hipEventElapsedTime(&ms, a, b));
  HIP_OK(hipEventDestroy(a));
  HIP_OK(hipEventDestroy(b));
  return ms / iters;
}

}  // namespace

int main() {
  hipDeviceProp_t prop{};
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  std::printf("device: %s arch=%s CUs=%d\n", prop.name, prop.gcnArchName,
              prop.multiProcessorCount);

  const int iters = 2000;
  hipStream_t s{};
  HIP_OK(hipStreamCreate(&s));
  void* dout = nullptr;
  HIP_OK(hipMalloc(&dout, kUpRows * sizeof(float)));
  HIP_OK(hipMemset(dout, 0, kUpRows * sizeof(float)));
  auto* O = static_cast<float*>(dout);

  {
    const size_t one = size_t{kDownRows} * kDownBlocks * sizeof(Q8_0Block);
    void* dw = nullptr;
    void* dx = nullptr;
    HIP_OK(hipMalloc(&dw, one * kCopies));
    HIP_OK(hipMalloc(&dx, size_t{kDownBlocks} * sizeof(Q8_1Block)));
    HIP_OK(hipMemset(dw, 1, one * kCopies));
    HIP_OK(hipMemset(dx, 1, size_t{kDownBlocks} * sizeof(Q8_1Block)));
    const auto* W0 = static_cast<const Q8_0Block*>(dw);
    const auto* X = static_cast<const Q8_1Block*>(dx);
    int rot = 0;
    const auto nextW = [&] {
      const auto* p = W0 + static_cast<long>(rot) * kDownRows * kDownBlocks;
      rot = (rot + 1) % kCopies;
      return p;
    };
    std::printf("\ndown 10240->320, %.2f MB per call\n\n", one / 1e6);
    std::printf("%-34s %8s %10s %10s\n", "variant", "blocks", "us", "GB/s");
    const auto report = [&](const char* name, int blocks, float ms) {
      const double us = ms * 1000.0;
      std::printf("%-34s %8d %10.2f %10.1f\n", name, blocks, us,
                  one / (us * 1e3));
    };
    for (int rpb : {8, 4, 2}) {
      char name[64];
      std::snprintf(name, sizeof(name), "prefetch2 rows/block=%d", rpb);
      report(name, kDownRows / rpb, Time(s, iters, [&] {
               const auto* w = nextW();
               Down<2><<<kDownRows / rpb, kThreads, 0, s>>>(w, X, O, rpb);
             }));
    }
    report("prefetch4 rows/block=8", kDownRows / 8, Time(s, iters, [&] {
             const auto* w = nextW();
             Down<4><<<kDownRows / 8, kThreads, 0, s>>>(w, X, O, 8);
           }));
    HIP_OK(hipStreamSynchronize(s));
    HIP_OK(hipFree(dw));
    HIP_OK(hipFree(dx));
  }

  {
    const size_t one = size_t{kUpRows} * kUpBlocks * sizeof(Q8_0Block);
    void* uw = nullptr;
    void* ux = nullptr;
    HIP_OK(hipMalloc(&uw, one * kCopies));
    HIP_OK(hipMalloc(&ux, size_t{kUpBlocks} * sizeof(Q8_1Block)));
    HIP_OK(hipMemset(uw, 1, one * kCopies));
    HIP_OK(hipMemset(ux, 1, size_t{kUpBlocks} * sizeof(Q8_1Block)));
    const auto* U0 = static_cast<const Q8_0Block*>(uw);
    const auto* UX = static_cast<const Q8_1Block*>(ux);
    int rot = 0;
    const auto nextW = [&] {
      const auto* p = U0 + static_cast<long>(rot) * kUpRows * kUpBlocks;
      rot = (rot + 1) % kCopies;
      return p;
    };
    std::printf("\nup 320->10240, %.2f MB per call\n\n", one / 1e6);
    std::printf("%-34s %8s %10s %10s\n", "variant", "blocks", "us", "GB/s");
    const auto report = [&](const char* name, int blocks, float ms) {
      const double us = ms * 1000.0;
      std::printf("%-34s %8d %10.2f %10.1f\n", name, blocks, us,
                  one / (us * 1e3));
    };
    for (int rpw : {16, 8, 4, 2}) {
      const int bx = (kUpRows + rpw * kWaves - 1) / (rpw * kWaves);
      char name[64];
      std::snprintf(name, sizeof(name), "rows/wave=%d", rpw);
      report(name, bx * kWaves, Time(s, iters, [&] {
               const auto* w = nextW();
               if (rpw == 16) {
                 Up<16><<<bx, kThreads, 0, s>>>(w, UX, O, kWaves);
               } else if (rpw == 8) {
                 Up<8><<<bx, kThreads, 0, s>>>(w, UX, O, kWaves);
               } else if (rpw == 4) {
                 Up<4><<<bx, kThreads, 0, s>>>(w, UX, O, kWaves);
               } else {
                 Up<2><<<bx, kThreads, 0, s>>>(w, UX, O, kWaves);
               }
             }));
    }
    HIP_OK(hipStreamSynchronize(s));
    HIP_OK(hipFree(uw));
    HIP_OK(hipFree(ux));
  }
  HIP_OK(hipFree(dout));
  return 0;
}
