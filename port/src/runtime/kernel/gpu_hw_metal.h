#pragma once

// R4 (PLAN-R4.md): own Metal backend for macOS. FROZEN CONTRACT (19-nov, before M1): the public surface below
// is exactly what gpu_render.cpp calls on the D3D11 backend (gpu_hw_d3d11.h), same names, same signatures, same
// meaning. gpu_render.cpp does not change; gpu_hw_backend.h picks this header under TTT2_GPU_HW_METAL.
//
// Plain C++ on purpose: no Metal or Objective-C type appears here. The state lives in `Impl`, defined and
// implemented in Objective-C++ files, one per area (each file is one constructor brief; two briefs never
// write the same file):
//   gpu_hw_metal_core.mm      device, queue, window (CAMetalLayer on the SDL window), present, ReadFront,
//                             Get(), ReportBackendFailure, RemovedReason
//   gpu_hw_metal_shaders.mm   pixel-shader cache: key -> airconv .metallib + .bind sidecar; pipelines;
//                             the fixed CPU-vertex VS; VertexShader() (M1: always null = CPU vertex stage)
//   gpu_hw_metal_textures.mm  guest texture upload/decode (same decode as gpu_hw_texture_upload.h), samplers
//   gpu_hw_metal_surfaces.mm  EDRAM colour/depth surfaces, Resolve, CompareResolve, DumpResolved,
//                             WriteBackRange, PendingCopyWaiter
//   gpu_hw_metal_draw.mm      Draw, DrawGpuVs (M1: always Fallback), states, constant upload, encoders
//
// Shader binding (m0/duelo/BINDINGS.md): fragment function "shader_main"; buffer(0) = cbuffer table (one
// 64-bit gpuAddress per declared cb, ascending register: cb0 = XenosPsConstants 4096 B, cb1 = XenosPsState =
// PsStateCB 1712 B); buffer(1) = argument table, dense, field order from the .bind sidecar written at
// conversion time from airconv's SM50GetArgumentsInfo (never from a formula). Today's corpus: per declared
// sampler N, 3 qwords {sN, cube_sN, meta_sN = lod-bias float bits}; then per declared SRV N, 2 qwords
// {tN, meta_tN = (arrayLength << 32) | minLOD float bits}. PS input TEXCOORDn -> user(reg{n+1}_0); the fixed VS
// writes all 16. Every texture, buffer and sampler reached through a table needs useResource:usage:stages:.
//
// Thread model: as D3D11, every call arrives under ttt2_hw::g_backendMutex (Guarded), except the
// std::function returned by PendingCopyWaiter, which runs without the lock.
// Errors: throw ttt2_hw::Failure; Guarded turns it into Off mode (gpu_hw_control.h).

#include "gpu_hw_api.h"
#include "gpu_hw_control.h"
#include "../host_window.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace ttt2_hw {

struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

// Same hash as gpu_hw_d3d11.h (gpu_render.cpp prints shader text keys with it; the metallib cache uses the
// D3D11 backend's exact keys, so ps_<KEY>.dxbc on the PC and ps_<KEY>.metallib on the Mac name one shader).
inline uint64_t Fnv(const void* data, size_t size, uint64_t h = 1469598103934665603ull)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

class Backend
{
public:
    Backend();
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Draw with the CPU vertex stage: `slots` index the packed frontend vertices (three per triangle); only
    // `interps` interpolators reach the shader. False = not drawn (the caller keeps going; same as D3D11).
    bool Draw(const uint32_t* regs, const std::vector<uint32_t>& psCode,
              const std::vector<CpuVertex>& vertices, const std::vector<uint32_t>& slots,
              uint32_t interps, const DrawState& st);

    // Translated vertex shader for this microcode, or null when it cannot run on the GPU (CPU vertex stage).
    VertexShaderHandle* VertexShader(const std::vector<uint32_t>& code);

    // Draw with the VS on the GPU. Fallback = nothing changed, the caller runs the CPU vertex stage.
    GpuVsResult DrawGpuVs(const uint32_t* regs, const std::vector<uint32_t>& psCode,
                          VertexShaderHandle& shader,
                          std::span<const uint32_t> indices, uint32_t minIndex, uint32_t maxIndex, const DrawState& st,
                          bool triangleStrip = false, uint32_t stripTriangles = 0, uint64_t indexIdentity = 0);

    // EDRAM -> guest memory copy (RB_COPY_*). destFormat 31 = 16_16_FLOAT keeps red/green as half floats;
    // every other colour destination is 8_8_8_8 (as gpu_hw_d3d11.h).
    void Resolve(uint32_t colorBase, uint32_t dest, uint32_t pitch, uint32_t height, bool hasLatch, uint32_t latch,
                 bool swapRB, uint32_t srcSelect, uint32_t depthBase, bool depthFloat, uint32_t destFormat = 6);
    // Compare mode: GPU resolve against the CPU renderer's linear result.
    void CompareResolve(uint32_t dest, const uint32_t* cpu, uint32_t pitch, uint32_t height);

    // Show the resolved front buffer. displaySwapRB / gamma as gpu_hw_d3d11.h Present.
    bool Present(uint32_t frontBuffer, bool displaySwapRB, const uint32_t* gammaRamp, uint32_t gammaVersion);
    bool IsExclusiveFullscreen() const;
    // Last presented image as 1280x720 BGRX (alpha cleared), for BMP filmstrips and the M1 image gate.
    bool ReadFront(uint32_t* out);
    // Diagnostics: the GPU texture registered for a resolve destination, as a BMP.
    bool DumpResolved(uint32_t dest, const char* path);

    // Guest-memory readback of resolved ranges. True = a copy is pending: the caller drops the lock and runs
    // PendingCopyWaiter(); `block` waits here instead.
    bool WriteBackRange(uint32_t start, uint32_t size, bool block = false);
    std::function<bool()> PendingCopyWaiter() const;

    void SetWindow(host_window::Window* window);
    // Failure-log diagnostics (D3D11: DXGI device-removed reason). 0 = no device error.
    long RemovedReason() const;

    // main.cpp --build-shaders and the start-up check. Metal: convert/verify the metallib cache.
    static int PrebuildShaders(const std::vector<std::filesystem::path>& files, unsigned threads);
    static bool PrebuildCurrent();

    struct Impl;   // Objective-C++ only (gpu_hw_metal_*.mm)
private:
    std::unique_ptr<Impl> impl_;
};

Backend& Get();
void ReportBackendFailure(const char* what, const std::exception& e);

} // namespace ttt2_hw
