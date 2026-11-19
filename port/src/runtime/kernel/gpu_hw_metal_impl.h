#pragma once
// R4 Metal backend: INTERNAL contract between the gpu_hw_metal_*.mm files (Objective-C++ only; gpu_render.cpp
// never includes this). Frozen per slice (PLAN-R4.md §8, M1 slices). Each section names the one file that owns
// its definitions; other files may call them but never define them.
//
// Porting rule: logic copied from gpu_hw_d3d11.h keeps a "// D<line>" marker so a diff against the reference
// backend shows which side moved. gpu_hw_d3d11.h is NOT refactored during M1-M4 (it is the image-gate reference).
// Unsupported paths fail by name: throw Failure("metal: <what> not implemented (M1 slice N)") or, where the
// contract allows a soft answer (VertexShader -> null, DrawGpuVs -> Fallback), return it. Never a fake success.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "gpu_hw_metal.h"
#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct SDL_Window;

namespace ttt2_hw {

// ---------------------------------------------------------------- shared value types
// Colour surface, depth surface and resolved texture (D2134-2163, trimmed to what M1 uses).
struct MtlColorSurface { id<MTLTexture> tex = nil; bool touched = false; uint32_t samplePitch = 0;
                         MTLPixelFormat format = MTLPixelFormatBGRA8Unorm; uint32_t extW = 0, extH = 0; float scale = 1; };
struct MtlDepthSurface { id<MTLTexture> tex = nil; bool touched = false; uint32_t sampX = 1, sampY = 1;
                         uint32_t extW = 0, extH = 0; float scale = 1; };
struct MtlResolved {
    id<MTLTexture> tex = nil;          // at the SOURCE surface's scale (D976)
    uint32_t pitch = 0, height = 0, copiedWidth = 0, size = 0;
    uint64_t seq = 0, revision = 0;
    bool valid = false, cpuCoherent = false, swapRB = false, depth = false, depthFloat = false, rg16f = false;
    float scale = 1;
};

// ---------------------------------------------------------------- Backend::Impl
struct Backend::Impl {
    // ===== core (gpu_hw_metal_core.mm) =====
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::atomic<SDL_Window*> window{nullptr};     // from SetWindow
    void* metalView = nullptr;                    // SDL_MetalView, created at the first Present (as DXVK did)
    CAMetalLayer* layer = nil;
    id<MTLTexture> lastFront = nil;               // what Present showed (logical size below), for ReadFront
    uint32_t lastFrontW = 0, lastFrontH = 0;
    bool lastSwap = false, gammaApplied = false;
    uint32_t gammaVersion = ~0u;
    uint8_t gamma8[3][256] = {};                  // D1521-1535: CPU copy for ReadFront
    id<MTLTexture> gammaLut = nil;                // 256x1 RGBA32Float
    float scale = 1;                              // render scale (D3869); slice 1: fixed by TTT2_RENDER_SCALE or 1
    std::atomic<bool> exclusive{false};
    id<MTLCommandBuffer> frameCmd = nil;          // the open command buffer all files encode into

    void Initialize();                            // device + queue + core pipelines; idempotent (D246)
    id<MTLCommandBuffer> Cmd();                   // frameCmd, created on demand
    void Submit(bool wait);                       // commit frameCmd (wait = waitUntilCompleted), frameCmd = nil
    // Show `source` (logical w x h, at any scale): display pass (swap, gamma), scale to the letterboxed 16:9
    // area, present the drawable. D1332, D2012-2100. FXAA/CAS/MetalFX: log once by name and skip (M5).
    bool PresentTexture(id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap);
    void UpdateGamma(const uint32_t* ramp, uint32_t version);   // D1517-1552
    // 1280x720 BGRX of `tex` resampled at logical pixel centres (D4886-4903), then DisplayPixel (D1488-1495).
    bool ReadLogical(id<MTLTexture> tex, uint32_t logicalW, uint32_t logicalH, std::vector<uint32_t>& out);

    // ===== surfaces (gpu_hw_metal_surfaces.mm) — slice 4; slice 1 stubs live in core =====
    std::unordered_map<uint32_t, MtlColorSurface> colors;     // key = colorBase (base | fmt << 12), D2888
    std::unordered_map<uint32_t, MtlDepthSurface> depths;     // key = depthBase, D2889
    std::unordered_map<uint32_t, MtlResolved> resolved;       // key = dest guest address, D2890
    // Front-buffer choice (D4906-4939): resolved[fb] if valid, >= 1280x720 and fresh; else a 1280x720 BGRA
    // texture decoded from guest memory (storedSwap = true). nil = nothing to show.
    id<MTLTexture> FrontTexture(uint32_t fb, bool& storedSwap, uint32_t& logicalW, uint32_t& logicalH);

    // ===== later slices (frozen when their brief goes out) =====
    // shaders  (gpu_hw_metal_shaders.mm): PS cache by Fnv(PixelShaderHlsl(generic)) text key -> ps_<KEY>.metallib
    //          + ps_<KEY>.layout.tsv (m0/duelo/layout.py format); fixed CPU-vertex VS; pipeline cache.
    // textures (gpu_hw_metal_textures.mm): BindTexture (D4443-4569), samplers (D202-239).
    // draw     (gpu_hw_metal_draw.mm): Draw / PixelStageAndDraw (D375-857), states (D4216-4339), encoders.
};

// Test hook (core): lets a harness without the game show a texture through the real present path.
// Not used by gpu_render.cpp.
bool MetalPresentForTest(Backend& b, id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap);
void MetalSetSdlWindowForTest(Backend& b, SDL_Window* window);   // harness has no host_window::Window
Backend::Impl& MetalImplForTest(Backend& b);

} // namespace ttt2_hw
