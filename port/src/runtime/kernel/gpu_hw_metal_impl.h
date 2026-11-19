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

namespace ttt2_hw { class DirtyTracker; }   // gpu_hw_dirty_tracker.h (needs host_vm: only the game links it)

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
    uint32_t sourceKey = 0, sourcePitch = 0;   // D981-982
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

    // ===== surfaces (gpu_hw_metal_surfaces.mm) — slice 4; slice 1 stubs live in gpu_hw_metal_stubs.mm =====
    std::unordered_map<uint32_t, MtlColorSurface> colors;     // key = colorBase (base | fmt << 12), D2888
    std::unordered_map<uint32_t, MtlDepthSurface> depths;     // key = depthBase, D2889
    std::unordered_map<uint32_t, MtlResolved> resolved;       // key = dest guest address, D2890
    // Front-buffer choice (D4906-4939): resolved[fb] if valid, >= 1280x720 and fresh; else a 1280x720 BGRA
    // texture decoded from guest memory (storedSwap = true). nil = nothing to show.
    id<MTLTexture> FrontTexture(uint32_t fb, bool& storedSwap, uint32_t& logicalW, uint32_t& logicalH);

    // ===== draw (gpu_hw_metal_draw.mm) — slice 2 =====
    // ONE render encoder may be open on frameCmd at a time. Every file that opens any other encoder (render,
    // blit) or commits frameCmd calls EndPass() first (core: Pass, ReadLogical, Submit). A pass reopened on a
    // target drawn earlier in the frame must LOAD, never clear.
    id<MTLRenderCommandEncoder> renderEnc = nil;
    id<MTLTexture> encColor = nil, encDepth = nil;   // targets of renderEnc (nil = none)
    void EndPass();                                  // ends renderEnc if open; idempotent
    // Test visibility: the exact bytes the last Draw bound as cb0 (first constantRows * 16 B) and cb1 (PsStateCB).
    std::vector<uint8_t> lastCb0, lastCb1;

    // ===== shaders (gpu_hw_metal_shaders.mm) — slice 2 =====
    struct ArgField { std::string type, name; uint32_t offset; };        // one row of layout.tsv, binding_table
    struct Ps {
        uint64_t textKey = 0;                        // Fnv(PixelShaderHlsl(generic)) = ps_<KEY> (D3417)
        id<MTLFunction> fn = nil;                    // "shader_main"
        std::vector<ArgField> args;                  // binding_table fields in offset order
        uint32_t argBytes = 0, cbCount = 0;          // binding_table size; cbuffer_table entries (cb0, cb1)
        uint32_t slots = 0, cubeSlots = 0, constantRows = 0;   // from the translation (D2892-2904)
        bool failed = false;
    };
    // Generic variant only in M1 (Depth / ResolvedColor / no-discard / texture-state variants: M3). Directory:
    // $TTT2_METAL_CACHE, else <shader cache dir>/metal. A miss logs "[gpu-metal] ps cache miss <KEY>" once per
    // key, counts it, and returns nullptr (Draw returns false, as D3D11 when a PS is not ready, D390).
    const Ps* PixelShader(const std::vector<uint32_t>& code);
    // Pipeline for (ps, interps, colour format or Invalid, depth format or Invalid, blend word, writeMask).
    id<MTLRenderPipelineState> Pipeline(const Ps& ps, uint32_t interps, MTLPixelFormat color, MTLPixelFormat depth,
                                        uint32_t blend, uint32_t writeMask);
    id<MTLDepthStencilState> DepthState(const DrawState& st);           // D4305-4339

    // ===== textures (gpu_hw_metal_textures.mm) — slice 3; the slice-2 harness provides its own =====
    struct Bound { id<MTLTexture> tex = nil; uint32_t swizzle = 0x688, mode = 0, width = 1, height = 1;
                   float scaleX = 1, scaleY = 1; uint32_t slices = 1, levels = 1; uint64_t mip = 0;
                   uint32_t info[4] = {2, 2, 0, 0}; };   // D2197 + MipSampling(ft, levels) (D730) + SamplerInfo(ft) (D709)
    Bound BindTexture(const uint32_t* regs, uint32_t slot, bool array);  // D4443-4569
    // Bound.info is SamplerInfo(DecodeTexture(regs, slot)) BEFORE the cube clamp; draw applies D711 itself.
    // Sampler for a slot (the slot is only a harness hook; the real one keys on info + mip, D2673-2685).
    // The three airconv fields: sN, cube_sN (D2 clamp variant, BINDINGS.md), meta_sN = LOD-bias float bits.
    struct SamplerSet { id<MTLSamplerState> s = nil, cube = nil; float lodBias = 0; };
    SamplerSet Sampler(uint32_t slot, const uint32_t info[4], uint64_t mip);

    // ===== surfaces (gpu_hw_metal_surfaces.mm) — slice 4; the slice-2 harness provides its own =====
    MtlColorSurface& Color(uint32_t key);                                // D3858-3863
    MtlDepthSurface& Depth(uint32_t base);                               // D3865
    void Grow(MtlColorSurface& s, uint32_t key, uint32_t w, uint32_t h, float scale);   // D4027-4047
    void Grow(MtlDepthSurface& s, uint32_t base, uint32_t w, uint32_t h, float scale);  // D4048-4063
    // Page-sequence tracker over guest memory (D2931 `DirtyTracker dirty_{g_base}`), created on first use.
    DirtyTracker& Dirty();
    // Synchronous readback of one resolve into guest memory (D4795-4805: stage, wait, scatter tiled words).
    // textures calls it for an overlapping non-exact view of resolved bytes (D4525-4535); WriteBackRange uses it
    // in M1 (the async staging of D1100-1190 / D4674-4791 is M2).
    void WriteBack(uint32_t dest, MtlResolved& r);
};

// Test hook (core): lets a harness without the game show a texture through the real present path.
// Not used by gpu_render.cpp.
bool MetalPresentForTest(Backend& b, id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap);
void MetalSetSdlWindowForTest(Backend& b, SDL_Window* window);   // harness has no host_window::Window
Backend::Impl& MetalImplForTest(Backend& b);

} // namespace ttt2_hw
