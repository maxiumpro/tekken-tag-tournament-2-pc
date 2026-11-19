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
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct SDL_Window;

namespace ttt2_hw { class DirtyTracker; }   // gpu_hw_dirty_tracker.h (needs host_vm: only the game links it)
namespace ttt2_hw { struct VsTranslation; }  // gpu_hw_translate.h (needs gpu_render.cpp's decoders: only shaders.mm), M2b

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

    // ----- stats (M2a; core owns the report, every file bumps its own counters) -----
    // Same names and meaning as D3D11's stats_ (D5062-5090); a counter the Metal path has no source for stays 0
    // and is still printed, so bench/analizar*.py parse the line unchanged.
    struct Stats {
        uint64_t draws = 0, triangles = 0, skippedDraws = 0, emptyDraws = 0;       // draw (D382-398, D855)
        uint64_t resolves = 0, depthResolves = 0, writebacks = 0, stagedReadbacks = 0;  // surfaces (D959, D1006, D1153)
        uint64_t presents = 0;                                                     // core (D1451)
        uint64_t compiles = 0, diskHits = 0, warmHits = 0, pendingSkips = 0;       // shaders: diskHits = metallib loads
        uint64_t uploads = 0, uploadBytes = 0, resolvedHits = 0, depthHits = 0;    // textures
        uint64_t gpuVsDraws = 0, vertexDataBytes = 0, ringWaits = 0;               // M2b (0 in M1)
        double ringWaitMs = 0;
        uint64_t resolveCreates = 0, resolvePoolHits = 0;                          // surfaces
        uint64_t shadowSkips = 0;
    };
    Stats stats, reported;
    size_t statShaders = 0, statTextures = 0, statVertexShaders = 0;   // cache sizes, set by their owners
    std::chrono::steady_clock::time_point lastStats = std::chrono::steady_clock::now();
    // Called at the end of Draw and Resolve (D856, D1090): every >= 5 s prints the D5080 "[gpu-hw] %.1fs: draws=..."
    // line byte-compatible with D3D11, then the D5044 non-Windows "frame cost:" line. GPU ms per frame = span from
    // the first GPUStartTime to the last GPUEndTime of the command buffers committed between two presents; plus a
    // "[gpu-metal]   gpu busy" line with the summed (GPUEndTime - GPUStartTime). NOT the same quantity as D3D11's
    // (DXVK submits several times per frame and its span includes GPU waits for the CPU, up to the whole frame when
    // CPU-bound; M1 commits ~once per frame, so span ~= busy). Compare backends on this number only when GPU-bound.
    void MaybeReportStats();
    // Submit/Cmd tag each command buffer with the present sequence it belongs to; the completed handler folds its
    // GPU times into that frame (handler thread: under statsMutex only).
    std::mutex statsMutex;
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

    // ===== vertex shaders (gpu_hw_metal_shaders.mm) — M2b =====
    // A converted vertex shader: vg_<KEY>.metallib + vg_<KEY>.layout.tsv in the metallib cache directory (the PS
    // one), KEY = Fnv(VertexShaderHlsl(TranslateVertexShader(code))) = D3D11's vg_<KEY>.dxbc (D3352-3355, D3417-3424).
    // Vertex function "shader_main", no stage_in (vertex id only):
    //   vertex buffer(0) = cbuffer_table: {cb0 @0, cb1 @8} (16 B) or {cb1 @0} (8 B, a shader reading no constant).
    //     cb0 = XenosVsConstants (the first constantRows rows of regs + 0x4000, c[512]), cb1 = XenosVsState = VsStateCB
    //     (2256 B). One gpuAddress each.
    //   vertex buffer(1) = binding_table, 16 B: t0 @0 = gpuAddress of the raw vertex data (host-order dwords,
    //     PackVertexFetches with baseDword 0: vfRegion counts from t0), meta_t0 @8 = its length in bytes
    //     (BINDINGS.md "SRV buffer": gpuAddress, byteLength). The data buffer needs useResource (vertex stage).
    //   Outputs: position + user(reg1_0)..user(reg16_0) = the PS inputs TEXCOORD0..15 (checked in 709/709 metallibs).
    // One entry per microcode key (VsKey, D3340): each run-time copy has its own fetch parameters; copies with one
    // HLSL text share one loaded function (D3356-3358 vsByText_). A key missing from the cache logs
    // "[gpu-metal] vs cache miss <KEY>" once per text key and the shader stays on the CPU (VertexShader -> null).
    struct Vs : VertexShaderHandle {
        uint64_t textKey = 0;                        // vg_<KEY>
        id<MTLFunction> fn = nil;                    // "shader_main" (vertex)
        std::vector<ArgField> args;                  // binding_table fields in offset order (t0, meta_t0)
        uint32_t argBytes = 0, cbCount = 0;          // binding_table size; cbuffer_table entries (2 = cb0 cb1, 1 = cb1)
        uint32_t constantRows = 0;                   // VsTranslation::constantRows (D596)
        uint32_t vfDesc[32][4] = {};                 // VsFetchDescriptors(t) (D619): depends on the translation only
        std::shared_ptr<const VsTranslation> t;      // opaque outside shaders.mm
    };
    // D472-500 without the TTT2_GPU_VS gate (Backend::VertexShader applies it, D325, D473). Null = CPU vertex stage.
    Vs* VertexShaderFor(const std::vector<uint32_t>& code);
    // D522-523 VsFetchRanges(vs.t, minIndex, maxIndex, lo, hi, used).
    void FetchRanges(const Vs& vs, uint32_t minIndex, uint32_t maxIndex, int64_t lo[96], int64_t hi[96], bool used[96]);
    // Pipeline with the converted VS instead of the fixed CPU-vertex one (no vertex descriptor); colour / depth /
    // blend / write mask exactly as Pipeline(ps, interps, ...).
    id<MTLRenderPipelineState> Pipeline(const Vs& vs, const Ps& ps, MTLPixelFormat color, MTLPixelFormat depth,
                                        uint32_t blend, uint32_t writeMask);

    // ===== readback (surfaces.mm) — M2c =====
    // Staged CPU readback of resolves (D1068-1190, D4674-4791) without blocking the CPU: a resolve's copy goes to a
    // shared buffer on frameCmd, frameCmd is committed at once (Submit(false) = the D4705 Flush), and a CPU fence fed
    // by that command buffer's completed handler says when it landed. The scatter into guest memory always runs under
    // g_backendMutex, after the fence passed (D1126-1150, D4766-4786). Everything here is under the lock except
    // ReadbackFence's own fields (completed handlers on a Metal thread; the PendingCopyWaiter function, lock-free).
    struct ReadbackFence;                            // surfaces.mm: mutex + condvar + done / failed (D3D11 readbackFence_)
    // The D2139-2163 Resolved fields MtlResolved has no room for, one per resolve destination (and one per small
    // snapshot). Reset whenever Resolve makes a new texture for the destination (D976 r = Resolved{}).
    struct Readback {
        id<MTLBuffer> staging = nil;                 // shared, pitch * height * 4 B, tight rows (D4693-4699 r.staging)
        bool staged = false;                         // a copy to `staging` is queued (D2154)
        uint64_t stagedSeq = 0;                      // the resolve's seq when it was queued (D2155)
        uint64_t stagedFence = 0;                    // fence value reached once the copy landed (D2157)
        std::chrono::steady_clock::time_point stagedAt{};   // D2156
        uint64_t generation = 0, cpuReadSerial = 0;  // D2143 (generation: a new resolve texture, D4178)
        std::vector<uint32_t> pixels;                // D2161 readbackPixels (logical size, tight rows)
        std::vector<uint32_t> tiledToLinear;         // D2162 (BuildResolvedTileLookup, once per resolve shape)
    };
    std::unordered_map<uint32_t, Readback> readbacks;   // key = dest (as `resolved`)
    // D4711-4713 smallReadbacks_ / smallPending_: a snapshot of a tiny resolve owning its own staging copy until
    // FinishSmallWritebacks consumes it; `pending` replaces the smallPending_ set.
    struct SmallReadback { MtlResolved r; Readback rb; bool pending = false; };
    std::unordered_map<uint32_t, SmallReadback> smallReadbacks;   // key = dest
    std::shared_ptr<ReadbackFence> readbackFence;    // made at the first staged copy
    uint64_t readbackFenceValue = 0;                 // D3D11 readbackFenceValue_: last value handed to a copy
    uint64_t pendingFence = 0;                       // D3D11 pendingFence_: what PendingCopyWaiter waits for
    // D4674-4710: resample (scaled) + copy to rb.staging on frameCmd, fence value, Submit(false); stagedReadbacks++.
    void StageReadback(MtlResolved& r, Readback& rb);
    // D4845-4860: the staged copy into `out` (width * height words). noWait: false while the copy has not landed;
    // otherwise waits. Throws if the copy's command buffer failed.
    bool ReadStaging(const Readback& rb, uint32_t width, uint32_t height, std::vector<uint32_t>& out, bool noWait);
    void CancelOverlappingSmallReadbacks(uint32_t dest, uint32_t size);              // D4716-4728
    void StageSmallReadback(uint32_t dest, const MtlResolved& r, const Readback& rb); // D4729-4748
    // D4749-4791: scatter the small copies that landed. Resolve calls it (D957); D3D11 also calls it at Present
    // (D1265): here FrontTexture does, at its top (core.mm's Present calls FrontTexture).
    void FinishSmallWritebacks();
};

// Test hook (core): lets a harness without the game show a texture through the real present path.
// Not used by gpu_render.cpp.
bool MetalPresentForTest(Backend& b, id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap);
void MetalSetSdlWindowForTest(Backend& b, SDL_Window* window);   // harness has no host_window::Window
Backend::Impl& MetalImplForTest(Backend& b);

} // namespace ttt2_hw
