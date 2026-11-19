// R4 Metal backend, surfaces (gpu_hw_metal_impl.h "surfaces" section, M1 slice 4): EDRAM-lite colour / depth
// surfaces (Color, Depth, Grow), Resolve, the CPU readback (synchronous WriteBack; M2c: the staged readback of
// the impl.h "readback" section, WriteBackRange / PendingCopyWaiter), the front-buffer choice (FrontTexture),
// the dirty tracker, and the diagnostics entry points (M2: fail by name).
// Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11 backend stays the reference).
//
// Content rule shared with draw.mm: Draw opens its passes with MTLLoadActionLoad, so every texture made here
// leaves this file fully defined (new surface area 0 / depth 1.0 + stencil 0, new resolve textures 0), through
// passes of its own on Cmd() after EndPass(). Two things here commit the frame's command buffer: WriteBack's
// own copy (Submit(true): it must wait for the GPU - see the note on WriteBack about calls from inside a Draw),
// and StageReadback (Submit(false), never waits: the D4705 Flush, so the copy starts now and its fence can pass
// before the next present).
//
// M2c staged readback (D1068-1190, D4674-4860): a copy to a shared buffer + a CPU fence that the command
// buffer's completed handler advances (Backend::Impl::ReadbackFence below; one value per staged copy, so the
// fence is "every value <= done landed" even if command buffers finish out of order). The scatter into guest
// memory runs under the backend lock only, after the fence passed. Not ported (each named):
//   TTT2_READBACK_FENCE=0 (D270, no fence: Map polling): the fence is always on here.
//   D4768-4773 small-copy event query: the fence value already says the copy landed.
//   D4759 sourceDepth: D3D11 never sets it (always ~0u), so the comparison is always equal.
//   lastStagedAgeMs_ / lastMapWaitMs_, frame_pacing_probe, writebackCauses_: diagnostics.
//
// Not ported in M1 (each named):
//   UpdateScale / Rescale (D956, D3960-3973, D4123-4157): core fixes the render scale at Initialize.
//   Native-size bake surfaces (D784-813, D3877-3935, D992-997 resolveDests_/UpdateNative, D1110-1117): M2.
//   Shadow-map skipping (D962 NoteShadowResolve, D1009-1013 ShadowMapOff, D2201-2244): M2; depth resolves
//     always copy the surface.
//   HW toggles (D1007 / D1047 toggle_ & 4, D1085 toggle_ & 8), census / scale logs (D983-991, D999-1002),
//     stats_.*: diagnostics.
//
// State the frozen Impl has no fields for lives in SurfaceState below (one Backend exists, Get()).
// Game symbols: g_base (kernel_shared.h, declared here directly: that header pulls in the PPC headers),
// DecodeFrontBufferUploadSurface (gpu_hw_texture_upload.h, body in gpu_render.cpp), GpuTiledOffset2D
// (gpu_render.h, through gpu_hw_writeback.h), the write-watch functions (gpu_hw_dirty_tracker.h), and
// MetalDropTexturesOverlapping (gpu_hw_metal_textures.mm, slice 3).

#include "gpu_hw_metal_impl.h"
#include "gpu_hw_dirty_tracker.h"
#include "gpu_hw_texture_upload.h"
#include "gpu_hw_writeback.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern uint8_t* g_base;   // kernel_shared.h: global guest memory base

namespace ttt2_hw {

// gpu_hw_metal_textures.mm (slice 3): D4431-4440, evict guest-decoded textures overlapping a resolve.
void MetalDropTexturesOverlapping(Backend::Impl& m, uint32_t start, uint32_t size);

// D3D11 readbackFence_ (ID3D11Fence + Signal), on the CPU: StageReadback hands each copy the next value; the
// completed handler of the command buffer holding that copy (a Metal thread) records it. Only `mutex` guards the
// fields: the handler and the PendingCopyWaiter function (no backend lock) touch nothing else.
struct Backend::Impl::ReadbackFence {
    std::mutex mutex;
    std::condition_variable landed;
    uint64_t done = 0;                     // every value <= done has landed
    std::unordered_set<uint64_t> early;    // landed values > done + 1 (a later command buffer finished first)
    bool failed = false;                   // a copy's command buffer failed: nothing may be scattered from now on
};

namespace {

using Impl = Backend::Impl;

// ---------------------------------------------------------------- surface shaders (MSL, compiled at run time)
const char* const kSurfaceMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 pos [[position]]; };

// D2787-2788 BlitVs: full-screen triangle.
vertex VOut surf_vs(uint id [[vertex_id]])
{
    float2 p = float2(float((id << 1) & 2u), float(id & 2u));
    VOut o;
    o.pos = float4(p * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

// D2713-2717 BlitColor: texel for texel. HLSL Load past the texture returns 0 and D1038 relies on it (the area
// no draw reached reads as the cleared store); MSL read past the texture is undefined, so check.
fragment float4 blit_color(VOut in [[stage_in]], texture2d<float> src [[texture(0)]])
{
    uint2 p = uint2(in.pos.xy);
    if (p.x >= src.get_width() || p.y >= src.get_height()) return float4(0.0);
    return src.read(p);
}
// D2713-2715: 16_16_FLOAT destination keeps red and green.
fragment float4 blit_red_green(VOut in [[stage_in]], texture2d<float> src [[texture(0)]])
{
    uint2 p = uint2(in.pos.xy);
    if (p.x >= src.get_width() || p.y >= src.get_height()) return float4(0.0);
    float4 c = src.read(p);
    return float4(c.r, c.g, 0.0, 0.0);
}
// D2751-2755 BlitDepth: pixel p of the resolve = sample p x samp of the surface; past it the cleared far depth.
fragment float4 blit_depth(VOut in [[stage_in]], depth2d<float> src [[texture(0)]], constant uint2& samp [[buffer(0)]])
{
    uint2 p = uint2(in.pos.xy) * samp;
    float d = (p.x < src.get_width() && p.y < src.get_height()) ? src.read(p) : 1.0;
    return float4(d, 0.0, 0.0, 0.0);
}

// D4076-4083 Resample: read src at uv = pixel centre * k.
constexpr sampler lin(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler pnt(coord::normalized, filter::nearest, address::clamp_to_edge);
fragment float4 resample_linear(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant float4& k [[buffer(0)]])
{ return src.sample(lin, in.pos.xy * k.xy, level(0.0)); }
fragment float4 resample_point(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant float4& k [[buffer(0)]])
{ return src.sample(pnt, in.pos.xy * k.xy, level(0.0)); }
// D4080-4082: depth through SV_Depth (point reads: D4058 / D4060 pass linear = false).
struct DepthOut { float depth [[depth(any)]]; };
fragment DepthOut resample_depth(VOut in [[stage_in]], depth2d<float> src [[texture(0)]], constant float4& k [[buffer(0)]])
{
    DepthOut o;
    o.depth = src.sample(pnt, in.pos.xy * k.xy, level(0.0));
    return o;
}
)MSL";

enum Fn : uint32_t { kBlitColor, kBlitRedGreen, kBlitDepth, kResampleLinear, kResamplePoint, kResampleDepth, kFnCount };
const char* const kFnNames[kFnCount] = { "blit_color", "blit_red_green", "blit_depth", "resample_linear", "resample_point",
                                         "resample_depth" };

constexpr MTLPixelFormat kDepthFormat = MTLPixelFormatDepth32Float_Stencil8;   // D3996-4016

struct SurfaceState {
    id<MTLLibrary> library = nil;
    id<MTLFunction> vs = nil;
    id<MTLFunction> fns[kFnCount] = {};
    std::unordered_map<uint64_t, id<MTLRenderPipelineState>> pipelines;   // (fn << 32) | target format
    id<MTLDepthStencilState> depthAlways = nil;                         // D4091-4093
    // D4165-4175 resolvePool_: textures a destination gave up, by (w, h, format), at most 4 each.
    std::unordered_map<uint64_t, std::vector<id<MTLTexture>>> resolvePool;
    // D2139-2163 fields MtlResolved does not carry: readbackHot (the CPU read this destination before), kept
    // across texture re-creation (D975-979); resolveRevision_ (D2930).
    std::unordered_set<uint32_t> readbackHot;
    uint64_t resolveRevision = 0;
    uint64_t resolveGeneration = 0;                                     // D2930 resolveGeneration_ (D4178)
    // The logical-size target of a scaled resolve's readback (D4680-4689 r.logical), one per shape here, shared by
    // every copy of that shape, staged ones included: each copy is "resample into it, then blit it out" on the one
    // queue, and Metal's hazard tracking orders the next resample after the previous blit. The synchronous
    // WriteBack's own readback buffer (staged copies have one each, Impl::Readback::staging).
    std::unordered_map<uint64_t, id<MTLTexture>> logical;
    id<MTLBuffer> readback = nil;
    id<MTLTexture> cpuFront = nil;                                      // D4927-4933 cpuFront_
};
SurfaceState& St() { static SurfaceState* s = new SurfaceState; return *s; }   // leaked with the Backend (D5140)

std::string Describe(NSError* e)
{
    if (!e) return "unknown error";
    const char* s = e.localizedDescription.UTF8String;
    return std::string(s ? s : "unknown error") + " (code " + std::to_string(long(e.code)) + ")";
}

// kernel_shared.h:11 GWrite32 (not included: it pulls in the PPC headers): big-endian guest store, address 0 skipped.
inline void GWrite32(uint8_t* base, uint32_t ga, uint32_t v)
{
    if (ga) *reinterpret_cast<uint32_t*>(base + ga) = __builtin_bswap32(v);
}

// D3874-3875: one rounding for every size and edge; textures at most 16384 (D3D11 and Apple GPUs alike).
uint32_t Px(double logical, float scale) { return uint32_t(std::floor(logical * scale + 0.5)); }
uint32_t AllocPx(double logical, float scale) { return std::clamp<uint32_t>(Px(logical, scale), 1, 16384); }

// D3977-3981: surface extents grow in steps, never past the EDRAM store.
uint32_t Extent(uint32_t need, uint32_t have, uint32_t step, uint32_t limit)
{
    need = std::min(std::max(need, 1u), limit);                                                 // D3979
    return need <= have ? have : std::min((need + step - 1) / step * step, limit);              // D3980
}

// D4341-4344
uint32_t ResolveBytes(uint32_t pitch, uint32_t height) { return ((pitch + 31) & ~31u) * ((height + 31) & ~31u) * 4; }

// D3841-3853 ColorDxgiFormat as Metal formats.
MTLPixelFormat ColorMtlFormat(uint32_t format)
{
    switch (format)
    {
    case 2: case 10:  return MTLPixelFormatRGB10A2Unorm;      // D3845: 2_10_10_10
    case 4: case 6:   return MTLPixelFormatRG16Float;         // D3846: 16_16, 16_16_FLOAT
    case 3: case 5: case 7: case 12:
                      return MTLPixelFormatRGBA16Float;       // D3847-3848
    case 14:          return MTLPixelFormatR32Float;          // D3849
    case 15:          return MTLPixelFormatRG32Float;         // D3850
    default:          return MTLPixelFormatBGRA8Unorm;        // D3851: 8_8_8_8, 8_8_8_8_GAMMA
    }
}

// D4119: formats that may not filter: resample with point reads.
bool Filterable(MTLPixelFormat f) { return f != MTLPixelFormatR32Float && f != MTLPixelFormatRG32Float; }

// Metal needs one sample count on every attachment of a pass; TTT2 renders one sample per pixel everywhere
// (D3988 / D4005 SampleDesc.Count = 1). Anything else is a bug: fail by name instead of a validation abort.
void CheckSamples(id<MTLTexture> t, const char* what)
{
    if (t && t.sampleCount != 1)
        throw Failure(std::string("metal: ") + what + ": texture with " + std::to_string(unsigned(t.sampleCount)) +
                      " samples; TTT2 renders 1 sample per pixel and a Metal pass needs one sample count on every attachment");
}

void EnsureLibrary(Impl& m)
{
    SurfaceState& s = St();
    if (s.library) return;
    NSError* err = nil;
    id<MTLLibrary> lib = [m.device newLibraryWithSource:[NSString stringWithUTF8String:kSurfaceMsl]
                                                options:[MTLCompileOptions new] error:&err];
    if (!lib) throw Failure("metal: surface shaders did not compile: " + Describe(err));
    id<MTLFunction> vs = [lib newFunctionWithName:@"surf_vs"];
    if (!vs) throw Failure("metal: surface shader function missing: surf_vs");
    for (uint32_t i = 0; i < kFnCount; ++i)
    {
        s.fns[i] = [lib newFunctionWithName:[NSString stringWithUTF8String:kFnNames[i]]];
        if (!s.fns[i]) throw Failure(std::string("metal: surface shader function missing: ") + kFnNames[i]);
    }
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];                            // D4091-4093 depthAlways_
    dd.depthCompareFunction = MTLCompareFunctionAlways;
    dd.depthWriteEnabled = YES;
    s.depthAlways = [m.device newDepthStencilStateWithDescriptor:dd];
    if (!s.depthAlways) throw Failure("metal: depth-always state");
    s.vs = vs;
    s.library = lib;
}

// `target` is the colour format, or kDepthFormat for kResampleDepth (depth + stencil attachment, no colour).
id<MTLRenderPipelineState> SurfacePipeline(Impl& m, Fn fn, MTLPixelFormat target)
{
    EnsureLibrary(m);
    SurfaceState& s = St();
    const uint64_t key = (uint64_t(fn) << 32) | uint64_t(target);
    if (auto it = s.pipelines.find(key); it != s.pipelines.end()) return it->second;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = s.vs;
    d.fragmentFunction = s.fns[fn];
    d.rasterSampleCount = 1;
    if (fn == kResampleDepth)
    {
        d.depthAttachmentPixelFormat = kDepthFormat;
        d.stencilAttachmentPixelFormat = kDepthFormat;   // as Impl::Pipeline: a D32F_S8 pass binds both
    }
    else d.colorAttachments[0].pixelFormat = target;
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [m.device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!p)
        throw Failure(std::string("metal: surface pipeline ") + kFnNames[fn] + " (format " + std::to_string(long(target)) +
                      "): " + Describe(err));
    return s.pipelines[key] = p;
}

id<MTLTexture> NewTexture(Impl& m, MTLPixelFormat format, uint32_t w, uint32_t h, const char* what)
{
    MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    t.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;   // D3987 / D4001 / D4183 (RT|SRV)
    t.storageMode = MTLStorageModePrivate;
    t.sampleCount = 1;                                                    // D3988 / D4005 / D4184
    id<MTLTexture> tex = [m.device newTextureWithDescriptor:t];
    if (!tex) throw Failure(std::string("metal: ") + what + " (" + std::to_string(w) + "x" + std::to_string(h) + ")");
    CheckSamples(tex, what);
    return tex;
}

// One pass on the frame's command buffer (the caller did EndPass()). A depth target is the combined
// depth/stencil texture, cleared to 1.0 / 0 when `load` is Clear.
id<MTLRenderCommandEncoder> BeginPass(Impl& m, id<MTLTexture> color, id<MTLTexture> depth, MTLLoadAction load,
                                      MTLClearColor clear, const char* what)
{
    CheckSamples(color, what);
    CheckSamples(depth, what);
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    if (color)
    {
        rp.colorAttachments[0].texture = color;
        rp.colorAttachments[0].loadAction = load;
        rp.colorAttachments[0].clearColor = clear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (depth)
    {
        rp.depthAttachment.texture = depth;
        rp.depthAttachment.loadAction = load;
        rp.depthAttachment.clearDepth = 1.0;                 // D4015: the CPU store starts at 1.0
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        rp.stencilAttachment.texture = depth;
        rp.stencilAttachment.loadAction = load;
        rp.stencilAttachment.clearStencil = 0;               // D4015
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> enc = [m.Cmd() renderCommandEncoderWithDescriptor:rp];
    if (!enc) throw Failure(std::string("metal: render command encoder (") + what + ")");
    return enc;
}

// A pass that only clears (its load action does the work).
void ClearPass(Impl& m, id<MTLTexture> color, id<MTLTexture> depth, MTLClearColor clear, const char* what)
{
    [BeginPass(m, color, depth, MTLLoadActionClear, clear, what) endEncoding];
}

// The full-screen triangle over [0, w) x [0, h) of the target (clamped to it; the shaders read window
// positions, so the clamp changes nothing they compute), source at texture(0), `bytes` at buffer(0).
void FullScreen(id<MTLRenderCommandEncoder> enc, id<MTLRenderPipelineState> ps, id<MTLTexture> target, uint32_t w,
                uint32_t h, id<MTLTexture> src, const void* bytes, size_t length, id<MTLDepthStencilState> ds = nil)
{
    w = std::min<uint32_t>(w, uint32_t(target.width));
    h = std::min<uint32_t>(h, uint32_t(target.height));
    if (w && h)
    {
        [enc setRenderPipelineState:ps];
        if (ds) [enc setDepthStencilState:ds];
        const MTLViewport vp{ 0.0, 0.0, double(w), double(h), 0.0, 1.0 };
        const MTLScissorRect sc{ 0, 0, w, h };
        [enc setViewport:vp];
        [enc setScissorRect:sc];
        [enc setFragmentTexture:src atIndex:0];
        if (bytes) [enc setFragmentBytes:bytes length:length atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [enc endEncoding];
}

// Texture-to-texture copy of [0, w) x [0, h) at the origin (CopySubresourceRegion); a zero-size copy is skipped.
void CopyRegion(Impl& m, id<MTLTexture> src, id<MTLTexture> dst, uint32_t w, uint32_t h)
{
    w = std::min({ w, uint32_t(src.width), uint32_t(dst.width) });
    h = std::min({ h, uint32_t(src.height), uint32_t(dst.height) });
    if (!w || !h) return;
    id<MTLBlitCommandEncoder> blit = [m.Cmd() blitCommandEncoder];
    if (!blit) throw Failure("metal: blit command encoder");
    [blit copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                toTexture:dst destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
}

// D4070-4117 Resample into a colour target over [0, w) x [0, h), loading or clearing (0) it first.
void ResampleColor(Impl& m, id<MTLTexture> src, id<MTLTexture> dst, MTLLoadAction load, uint32_t w, uint32_t h,
                   float kx, float ky, bool linear, const char* what)
{
    const float k[4] = { kx, ky, 0.0f, 0.0f };                                                    // D4097
    id<MTLRenderPipelineState> ps = SurfacePipeline(m, linear ? kResampleLinear : kResamplePoint, dst.pixelFormat);
    FullScreen(BeginPass(m, dst, nil, load, MTLClearColorMake(0, 0, 0, 0), what), ps, dst, w, h, src, k, sizeof(k));
}

// D4070-4117 Resample into a depth target through the fragment depth (depthAlways_), the target cleared to
// 1.0 / stencil 0 first (the stencil is not carried, as D3D11's SV_Depth resample).
void ResampleDepth(Impl& m, id<MTLTexture> src, id<MTLTexture> dst, uint32_t w, uint32_t h, float kx, float ky)
{
    const float k[4] = { kx, ky, 0.0f, 0.0f };
    id<MTLRenderPipelineState> ps = SurfacePipeline(m, kResampleDepth, kDepthFormat);
    FullScreen(BeginPass(m, nil, dst, MTLLoadActionClear, MTLClearColorMake(0, 0, 0, 0), "depth resample"), ps, dst, w, h,
               src, k, sizeof(k), St().depthAlways);
}

void LogGrowth(const char* kind, uint32_t key, uint32_t w, uint32_t h, float scale)   // D4017-4023
{
    static int logged = 0;
    if (logged++ < 64)
        std::fprintf(stderr, "[gpu-metal] %s surface %05X covers %ux%u samples (%ux%u at scale %.2f)\n", kind, key, w, h,
                     AllocPx(w, scale), AllocPx(h, scale), double(scale));
}

uint64_t PoolKey(uint32_t w, uint32_t h, MTLPixelFormat f) { return (uint64_t(w) << 40) | (uint64_t(h) << 16) | uint64_t(f); }   // D4166

// D4167-4175 RetireResolvedTexture.
void RetireResolved(Impl& m, MtlResolved& r)
{
    static const bool on = [] { const char* e = std::getenv("TTT2_RESOLVE_POOL"); return !(e && *e == '0'); }();   // D4169
    if (!on || !r.tex || r.tex == m.lastFront) return;                                             // D4170: the screenshot source keeps its content
    auto& v = St().resolvePool[PoolKey(uint32_t(r.tex.width), uint32_t(r.tex.height), r.tex.pixelFormat)];   // D4172-4173
    if (v.size() < 4) v.push_back(r.tex);                                                          // D4174
}

// D4176-4203 MakeResolvedTexture (at the source surface's scale); the caller clears it in its first pass (D4200-4201).
void MakeResolved(Impl& m, MtlResolved& r, float scale)
{
    const uint32_t w = AllocPx(r.pitch, scale), h = AllocPx(r.height, scale);                     // D4180
    const MTLPixelFormat f = r.depth ? MTLPixelFormatR32Float : r.rg16f ? MTLPixelFormatRG16Float : MTLPixelFormatBGRA8Unorm;   // D4182
    r.tex = nil;
    auto pooled = St().resolvePool.find(PoolKey(w, h, f));                                         // D4186
    if (pooled != St().resolvePool.end() && !pooled->second.empty())
    {
        r.tex = pooled->second.back();                                                             // D4189-4191
        pooled->second.pop_back();
        ++m.stats.resolvePoolHits;                                                                 // D4191 (reused)
    }
    else { r.tex = NewTexture(m, f, w, h, "resolve texture"); ++m.stats.resolveCreates; }          // D4196-4198 (new)
    r.scale = scale;                                                                               // D4202
}

bool IsHot(uint32_t dest) { return St().readbackHot.count(dest) != 0; }
void SetHot(uint32_t dest) { St().readbackHot.insert(dest); }

bool Overlaps(uint32_t a, uint32_t aSize, uint32_t b, uint32_t bSize)
{
    return uint64_t(a) < uint64_t(b) + bSize && uint64_t(b) < uint64_t(a) + aSize;
}

// D4676-4701: resample a scaled resolve to its logical size (as ReadLogical), then copy pitch x height texels,
// tight rows (4 B per texel in all three resolve formats), into `buffer` - all on frameCmd, nothing waits. The
// caller did EndPass() and sized `buffer`.
void EncodeLogicalCopy(Impl& m, const MtlResolved& r, id<MTLBuffer> buffer)
{
    SurfaceState& st = St();
    __strong id<MTLTexture> source = r.tex;                                                           // D4678
    if (uint32_t(r.tex.width) != r.pitch || uint32_t(r.tex.height) != r.height)                        // D4679
    {
        __strong id<MTLTexture>& logical = st.logical[PoolKey(r.pitch, r.height, r.tex.pixelFormat)];   // D4681-4688 r.logical
        if (!logical) logical = NewTexture(m, r.tex.pixelFormat, r.pitch, r.height, "readback logical target");
        ResampleColor(m, r.tex, logical, MTLLoadActionDontCare, r.pitch, r.height, 1.0f / float(r.pitch),
                      1.0f / float(r.height), !r.depth && Filterable(r.tex.pixelFormat), "readback resample");   // D4689-4690
        source = logical;                                                                              // D4691
    }
    const size_t rowBytes = size_t(r.pitch) * 4, bytes = rowBytes * r.height;
    if (buffer.length < bytes) throw Failure("metal: readback buffer smaller than the resolve");
    // D4700-4701 (CopySubresourceRegion to the staging texture) + D4853-4857 (Map): one copy to a shared buffer.
    id<MTLBlitCommandEncoder> blit = [m.Cmd() blitCommandEncoder];
    if (!blit) throw Failure("metal: blit command encoder");
    [blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(r.pitch, r.height, 1) toBuffer:buffer destinationOffset:0
   destinationBytesPerRow:rowBytes destinationBytesPerImage:bytes];
    [blit endEncoding];
}

// D4809-4820 Scatter: the logical-size pixels of `r` into guest memory (tiled, in the guest's byte and channel
// order, gpu_hw_writeback.h); the resolve is CPU-coherent from now on. Under the backend lock.
void Scatter(Impl& m, uint32_t dest, MtlResolved& r, std::span<const uint32_t> px)
{
    const ResolveWritebackInfo info{ dest, r.pitch, r.height, r.depth, r.depthFloat, r.rg16f, r.swapRB };   // D4813
    if (!ScatterResolvedWords(info, px, [](uint32_t address, uint32_t word) { GWrite32(g_base, address, word); }))   // D4814-4815
        throw Failure("metal: invalid resolved guest writeback");                                     // D4816
    m.Dirty().Sync(dest, r.size);                                                                      // D4817
    r.seq = m.Dirty().Current();                                                                       // D4818
    r.cpuCoherent = true;                                                                              // D4819
    ++m.stats.writebacks;                                                                              // D4820
}

// D154-163 HostWaitFence on the CPU fence: true once `value` landed, false after `timeout` (or if a copy failed:
// the next call under the lock throws). Takes the fence's mutex only.
bool WaitFence(Impl::ReadbackFence& f, uint64_t value, std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(f.mutex);
    f.landed.wait_for(lock, timeout, [&] { return f.failed || f.done >= value; });
    return !f.failed && f.done >= value;
}

} // namespace

// ================================================================ Backend::Impl (surfaces section)
// D3855-3863: key = EDRAM base | colour format << 12; each format its own surface. Grow makes the texture.
MtlColorSurface& Backend::Impl::Color(uint32_t key)
{
    auto [it, added] = colors.try_emplace(key);                                                    // D3860
    if (added) it->second.format = ColorMtlFormat((key >> 12) & 0xF);                              // D3861
    return it->second;                                                                             // D3862
}

MtlDepthSurface& Backend::Impl::Depth(uint32_t base) { return depths[base]; }                       // D3865

// D4027-4047: cover logical samples [0, w) x [0, h) at `scale`: re-created larger with the old content copied
// (same scale) or resampled (another scale); the new area starts at 0, as the CPU store does.
void Backend::Impl::Grow(MtlColorSurface& s, uint32_t key, uint32_t w, uint32_t h, float scale)
{
    w = Extent(w, s.extW, 128, SurfaceW); h = Extent(h, s.extH, 64, SurfaceH);                     // D4029
    if (s.tex && w == s.extW && h == s.extH && s.scale == scale) return;                           // D4030 (Draw keeps its encoder)
    @autoreleasepool {
        id<MTLTexture> old = s.tex;                                                                // D4031-4032
        const float oldScale = s.scale;                                                            // D4033
        const uint32_t oldW = AllocPx(s.extW, oldScale), oldH = AllocPx(s.extH, oldScale);         // D4034
        const uint32_t oldExtW = s.extW, oldExtH = s.extH;                                         // D4035
        s.extW = w; s.extH = h;                                                                    // D4036
        // D3982-3995 MakeColorTexture
        s.scale = scale;
        s.tex = NewTexture(*this, s.format, AllocPx(s.extW, scale), AllocPx(s.extH, scale), "colour surface");
        EndPass();                                                    // the passes below are encoders of their own
        if (old && oldScale == scale)                                                              // D4038
        {
            ClearPass(*this, s.tex, nil, MTLClearColorMake(0, 0, 0, 0), "colour surface clear");   // D3994
            CopyRegion(*this, old, s.tex, oldW, oldH);                                             // D4040-4041
        }
        else if (old)                                                                              // D4043-4045: one pass, clear + resample
        {
            const uint32_t rw = AllocPx(oldExtW, scale), rh = AllocPx(oldExtH, scale);
            ResampleColor(*this, old, s.tex, MTLLoadActionClear, rw, rh, 1.0f / float(rw), 1.0f / float(rh),
                          Filterable(s.format), "colour surface resample");
        }
        else ClearPass(*this, s.tex, nil, MTLClearColorMake(0, 0, 0, 0), "colour surface clear");  // D3994
        LogGrowth("colour", key, w, h, scale);                                                     // D4046
    }
}

// D4048-4063. The same-scale growth is a resample too, as D4058: a point read at the same texel grid copies
// the depth exactly, and the stencil restarts at 0 exactly as on D3D11 (the image-gate reference). A blit of
// the combined depth/stencil texture would keep the stencil; it is a possible M2 change, not parity.
void Backend::Impl::Grow(MtlDepthSurface& s, uint32_t base, uint32_t w, uint32_t h, float scale)
{
    w = Extent(w, s.extW, 128, SurfaceW); h = Extent(h, s.extH, 64, SurfaceH);                     // D4050
    if (s.tex && w == s.extW && h == s.extH && s.scale == scale) return;                           // D4051
    @autoreleasepool {
        id<MTLTexture> old = s.tex;                                                                // D4052
        const uint32_t oldW = AllocPx(s.extW, s.scale), oldH = AllocPx(s.extH, s.scale);           // D4053
        const uint32_t oldExtW = s.extW, oldExtH = s.extH;                                         // D4054
        const float oldScale = s.scale;                                                            // D4055
        s.extW = w; s.extH = h;                                                                    // D4056
        // D3996-4016 MakeDepthTexture: 32-bit float depth + the 8-bit stencil (D4002-4004).
        s.scale = scale;
        s.tex = NewTexture(*this, kDepthFormat, AllocPx(s.extW, scale), AllocPx(s.extH, scale), "depth surface");
        EndPass();
        if (old && oldScale == scale) ResampleDepth(*this, old, s.tex, oldW, oldH, 1.0f / float(oldW), 1.0f / float(oldH));   // D4058
        else if (old)                                                                              // D4059-4061
        {
            const uint32_t rw = AllocPx(oldExtW, scale), rh = AllocPx(oldExtH, scale);
            ResampleDepth(*this, old, s.tex, rw, rh, 1.0f / float(rw), 1.0f / float(rh));
        }
        else ClearPass(*this, nil, s.tex, MTLClearColorMake(0, 0, 0, 0), "depth surface clear");   // D4015
        LogGrowth("depth", base, w, h, scale);                                                     // D4062
    }
}

// D2931 `DirtyTracker dirty_{g_base}`: one per process (one Backend), made at the first use.
DirtyTracker& Backend::Impl::Dirty()
{
    static DirtyTracker* d = new DirtyTracker(g_base);
    return *d;
}

// D4795-4805 WriteBack, synchronous: the pixels a partial WriteBackRange already read, else the staged copy of this
// very resolve (waits for it, no new copy, no commit), else a copy now (resample a scaled resolve to its logical
// size, D4680-4691, copy it to CPU-visible memory, Submit(true): the D4802 ReadLogical); then Scatter.
//
// Submit(true) commits the frame's command buffer and waits. Called from textures.mm's BindTexture inside a Draw,
// that command buffer holds the draw's arena (vertex / index slices allocated before the texture loop), which
// returns to the free list here; draw.mm (slice 3 fix) copies vb / ib into the new command buffer's arena when
// frameCmd changed across the texture loop, before any new Alloc. The first two sources never change frameCmd.
void Backend::Impl::WriteBack(uint32_t dest, MtlResolved& r)
{
    if (!r.tex) throw Failure("metal: WriteBack of a resolve without a texture");
    @autoreleasepool {
        Initialize();
        Readback& rb = readbacks[dest];
        ++rb.cpuReadSerial;                                                                        // D4797
        std::vector<uint32_t> px;                                                                  // D4799
        std::span<const uint32_t> words;
        if (!rb.pixels.empty()) { px = std::move(rb.pixels); rb.pixels.clear(); words = px; }      // D4800
        else if (rb.staged && rb.stagedSeq == r.seq)                                               // D4801: copied at resolve time
        {
            ReadStaging(rb, r.pitch, r.height, px, false);
            words = px;
        }
        else                                                                                       // D4802 ReadLogical
        {
            SurfaceState& st = St();
            EndPass();
            const size_t bytes = size_t(r.pitch) * 4 * r.height;                                   // 4 B per texel in all three formats
            if (!st.readback || st.readback.length < bytes)
            {
                st.readback = [device newBufferWithLength:std::max<size_t>(bytes, 4096) options:MTLResourceStorageModeShared];
                if (!st.readback) throw Failure("metal: readback buffer (" + std::to_string(bytes) + " B)");
            }
            EncodeLogicalCopy(*this, r, st.readback);
            Submit(true);   // everything queued before (the draws and the resolve) lands first
            words = std::span<const uint32_t>(static_cast<const uint32_t*>(st.readback.contents), size_t(r.pitch) * r.height);
        }
        rb.staged = false;                                                                         // D4803
        Scatter(*this, dest, r, words);                                                            // D4804
    }
}

// ================================================================ Backend::Impl (readback section, M2c)
// D4674-4710 StageReadback. EndPass, (resample +) blit to rb.staging, the fence value in a completed handler, then
// Submit(false): the D4705 Flush. Metal runs nothing before its command buffer is committed, so without it the copy
// would wait for the next Present's commit and a CPU read before that would find it never started. Never waits.
// Called from Resolve (hot and small destinations, D1084-1089) and WriteBackRange (D1123), both outside a Draw
// unless a CPU fault reaches WriteBackRange from inside one (then frameCmd changes mid-Draw, as with WriteBack).
void Backend::Impl::StageReadback(MtlResolved& r, Readback& rb)
{
    if (!r.tex) throw Failure("metal: StageReadback of a resolve without a texture");
    @autoreleasepool {
        Initialize();
        EndPass();
        const size_t bytes = size_t(r.pitch) * 4 * r.height;
        if (!rb.staging || rb.staging.length < bytes)                                              // D4693-4699
        {
            rb.staging = [device newBufferWithLength:std::max<size_t>(bytes, 4096) options:MTLResourceStorageModeShared];
            if (!rb.staging) throw Failure("metal: readback staging buffer (" + std::to_string(bytes) + " B)");
        }
        EncodeLogicalCopy(*this, r, rb.staging);                                                   // D4676-4701
        if (!readbackFence) readbackFence = std::make_shared<ReadbackFence>();
        const uint64_t value = ++readbackFenceValue;                                               // D4702 Signal
        std::shared_ptr<ReadbackFence> fence = readbackFence;   // the block keeps its own copy (Metal thread)
        [Cmd() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            {
                std::lock_guard<std::mutex> lock(fence->mutex);
                if (cb.status == MTLCommandBufferStatusError) fence->failed = true;
                else
                {
                    fence->early.insert(value);
                    while (fence->early.erase(fence->done + 1)) ++fence->done;
                }
            }
            fence->landed.notify_all();
        }];
        Submit(false);                                                                             // D4705 Flush
        rb.stagedFence = value;                                                                    // D4702 r.stagedFence
        rb.staged = true;                                                                          // D4706
        rb.stagedSeq = r.seq;                                                                      // D4707
        rb.stagedAt = std::chrono::steady_clock::now();                                            // D4708
        ++stats.stagedReadbacks;                                                                   // D4709
    }
}

// D4845-4860 ReadStaging: D3D11 maps with DO_NOT_WAIT (noWait) or waits in Map; here the fence says whether the
// copy landed, and the blocking form waits on it (under the backend lock, as Map does; no timeout, as Map: a GPU
// hang ends as a failed command buffer). The staged rows are tight, so one memcpy.
bool Backend::Impl::ReadStaging(const Readback& rb, uint32_t width, uint32_t height, std::vector<uint32_t>& out, bool noWait)
{
    if (!rb.staging || !rb.stagedFence || !readbackFence) throw Failure("metal: ReadStaging without a staged copy");
    const size_t words = size_t(width) * height;
    if (rb.staging.length < words * 4) throw Failure("metal: readback staging buffer smaller than the resolve");
    {
        ReadbackFence& f = *readbackFence;
        std::unique_lock<std::mutex> lock(f.mutex);
        if (noWait) { if (!f.failed && f.done < rb.stagedFence) return false; }                    // D4852 DXGI_ERROR_WAS_STILL_DRAWING
        else f.landed.wait(lock, [&] { return f.failed || f.done >= rb.stagedFence; });            // D4849 Map (waits)
        if (f.failed) throw Failure("metal: a staged readback's command buffer failed on the GPU");   // D4853 Check
    }
    out.resize(words);                                                                             // D4855
    std::memcpy(out.data(), rb.staging.contents, words * 4);                                       // D4856-4857
    return true;
}

// D4716-4728
void Backend::Impl::CancelOverlappingSmallReadbacks(uint32_t dest, uint32_t size)
{
    for (auto& [d, copy] : smallReadbacks)
        if (copy.pending && d != dest && Overlaps(dest, size, d, copy.r.size))                    // D4721
        {
            copy.rb.staged = false;                                                                // D4723
            copy.pending = false;                                                                  // D4724
        }
}

// D4729-4748: a snapshot of `r` with its own staging copy (a later resolve of `dest` may change the GPU texture,
// never this copy).
void Backend::Impl::StageSmallReadback(uint32_t dest, const MtlResolved& r, const Readback& rb)
{
    SmallReadback& copy = smallReadbacks[dest];                                                    // D4732
    if (copy.pending) return;                                                                      // D4731
    if (copy.r.pitch != r.pitch || copy.r.height != r.height || copy.r.depth != r.depth || copy.r.rg16f != r.rg16f)   // D4733
        copy = SmallReadback{};                                                                    // D4734
    copy.r.tex = r.tex;                                                                            // D4735
    copy.r.pitch = r.pitch; copy.r.height = r.height; copy.r.size = r.size;                       // D4736
    copy.r.depth = r.depth; copy.r.depthFloat = r.depthFloat; copy.r.rg16f = r.rg16f; copy.r.swapRB = r.swapRB;   // D4737
    copy.r.seq = r.seq; copy.r.revision = r.revision; copy.rb.generation = rb.generation;         // D4738
    copy.rb.cpuReadSerial = rb.cpuReadSerial;                                                      // D4739
    copy.r.sourceKey = r.sourceKey; copy.r.sourcePitch = r.sourcePitch;                            // D4740 (sourceDepth: header)
    copy.r.scale = r.scale; copy.r.valid = true;
    StageReadback(copy.r, copy.rb);                // D4746 (its fence value doubles as the D4741-4745 event query)
    copy.pending = true;                                                                           // D4747
}

// D4749-4791. Never waits: a copy that has not landed stays pending for the next call.
void Backend::Impl::FinishSmallWritebacks()
{
    for (auto& [dest, copy] : smallReadbacks)                                                      // D4751
    {
        if (!copy.pending) continue;
        const auto f = resolved.find(dest);                                                        // D4755
        const auto b = readbacks.find(dest);   // generation / cpuReadSerial of the live resolve (impl.h readback)
        if (f == resolved.end() || b == readbacks.end() || !f->second.valid || f->second.cpuCoherent ||   // D4756
            b->second.generation != copy.rb.generation || b->second.cpuReadSerial != copy.rb.cpuReadSerial ||   // D4757
            f->second.swapRB != copy.r.swapRB || f->second.depthFloat != copy.r.depthFloat ||      // D4758
            f->second.sourceKey != copy.r.sourceKey || f->second.sourcePitch != copy.r.sourcePitch)   // D4759-4760
        {
            copy.rb.staged = false;                                                                // D4762
            copy.pending = false;                                                                  // D4763
            continue;
        }
        std::vector<uint32_t> px;
        if (!ReadStaging(copy.rb, copy.r.pitch, copy.r.height, px, true)) continue;               // D4767-4776: not landed yet
        MtlResolved& r = f->second;                                                                // D4777
        Dirty().Sync(dest, copy.r.size);                                                           // D4778
        if (Dirty().Newest(dest, copy.r.size) <= copy.r.seq)                                       // D4779
        {
            Scatter(*this, dest, copy.r, px);                                                      // D4781
            // Scattering updates guest write-watch. Keep the latest GPU resolve valid, but only mark it coherent
            // if these pixels are from that same resolve.
            r.seq = copy.r.seq;                                                                    // D4784
            r.cpuCoherent = r.revision == copy.r.revision;                                         // D4785
        }
        copy.rb.staged = false;                                                                    // D4788
        copy.pending = false;                                                                      // D4789
    }
}

// logicalW / logicalH: the guest size the returned texture holds (scaled or not). D4906-4939.
id<MTLTexture> Backend::Impl::FrontTexture(uint32_t fb, bool& storedSwap, uint32_t& logicalW, uint32_t& logicalH)
{
    // D1265 (Present): core.mm's Present calls FrontTexture first thing after UpdateGamma, so the small copies
    // that landed during the frame reach guest memory here, as on D3D11. No GPU work, no commit.
    FinishSmallWritebacks();
    storedSwap = false;                                                                            // D4908
    if (!fb) return nil;                                                                           // D4909
    auto it = resolved.find(fb);                                                                   // D4910
    if (it != resolved.end() && it->second.valid && it->second.tex && it->second.pitch >= 1280 && it->second.height >= 720)   // D4911
    {
        Dirty().Sync(fb, it->second.size);                                                         // D4913
        if (Dirty().Newest(fb, it->second.size) <= it->second.seq)                                 // D4914
        {
            storedSwap = it->second.swapRB;                                                        // D4916
            logicalW = it->second.pitch; logicalH = it->second.height;                             // D4917
            return it->second.tex;                                                                 // D4918
        }
        it->second.valid = false;                                                                  // D4920
    }
    logicalW = 1280; logicalH = 720;                                                               // D4922
    // Not GPU-resolved (or overwritten by the CPU): decode guest memory like the GDI path. The B8G8R8A8 view
    // reads the word as ARGB, red and blue exchanged against the fetch order (low byte = x), the same as a
    // copy_dest_swap resolve.
    storedSwap = true;                                                                             // D4926
    Initialize();
    SurfaceState& st = St();
    if (!st.cpuFront)                                                                              // D4927-4933
    {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:1280 height:720 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;   // written by blit only
        st.cpuFront = [device newTextureWithDescriptor:d];
        if (!st.cpuFront) throw Failure("metal: CPU front-buffer texture");
    }
    static TextureUploadSurface decoded;                                                           // D4934
    if (!DecodeFrontBufferUploadSurface(g_base, fb, 1280, 720, decoded))                           // D4935-4936
        throw Failure("metal: invalid guest front-buffer upload");
    // D4937 (UpdateSubresource): staging buffer -> texture, ordered in the frame's command buffer, so a frame
    // still on the GPU never sees the next frame's CPU write.
    id<MTLBuffer> staging = [device newBufferWithBytes:decoded.bytes.data() length:size_t(decoded.rowPitch) * 720
                                               options:MTLResourceStorageModeShared];
    if (!staging) throw Failure("metal: front-buffer staging buffer");
    EndPass();
    id<MTLBlitCommandEncoder> blit = [Cmd() blitCommandEncoder];
    if (!blit) throw Failure("metal: blit command encoder");
    [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:decoded.rowPitch sourceBytesPerImage:size_t(decoded.rowPitch) * 720
              sourceSize:MTLSizeMake(1280, 720, 1) toTexture:st.cpuFront destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return st.cpuFront;                                                                            // D4938
}

// ================================================================ Backend (public, surfaces part)
// D952-1091. srcSelect 0-3 colour, 4 depth; destFormat 31 (16_16_FLOAT) keeps red / green as halves, every
// other colour destination is 8_8_8_8 (D945-951).
void Backend::Resolve(uint32_t colorBase, uint32_t dest, uint32_t pitch, uint32_t height, bool hasLatch, uint32_t latch,
                      bool swapRB, uint32_t srcSelect, uint32_t depthBase, bool depthFloat, uint32_t destFormat)
{
    @autoreleasepool {
        Impl& m = *impl_;
        m.Initialize();                                                                            // D955
        // D956 UpdateScale: not in M1 (header).
        m.FinishSmallWritebacks();                                                                 // D957
        m.CancelOverlappingSmallReadbacks(dest, ResolveBytes(pitch, height));                      // D958
        ++m.stats.resolves;                                                                        // D959
        const bool depth = srcSelect == 4;                                                         // D960
        const bool rg16f = !depth && destFormat == 31;                                             // D961
        // D962 NoteShadowResolve: shadow-map skipping is not in M1.
        MtlResolved& r = m.resolved[dest];                                                         // D963
        float sourceScale = m.scale;                                                               // D964
        if (depth)                                                                                 // D965-969
        {
            auto d = m.depths.find(depthBase);
            if (d != m.depths.end() && d->second.tex) sourceScale = d->second.scale;
        }
        else if (auto c = m.colors.find(colorBase); c != m.colors.end() && c->second.tex)         // D970-971
            sourceScale = c->second.scale;
        bool fresh = false;   // a new (or pooled) texture: undefined until the first pass clears it (D4200-4201)
        if (!r.tex || r.pitch != pitch || r.height != height || r.depth != depth || r.rg16f != rg16f || r.scale != sourceScale)   // D972
        {
            RetireResolved(m, r);                                                                  // D975
            r = MtlResolved{};                                                                     // D976 (readbackHot stays: SurfaceState)
            r.pitch = pitch; r.height = height; r.depth = depth; r.rg16f = rg16f;                  // D977
            MakeResolved(m, r, sourceScale);                                                       // D979
            fresh = true;
        }
        Impl::Readback& rb = m.readbacks[dest];   // the D2139-2163 readback fields (impl.h readback section)
        if (fresh)
        {
            rb = Impl::Readback{};                                                                 // D976 (staging, pixels, lookup go)
            rb.generation = ++St().resolveGeneration;                                              // D4178
        }
        r.sourceKey = depth ? ~0u : colorBase;                                                     // D981
        if (!depth) { const auto c = m.colors.find(colorBase); r.sourcePitch = c != m.colors.end() ? c->second.samplePitch : 0; }   // D982
        // D992-997 resolveDests_ / UpdateNative: native-size surfaces are not in M1.
        r.depthFloat = depthFloat;                                                                 // D998
        m.EndPass();                                                    // every fill below is an encoder of its own
        const MTLLoadAction load = fresh ? MTLLoadActionClear : MTLLoadActionLoad;
        const MTLClearColor zero = MTLClearColorMake(0, 0, 0, 0);
        auto it = m.colors.find(colorBase);                                                        // D1003
        if (depth)                                                                                 // D1004
        {
            ++m.stats.depthResolves;                                                               // D1006
            auto d = m.depths.find(depthBase);                                                     // D1007
            // D1009-1013 (shadow map off -> 0): not in M1.
            if (d != m.depths.end() && d->second.touched && d->second.tex)                         // D1014: BlitDepth (D2745-2783)
            {
                MtlDepthSurface& s = d->second;
                const uint32_t sx = std::clamp<uint32_t>(s.sampX, 1, 4), sy = std::clamp<uint32_t>(s.sampY, 1, 4);   // D2747
                const uint32_t lw = std::min<uint32_t>(r.pitch, SurfaceW / sx), lh = std::min<uint32_t>(r.height, SurfaceH / sy);   // D2762
                const uint32_t w = std::min(Px(lw, s.scale), AllocPx(r.pitch, r.scale));           // D2768
                const uint32_t h = std::min(Px(lh, s.scale), AllocPx(r.height, r.scale));
                const uint32_t samp[2] = { sx, sy };
                FullScreen(BeginPass(m, r.tex, nil, load, zero, "depth resolve"), SurfacePipeline(m, kBlitDepth, r.tex.pixelFormat),
                           r.tex, w, h, s.tex, samp, sizeof(samp));
            }
            else ClearPass(m, r.tex, nil, MTLClearColorMake(1, 1, 1, 1), "depth resolve clear");  // D1015-1019: surfaces start at 1.0
            r.copiedWidth = pitch;                                                                 // D1020
        }
        else if (it == m.colors.end() || !it->second.touched)                                      // D1022
        {
            // Never rasterized: the CPU path fills the fast-clear latch (or zero). RB_COLOR_CLEAR holds the
            // 8_8_8_8 pixel itself, red in the low byte (Xenia's k_8_8_8_8 clear).
            const uint32_t px = hasLatch ? latch : 0;                                              // D1026
            ClearPass(m, r.tex, nil,
                      MTLClearColorMake((px & 0xFF) / 255.0, ((px >> 8) & 0xFF) / 255.0, ((px >> 16) & 0xFF) / 255.0,
                                        ((px >> 24) & 0xFF) / 255.0), "resolve latch clear");      // D1027-1029
            r.copiedWidth = pitch;                                                                 // D1030
        }
        else
        {
            MtlColorSurface& s = it->second;                                                       // D1034
            const uint32_t width = std::min({ s.samplePitch ? s.samplePitch : pitch, pitch, SurfaceW });   // D1035
            const uint32_t rows = std::min<uint32_t>(height, SurfaceH);                           // D1036
            // Nothing was ever drawn beyond the surface's extent (it covers every draw's reach): that part
            // reads as the cleared store, zero - the shader blits' out-of-range loads.
            const float k = r.scale;                                                               // D1039: == s.scale, texel for texel
            const uint32_t copyW = std::min(Px(width, k), AllocPx(r.pitch, k)), copyH = std::min(Px(rows, k), AllocPx(r.height, k));   // D1040
            const uint32_t haveW = s.tex ? std::min(copyW, AllocPx(s.extW, k)) : 0;               // D1041
            const uint32_t haveH = s.tex ? std::min(copyH, AllocPx(s.extH, k)) : 0;
            const bool clear = fresh || (width != r.copiedWidth && width < pitch) || haveW < copyW || haveH < copyH;   // D1042
            if (!s.tex) { if (clear) ClearPass(m, r.tex, nil, zero, "resolve clear"); }           // D1047 (D1044-1045 clear)
            else if (r.rg16f || s.format != MTLPixelFormatBGRA8Unorm)                              // D1048-1049: BlitColor (D2707-2742)
            {
                // D2728: viewport / scissor Px(width, s.scale) x Px(rows, s.scale), here clamped to the target.
                FullScreen(BeginPass(m, r.tex, nil, clear ? MTLLoadActionClear : MTLLoadActionLoad, zero, "colour resolve"),
                           SurfacePipeline(m, r.rg16f ? kBlitRedGreen : kBlitColor, r.tex.pixelFormat), r.tex,
                           Px(width, s.scale), Px(rows, s.scale), s.tex, nullptr, 0);
            }
            else
            {
                if (clear) ClearPass(m, r.tex, nil, zero, "resolve clear");                        // D1044-1045
                CopyRegion(m, s.tex, r.tex, haveW, haveH);                                         // D1052-1053
            }
            r.copiedWidth = width;                                                                 // D1055
        }
        // The resolve owns its destination from now on; any older decode of these bytes is stale. Consume
        // earlier CPU writes so only LATER ones invalidate this entry.
        r.size = ResolveBytes(pitch, height);                                                      // D1059
        m.Dirty().Sync(dest, r.size);                                                              // D1060
        r.seq = m.Dirty().Current();                                                               // D1061
        r.revision = ++St().resolveRevision;                                                       // D1062
        r.valid = true;                                                                            // D1063
        r.cpuCoherent = false;                                                                     // D1064
        rb.pixels.clear();                                                                         // D1065
        r.swapRB = swapRB;                                                                         // D1066: copy_dest_swap
        MetalDropTexturesOverlapping(m, dest, r.size);                                             // D1067
        // D1068-1089 CPU readback policy. Tiny resolves are CPU readbacks (the game's 32x1 auto-exposure average,
        // D1069-1075): staged, and scattered once the copy landed (FinishSmallWritebacks, next Resolve or Present)
        // instead of waiting here (D1080-1085). A destination the CPU read before starts its copy now, so the read
        // finds it landed (D1089). TTT2_GPU_HW_WRITEBACK=1 / TTT2_SMALL_WRITEBACK_SYNC=1 wait here, as before.
        static const bool writeback = [] { const char* e = std::getenv("TTT2_GPU_HW_WRITEBACK"); return e && std::strcmp(e, "1") == 0; }();   // D1068
        static const uint32_t smallWriteback = [] {                                                // D1076-1079
            const char* e = std::getenv("TTT2_SMALL_WRITEBACK");
            return e ? uint32_t(std::strtoul(e, nullptr, 0)) : 32u;
        }();
        static const bool smallSync = [] { const char* e = std::getenv("TTT2_SMALL_WRITEBACK_SYNC"); return e && *e == '1'; }();   // D1086
        if (writeback || (pitch * height <= smallWriteback && smallSync)) m.WriteBack(dest, r);    // D1087
        else if (pitch * height <= smallWriteback) m.StageSmallReadback(dest, r, rb);             // D1088 (toggle_ & 8: header)
        else if (IsHot(dest)) m.StageReadback(r, rb);                                              // D1089
        m.MaybeReportStats();                                                                      // D1090
    }
}

void Backend::CompareResolve(uint32_t, const uint32_t*, uint32_t, uint32_t)
{
    throw Failure("metal: CompareResolve not implemented (M2)");   // D1193-1229: compare-mode diagnostic
}

bool Backend::DumpResolved(uint32_t, const char*)
{
    throw Failure("metal: DumpResolved not implemented (M2)");     // D1232-1248: BMP diagnostic
}

// D1100-1172. The CPU is about to read guest bytes in [start, start + size): make every GPU-only resolve
// overlapping them coherent, but only the requested bytes (a cached physical-view fault asks for one 64 KiB chunk
// of what may be a 4 MiB bake). True while a copy is still on its way: the caller (GpuHwWriteBackRange) drops the
// backend lock and runs PendingCopyWaiter() before calling again, so the command processor keeps drawing and
// presenting meanwhile. `block` waits here, under the lock, instead (the caller's last resort).
bool Backend::WriteBackRange(uint32_t start, uint32_t size, bool block)
{
    @autoreleasepool {
        Impl& m = *impl_;
        bool pending = false;                                                                      // D1102
        m.pendingFence = 0;                                                                        // D1103
        for (auto& [rs, r] : m.resolved)                                                           // D1104
        {
            if (!r.valid || r.cpuCoherent || !r.tex) continue;                                     // D1106
            if (!Overlaps(start, size, rs, r.size)) continue;                                      // D1107
            m.Dirty().Sync(rs, r.size);                                                            // D1108
            if (m.Dirty().Newest(rs, r.size) > r.seq) { r.valid = false; continue; }               // D1109: the CPU overwrote it
            SetHot(rs);                                       // D1110-1117 (UpdateNative: M2): its next resolves are staged
            Impl::Readback& rb = m.readbacks[rs];
            const auto t0 = std::chrono::steady_clock::now();                                      // D1118
            if (rb.pixels.empty())                                                                 // D1119
            {
                if (!rb.staged || rb.stagedSeq != r.seq)                                           // D1121
                {
                    m.StageReadback(r, rb);                                                        // D1123: first read: copy now, read later
                    if (!block) { pending = true; m.pendingFence = std::max(m.pendingFence, rb.stagedFence); continue; }   // D1124
                }
                if (!m.ReadStaging(rb, r.pitch, r.height, rb.pixels, !block))                      // D1126
                {
                    pending = true;                                                                // D1128
                    m.pendingFence = std::max(m.pendingFence, rb.stagedFence);                     // D1129
                    continue;
                }
                rb.staged = false;                                                                 // D1132
            }
            static int logged = 0;                                                                 // D1136-1142
            if (logged < 40)
            {
                ++logged;
                std::fprintf(stderr, "[gpu-metal] cpu-read writeback dest=%08X %ux%u requested=%08X+%u swapRB=%d rg16f=%d depth=%d\n",
                             rs, r.pitch, r.height, start, size, int(r.swapRB), int(r.rg16f), int(r.depth));
            }
            ++rb.cpuReadSerial;                                       // D1143: an older asynchronous snapshot must not replace this read
            const auto scatterStart = std::chrono::steady_clock::now();                            // D1144
            const ResolveWritebackInfo info{ rs, r.pitch, r.height, r.depth, r.depthFloat, r.rg16f, r.swapRB };   // D1145
            if (rb.tiledToLinear.empty() && !BuildResolvedTileLookup(info, rb.tiledToLinear))      // D1146
                throw Failure("metal: invalid resolved tile lookup");                              // D1147
            if (!ScatterResolvedWordsRange(info, rb.pixels, rb.tiledToLinear, start, size,         // D1148-1149
                    [](uint32_t address, uint32_t word) { GWrite32(g_base, address, word); }))
                throw Failure("metal: invalid partial resolved guest writeback");                  // D1150
            m.Dirty().Sync(rs, r.size);                                                            // D1151
            r.seq = m.Dirty().Current();                                                           // D1152
            ++m.stats.writebacks;                                                                  // D1153
            if (uint64_t(start) <= rs && uint64_t(start) + size >= uint64_t(rs) + r.size)          // D1154
            {
                r.cpuCoherent = true;                                                              // D1156
                rb.pixels.clear();                                                                 // D1157
            }
            static const bool timing = [] { const char* e = std::getenv("TTT2_FRAMETIME"); return e && *e == '1'; }();   // D1159
            if (timing && r.pitch * r.height > 64)                                                 // D1160-1169
            {
                static int loggedRange = 0;
                if (loggedRange++ < 200)
                    std::fprintf(stderr, "[frame] partial writeback %08X %ux%u: read %.2f ms, scatter %.2f ms, request %08X+%u\n",
                                 rs, r.pitch, r.height,
                                 std::chrono::duration<double, std::milli>(scatterStart - t0).count(),
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scatterStart).count(),
                                 start, size);
            }
        }
        return pending;                                                                            // D1171
    }
}

// D1176-1190. Called under the lock; the function it returns runs WITHOUT it, so it holds only copies: the fence
// (shared_ptr) and the value to reach. It touches nothing but the fence's own mutex-guarded fields. Waits at most
// 100 ms and says whether the copy landed, as HostWaitFence (D154-163, D1183; the non-Windows form); empty when nothing is pending.
std::function<bool()> Backend::PendingCopyWaiter() const
{
    const uint64_t value = impl_->pendingFence;                                                    // D1178
    std::shared_ptr<Impl::ReadbackFence> fence = value ? impl_->readbackFence : nullptr;           // D1179
    if (!fence) return {};                                                                         // D1180
    return [fence, value] { return WaitFence(*fence, value, std::chrono::milliseconds(100)); };   // D1181-1189
}

} // namespace ttt2_hw
