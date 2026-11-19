// R4 Metal backend, surfaces (gpu_hw_metal_impl.h "surfaces" section, M1 slice 4): EDRAM-lite colour / depth
// surfaces (Color, Depth, Grow), Resolve, the synchronous WriteBack, WriteBackRange / PendingCopyWaiter, the
// front-buffer choice (FrontTexture), the dirty tracker, and the diagnostics entry points (M2: fail by name).
// Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11 backend stays the reference).
//
// Content rule shared with draw.mm: Draw opens its passes with MTLLoadActionLoad, so every texture made here
// leaves this file fully defined (new surface area 0 / depth 1.0 + stencil 0, new resolve textures 0), through
// passes of its own on Cmd() after EndPass(). Nothing here commits the frame's command buffer except WriteBack
// (it must wait for the GPU) - see the note on WriteBack about calls from inside a Draw.
//
// Not ported in M1 (each named):
//   UpdateScale / Rescale (D956, D3960-3973, D4123-4157): core fixes the render scale at Initialize.
//   Native-size bake surfaces (D784-813, D3877-3935, D992-997 resolveDests_/UpdateNative, D1110-1117): M2.
//   Shadow-map skipping (D962 NoteShadowResolve, D1009-1013 ShadowMapOff, D2201-2244): M2; depth resolves
//     always copy the surface.
//   Asynchronous readbacks (D957-958 FinishSmallWritebacks / CancelOverlappingSmallReadbacks, D1084-1088
//     StageSmallReadback / StageReadback, D1119-1158 staged WriteBackRange, D1176-1190 fence waiter,
//     D4674-4791): M2. M1 writes back synchronously (Resolve: small, TTT2_GPU_HW_WRITEBACK=1 and
//     readback-hot ones; WriteBackRange: every overlapping resolve).
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern uint8_t* g_base;   // kernel_shared.h: global guest memory base

namespace ttt2_hw {

// gpu_hw_metal_textures.mm (slice 3): D4431-4440, evict guest-decoded textures overlapping a resolve.
void MetalDropTexturesOverlapping(Backend::Impl& m, uint32_t start, uint32_t size);

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
    // WriteBack: the logical-size target of a scaled resolve (D4680-4689 r.logical, one per shape here: the
    // readback is synchronous, so the texture is free again when WriteBack returns) and the shared readback buffer.
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

// D4795-4805 WriteBack + D4809-4841 Scatter, synchronous: resample a scaled resolve to its logical size
// (D4680-4691, as StageReadback / ReadLogical), copy it to CPU-visible memory, wait, scatter the tiled guest words.
//
// Submit(true) commits the frame's command buffer and waits. Called from textures.mm's BindTexture inside a Draw,
// that command buffer holds the draw's arena (vertex / index slices allocated before the texture loop), which
// returns to the free list here; draw.mm (slice 3 fix) copies vb / ib into the new command buffer's arena when
// frameCmd changed across the texture loop, before any new Alloc.
void Backend::Impl::WriteBack(uint32_t dest, MtlResolved& r)
{
    if (!r.tex) throw Failure("metal: WriteBack of a resolve without a texture");
    @autoreleasepool {
        Initialize();
        SurfaceState& st = St();
        EndPass();
        id<MTLTexture> source = r.tex;                                                             // D4678
        if (uint32_t(r.tex.width) != r.pitch || uint32_t(r.tex.height) != r.height)                // D4679
        {
            __strong id<MTLTexture>& logical = st.logical[PoolKey(r.pitch, r.height, r.tex.pixelFormat)];   // D4681-4688 r.logical
            if (!logical) logical = NewTexture(*this, r.tex.pixelFormat, r.pitch, r.height, "readback logical target");
            ResampleColor(*this, r.tex, logical, MTLLoadActionDontCare, r.pitch, r.height, 1.0f / float(r.pitch),
                          1.0f / float(r.height), !r.depth && Filterable(r.tex.pixelFormat), "readback resample");   // D4689-4690
            source = logical;                                                                      // D4691
        }
        const size_t rowBytes = size_t(r.pitch) * 4, bytes = rowBytes * r.height;                 // 4 B per texel in all three formats
        if (!st.readback || st.readback.length < bytes)
        {
            st.readback = [device newBufferWithLength:std::max<size_t>(bytes, 4096) options:MTLResourceStorageModeShared];
            if (!st.readback) throw Failure("metal: readback buffer (" + std::to_string(bytes) + " B)");
        }
        // D4700-4701 (CopySubresourceRegion to the staging texture) + D4853-4857 (Map): one copy to a shared buffer.
        id<MTLBlitCommandEncoder> blit = [Cmd() blitCommandEncoder];
        if (!blit) throw Failure("metal: blit command encoder");
        [blit copyFromTexture:source sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(r.pitch, r.height, 1) toBuffer:st.readback destinationOffset:0
       destinationBytesPerRow:rowBytes destinationBytesPerImage:bytes];
        [blit endEncoding];
        Submit(true);   // everything queued before (the draws and the resolve) lands first
        const uint32_t* px = static_cast<const uint32_t*>(st.readback.contents);
        // D4813-4816 Scatter: tiled, in the guest's byte and channel order (gpu_hw_writeback.h).
        const ResolveWritebackInfo info{ dest, r.pitch, r.height, r.depth, r.depthFloat, r.rg16f, r.swapRB };
        if (!ScatterResolvedWords(info, std::span<const uint32_t>(px, size_t(r.pitch) * r.height),
                                  [](uint32_t address, uint32_t word) { GWrite32(g_base, address, word); }))
            throw Failure("metal: invalid resolved guest writeback");
        Dirty().Sync(dest, r.size);                                                                // D4817
        ++stats.writebacks;                                                                        // D1153
        r.seq = Dirty().Current();                                                                 // D4818
        r.cpuCoherent = true;                                                                      // D4819
    }
}

// logicalW / logicalH: the guest size the returned texture holds (scaled or not). D4906-4939.
id<MTLTexture> Backend::Impl::FrontTexture(uint32_t fb, bool& storedSwap, uint32_t& logicalW, uint32_t& logicalH)
{
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
        ++m.stats.resolves;                                                                        // D959
        // D956-958: UpdateScale, FinishSmallWritebacks, CancelOverlappingSmallReadbacks - not in M1 (header).
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
        r.cpuCoherent = false;                                                                     // D1064 (D1065 readbackPixels: M2)
        r.swapRB = swapRB;                                                                         // D1066: copy_dest_swap
        MetalDropTexturesOverlapping(m, dest, r.size);                                             // D1067
        // D1068-1089 CPU readback policy. M1: every write-back is synchronous (the staged, asynchronous small
        // readback D1084-1088 and the staged hot readback D1089 are M2). Tiny resolves are CPU readbacks (the
        // game's 32x1 auto-exposure average, D1069-1075), and a destination the CPU read before is read again.
        static const bool writeback = [] { const char* e = std::getenv("TTT2_GPU_HW_WRITEBACK"); return e && std::strcmp(e, "1") == 0; }();   // D1068
        static const uint32_t smallWriteback = [] {                                                // D1076-1079
            const char* e = std::getenv("TTT2_SMALL_WRITEBACK");
            return e ? uint32_t(std::strtoul(e, nullptr, 0)) : 32u;
        }();
        if (writeback || pitch * height <= smallWriteback || IsHot(dest)) m.WriteBack(dest, r);   // D1087-1089 (synchronous in M1)
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

// D1100-1172, M1: every valid GPU-only resolve overlapping the range is written back synchronously, so nothing
// is ever pending (false). The staged / partial-range path (StageReadback, ReadStaging, ScatterResolvedWordsRange)
// is M2.
bool Backend::WriteBackRange(uint32_t start, uint32_t size, bool block)
{
    (void)block;   // nothing waits outside the lock in M1
    @autoreleasepool {
        Impl& m = *impl_;
        for (auto& [rs, r] : m.resolved)                                                           // D1104
        {
            if (!r.valid || r.cpuCoherent || !r.tex) continue;                                     // D1106
            if (!(uint64_t(start) < uint64_t(rs) + r.size && uint64_t(rs) < uint64_t(start) + size)) continue;   // D1107
            m.Dirty().Sync(rs, r.size);                                                            // D1108
            if (m.Dirty().Newest(rs, r.size) > r.seq) { r.valid = false; continue; }               // D1109: the CPU overwrote it
            SetHot(rs);                                       // D1110-1117 (UpdateNative: M2): its next resolves are read back too
            static int logged = 0;                                                                 // D1137-1142
            if (logged < 40)
            {
                ++logged;
                std::fprintf(stderr, "[gpu-metal] cpu-read writeback dest=%08X %ux%u requested=%08X+%u swapRB=%d rg16f=%d depth=%d\n",
                             rs, r.pitch, r.height, start, size, int(r.swapRB), int(r.rg16f), int(r.depth));
            }
            m.WriteBack(rs, r);                                                                    // M1: whole resolve, synchronous
        }
        return false;
    }
}

std::function<bool()> Backend::PendingCopyWaiter() const
{
    return {};   // D1176-1190: M2 (WriteBackRange never leaves a copy pending in M1)
}

} // namespace ttt2_hw
