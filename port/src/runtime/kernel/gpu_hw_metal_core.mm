// R4 Metal backend, core (gpu_hw_metal_impl.h "core" section): device, queue, the window's CAMetalLayer,
// Present / ReadFront, gamma, Get(), ReportBackendFailure, RemovedReason, and the test hooks.
// Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11 backend stays the reference).
//
// External symbols this file needs (the core harness m1/core_test.mm defines its own copies, so it links
// without the game): display_settings::GetVSync / GetSharpness / GetAntiAliasing, host_window::SDLHandle,
// and Backend::Impl::FrontTexture (gpu_hw_metal_stubs.mm in the game).

#include "gpu_hw_metal_impl.h"
#include "display_settings.h"

#import <Foundation/Foundation.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_error.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace ttt2_hw {

namespace {

// ---------------------------------------------------------------- core shaders (MSL, compiled at run time)
// The Mac build has no offline `metal` compiler (Command Line Tools only), so the present shaders are
// source here and go through newLibraryWithSource once, in Initialize.
const char* const kCoreMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

// D1812: PresentCB { srcW, srcH, dstW, dstH, originX, originY, sharp, mode } (32 bytes).
struct PresentCB { float2 srcSize; float2 dstSize; float2 origin; float sharp; uint mode; };
struct VOut { float4 pos [[position]]; float2 uv; };

// D1846-1849: full-screen triangle, uv 0..1 over the viewport.
vertex VOut present_vs(uint id [[vertex_id]])
{
    float2 t = float2(float((id << 1) & 2u), float(id & 2u));
    VOut o;
    o.pos = float4(t * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = t;
    return o;
}

inline uint2 Texel(int2 p, texture2d<float> t)
{
    return uint2(clamp(p, int2(0), int2(int(t.get_width()) - 1, int(t.get_height()) - 1)));
}

// D1861-1872: display pass, variants [swap | gamma << 1].
inline float4 Display(texture2d<float> src, texture2d<float> lut, float2 uv, constant PresentCB& cb, bool swap, bool gamma)
{
    float4 c = src.read(Texel(int2(uv * cb.dstSize), src));                                      // D1865
    if (swap) c = c.bgra;                                                                         // D1866
    if (gamma)
    {
        uint3 i = uint3(round(saturate(c.rgb) * 255.0));                                          // D1868
        c.rgb = float3(lut.read(uint2(i.r, 0u)).r, lut.read(uint2(i.g, 0u)).g, lut.read(uint2(i.b, 0u)).b);  // D1869
    }
    return float4(c.rgb, 1.0);                                                                    // D1870
}

fragment float4 display_0(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant PresentCB& cb [[buffer(0)]])
{ return Display(src, src, in.uv, cb, false, false); }
fragment float4 display_1(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant PresentCB& cb [[buffer(0)]])
{ return Display(src, src, in.uv, cb, true, false); }
fragment float4 display_2(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], texture2d<float> lut [[texture(1)]],
                          constant PresentCB& cb [[buffer(0)]])
{ return Display(src, lut, in.uv, cb, false, true); }
fragment float4 display_3(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], texture2d<float> lut [[texture(1)]],
                          constant PresentCB& cb [[buffer(0)]])
{ return Display(src, lut, in.uv, cb, true, true); }

// D1852: linear clamp sampler (single-level textures here).
constexpr sampler lin(coord::normalized, filter::linear, address::clamp_to_edge);
constexpr sampler pnt(coord::normalized, filter::nearest, address::clamp_to_edge);

// D1941-1974: scale, mode 0 texel for texel, 1 Catmull-Rom (9 bilinear taps), 2 area filter.
fragment float4 scale_ps(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant PresentCB& cb [[buffer(0)]])
{
    float2 p = in.uv * cb.dstSize;                                                                // D1943
    if (cb.mode == 0u) return float4(src.read(Texel(int2(p), src)).rgb, 1.0);                    // D1944
    float2 srcSize = cb.srcSize;
    float2 s = p / cb.dstSize * srcSize;                                                          // D1945
    if (cb.mode == 1u)
    {
        float2 t1 = floor(s - 0.5) + 0.5, f = s - t1;                                             // D1947
        float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);       // D1948
        float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);               // D1949
        float2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / srcSize, t0 = (t1 - 1.0) / srcSize, t3 = (t1 + 2.0) / srcSize;  // D1950
        float3 r = src.sample(lin, float2(t0.x, t0.y), level(0.0)).rgb * w0.x * w0.y            // D1951-1959
                 + src.sample(lin, float2(t12.x, t0.y), level(0.0)).rgb * w12.x * w0.y
                 + src.sample(lin, float2(t3.x, t0.y), level(0.0)).rgb * w3.x * w0.y
                 + src.sample(lin, float2(t0.x, t12.y), level(0.0)).rgb * w0.x * w12.y
                 + src.sample(lin, float2(t12.x, t12.y), level(0.0)).rgb * w12.x * w12.y
                 + src.sample(lin, float2(t3.x, t12.y), level(0.0)).rgb * w3.x * w12.y
                 + src.sample(lin, float2(t0.x, t3.y), level(0.0)).rgb * w0.x * w3.y
                 + src.sample(lin, float2(t12.x, t3.y), level(0.0)).rgb * w12.x * w3.y
                 + src.sample(lin, float2(t3.x, t3.y), level(0.0)).rgb * w3.x * w3.y;
        return float4(saturate(r), 1.0);                                                          // D1960
    }
    float2 ratio = srcSize / cb.dstSize;                                                          // D1962
    float2 a = floor(p) * ratio, b = a + ratio;                                                   // D1963
    int2 i0 = int2(floor(a)), i1 = min(int2(ceil(b)), int2(srcSize)) - 1;                         // D1964
    float3 sum = float3(0.0); float wsum = 0.0;                                                   // D1965
    for (int y = i0.y; y <= i1.y && y < i0.y + 12; ++y)                                           // D1966
    {
        float wy = min(b.y, float(y) + 1.0) - max(a.y, float(y));                                 // D1967
        for (int x = i0.x; x <= i1.x && x < i0.x + 12; ++x)                                       // D1968
        {
            float w = wy * (min(b.x, float(x) + 1.0) - max(a.x, float(x)));                       // D1969
            sum += src.read(Texel(int2(x, y), src)).rgb * w; wsum += w;                            // D1970
        }
    }
    return float4(sum / max(wsum, 1e-6f), 1.0);                                                    // D1973
}

// D4078: resample at uv = pixel centre * k (ReadLogical's logical-size read).
fragment float4 resample_linear(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant float4& k [[buffer(0)]])
{ return src.sample(lin, in.pos.xy * k.xy, level(0.0)); }
fragment float4 resample_point(VOut in [[stage_in]], texture2d<float> src [[texture(0)]], constant float4& k [[buffer(0)]])
{ return src.sample(pnt, in.pos.xy * k.xy, level(0.0)); }
)MSL";

// D1812
struct PresentCB { float srcW, srcH, dstW, dstH, originX, originY, sharp; uint32_t mode; };
static_assert(sizeof(PresentCB) == 32, "PresentCB must match the MSL struct");

constexpr uint32_t kFrontW = 1280, kFrontH = 720;   // D1482: ReadFront is always 1280x720

// State the frozen Impl has no fields for (pipelines, post target, readback buffer, error code, the
// Backend -> Impl link the test hooks need). One Backend exists (Get()), so one instance.
struct CoreState {
    id<MTLLibrary> library = nil;
    id<MTLRenderPipelineState> display[4] = { nil, nil, nil, nil };   // D1814: displayPs_[swap | gamma << 1]
    id<MTLRenderPipelineState> scale = nil;                             // D1814: scalePs_
    id<MTLRenderPipelineState> resampleLinear = nil, resamplePoint = nil;   // D4067: resamplePs_ (linear/point)
    id<MTLTexture> post0 = nil;                                         // D1817-1832: Post(0), display pass output
    id<MTLTexture> logicalRead = nil;                                   // D4892-4899: ReadLogical's 1280x720 target
    id<MTLBuffer> readBuffer = nil;                                     // ReadTexture's staging (shared)
    SDL_Window* viewWindow = nullptr;                                   // the window metalView belongs to
    bool loggedNoDrawable = false, loggedFxaa = false, loggedCas = false, loggedMetalFx = false;
};
CoreState& Core() { static CoreState* s = new CoreState; return *s; }   // leaked with the Backend (D5140)

// RemovedReason(): the last command-buffer error, recorded by the completed handler (never thrown there);
// pendingError makes the next Cmd() / Submit() throw it.
std::atomic<long> g_lastCommandError{0};
std::atomic<bool> g_pendingCommandError{false};

// The test hooks are free functions and Backend::impl_ is private: the constructor records the link.
std::atomic<Backend*> g_testBackend{nullptr};
std::atomic<Backend::Impl*> g_testImpl{nullptr};

std::string Describe(NSError* e)
{
    if (!e) return "unknown error";
    const char* s = e.localizedDescription.UTF8String;
    return std::string(s ? s : "unknown error") + " (code " + std::to_string(long(e.code)) + ")";
}

void ThrowPendingCommandError()
{
    if (g_pendingCommandError.exchange(false))
        throw Failure("metal: a command buffer failed on the GPU (error " + std::to_string(g_lastCommandError.load()) + ")");
}

id<MTLRenderPipelineState> Pipeline(id<MTLDevice> device, id<MTLLibrary> lib, NSString* fs)
{
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"present_vs"];
    d.fragmentFunction = [lib newFunctionWithName:fs];
    if (!d.vertexFunction || !d.fragmentFunction)
        throw Failure(std::string("metal: core shader function missing: ") + fs.UTF8String);
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;   // drawables, Post(0), logical reads: all BGRA8
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&err];
    if (!p) throw Failure(std::string("metal: core pipeline ") + fs.UTF8String + ": " + Describe(err));
    return p;
}

id<MTLTexture> RenderTexture(id<MTLDevice> device, uint32_t w, uint32_t h, const char* what)
{
    MTLTextureDescriptor* t = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                 width:w height:h mipmapped:NO];
    t.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    t.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tex = [device newTextureWithDescriptor:t];
    if (!tex) throw Failure(std::string("metal: ") + what);
    return tex;
}

// One full-screen-triangle pass (D1995-2011 PostPass): viewport + scissor, t0 / t1, fragment bytes at buffer(0).
void Pass(id<MTLCommandBuffer> cmd, id<MTLTexture> target, bool clearBlack, const MTLViewport& vp, const MTLScissorRect& sc,
          id<MTLRenderPipelineState> ps, id<MTLTexture> t0, id<MTLTexture> t1, const void* bytes, size_t length)
{
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = clearBlack ? MTLLoadActionClear : MTLLoadActionDontCare;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);   // D2030: black letterbox
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
    if (!enc) throw Failure("metal: render command encoder");
    [enc setRenderPipelineState:ps];
    [enc setViewport:vp];
    [enc setScissorRect:sc];
    [enc setFragmentTexture:t0 atIndex:0];
    if (t1) [enc setFragmentTexture:t1 atIndex:1];
    [enc setFragmentBytes:bytes length:length atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}

// D1507-1511
bool GammaRampEnabled()
{
    static const bool on = [] { const char* e = std::getenv("TTT2_GAMMA_RAMP"); return !e || std::strcmp(e, "0") != 0; }();
    return on;
}

// gpu_hw_metalfx.h:11: TTT2_METALFX=1 asks for MetalFX (M5 here: logged and skipped).
bool MetalFxRequested()
{
    static const bool on = [] { const char* e = std::getenv("TTT2_METALFX"); return e && *e == '1'; }();
    return on;
}

// D4119: formats that may not filter: resample with point reads.
bool Filterable(MTLPixelFormat f) { return f != MTLPixelFormatR32Float && f != MTLPixelFormatRG32Float; }

// D1488-1495: a stored front-buffer texel as shown: red / blue as presented, the gamma ramp, alpha cleared.
uint32_t DisplayPixel(const Backend::Impl& m, uint32_t v)
{
    if (m.lastSwap) v = (v & 0xFF00FF00u) | ((v >> 16) & 0xFFu) | ((v & 0xFFu) << 16);           // D1490
    if (m.gammaApplied)                                                                          // D1491-1493
        v = (uint32_t(m.gamma8[0][(v >> 16) & 0xFF]) << 16) | (uint32_t(m.gamma8[1][(v >> 8) & 0xFF]) << 8) |
            m.gamma8[2][v & 0xFF];
    return v & 0x00FFFFFFu;                                                                      // D1494
}

// The CAMetalLayer on the SDL window, made on the first present in the presenting thread (as DXVK made its
// surface there); remade if SetWindow moved to another window.
bool EnsureLayer(Backend::Impl& m)
{
    SDL_Window* w = m.window.load();
    if (!w) return false;
    CoreState& c = Core();
    if (m.metalView && c.viewWindow != w)
    {
        SDL_Metal_DestroyView(static_cast<SDL_MetalView>(m.metalView));
        m.metalView = nullptr;
        m.layer = nil;
    }
    if (m.metalView) return true;
    SDL_MetalView view = SDL_Metal_CreateView(w);
    if (!view) throw Failure(std::string("metal: SDL_Metal_CreateView: ") + SDL_GetError());
    CAMetalLayer* layer = (__bridge CAMetalLayer*)SDL_Metal_GetLayer(view);
    if (!layer)
    {
        SDL_Metal_DestroyView(view);
        throw Failure(std::string("metal: SDL_Metal_GetLayer: ") + SDL_GetError());
    }
    layer.device = m.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;   // D1304
    layer.framebufferOnly = YES;
    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(w, &pw, &ph);
    if (pw > 0 && ph > 0) layer.drawableSize = CGSizeMake(pw, ph);
    layer.displaySyncEnabled = display_settings::GetVSync() ? YES : NO;
    m.metalView = view;
    m.layer = layer;
    c.viewWindow = w;
    std::fprintf(stderr, "[gpu-metal] presenting through a CAMetalLayer (%dx%d), vsync %s\n", pw, ph,
                 display_settings::GetVSync() ? "on" : "off");
    return true;
}

} // namespace

// ================================================================ Backend (public, core part)
Backend::Backend() : impl_(std::make_unique<Impl>())
{
    g_testBackend.store(this);
    g_testImpl.store(impl_.get());
}

Backend::~Backend()
{
    if (g_testBackend.load() == this) { g_testBackend.store(nullptr); g_testImpl.store(nullptr); }
}

Backend& Get()
{
    static Backend* b = new Backend;     // D5140: leaked on purpose, no device teardown at exit
    return *b;
}

void ReportBackendFailure(const char* what, const std::exception& e)
{
    // D5144-5148 (the D3D11 line's prefix, so log greps for "[gpu-hw] DISABLED" keep working)
    std::fprintf(stderr, "[gpu-hw] DISABLED after failure in %s: %s (metal command buffer error %ld)\n",
                 what, e.what(), Get().RemovedReason());
}

long Backend::RemovedReason() const { return g_lastCommandError.load(); }

void Backend::SetWindow(host_window::Window* window)
{
    impl_->window.store(host_window::SDLHandle(window));   // D1251-1254
}

bool Backend::IsExclusiveFullscreen() const { return impl_->exclusive.load(); }   // D2102 (M1: never exclusive)

bool Backend::Present(uint32_t frontBuffer, bool displaySwapRB, const uint32_t* gammaRamp, uint32_t gammaVersion)
{
    @autoreleasepool {
        Impl& m = *impl_;
        m.Initialize();                                                          // D1263
        m.EndPass();   // FrontTexture may blit (CPU front buffer); no render encoder may stay open
        ThrowPendingCommandError();
        m.UpdateGamma(gammaRamp, gammaVersion);                                  // D1267
        bool storedSwap = false;
        uint32_t logicalW = 1280, logicalH = 720;                                // D1269
        id<MTLTexture> source = m.FrontTexture(frontBuffer, storedSwap, logicalW, logicalH);   // D1270
        if (!source) return false;                                               // D1271
        if (!m.window.load()) return false;                                      // D1272-1273
        const bool swap = storedSwap != displaySwapRB;                           // D1332
        if (!m.PresentTexture(source, logicalW, logicalH, swap)) return false;   // D1334
        m.lastSwap = swap;                                                       // D1335
        m.lastFront = source; m.lastFrontW = logicalW; m.lastFrontH = logicalH; // D1436
        return true;
    }
}

bool Backend::ReadFront(uint32_t* out)
{
    @autoreleasepool {
        Impl& m = *impl_;
        if (!m.lastFront) return false;                                          // D1480
        std::vector<uint32_t> px;
        if (!m.ReadLogical(m.lastFront, m.lastFrontW, m.lastFrontH, px)) return false;   // D1482 (+ DisplayPixel)
        std::memcpy(out, px.data(), px.size() * sizeof(uint32_t));               // D1483
        return true;
    }
}

// ================================================================ Backend::Impl (core section)
void Backend::Impl::Initialize()
{
    if (device) return;                                                          // D248
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) throw Failure("metal: MTLCreateSystemDefaultDevice returned nil");
        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) throw Failure("metal: newCommandQueue");
        NSError* err = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kCoreMsl] options:options error:&err];
        if (!lib) throw Failure("metal: core shaders did not compile: " + Describe(err));
        CoreState& c = Core();
        c.library = lib;
        c.display[0] = Pipeline(dev, lib, @"display_0");
        c.display[1] = Pipeline(dev, lib, @"display_1");
        c.display[2] = Pipeline(dev, lib, @"display_2");
        c.display[3] = Pipeline(dev, lib, @"display_3");
        c.scale = Pipeline(dev, lib, @"scale_ps");
        c.resampleLinear = Pipeline(dev, lib, @"resample_linear");
        c.resamplePoint = Pipeline(dev, lib, @"resample_point");
        // D3948: TTT2_RENDER_SCALE, clamped to [1, 4] off Android; slice 1 has no settings-driven scale.
        if (const char* e = std::getenv("TTT2_RENDER_SCALE"); e && *e)
            scale = std::clamp(float(std::atof(e)), 1.0f, 4.0f);
        else
            scale = 1.0f;
        queue = q;
        device = dev;                                                            // last: Initialize is done
        std::fprintf(stderr, "[gpu-metal] active on %s, mode=%s, render scale %.2f\n", dev.name.UTF8String,
                     RequestedMode() == Mode::Compare ? "compare" : "on", double(scale));
    }
}

id<MTLCommandBuffer> Backend::Impl::Cmd()
{
    ThrowPendingCommandError();
    if (frameCmd) return frameCmd;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (!cb) throw Failure("metal: commandBuffer returned nil");
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (done.status == MTLCommandBufferStatusError)
        {
            NSError* e = done.error;
            const long code = e ? long(e.code) : -1;
            g_lastCommandError.store(code ? code : -1);
            g_pendingCommandError.store(true);
        }
    }];
    frameCmd = cb;
    return cb;
}

void Backend::Impl::Submit(bool wait)
{
    EndPass();                                   // draw's render encoder ends before the commit (impl.h, draw section)
    id<MTLCommandBuffer> cb = frameCmd;
    frameCmd = nil;
    ThrowPendingCommandError();
    if (!cb) return;
    [cb commit];
    if (!wait) return;
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError)
    {
        NSError* e = cb.error;
        const long code = e ? long(e.code) : -1;
        g_lastCommandError.store(code ? code : -1);
        g_pendingCommandError.store(false);   // thrown here, not again
        throw Failure("metal: command buffer failed: " + Describe(e));
    }
}

void Backend::Impl::UpdateGamma(const uint32_t* ramp, uint32_t version)
{
    if (!ramp || version == gammaVersion) return;                                // D1519
    gammaVersion = version;                                                      // D1520
    float table[256][4];                                                         // D1521
    bool identity = true;
    for (uint32_t i = 0; i < 256; ++i)                                           // D1523-1534
    {
        const uint32_t e = ramp[i], identityValue = i * 0x3FF / 0xFF;
        const uint32_t ch[3] = { (e >> 20) & 0x3FF, (e >> 10) & 0x3FF, e & 0x3FF };   // red, green, blue
        for (int k = 0; k < 3; ++k)
        {
            table[i][k] = float(ch[k]) / 1023.0f;
            gamma8[k][i] = uint8_t((ch[k] * 255 + 511) / 1023);
            identity = identity && ch[k] == identityValue;
        }
        table[i][3] = 1.0f;
    }
    gammaApplied = GammaRampEnabled() && !identity;                              // D1535
    static int logged = 0;                                                       // D1536-1540
    if (logged++ < 4)
        std::fprintf(stderr, "[gpu-metal] display gamma ramp %s (entry 128 = %.4f, identity %.4f)%s\n",
                     identity ? "identity" : "loaded", table[128][0], 128.0f / 255.0f,
                     !identity && !GammaRampEnabled() ? " - not applied (TTT2_GAMMA_RAMP=0)" : "");
    if (!gammaApplied) return;                                                   // D1541
    Initialize();
    // D1542-1551: 256x1 RGBA32F. A new texture per ramp (cheap, rare), so a frame still in flight keeps the
    // old one instead of seeing a CPU write mid-frame (D3D11's UpdateSubresource was ordered; replaceRegion is not).
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                 width:256 height:1 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeManaged;   // CPU-writable (replaceRegion); never Private
    id<MTLTexture> lut = [device newTextureWithDescriptor:d];
    if (!lut) throw Failure("metal: gamma LUT texture");
    [lut replaceRegion:MTLRegionMake2D(0, 0, 256, 1) mipmapLevel:0 withBytes:table bytesPerRow:sizeof(table)];
    gammaLut = lut;
}

bool Backend::Impl::PresentTexture(id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap)
{
    (void)logicalW; (void)logicalH;   // D2042: the passes use the texture's own size, as PresentScaled
    if (!source) return false;
    @autoreleasepool {
        Initialize();
        if (!EnsureLayer(*this)) return false;
        CoreState& c = Core();
        SDL_Window* w = window.load();
        // D1777-1784 (ApplyDisplay's follow-the-client-size, reduced): the drawable tracks the window's pixels.
        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(w, &pw, &ph);
        if (pw <= 0 || ph <= 0) { Submit(false); return true; }   // minimised / zero-size: nothing shown (as OCCLUDED)
        const CGSize want = CGSizeMake(pw, ph);
        if (!CGSizeEqualToSize(layer.drawableSize, want)) layer.drawableSize = want;
        const BOOL vsync = display_settings::GetVSync() ? YES : NO;   // D1649
        if (layer.displaySyncEnabled != vsync) layer.displaySyncEnabled = vsync;
        id<CAMetalDrawable> drawable = [layer nextDrawable];
        if (!drawable)
        {
            if (!c.loggedNoDrawable)
            {
                c.loggedNoDrawable = true;
                std::fprintf(stderr, "[gpu-metal] no drawable this frame (window %dx%d); frame skipped\n", pw, ph);
            }
            Submit(false);   // the frame's work (e.g. the front-buffer upload) still goes
            return true;     // D1376 / D1433: a frame that does not reach the screen is not an error
        }
        id<MTLTexture> back = drawable.texture;
        const uint32_t bufW = uint32_t(back.width), bufH = uint32_t(back.height);
        // D2024-2027: the window's 16:9 area, whole pixels.
        const float s = std::min(float(bufW) / 1280.0f, float(bufH) / 720.0f);
        const uint32_t vw = std::max(1u, uint32_t(std::lround(1280.0f * s))), vh = std::max(1u, uint32_t(std::lround(720.0f * s)));
        const uint32_t vx = (bufW - std::min(vw, bufW)) / 2, vy = (bufH - std::min(vh, bufH)) / 2;
        const MTLViewport out{ double(vx), double(vy), double(vw), double(vh), 0.0, 1.0 };       // D2027
        const MTLScissorRect outSc{ vx, vy, std::min(vw, bufW - vx), std::min(vh, bufH - vy) };  // D2001 (inside the drawable)
        id<MTLRenderPipelineState> display = c.display[(swap ? 1 : 0) | (gammaApplied ? 2 : 0)];   // D2041
        id<MTLTexture> lut = gammaApplied ? gammaLut : nil;
        const uint32_t sw = uint32_t(source.width), sh = uint32_t(source.height);               // D2042
        const int sharpness = display_settings::GetSharpness(), aa = display_settings::GetAntiAliasing();   // D2043
        if (aa && !c.loggedFxaa) { c.loggedFxaa = true; std::fprintf(stderr, "[gpu-metal] FXAA not implemented (M5)\n"); }
        if (sharpness && !c.loggedCas) { c.loggedCas = true; std::fprintf(stderr, "[gpu-metal] CAS not implemented (M5)\n"); }
        const bool resample = sw != vw || sh != vh;                                              // D2044
        PresentCB cb{ float(sw), float(sh), float(vw), float(vh), float(vx), float(vy), 0.0f, 0 };   // D2045
        EndPass();                                                                               // before Pass opens its encoders
        id<MTLCommandBuffer> cmd = Cmd();
        // D2046: sizes equal (FXAA / CAS skipped in M1): the display pass writes the drawable directly.
        if (!resample) Pass(cmd, back, true, out, outSc, display, source, lut, &cb, sizeof(cb));
        else
        {
            // D2047-2051: display pass at render size into Post(0).
            if (!c.post0 || c.post0.width != sw || c.post0.height != sh)
                c.post0 = RenderTexture(device, sw, sh, "present post target");
            const MTLViewport full{ 0.0, 0.0, double(sw), double(sh), 0.0, 1.0 };
            const MTLScissorRect fullSc{ 0, 0, sw, sh };
            const PresentCB at0{ float(sw), float(sh), float(sw), float(sh), 0.0f, 0.0f, 0.0f, 0 };
            Pass(cmd, c.post0, false, full, fullSc, display, source, lut, &at0, sizeof(at0));
            cb.mode = vw > sw ? 1u : 2u;                                                         // D2058
            if (cb.mode == 1u && MetalFxRequested() && !c.loggedMetalFx)                          // D2062
            {
                c.loggedMetalFx = true;
                std::fprintf(stderr, "[gpu-metal] MetalFX not implemented (M5)\n");
            }
            Pass(cmd, back, true, out, outSc, c.scale, c.post0, nil, &cb, sizeof(cb));           // D2099
        }
        [cmd presentDrawable:drawable];
        Submit(false);
        return true;
    }
}

bool Backend::Impl::ReadLogical(id<MTLTexture> tex, uint32_t logicalW, uint32_t logicalH, std::vector<uint32_t>& out)
{
    constexpr uint32_t w = kFrontW, h = kFrontH;
    if (!tex || logicalW < w || logicalH < h) return false;
    @autoreleasepool {
        Initialize();
        CoreState& c = Core();
        if (!c.readBuffer)
        {
            c.readBuffer = [device newBufferWithLength:size_t(w) * h * 4 options:MTLResourceStorageModeShared];
            if (!c.readBuffer) throw Failure("metal: readback buffer");
        }
        EndPass();                                                                               // before Pass / the blit encoder
        id<MTLCommandBuffer> cmd = Cmd();
        id<MTLTexture> from = tex;
        // D4891: at scale 1 a plain copy (BGRA8 already); otherwise read on the GPU at logical pixel centres.
        if (!(tex.width == logicalW && tex.height == logicalH && tex.pixelFormat == MTLPixelFormatBGRA8Unorm))
        {
            if (!c.logicalRead) c.logicalRead = RenderTexture(device, w, h, "logical read target");   // D4892-4899
            const float k[4] = { 1.0f / float(logicalW), 1.0f / float(logicalH), 0.0f, 0.0f };      // D4901
            const MTLViewport vp{ 0.0, 0.0, double(w), double(h), 0.0, 1.0 };
            const MTLScissorRect sc{ 0, 0, w, h };
            Pass(cmd, c.logicalRead, false, vp, sc, Filterable(tex.pixelFormat) ? c.resampleLinear : c.resamplePoint,
                 tex, nil, k, sizeof(k));
            from = c.logicalRead;
        }
        // D4862-4880 (ReadTexture): copy the first w x h texels to CPU-visible memory.
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (!blit) throw Failure("metal: blit command encoder");
        [blit copyFromTexture:from sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
                     toBuffer:c.readBuffer destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:size_t(w) * h * 4];
        [blit endEncoding];
        Submit(true);   // everything queued before (the frame) lands first
        out.resize(size_t(w) * h);
        std::memcpy(out.data(), c.readBuffer.contents, out.size() * sizeof(uint32_t));
        for (uint32_t& v : out) v = DisplayPixel(*this, v);                                      // D1483
        return true;
    }
}

// ================================================================ test hooks (not used by gpu_render.cpp)
Backend::Impl& MetalImplForTest(Backend& b)
{
    if (g_testBackend.load() != &b || !g_testImpl.load()) throw Failure("metal: test hook on an unknown Backend");
    return *g_testImpl.load();
}

void MetalSetSdlWindowForTest(Backend& b, SDL_Window* window)
{
    MetalImplForTest(b).window.store(window);
}

bool MetalPresentForTest(Backend& b, id<MTLTexture> source, uint32_t logicalW, uint32_t logicalH, bool swap)
{
    @autoreleasepool {
        Backend::Impl& m = MetalImplForTest(b);
        m.Initialize();
        if (!source || !m.window.load()) return false;
        if (!m.PresentTexture(source, logicalW, logicalH, swap)) return false;
        m.lastSwap = swap;                                                       // as Present, D1335
        m.lastFront = source; m.lastFrontW = logicalW; m.lastFrontH = logicalH; // D1436
        return true;
    }
}

} // namespace ttt2_hw
