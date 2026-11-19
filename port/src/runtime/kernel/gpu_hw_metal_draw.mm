// R4 Metal backend, draw (gpu_hw_metal_impl.h "draw" section, M1 slice 2): Draw (CPU vertex stage),
// VertexShader / DrawGpuVs (M2: null / Fallback), the shared pixel stage, constant upload, the render encoder.
// Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11 backend stays the reference).
//
// Surfaces (Color / Depth / Grow) and textures (BindTexture / Sampler) are other slices; the slice-2 harness
// (m1/draw_test.mm) defines its own.
//
// Clear on load: the render encoder ALWAYS loads its attachments (MTLLoadActionLoad). Grow owns the content of
// a new or grown surface: it leaves the texture fully defined (old area copied, new area 0 / depth 1.0 stencil 0,
// D3994, D4015, D4038-4045) using its own passes on Cmd() after EndPass(). So no "clear on load" flag crosses
// the contract; a pass opened by Draw never clears anything.
//
// Not ported in M1 (each named): ShadowCaster (D381, shadow tracking arrives with Resolve, slice 4),
// UpdateScale (D380: core fixes the scale at Initialize), RasterPool (D413-418: vertices convert serially),
// IndexUploadCache bookkeeping (D444), native-size surfaces (D784-813: sc = render scale), texture-state /
// texture-spec / no-discard variants (D695-756: generic shader only, M3), HW toggles / overrides / dumps
// (D713-723, D825-837, D850), stats (stats_.*).

#include "gpu_hw_metal_impl.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ttt2_hw {

namespace {

using Impl = Backend::Impl;

constexpr size_t VertexRingBytes = 64u << 20, IndexRingBytes = 16u << 20;   // D2105 (per-draw limits, D398)
constexpr size_t kChunkBytes = 8u << 20;      // arena chunk; a bigger request gets a buffer of its own
constexpr size_t kAlign = 256;                // every allocation (vertex, index, cb, tables) on 256 B
constexpr size_t kCb0Bytes = 256 * 16;        // D282: c[256]

// ---- per-command-buffer upload arena (replaces D3D11's vertex / index / constant rings and their fences,
// D2799-2868, D2876-3040). Every byte a draw uploads lives in the arena of the frame's command buffer; the
// arena goes back to the free list only from that command buffer's completed handler, so nothing the GPU may
// still read is ever overwritten (Present commits without waiting).
struct Chunk { id<MTLBuffer> buf = nil; size_t used = 0; };
struct Arena { std::vector<Chunk> chunks; };
struct Slice { id<MTLBuffer> buf = nil; size_t offset = 0; uint8_t* ptr = nullptr; };

struct DrawShared {
    std::mutex freeMutex;                             // the completed handler runs on a Metal thread
    std::vector<std::shared_ptr<Arena>> freeArenas;
    std::shared_ptr<Arena> arena;                     // the arena of `arenaCmd`
    id<MTLCommandBuffer> arenaCmd = nil;              // strong: its address cannot be reused while held
};
DrawShared& G() { static DrawShared* s = new DrawShared; return *s; }   // leaked with the Backend (D5140)

Slice Alloc(Impl& m, size_t bytes)
{
    DrawShared& g = G();
    id<MTLCommandBuffer> cmd = m.Cmd();
    if (cmd != g.arenaCmd)
    {
        std::shared_ptr<Arena> a;
        {
            std::lock_guard lock(g.freeMutex);
            if (!g.freeArenas.empty()) { a = std::move(g.freeArenas.back()); g.freeArenas.pop_back(); }
        }
        if (!a) a = std::make_shared<Arena>();
        for (Chunk& c : a->chunks) c.used = 0;
        std::shared_ptr<Arena> back = a;
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
            (void)done;
            std::lock_guard lock(G().freeMutex);
            G().freeArenas.push_back(back);
        }];
        g.arena = std::move(a);
        g.arenaCmd = cmd;
    }
    bytes = std::max<size_t>(bytes, 1);
    Arena& a = *g.arena;
    for (Chunk& c : a.chunks)
    {
        const size_t at = (c.used + kAlign - 1) / kAlign * kAlign;
        if (at + bytes <= c.buf.length)
        {
            c.used = at + bytes;
            return { c.buf, at, static_cast<uint8_t*>(c.buf.contents) + at };
        }
    }
    id<MTLBuffer> buf = [m.device newBufferWithLength:std::max(kChunkBytes, bytes) options:MTLResourceStorageModeShared];
    if (!buf) throw Failure("metal: upload arena buffer (" + std::to_string(bytes) + " B)");
    a.chunks.push_back({ buf, bytes });
    return { buf, 0, static_cast<uint8_t*>(buf.contents) };
}

uint32_t Px(double logical, float scale) { return uint32_t(std::floor(logical * scale + 0.5)); }   // D3874

// D4296-4301 StencilRefMask / StencilRef.
uint32_t StencilRefMask(const DrawState& st)
{
    const bool backOnly = ((st.depthCtl >> 7) & 1) && (st.cull & 3) == 1;                                // D4298
    return backOnly ? st.stencilRefMaskBf : st.stencilRefMask;                                           // D4299
}
uint32_t StencilRef(const DrawState& st) { return st.stencil ? (StencilRefMask(st) & 0xFF) : 0; }        // D4301

uint64_t Rid(MTLResourceID r) { uint64_t u; std::memcpy(&u, &r, 8); return u; }
uint32_t FloatBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// "<prefix><digits>" -> the number, else -1 (airconv field names: sN, cube_sN, meta_sN, tN, meta_tN).
int Slot(const std::string& name, const char* prefix)
{
    const size_t n = std::strlen(prefix);
    if (name.size() <= n || name.compare(0, n, prefix) != 0) return -1;
    int v = 0;
    for (size_t i = n; i < name.size(); ++i)
    {
        if (name[i] < '0' || name[i] > '9' || v > 1000) return -1;
        v = v * 10 + (name[i] - '0');
    }
    return v;
}

// D671-857 PixelStageAndDraw, CPU vertex path (vertex input already uploaded): constants, textures and samplers,
// output merger, viewport, scissor, rasterizer state, the draw.
void PixelStageAndDraw(Impl& m, const uint32_t* regs, const Impl::Ps& ps, MtlColorSurface& color, const DrawState& st,
                       const Slice& vb, uint32_t interps, const Slice& ib, uint32_t indexCount, float reachX, float reachY)
{
    const uint32_t textureSlots = ps.slots, cubeSlots = ps.cubeSlots;                                    // D391 (lastSlots_, lastCubeSlots_)
    // ---- constants and per-draw shader state
    const uint32_t psRows = std::min<uint32_t>(ps.constantRows, 256);                                    // D681 (lastPsRows_)
    PsStateCB cb{};                                                                                      // D689
    std::memcpy(cb.boolConsts, regs + 0x4900, sizeof(cb.boolConsts));                                    // D690
    std::memcpy(cb.loopConsts, regs + 0x4908, sizeof(cb.loopConsts));                                    // D691
    cb.alphaRef = st.alphaRef; cb.alphaTest = st.alphaTest ? 1 : 0; cb.alphaFunc = st.alphaFunc;        // D692
    id<MTLTexture> textures[32] = {};                                                                    // D693 srvs
    Impl::SamplerSet samplers[16];                                                                       // D694
    for (uint32_t slot = 0; slot < 32; ++slot)                                                           // D701
    {
        if (!(textureSlots & (1u << slot))) continue;                                                    // D703
        const Impl::Bound b = m.BindTexture(regs, slot, ((cubeSlots >> slot) & 1) != 0);                 // D704
        textures[slot] = b.tex;                                                                          // D705
        cb.texInfo[slot][0] = b.swizzle; cb.texInfo[slot][1] = b.mode;                                   // D706
        cb.texInfo[slot][2] = b.width; cb.texInfo[slot][3] = b.height;                                   // D707
        cb.texScale[slot][0] = b.scaleX; cb.texScale[slot][1] = b.scaleY; cb.texScale[slot][2] = float(b.slices);   // D708
        std::memcpy(cb.texSampler[slot], b.info, sizeof(cb.texSampler[slot]));                           // D709-710 (SamplerInfo, from BindTexture)
        if (b.slices > 1) cb.texSampler[slot][0] = cb.texSampler[slot][1] = 2;                           // D711: cube faces clamp
        samplers[slot & 15] = m.Sampler(slot, cb.texSampler[slot], b.mip);                               // D728-729
    }
    // D695-756: the texture-state, texture-spec and no-discard variants are M3; the generic shader draws.

    // ---- output merger: the surfaces cover the scissored area (logical samples); a bound colour / depth pair is
    // grown to one size (Metal attachments, like D3D11 views, must agree).
    const int64_t x1 = std::clamp<int64_t>(int64_t(st.scX1) * st.sampX, 0, SurfaceW);                   // D765
    const int64_t y1 = std::clamp<int64_t>(int64_t(st.scY1) * st.sampY, 0, SurfaceH);                   // D766
    MtlDepthSurface* depth = nullptr;                                                                    // D767
    if (st.zEnable || st.stencil)                                                                        // D768
    {
        depth = &m.Depth(st.depthBase);                                                                  // D770
        depth->touched = true; depth->sampX = st.sampX; depth->sampY = st.sampY;                         // D771
    }
    auto reach = [](int64_t scissor, float geometry, uint32_t samples) {                                 // D774-777
        return uint32_t(std::isfinite(geometry) ? std::clamp<int64_t>(int64_t(std::min(geometry, 65536.0f)) * samples, 0, scissor)
                                                : scissor);
    };
    uint32_t needW = reach(x1, reachX, st.sampX), needH = reach(y1, reachY, st.sampY);                  // D778
    if (st.writeMask && depth)                                                                           // D779
    {
        needW = std::max({ needW, color.extW, depth->extW });                                            // D781
        needH = std::max({ needH, color.extH, depth->extH });                                            // D782
    }
    // D784-811 native-size surfaces: not in M1 (needs Resolve's readback tracking), always the render scale.
    const float sc = m.scale;                                                                            // D810
    if (st.writeMask) m.Grow(color, st.colorBase, needW, needH, sc);                                     // D812
    if (depth) m.Grow(*depth, st.depthBase, needW, needH, sc);                                           // D813
    id<MTLTexture> colorTex = st.writeMask ? color.tex : nil;                                            // D814 (re-read after Grow)
    id<MTLTexture> depthTex = depth ? depth->tex : nil;                                                  // D815
    if (st.writeMask && !colorTex) throw Failure("metal: Grow left colour surface without a texture");
    if (depth && !depthTex) throw Failure("metal: Grow left depth surface without a texture");
    if (!colorTex && !depthTex) return;                       // D814-816: nothing bound, nothing written (no pass to open)

    // D821 viewport: the full sample space (2048 rows), times the render scale; vertices stay logical.
    const MTLViewport vp{ 0.0, 0.0, double(st.pitch * st.sampX) * sc, double(SurfaceH) * sc, 0.0, 1.0 };
    // D823-824 scissor, clipped to the attachment (Metal refuses a scissor outside the render target).
    uint32_t attW = uint32_t(colorTex ? colorTex.width : depthTex.width), attH = uint32_t(colorTex ? colorTex.height : depthTex.height);
    if (colorTex && depthTex) { attW = std::min(attW, uint32_t(depthTex.width)); attH = std::min(attH, uint32_t(depthTex.height)); }
    const uint32_t left = std::min(Px(std::max(st.scX0, 0) * int(st.sampX), sc), attW);                 // D823
    const uint32_t top = std::min(Px(std::max(st.scY0, 0) * int(st.sampY), sc), attH);
    const uint32_t right = std::min(Px(double(x1), sc), attW), bottom = std::min(Px(double(y1), sc), attH);   // D824
    if ((st.cull & 3) == 3) return;                                                                      // D828: both faces culled
    if (right <= left || bottom <= top) return;               // empty scissor: draws nothing (as D3D11), no zero-size rect

    id<MTLRenderPipelineState> pipeline = m.Pipeline(ps, interps, colorTex ? colorTex.pixelFormat : MTLPixelFormatInvalid,
                                                     depthTex ? depthTex.pixelFormat : MTLPixelFormatInvalid, st.blend, st.writeMask);
    id<MTLDepthStencilState> depthState = m.DepthState(st);                                              // D818

    // ---- cb0 (D682-688: the first psRows rows of regs + 0x4000 + 256 * 4) and cb1 (PsStateCB, D757-761) into the
    // frame's arena; their gpuAddress goes to the cbuffer table (BINDINGS.md: {cb0, cb1}, or {cb1} when the
    // shader declares no cb0, checked when the layout was loaded).
    std::vector<id<MTLBuffer>> used;                                                                     // reached through tables: useResource
    auto use = [&used](id<MTLBuffer> b) { if (std::find(used.begin(), used.end(), b) == used.end()) used.push_back(b); };
    Slice cb1 = Alloc(m, sizeof(PsStateCB));
    std::memcpy(cb1.ptr, &cb, sizeof(cb));
    use(cb1.buf);
    Slice cbTable = Alloc(m, 8 * size_t(ps.cbCount));
    uint64_t cbAddr[2] = { 0, 0 };
    m.lastCb0.clear();
    if (ps.cbCount == 2)
    {
        Slice cb0 = Alloc(m, kCb0Bytes);
        std::memcpy(cb0.ptr, regs + 0x4000 + 256 * 4, size_t(psRows) * 16);                             // D684-685
        std::memset(cb0.ptr + size_t(psRows) * 16, 0, kCb0Bytes - size_t(psRows) * 16);
        use(cb0.buf);
        cbAddr[0] = cb0.buf.gpuAddress + cb0.offset;
        cbAddr[1] = cb1.buf.gpuAddress + cb1.offset;
        m.lastCb0.assign(cb0.ptr, cb0.ptr + size_t(psRows) * 16);
    }
    else if (ps.cbCount == 1) cbAddr[0] = cb1.buf.gpuAddress + cb1.offset;
    else throw Failure("metal: cbuffer table with " + std::to_string(ps.cbCount) + " entries not implemented (M1 slice 2)");
    std::memcpy(cbTable.ptr, cbAddr, 8 * size_t(ps.cbCount));
    m.lastCb1.assign(reinterpret_cast<const uint8_t*>(&cb), reinterpret_cast<const uint8_t*>(&cb) + sizeof(cb));

    // ---- binding table (BINDINGS.md), filled by field name from the layout: sN = sampler register N (xs[slot & 15],
    // D729), cube_sN its cube variant, meta_sN = LOD-bias float bits, tN = texture slot N, meta_tN =
    // (arrayLength << 32) | bits(minLOD = 0).
    std::vector<id<MTLTexture>> usedTextures;
    Slice argTable;
    if (ps.argBytes)
    {
        argTable = Alloc(m, ps.argBytes);
        std::memset(argTable.ptr, 0, ps.argBytes);
        for (const Impl::ArgField& f : ps.args)
        {
            if (f.offset + 8 > ps.argBytes) throw Failure("metal: binding_table field " + f.name + " past the table");
            uint64_t v = 0;
            int n;
            if ((n = Slot(f.name, "s")) >= 0 || (n = Slot(f.name, "cube_s")) >= 0)
            {
                const bool cube = f.name[0] == 'c';
                id<MTLSamplerState> s = n < 16 ? (cube ? samplers[n].cube : samplers[n].s) : nil;
                if (!s) throw Failure("metal: binding_table " + f.name + ": no sampler for register " + std::to_string(n));
                v = Rid(s.gpuResourceID);
            }
            else if ((n = Slot(f.name, "meta_s")) >= 0)
            {
                if (n >= 16) throw Failure("metal: binding_table " + f.name + ": sampler register out of range");
                v = FloatBits(samplers[n].lodBias);
            }
            else if ((n = Slot(f.name, "t")) >= 0 || (n = Slot(f.name, "meta_t")) >= 0)
            {
                id<MTLTexture> t = n < 32 ? textures[n] : nil;
                if (!t) throw Failure("metal: binding_table " + f.name + ": no texture for slot " + std::to_string(n));
                if (f.name[0] == 'm') v = (uint64_t(t.arrayLength) << 32) | FloatBits(0.0f);
                else
                {
                    v = Rid(t.gpuResourceID);
                    if (std::find(usedTextures.begin(), usedTextures.end(), t) == usedTextures.end()) usedTextures.push_back(t);
                }
            }
            else throw Failure("metal: binding_table field " + f.name + " (" + f.type + ") not implemented (M1 slice 2)");
            std::memcpy(argTable.ptr + f.offset, &v, 8);
        }
    }

    // ---- the render encoder: one open at a time, reused while the targets stay the same.
    if (!m.renderEnc || m.encColor != colorTex || m.encDepth != depthTex)                                // D816 (BindRenderTargets)
    {
        m.EndPass();
        MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
        if (colorTex)
        {
            rp.colorAttachments[0].texture = colorTex;
            rp.colorAttachments[0].loadAction = MTLLoadActionLoad;      // Grow owns new content (header comment)
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
        if (depthTex)
        {
            rp.depthAttachment.texture = depthTex;
            rp.depthAttachment.loadAction = MTLLoadActionLoad;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            if (depthTex.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
            {
                rp.stencilAttachment.texture = depthTex;
                rp.stencilAttachment.loadAction = MTLLoadActionLoad;
                rp.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
        id<MTLRenderCommandEncoder> enc = [m.Cmd() renderCommandEncoderWithDescriptor:rp];
        if (!enc) throw Failure("metal: render command encoder");
        m.renderEnc = enc;
        m.encColor = colorTex;
        m.encDepth = depthTex;
    }
    id<MTLRenderCommandEncoder> enc = m.renderEnc;
    // Every piece of encoder state, every draw (the encoder may carry the previous draw's).
    [enc setRenderPipelineState:pipeline];
    [enc setBlendColorRed:st.blendConst[0] green:st.blendConst[1] blue:st.blendConst[2] alpha:st.blendConst[3]];   // D817 (factor)
    [enc setDepthStencilState:depthState];                                                               // D818
    [enc setStencilReferenceValue:StencilRef(st)];                                                       // D818, D4301
    [enc setViewport:vp];                                                                                // D822
    const MTLScissorRect scissor{ left, top, right - left, bottom - top };
    [enc setScissorRect:scissor];                                                                        // D827
    // D4274-4291 RasterState: cull bit0 front, bit1 back; front face from frontIsCW (the viewport flips y the same
    // way in both APIs, D4272); no depth clip -> clamp (Xenos clip_disable, D4285-4287).
    [enc setCullMode:(st.cull & 1) ? MTLCullModeFront : (st.cull & 2) ? MTLCullModeBack : MTLCullModeNone];   // D4281
    [enc setFrontFacingWinding:st.frontIsCW ? MTLWindingClockwise : MTLWindingCounterClockwise];         // D4282
    [enc setDepthClipMode:st.depthClip ? MTLDepthClipModeClip : MTLDepthClipModeClamp];                  // D4286
    [enc setVertexBuffer:vb.buf offset:vb.offset atIndex:0];                                             // D453 (IASetVertexBuffers)
    [enc setFragmentBuffer:cbTable.buf offset:cbTable.offset atIndex:0];                                 // D838-848 (b0, b1)
    if (argTable.buf) [enc setFragmentBuffer:argTable.buf offset:argTable.offset atIndex:1];             // D849-850 (t0..31, s0..15)
    for (id<MTLBuffer> b : used) [enc useResource:b usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    for (id<MTLTexture> t : usedTextures) [enc useResource:t usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:indexCount indexType:MTLIndexTypeUInt32
                   indexBuffer:ib.buf indexBufferOffset:ib.offset];                                      // D853 (DrawIndexed)
}

} // namespace

// ================================================================ Backend (public, draw part)
bool Backend::Draw(const uint32_t* regs, const std::vector<uint32_t>& psCode,
                   const std::vector<CpuVertex>& vertices, const std::vector<uint32_t>& slots,
                   uint32_t interps, const DrawState& st)
{
    @autoreleasepool {
        Impl& m = *impl_;
        m.Initialize();                                                                                  // D379
        // D380 UpdateScale, D381 ShadowCaster: not in M1 (header comment).
        MtlColorSurface& color = m.Color(st.colorBase);                                                  // D383
        color.touched = true;                                                                            // D385
        color.samplePitch = st.pitch * st.sampX;                                                         // D386
        if (slots.empty()) return true;                                                                  // D387

        const Impl::Ps* ps = m.PixelShader(psCode);                                                      // D389
        if (!ps) return false;                                                                           // D390

        // ---- vertices: window space -> clip space for the sample-space viewport
        interps = std::min<uint32_t>(interps, 16);                                                       // D394
        const size_t stride = 16 + 16 * size_t(interps);                                                 // D395
        const size_t total = vertices.size();                                                            // D396
        const size_t vbBytes = total * stride, ibBytes = slots.size() * 4;                               // D397
        if (vbBytes + stride > VertexRingBytes || ibBytes > IndexRingBytes) return false;                // D398
        const Slice vb = Alloc(m, vbBytes);                                                              // D399-408 (arena, not a ring)
        const bool noClip = !st.depthClip;                                                               // D408
        for (size_t i = 0; i < total; ++i)                                                               // D409-418 (serial: no RasterPool)
        {
            const HwVertex packed = ConvertCpuVertex(vertices[i], st, interps);                          // D410
            std::memcpy(vb.ptr + i * stride, &packed, stride);                                           // D411
        }
        // How far right / down the draw can reach (logical pixels), for surface sizing: the vertices' window
        // positions bound every triangle unless one is clipped at W = 0 (negative W with clipping on).
        float reachX = std::numeric_limits<float>::infinity(), reachY = reachX;                         // D425
        {
            float mx = 0.0f, my = 0.0f;                                                                  // D427
            bool bounded = true;                                                                         // D428
            for (size_t i = 0; i < total && bounded; ++i)                                                // D429
            {
                const CpuVertex& v = vertices[i];                                                        // D431
                const float w = v.invW != 0.0f ? 1.0f / v.invW : 1.0f;                                   // D432
                bounded = (w > 0.0f || noClip) && std::isfinite(v.x) && std::isfinite(v.y);              // D433
                mx = std::max(mx, v.x); my = std::max(my, v.y);                                          // D434
            }
            if (bounded) { reachX = std::ceil(mx); reachY = std::ceil(my); }                             // D436
        }
        // D439-448 indices: the u32 slots verbatim (three per triangle, indexing the packed vertices).
        const Slice ib = Alloc(m, ibBytes);
        std::memcpy(ib.ptr, slots.data(), ibBytes);                                                      // D445
        // D450-456: fixed VS Passthrough(interps), TRIANGLELIST, R32_UINT - in the pipeline (Impl::Pipeline).
        PixelStageAndDraw(m, regs, *ps, color, st, vb, interps, ib, uint32_t(slots.size()), reachX, reachY);   // D457
        return true;                                                                                     // D458
    }
}

VertexShaderHandle* Backend::VertexShader(const std::vector<uint32_t>&)
{
    return nullptr;   // M1: always the CPU vertex stage (gpu_hw_metal.h); the GPU VS path is M2
}

GpuVsResult Backend::DrawGpuVs(const uint32_t*, const std::vector<uint32_t>&, VertexShaderHandle&,
                               std::span<const uint32_t>, uint32_t, uint32_t, const DrawState&, bool, uint32_t, uint64_t)
{
    return GpuVsResult::Fallback;   // M2; nothing changed, the caller runs the CPU vertex stage
}

// ================================================================ Backend::Impl (draw section)
void Backend::Impl::EndPass()
{
    if (renderEnc) [renderEnc endEncoding];
    renderEnc = nil;
    encColor = nil;
    encDepth = nil;
}

} // namespace ttt2_hw
