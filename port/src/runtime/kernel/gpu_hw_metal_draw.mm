// R4 Metal backend, draw (gpu_hw_metal_impl.h "draw" section, M1 slice 2): Draw (CPU vertex stage),
// DrawGpuVs (M2b: the converted VS on the GPU), the shared pixel stage, constant upload, the render encoder.
// Backend::VertexShader lives in gpu_hw_metal_shaders.mm (it needs the translator).
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
//
// DrawGpuVs (M2b) ports D515-666. Not ported (each named): UpdateScale (D520, as Draw), CensusVertex (D526),
// toggle_ & 2 (D529), ShadowCaster (D530, as Draw), GpuNoteMemory / VcacheStat (D544-550: diagnostics; textures
// skips them too), the vertex-data ring and its fences / NO_OVERWRITE / VertexUploadCache reuse (D555-583: the
// per-command-buffer arena replaces every ring, so each draw packs its own data), the "upload only when changed"
// VS constants (D601-606: the arena is per draw), the IndexUploadCache (D622-639, indexIdentity unused).

// Before the Metal headers (as shaders.mm does with the translator): plain C++, no Apple macro touches its names.
#include "gpu_hw_vertex_fetch.h"

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

extern uint8_t* g_base;   // kernel_shared.h: global guest memory base (declared directly, as surfaces.mm does)

namespace ttt2_hw {

namespace {

using Impl = Backend::Impl;

constexpr size_t VertexRingBytes = 64u << 20, IndexRingBytes = 16u << 20;   // D2105 (per-draw limits, D398)
constexpr size_t VdataRingBytes = 64u << 20;                                // D2105 (GPU-VS per-draw limit, D528)
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
float AsFloat(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }                                // D613 AsFloat

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

// What feeds the vertex stage (D3D11: what the caller bound before PixelStageAndDraw, D450-456 / D641-661).
//   Draw (vs == nullptr): vb = the packed clip-space vertices (stage_in, vertex buffer 0, fixed VS for `interps`).
//   DrawGpuVs: vb = the raw vertex data the converted VS reads as t0 (packed with baseDword 0, so vfRegion counts
//   dwords from vb's offset and stays valid if vb moves), vbBytes = its length (meta_t0); vsState = its cb1,
//   vsRows rows of regs + 0x4000 = its cb0. Both are uploaded here, after the BindTexture loop (see below).
struct VertexInput {
    Slice vb; size_t vbBytes = 0;
    uint32_t interps = 0;                       // Draw
    const Impl::Vs* vs = nullptr;               // DrawGpuVs
    const VsStateCB* vsState = nullptr;
    uint32_t vsRows = 0;
    bool strip = false;                         // D642-643 TRIANGLESTRIP (UINT32_MAX restarts) instead of TRIANGLELIST
    uint32_t triangles = UINT32_MAX;            // D664 / D855 triangleCount (UINT32_MAX: indexCount / 3)
};

// D668-857 PixelStageAndDraw, shared tail of both vertex paths (vertex input already uploaded): constants, textures
// and samplers, output merger, viewport, scissor, rasterizer state, the draw.
void PixelStageAndDraw(Impl& m, const uint32_t* regs, const Impl::Ps& ps, MtlColorSurface& color, const DrawState& st,
                       VertexInput vin, Slice ib, uint32_t indexCount, float reachX, float reachY)
{
    Slice& vb = vin.vb;
    const size_t vbBytes = vin.vbBytes;
    id<MTLCommandBuffer> uploadCmd = m.frameCmd;   // the command buffer whose arena holds vb / ib
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
    // A synchronous WriteBack inside BindTexture (D4525-4535) commits and waits for the frame's command buffer: its
    // arena returns to the free list and the next Alloc may reuse it from offset 0. Copy vb / ib out BEFORE any new
    // Alloc, then upload them again into the new command buffer's arena.
    if (m.frameCmd != uploadCmd)
    {
        const size_t ibBytes = size_t(indexCount) * 4;
        std::vector<uint8_t> keep(vbBytes + ibBytes);
        std::memcpy(keep.data(), vb.ptr, vbBytes);
        std::memcpy(keep.data() + vbBytes, ib.ptr, ibBytes);
        vb = Alloc(m, vbBytes); std::memcpy(vb.ptr, keep.data(), vbBytes);
        ib = Alloc(m, ibBytes); std::memcpy(ib.ptr, keep.data() + vbBytes, ibBytes);
    }

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
    if ((st.cull & 3) == 3) { ++m.stats.emptyDraws; return; }                                            // D828: both faces culled
    m.stats.triangles += vin.triangles == UINT32_MAX ? indexCount / 3 : vin.triangles;                   // D855 (D3D11 counts them from here on, even for an empty scissor)
    if (right <= left || bottom <= top) return;               // empty scissor: draws nothing (as D3D11), no zero-size rect

    const MTLPixelFormat colorFormat = colorTex ? colorTex.pixelFormat : MTLPixelFormatInvalid;
    const MTLPixelFormat depthFormat = depthTex ? depthTex.pixelFormat : MTLPixelFormatInvalid;
    id<MTLRenderPipelineState> pipeline = vin.vs ? m.Pipeline(*vin.vs, ps, colorFormat, depthFormat, st.blend, st.writeMask)   // D647
                                                 : m.Pipeline(ps, vin.interps, colorFormat, depthFormat, st.blend, st.writeMask);
    id<MTLDepthStencilState> depthState = m.DepthState(st);                                              // D818

    // ---- cb0 (D682-688: the first psRows rows of regs + 0x4000 + 256 * 4) and cb1 (PsStateCB, D757-761) into the
    // frame's arena; their gpuAddress goes to the cbuffer table (BINDINGS.md: {cb0, cb1}, or {cb1} when the
    // shader declares no cb0, checked when the layout was loaded).
    // Reached through tables: useResource, ONE call per buffer with every stage that reads it (an arena chunk can hold
    // vertex and fragment data at once).
    std::vector<std::pair<id<MTLBuffer>, MTLRenderStages>> used;
    auto use = [&used](id<MTLBuffer> b, MTLRenderStages stages) {
        for (auto& u : used)
            if (u.first == b) { u.second |= stages; return; }
        used.push_back({ b, stages });
    };
    Slice cb1 = Alloc(m, sizeof(PsStateCB));
    std::memcpy(cb1.ptr, &cb, sizeof(cb));
    use(cb1.buf, MTLRenderStageFragment);
    Slice cbTable = Alloc(m, 8 * size_t(ps.cbCount));
    uint64_t cbAddr[2] = { 0, 0 };
    m.lastCb0.clear();
    if (ps.cbCount == 2)
    {
        Slice cb0 = Alloc(m, kCb0Bytes);
        std::memcpy(cb0.ptr, regs + 0x4000 + 256 * 4, size_t(psRows) * 16);                             // D684-685
        std::memset(cb0.ptr + size_t(psRows) * 16, 0, kCb0Bytes - size_t(psRows) * 16);
        use(cb0.buf, MTLRenderStageFragment);
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

    // ---- M2b, the converted VS's tables (gpu_hw_metal_impl.h "vertex shaders"), allocated here, after the frameCmd
    // check above, so no gpuAddress they hold can point into a recycled arena.
    //   cb0 (D597-609, XenosVsConstants): the first vsRows rows of regs + 0x4000, vsRows * 16 B (at least 16 B).
    //   cb1 (D610-620, XenosVsState): VsStateCB, 2256 B.
    //   cbuffer_table: {cb0, cb1} 16 B or {cb1} 8 B. binding_table 16 B: t0 = vb's gpuAddress, meta_t0 = vbBytes.
    Slice vsCbTable, vsArgTable;
    if (vin.vs)
    {
        const Impl::Vs& vs = *vin.vs;
        if (!vin.vsState) throw Failure("metal: GPU-VS draw without its VsStateCB");
        Slice vsCb1 = Alloc(m, sizeof(VsStateCB));
        std::memcpy(vsCb1.ptr, vin.vsState, sizeof(VsStateCB));                                          // D620
        use(vsCb1.buf, MTLRenderStageVertex);
        uint64_t vsCbAddr[2] = { 0, 0 };
        if (vs.cbCount == 2)
        {
            const size_t cb0Bytes = size_t(vin.vsRows) * 16;
            Slice vsCb0 = Alloc(m, std::max<size_t>(cb0Bytes, 16));
            if (cb0Bytes) std::memcpy(vsCb0.ptr, regs + 0x4000, cb0Bytes);                               // D606-607
            else std::memset(vsCb0.ptr, 0, 16);
            use(vsCb0.buf, MTLRenderStageVertex);
            vsCbAddr[0] = vsCb0.buf.gpuAddress + vsCb0.offset;
            vsCbAddr[1] = vsCb1.buf.gpuAddress + vsCb1.offset;
        }
        else if (vs.cbCount == 1) vsCbAddr[0] = vsCb1.buf.gpuAddress + vsCb1.offset;
        else throw Failure("metal: VS cbuffer table with " + std::to_string(vs.cbCount) + " entries not implemented (M2b)");
        vsCbTable = Alloc(m, 8 * size_t(vs.cbCount));
        std::memcpy(vsCbTable.ptr, vsCbAddr, 8 * size_t(vs.cbCount));                                    // D648-659 (VS b0, b1)
        if (vs.argBytes != 16) throw Failure("metal: VS binding_table of " + std::to_string(vs.argBytes) + " B not implemented (M2b)");
        vsArgTable = Alloc(m, vs.argBytes);
        std::memset(vsArgTable.ptr, 0, vs.argBytes);
        for (const Impl::ArgField& f : vs.args)
        {
            uint64_t v = 0;
            if (f.name == "t0") v = vb.buf.gpuAddress + vb.offset;                                       // D660-661 (VS t0 = vdata)
            else if (f.name == "meta_t0") v = uint64_t(vbBytes);                                         // BINDINGS.md "SRV buffer": byteLength
            else throw Failure("metal: VS binding_table field " + f.name + " (" + f.type + ") not implemented (M2b)");
            if (f.offset + 8 > vs.argBytes) throw Failure("metal: VS binding_table field " + f.name + " past the table");
            std::memcpy(vsArgTable.ptr + f.offset, &v, 8);
        }
        use(vb.buf, MTLRenderStageVertex);
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
    if (vin.vs)
    {
        [enc setVertexBuffer:vsCbTable.buf offset:vsCbTable.offset atIndex:0];                           // D648-659 (VS b0, b1)
        [enc setVertexBuffer:vsArgTable.buf offset:vsArgTable.offset atIndex:1];                         // D660-661 (VS t0)
    }
    else [enc setVertexBuffer:vb.buf offset:vb.offset atIndex:0];                                        // D453 (IASetVertexBuffers)
    [enc setFragmentBuffer:cbTable.buf offset:cbTable.offset atIndex:0];                                 // D838-848 (b0, b1)
    if (argTable.buf) [enc setFragmentBuffer:argTable.buf offset:argTable.offset atIndex:1];             // D849-850 (t0..31, s0..15)
    for (auto& u : used) [enc useResource:u.first usage:MTLResourceUsageRead stages:u.second];
    for (id<MTLTexture> t : usedTextures) [enc useResource:t usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    // D642-643 topology. Metal always restarts a strip at index 0xFFFFFFFF with 32-bit indices (D3D11's strip cut).
    [enc drawIndexedPrimitives:vin.strip ? MTLPrimitiveTypeTriangleStrip : MTLPrimitiveTypeTriangle
                    indexCount:indexCount indexType:MTLIndexTypeUInt32
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
        // D380 UpdateScale, D381 ShadowCaster: not in M1 (header comment), so shadowSkips stays 0.
        ++m.stats.draws;                                                                                 // D382
        MtlColorSurface& color = m.Color(st.colorBase);                                                  // D383
        color.touched = true;                                                                            // D385
        color.samplePitch = st.pitch * st.sampX;                                                         // D386
        if (slots.empty()) { ++m.stats.emptyDraws; return true; }                                        // D387

        const Impl::Ps* ps = m.PixelShader(psCode);                                                      // D389
        if (!ps) { ++m.stats.skippedDraws; return false; }                                              // D390

        // ---- vertices: window space -> clip space for the sample-space viewport
        interps = std::min<uint32_t>(interps, 16);                                                       // D394
        const size_t stride = 16 + 16 * size_t(interps);                                                 // D395
        const size_t total = vertices.size();                                                            // D396
        const size_t vbBytes = total * stride, ibBytes = slots.size() * 4;                               // D397
        if (vbBytes + stride > VertexRingBytes || ibBytes > IndexRingBytes) { ++m.stats.skippedDraws; return false; }   // D398
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
        VertexInput vin;
        vin.vb = vb;
        vin.vbBytes = vbBytes;
        vin.interps = interps;
        PixelStageAndDraw(m, regs, *ps, color, st, vin, ib, uint32_t(slots.size()), reachX, reachY);    // D457
        m.MaybeReportStats();                                                                            // D856
        return true;                                                                                     // D458
    }
}

// Backend::VertexShader: gpu_hw_metal_shaders.mm (M2b).

// D513-666 DrawGpuVs. `indices` are a triangle list or a strip with UINT32_MAX restarts; minIndex..maxIndex bound
// the guest vertex data packed. Fallback = nothing changed (the caller runs the CPU vertex stage).
GpuVsResult Backend::DrawGpuVs(const uint32_t* regs, const std::vector<uint32_t>& psCode,
                               VertexShaderHandle& shader,
                               std::span<const uint32_t> indices, uint32_t minIndex, uint32_t maxIndex, const DrawState& st,
                               bool triangleStrip, uint32_t stripTriangles, uint64_t indexIdentity)
{
    @autoreleasepool {
        (void)indexIdentity;                                     // D624 IndexUploadCache: not ported (header comment)
        Impl& m = *impl_;
        auto& vs = static_cast<Impl::Vs&>(shader);                                                       // D518
        m.Initialize();                                                                                  // D519
        // D520 UpdateScale: not ported (as Draw, D380).
        VsFetchPlan fetchPlan;                                                                           // D521
        m.FetchRanges(vs, minIndex, maxIndex, fetchPlan.lo, fetchPlan.hi, fetchPlan.used);              // D522-523
        if (!fetchPlan.Validate((16u << 20) / 4))                                                        // D524
            return GpuVsResult::Fallback;                                                                // D525
        // D526 CensusVertex: diagnostics, not ported.
        const uint64_t totalDwords = fetchPlan.totalDwords;                                              // D527
        if (totalDwords * 4 + 64 > VdataRingBytes || indices.size() * 4 > IndexRingBytes) return GpuVsResult::Fallback;   // D528
        // D529 toggle_ & 2, D530 ShadowCaster: not ported (as Draw, D381), so shadowSkips stays 0.

        ++m.stats.draws;                                                                                 // D532
        ++m.stats.gpuVsDraws;                                                                            // D533
        MtlColorSurface& color = m.Color(st.colorBase);                                                  // D534
        color.touched = true;                                                                            // D536
        color.samplePitch = st.pitch * st.sampX;                                                         // D537
        if (indices.empty()) { ++m.stats.emptyDraws; return GpuVsResult::Drawn; }                        // D538
        const Impl::Ps* ps = m.PixelShader(psCode);                                                      // D539
        if (!ps) { ++m.stats.skippedDraws; return GpuVsResult::Skipped; }                                // D540
        // D543-551 GpuNoteMemory / VcacheStat: diagnostics, not ported.

        // ---- vertex data (D553-593): packed into the frame's arena, slice-relative (baseDword 0: t0 = the slice),
        // so vfRegion stays valid if PixelStageAndDraw has to upload the slice again.
        VsStateCB vst{};                                                                                 // D553
        Slice vdata = Alloc(m, size_t(totalDwords) * 4);
        size_t vdataBytes = 0;
        if (totalDwords)                                                                                 // D554
        {
            const uint64_t cursor = PackVertexFetches(fetchPlan, regs, g_base, reinterpret_cast<uint32_t*>(vdata.ptr), 0,
                                                      vst, nullptr, nullptr);                            // D588 (no VertexUploadCache)
            if (cursor > totalDwords) throw Failure("metal: PackVertexFetches wrote past its " + std::to_string(totalDwords) + " dwords");
            vdataBytes = size_t(cursor) * 4;                                                             // D591
            m.stats.vertexDataBytes += vdataBytes;                                                       // D592
        }

        // ---- vertex constants and state (D596-620); uploaded by PixelStageAndDraw after its texture loop.
        const uint32_t vsRows = std::min<uint32_t>(vs.constantRows, 512);                                // D597 (c[512], D287)
        std::memcpy(vst.boolConsts, regs + 0x4900, sizeof(vst.boolConsts));                              // D610
        std::memcpy(vst.loopConsts, regs + 0x4908, sizeof(vst.loopConsts));                              // D611
        vst.vte = regs[0x2206];                                                                          // D612
        vst.xs = AsFloat(regs[0x210F]); vst.xo = AsFloat(regs[0x2110]);                                  // D613
        vst.ys = AsFloat(regs[0x2111]); vst.yo = AsFloat(regs[0x2112]);                                  // D614
        vst.zs = AsFloat(regs[0x2113]); vst.zo = AsFloat(regs[0x2114]);                                  // D615
        vst.pitch = float(st.pitch);                                                                     // D616
        vst.rows = float(SurfaceH / st.sampY);                                                           // D617
        vst.noClip = (regs[0x2204] >> 16) & 1;                                                           // D618
        static_assert(sizeof(vst.vfDesc) == sizeof(vs.vfDesc));
        std::memcpy(vst.vfDesc, vs.vfDesc, sizeof(vst.vfDesc));                                          // D619 (VsFetchDescriptors, at load)

        // ---- indices (D622-639): the list / strip verbatim into the frame's arena.
        const size_t ibBytes = indices.size() * 4;                                                       // D626
        const Slice ib = Alloc(m, ibBytes);
        std::memcpy(ib.ptr, indices.data(), ibBytes);                                                    // D634

        // D641-661: no input layout, no vertex buffer, topology, the VS and its b0 / b1 / t0 - in PixelStageAndDraw.
        VertexInput vin;
        vin.vb = vdata;
        vin.vbBytes = vdataBytes;
        vin.vs = &vs;
        vin.vsState = &vst;
        vin.vsRows = vsRows;
        vin.strip = triangleStrip;                                                                       // D642
        vin.triangles = triangleStrip ? stripTriangles : uint32_t(indices.size() / 3);                   // D664
        PixelStageAndDraw(m, regs, *ps, color, st, vin, ib, uint32_t(indices.size()),
                          std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());   // D662-663
        m.MaybeReportStats();                                                                            // D856
        return GpuVsResult::Drawn;                                                                       // D665
    }
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
