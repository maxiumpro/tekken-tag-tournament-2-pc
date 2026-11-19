// R4 Metal backend, textures (gpu_hw_metal_impl.h "textures" section, M1 slice 3): BindTexture (guest texture
// fetch -> a Metal texture: resolved colour / depth hits, the decoded-texture cache, uploads with mips and cube
// faces) and Sampler (fetch-constant sampler state -> MTLSamplerState, with the cube variant and the LOD bias
// airconv reads from meta_sN). Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11
// backend stays the reference); lines copied from gpu_render.cpp carry "// R<line>".
//
// Frontend helpers: DecodeTexture, TextureMipLevel / MipLevel, FormatBpp, IsBlockCompressed, TextureBlockBytes
// and the Fmt values live in an ANONYMOUS namespace in gpu_render.cpp (the D3D11 backend is included into that
// translation unit; this file is not). They are copied below, verbatim apart from names, as private helpers.
// Contract candidate: export them from the frontend (e.g. gpu_hw_texture_upload.h) and delete the copies.
// The face loop (D4603-4609) uses the exported DecodeTextureUploadFaces, so TextureFaceStride is not copied.
//
// State the frozen Impl has no fields for (decoded-texture cache, its byte count, default textures, sampler
// cache) is file static, as core / shaders / draw did: one Backend exists (Get()), so one instance.
//
// Uploads are blits on the frame's command buffer (Cmd()), after EndPass(); this file never commits it (the
// draw in progress keeps its upload arena in that command buffer).
//
// Not ported in M1 (each named): g_gpuFrameNote frame-log lines (D4446-4455), CensusTexture / MipCensus /
// MipVerify (D4478, D4491, D4499, D4521, D4546-4547, D4563), stats_ (depthHits, resolvedHits, uploads,
// uploadBytes, D4477, D4520, D4657-4658), undecodable_ / writebackCauses_ / uploadByFormat_ counters
// (D4490, D4498, D4529-4533, D4659-4661), GpuNoteMemory (D4544-4545), TTT2_MIP_TOGGLE / FollowMipToggle
// (D4374-4405: TTT2_MIPS=0 is honoured), whiteSrv_ / TTT2_HW_OVERRIDE (draw does not port D713-723 either).

#include "gpu_hw_metal_impl.h"
#include "gpu_hw_texture_upload.h"
#include "gpu_hw_dirty_tracker.h"
#include "display_settings.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace ttt2_hw {

namespace {

using Impl = Backend::Impl;
using Texture = TextureInfo;

// ================================================================ frontend copies (gpu_render.cpp, anonymous ns)
constexpr uint32_t REG_FETCH_CONST = 0x4800;                                                             // R257

enum Fmt : uint32_t                                                                                      // R405-415
{
    FMT_8 = 2, FMT_1_5_5_5 = 3, FMT_5_6_5 = 4, FMT_6_5_5 = 5, FMT_8_8_8_8 = 6,
    FMT_2_10_10_10 = 7, FMT_8_A = 8, FMT_8_B = 9, FMT_8_8 = 10, FMT_4_4_4_4 = 15,
    FMT_10_11_11 = 16, FMT_11_11_10 = 17, FMT_DXT1 = 18, FMT_DXT2_3 = 19, FMT_DXT4_5 = 20,
    FMT_24_8 = 22, FMT_24_8_FLOAT = 23, FMT_16 = 24, FMT_16_16 = 25, FMT_16_16_16_16 = 26,
    FMT_16_FLOAT = 30, FMT_16_16_FLOAT = 31, FMT_16_16_16_16_FLOAT = 32,
    FMT_32 = 33, FMT_32_32 = 34, FMT_32_32_32_32 = 35,
    FMT_32_FLOAT = 36, FMT_32_32_FLOAT = 37, FMT_32_32_32_32_FLOAT = 38,
    FMT_DXN = 49, FMT_32_32_32_FLOAT = 57, FMT_DXT3A = 58, FMT_DXT5A = 59, FMT_CTX1 = 60
};

uint32_t FormatBpp(uint32_t f)                                                                           // R418-433
{
    switch (f)
    {
    case FMT_8: case FMT_8_A: case FMT_8_B:                                   return 8;
    case FMT_1_5_5_5: case FMT_5_6_5: case FMT_6_5_5: case FMT_4_4_4_4:
    case FMT_8_8: case FMT_16: case FMT_16_FLOAT:                             return 16;
    case FMT_8_8_8_8: case FMT_2_10_10_10: case FMT_10_11_11: case FMT_11_11_10:
    case FMT_16_16: case FMT_16_16_FLOAT: case FMT_32: case FMT_32_FLOAT:     return 32;
    case FMT_16_16_16_16: case FMT_16_16_16_16_FLOAT:
    case FMT_32_32: case FMT_32_32_FLOAT:                                     return 64;
    case FMT_32_32_32_FLOAT:                                                  return 96;
    case FMT_32_32_32_32: case FMT_32_32_32_32_FLOAT:                         return 128;
    default:                                                                  return 0;
    }
}

bool IsBlockCompressed(uint32_t f)                                                                       // R435-439
{
    return f == FMT_DXT1 || f == FMT_DXT2_3 || f == FMT_DXT4_5 || f == FMT_DXN ||
           f == FMT_DXT3A || f == FMT_DXT5A || f == FMT_CTX1;
}

uint32_t PhysToGuest(uint32_t addr)                                                                      // R73-76
{
    return (addr && addr < 0x20000000u) ? (addr | 0xA0000000u) : addr;
}

Texture DecodeTexture(const uint32_t* regs, uint32_t slot)                                               // R680-716
{
    Texture t;
    if (slot >= 32) return t;
    const uint32_t* c = &regs[REG_FETCH_CONST + slot * 6];
    if ((c[0] & 3) != 2) return t;                       // type 2 == texture fetch constant
    t.pitch  = ((c[0] >> 22) & 0x1FF) * 32;
    t.tiled  = (c[0] >> 31) & 1;
    t.format = c[1] & 0x3F;
    t.endian = (c[1] >> 6) & 3;
    t.base   = PhysToGuest((c[1] >> 12) << 12);          // the page number is a PHYSICAL address (R689-694)
    t.width  = (c[2] & 0x1FFF) + 1;
    t.height = ((c[2] >> 13) & 0x1FFF) + 1;
    t.swizzle = (c[3] >> 1) & 0xFFF;
    t.clampX = (c[0] >> 10) & 7;
    t.clampY = (c[0] >> 13) & 7;
    t.magFilter = (c[3] >> 19) & 3;
    t.border = c[5] & 3;
    t.dimension = (c[5] >> 9) & 3;
    t.signs = (c[0] >> 2) & 0xFF;
    t.minFilter = (c[3] >> 21) & 3;
    t.mipFilter = (c[3] >> 23) & 3;
    t.aniso = (c[3] >> 25) & 7;
    t.mipMin = (c[4] >> 2) & 0xF;
    t.mipMax = (c[4] >> 6) & 0xF;
    t.lodBias = int32_t(c[4] << 10) >> 22;
    t.packedMips = (c[5] >> 11) & 1;
    t.mipBase = (c[5] >> 12) ? PhysToGuest((c[5] >> 12) << 12) : 0;
    if (!t.pitch) t.pitch = t.width;
    t.valid = t.base != 0 && t.width > 0 && t.height > 0;
    return t;
}

// Where a mip level is stored (R718-724: Xenia's GetGuestTextureLayout / GetPackedMipOffset).
struct MipLevel { uint32_t base = 0, pitch = 1, rows = 1, x = 0, y = 0, width = 1, height = 1; };       // R725
inline uint32_t Log2Ceil(uint32_t v) { return v <= 1 ? 0u : 32u - uint32_t(__builtin_clz(v - 1)); }     // R726
inline void PackedMipOffset(uint32_t width, uint32_t height, uint32_t level, uint32_t& x, uint32_t& y)    // R727-735
{
    const uint32_t lw = Log2Ceil(width), lh = Log2Ceil(height), ls = std::min(lw, lh);
    const uint32_t packedBase = ls > 4 ? ls - 4 : 0, p = level - packedBase;
    x = y = 0;
    if (p < 3) { if (lw > lh) y = 16u >> p; else x = 16u >> p; }
    else if (lw > lh) x = (1u << (lw - packedBase)) >> (p - 2);
    else y = (1u << (lh - packedBase)) >> (p - 2);
}
inline uint32_t TextureBlockBytes(uint32_t format)                                                       // R736-740
{
    if (!IsBlockCompressed(format)) return std::max<uint32_t>(FormatBpp(format) / 8, 1);
    return (format == FMT_DXT1 || format == FMT_DXT3A || format == FMT_DXT5A || format == FMT_CTX1) ? 8 : 16;
}
bool TextureMipLevel(const Texture& t, uint32_t level, MipLevel& m)                                      // R741-764 (variant 0)
{
    const uint32_t bw = IsBlockCompressed(t.format) ? 4 : 1, bpb = TextureBlockBytes(t.format);
    m.width = std::max(t.width >> level, 1u); m.height = std::max(t.height >> level, 1u);
    const uint32_t ls = Log2Ceil(std::min(t.width, t.height));
    const uint32_t packedLevel = t.packedMips ? (ls > 4 ? ls - 4 : 0) : 99u;
    m.x = m.y = 0;
    if (level >= packedLevel) PackedMipOffset(t.width, t.height, level, m.x, m.y);
    if (level == 0 || packedLevel == 0) { m.base = t.base; m.pitch = t.pitch; m.rows = t.height; return true; }
    if (!t.mipBase) return false;
    const uint32_t w2 = 1u << Log2Ceil(t.width), h2 = 1u << Log2Ceil(t.height);
    uint32_t addr = t.mipBase;
    const uint32_t last = std::min(level, packedLevel);
    for (uint32_t l = 1; l <= 15; ++l)
    {
        const uint32_t lw = std::max(w2 >> l, 1u), lh = std::max(h2 >> l, 1u);
        uint32_t cols = (lw + bw - 1) / bw, rows = (lh + bw - 1) / bw;
        if (t.tiled) { cols = (cols + 31) & ~31u; rows = (rows + 31) & ~31u; }
        else cols = ((cols * bpb + 255) & ~255u) / bpb;
        if (l == last) { m.base = addr; m.pitch = cols * bw; m.rows = rows * bw; return true; }
        addr += (cols * rows * bpb + 4095) & ~4095u;
    }
    return false;
}

// ================================================================ backend-neutral logic from gpu_hw_d3d11.h
// D198-207 MipSampling: bit 0 min filter linear, 1-2 mip filter (0 point, 1 linear, 2 none), 3-5 anisotropic
// filter (0 off, 1..5 = 1:1..16:1), 6-9 / 10-13 first / last level, 14-23 LOD bias in 1/32 steps (signed).
uint32_t MipSampling(const Texture& t, uint32_t levels)
{
    const uint32_t maxLevel = levels > 1 && t.mipFilter != 2 ? std::min<uint32_t>(t.mipMax, levels - 1) : 0;   // D204
    return (t.minFilter == 0 ? 0u : 1u) | (std::min<uint32_t>(t.mipFilter, 2) << 1) | (std::min<uint32_t>(t.aniso, 5) << 3) |
           (std::min(t.mipMin, maxLevel) << 6) | (maxLevel << 10) | ((uint32_t(t.lodBias) & 0x3FF) << 14);   // D205-206
}

// D4372-4373: TTT2_MIPS=0 uploads the base level only (TTT2_MIP_TOGGLE is not ported, header comment).
bool MipsOff()
{
    static const bool off = [] { const char* e = std::getenv("TTT2_MIPS"); return e && *e == '0'; }();
    return off;
}

uint32_t MipLevelsFor(const Texture& t, bool array)                                                      // D4406
{
    if (MipsOff() || array || t.mipMax == 0 || t.mipFilter == 2) return 1;                               // D4408
    if ((t.width & (t.width - 1)) || (t.height & (t.height - 1))) return 1;                              // D4409
    uint32_t levels = 1;                                                                                 // D4410
    for (uint32_t l = 1; l <= t.mipMax && l < 14; ++l)                                                   // D4411
    {
        MipLevel m;                                                                                      // D4413
        if (std::max(t.width >> l, t.height >> l) < 4 || !TextureMipLevel(t, l, m)) break;               // D4414
        levels = l + 1;                                                                                  // D4415
    }
    return levels;                                                                                       // D4417
}

// D4419-4429: the guest bytes of levels 1+ outside the base's storage.
void MipRange(const Texture& t, uint32_t levels, uint32_t& start, uint32_t& size)
{
    start = size = 0;                                                                                    // D4423
    MipLevel m;                                                                                          // D4424
    if (levels <= 1 || !TextureMipLevel(t, levels - 1, m) || m.base == t.base || m.base < t.mipBase) return;   // D4425
    const uint32_t bw = IsBlockCompressed(t.format) ? 4 : 1;                                             // D4426
    start = t.mipBase;                                                                                   // D4427
    size = uint32_t(std::min<uint64_t>(uint64_t(m.base - t.mipBase) + uint64_t(m.pitch / bw) * ((m.rows + bw - 1) / bw) * TextureBlockBytes(t.format),
                                       64ull << 20));                                                    // D4428
}

// ================================================================ file state (not in the frozen Impl)
struct TexKey                                                                                            // D2164-2169
{
    uint32_t base, format, width, height, pitch, tiled, endian, swizzle, slices;   // slices: count | 0x100 if array
    uint32_t mipBase, levels;                                                      // levels | 0x100 if packed
    bool operator==(const TexKey& o) const { return std::memcmp(this, &o, sizeof(*this)) == 0; }
};
struct TexKeyHash                                                                                        // D2170-2189 (no AB probe)
{
    size_t operator()(const TexKey& k) const
    {
        uint64_t h = 1469598103934665603ull;
        for (uint32_t word : { k.base, k.format, k.width, k.height, k.pitch, k.tiled, k.endian, k.swizzle, k.slices })
            h = (h ^ word) * 1099511628211ull;
        return size_t(h ^ (h >> 32));
    }
};
struct CachedTexture                                                                                     // D2190-2196
{
    id<MTLTexture> tex = nil;
    uint32_t mode = 0, rangeStart = 0, rangeSize = 0, seq = 0, allocW = 1, allocH = 1, slices = 1;
    uint32_t levels = 1, mipStart = 0, mipSize = 0;     // mip levels held; their guest bytes (levels 1+)
    bool genericSwizzled = false;
};

constexpr uint64_t TextureBudgetBytes = 1536ull << 20;                                                  // D2528

struct TextureState {
    std::unordered_map<TexKey, CachedTexture, TexKeyHash> textures;                                     // D2891 textures_
    uint64_t textureBytes = 0;                                                                           // D4648 textureBytes_
    id<MTLTexture> defaultTex = nil, defaultArrayTex = nil;                                              // D342-365
    std::unordered_map<uint64_t, Impl::SamplerSet> samplers;                                             // D2672 samplers_
};
TextureState& TexState() { static TextureState* s = new TextureState; return *s; }   // leaked with the Backend (D5140)

// One staged copy into a Private texture: `bytes` at `offset` of the staging buffer.
struct Region { size_t offset = 0, bytes = 0; uint32_t rowPitch = 0, width = 1, height = 1, slice = 0, level = 0; };

// Copy `regions` (already written into `staging`) into `tex` with one blit encoder on the frame's command buffer.
// EndPass first: one encoder at a time on frameCmd (gpu_hw_metal_impl.h, draw section). Never commits.
void BlitUpload(Impl& m, id<MTLBuffer> staging, id<MTLTexture> tex, const std::vector<Region>& regions, bool compressed)
{
    m.EndPass();
    id<MTLBlitCommandEncoder> blit = [m.Cmd() blitCommandEncoder];
    if (!blit) throw Failure("metal: blit command encoder");
    for (const Region& r : regions)
    {
        // The Metal level's own size (allocW >> l, as D3D11's level sizes, D4612-4613), never more than the
        // decoded rows hold; a block-compressed copy may end mid-block only at the texture's edge.
        const uint32_t levelW = std::max<uint32_t>(uint32_t(tex.width) >> r.level, 1u);
        const uint32_t levelH = std::max<uint32_t>(uint32_t(tex.height) >> r.level, 1u);
        const uint32_t w = std::min(levelW, r.width), h = std::min(levelH, r.height);
        const uint32_t rows = compressed ? (h + 3) / 4 : h;
        [blit copyFromBuffer:staging sourceOffset:r.offset sourceBytesPerRow:r.rowPitch
         sourceBytesPerImage:size_t(r.rowPitch) * rows sourceSize:MTLSizeMake(w, h, 1)
                   toTexture:tex destinationSlice:r.slice destinationLevel:r.level destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit endEncoding];
}

// Shared staging buffer for `parts` (each 256-byte aligned, which covers every pixel / block size used here).
id<MTLBuffer> Staging(Impl& m, const std::vector<const std::vector<uint8_t>*>& parts, std::vector<size_t>& offsets)
{
    size_t total = 0;
    offsets.clear();
    for (const auto* p : parts) { offsets.push_back(total); total += (p->size() + 255) / 256 * 256; }
    id<MTLBuffer> buf = [m.device newBufferWithLength:std::max<size_t>(total, 1) options:MTLResourceStorageModeShared];
    if (!buf) throw Failure("metal: texture staging buffer (" + std::to_string(total) + " B)");
    uint8_t* dst = static_cast<uint8_t*>(buf.contents);
    for (size_t i = 0; i < parts.size(); ++i) std::memcpy(dst + offsets[i], parts[i]->data(), parts[i]->size());
    return buf;
}

// D342-350 / D362-365: the interpreter's answer for an invalid texture fetch constant, (0, 0, 0, 1), 1x1
// RGBA32F; the array variant (cube-fetch slots) is a 2D array of 6 such slices (brief: any slice the shader
// picks reads the same texel). Created on first use; filled through a blit on the frame's command buffer.
id<MTLTexture> DefaultTexture(Impl& m, bool array)
{
    TextureState& s = TexState();
    __strong id<MTLTexture>& tex = array ? s.defaultArrayTex : s.defaultTex;
    if (tex) return tex;
    const uint32_t slices = array ? 6u : 1u;
    MTLTextureDescriptor* d = [MTLTextureDescriptor new];
    d.textureType = array ? MTLTextureType2DArray : MTLTextureType2D;
    d.pixelFormat = MTLPixelFormatRGBA32Float;                                                           // D345
    d.width = d.height = 1; d.mipmapLevelCount = 1; d.arrayLength = slices;
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> made = [m.device newTextureWithDescriptor:d];
    if (!made) throw Failure("metal: default texture");
    const float black[4] = { 0, 0, 0, 1 };                                                              // D343
    const std::vector<uint8_t> texel(reinterpret_cast<const uint8_t*>(black), reinterpret_cast<const uint8_t*>(black) + 16);
    std::vector<const std::vector<uint8_t>*> parts(slices, &texel);
    std::vector<size_t> offsets;
    id<MTLBuffer> staging = Staging(m, parts, offsets);
    std::vector<Region> regions;
    for (uint32_t i = 0; i < slices; ++i) regions.push_back({ offsets[i], 16, 16, 1, 1, i, 0 });
    BlitUpload(m, staging, made, regions, false);
    tex = made;
    return tex;
}

MTLPixelFormat UploadPixelFormat(TextureUploadFormat f)                                                  // D4578-4587
{
    switch (f)
    {
    case TextureUploadFormat::Rgba8: return MTLPixelFormatRGBA8Unorm;                                    // D4580
    case TextureUploadFormat::Bgra8: return MTLPixelFormatBGRA8Unorm;                                    // D4581
    case TextureUploadFormat::R8: return MTLPixelFormatR8Unorm;                                          // D4582
    case TextureUploadFormat::Bc1: return MTLPixelFormatBC1_RGBA;                                        // D4583
    case TextureUploadFormat::Bc2: return MTLPixelFormatBC2_RGBA;                                        // D4584
    case TextureUploadFormat::Bc3: return MTLPixelFormatBC3_RGBA;                                        // D4585
    case TextureUploadFormat::Rgba32Float: return MTLPixelFormatRGBA32Float;                             // D4586
    }
    throw Failure("metal: texture upload format " + std::to_string(int(f)) + " not implemented (M1 slice 3)");
}

bool Compressed(TextureUploadFormat f)
{
    return f == TextureUploadFormat::Bc1 || f == TextureUploadFormat::Bc2 || f == TextureUploadFormat::Bc3;
}

// D4596-4662 UploadTexture. `array` makes a 2D array texture (cube-fetch slots); a cube then has 6 slices, its
// faces at TextureFaceStride intervals (DecodeTextureUploadFaces walks them as D4603-4609 does).
void UploadTexture(Impl& m, const Texture& t, CachedTexture& c, uint32_t slices, bool array, uint32_t levels)
{
    std::vector<TextureUploadSurface> faces;                                                             // D4602
    if (!DecodeTextureUploadFaces(t, slices, faces) || faces.size() != slices)                           // D4604-4609 (D4577)
        throw Failure("metal: invalid guest texture upload");
    const TextureUploadFormat uploadFormat = faces[0].format;
    const MTLPixelFormat format = UploadPixelFormat(uploadFormat);                                       // D4578-4587
    const uint32_t allocW = faces[0].width, allocH = faces[0].height;                                    // D4588-4589
    c.mode = faces[0].remapMode;                                                                         // D4591
    c.genericSwizzled = faces[0].genericSwizzled;                                                        // D4592
    size_t total = 0;                                                                                    // D4610
    for (const auto& f : faces) total += f.bytes.size();                                                 // D4611
    // Levels 1+ (2D textures only, MipLevelsFor): each decoded like the base from where TextureMipLevel puts it.
    if (slices != 1) levels = 1;                                                                         // D4614
    std::vector<TextureUploadSurface> mips(levels - 1);                                                  // D4615-4616
    for (uint32_t l = 1; l < levels; ++l)                                                                // D4617
    {
        MipLevel ml;                                                                                     // D4619
        TextureMipLevel(t, l, ml);                                                                       // D4620
        Texture lt = t;                                                                                  // D4621
        lt.base = ml.base; lt.pitch = ml.pitch; lt.width = ml.width; lt.height = ml.height; lt.originX = ml.x; lt.originY = ml.y;   // D4622
        if (!DecodeTextureUploadSurface(lt, mips[l - 1])) throw Failure("metal: invalid guest texture upload");   // D4626 (D4577)
        total += mips[l - 1].bytes.size();                                                               // D4627
    }
    if (!c.tex || c.allocW != allocW || c.allocH != allocH || c.slices != slices || c.levels != levels)  // D4632
    {
        c.tex = nil;                                                                                     // D4634
        MTLTextureDescriptor* d = [MTLTextureDescriptor new];                                            // D4629-4631
        d.textureType = array ? MTLTextureType2DArray : MTLTextureType2D;                                // D4641-4646 (SRV dimension)
        d.pixelFormat = format;
        d.width = allocW; d.height = allocH; d.mipmapLevelCount = levels; d.arrayLength = slices;
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;   // written by blit only
        c.tex = [m.device newTextureWithDescriptor:d];                                                   // D4638
        if (!c.tex) throw Failure("metal: texture " + std::to_string(allocW) + "x" + std::to_string(allocH) +
                                  " format " + std::to_string(int(format)));
        TexState().textureBytes += total;                                                                       // D4648
    }
    // D4635-4637 initial data / D4650-4656 UpdateSubresource: one path in Metal, a blit into the (new or
    // reused) texture, ordered after earlier draws of this command buffer that sampled the old content.
    std::vector<const std::vector<uint8_t>*> parts;
    for (const auto& f : faces) parts.push_back(&f.bytes);
    for (const auto& mp : mips) parts.push_back(&mp.bytes);
    std::vector<size_t> offsets;
    id<MTLBuffer> staging = Staging(m, parts, offsets);
    std::vector<Region> regions;
    for (uint32_t s = 0; s < slices; ++s)                                                                // D4636 / D4652-4653
        regions.push_back({ offsets[s], faces[s].bytes.size(), faces[s].rowPitch, faces[s].width, faces[s].height, s, 0 });
    for (uint32_t l = 1; l < levels; ++l)                                                                // D4637 / D4654-4655
    {
        const TextureUploadSurface& mp = mips[l - 1];
        regions.push_back({ offsets[slices + l - 1], mp.bytes.size(), mp.rowPitch, mp.width, mp.height, 0, l });
    }
    BlitUpload(m, staging, c.tex, regions, Compressed(uploadFormat));
    ++m.stats.uploads; m.stats.uploadBytes += total;                                                     // D4658-4659
    c.allocW = allocW; c.allocH = allocH; c.slices = slices; c.levels = levels;                          // D4657
}

// Bound with the sampler fields every return carries: mip = MipSampling(ft, levels) (D730) and
// info = SamplerInfo(ft) (D709), ft = the decoded fetch constant, BEFORE draw's cube clamp (D711).
Impl::Bound Make(const Texture& ft, id<MTLTexture> tex, uint32_t swizzle, uint32_t mode, uint32_t width, uint32_t height,
                 float scaleX, float scaleY, uint32_t slices = 1, uint32_t levels = 1)
{
    Impl::Bound b;
    b.tex = tex; b.swizzle = swizzle; b.mode = mode; b.width = width; b.height = height;
    b.scaleX = scaleX; b.scaleY = scaleY; b.slices = slices; b.levels = levels;
    b.mip = MipSampling(ft, levels);
    SamplerInfo(ft, b.info);
    return b;
}

// D4458-4569 BindTextureFrom.
Impl::Bound BindTextureFrom(Impl& m, const uint32_t* regs, uint32_t slot, bool array)
{
    TextureState& s = TexState();
    Texture t = DecodeTexture(regs, slot);                                                               // D4460
    if (!t.valid || t.width > 8192 || t.height > 8192)                                                   // D4462
        return Make(t, DefaultTexture(m, array), 0x688, 0, 1, 1, 1.0f, 1.0f);                            // D4461, D4463
    // Depth textures (24_8 / 24_8_FLOAT) over a GPU depth resolve: the R32F copy, depth replicated (mode 3).
    if (!array && (t.format == FMT_24_8 || t.format == FMT_24_8_FLOAT))                                  // D4466
    {
        auto it = m.resolved.find(t.base);                                                               // D4468
        if (it != m.resolved.end() && it->second.valid && it->second.depth && t.tiled &&                 // D4469
            ((t.pitch + 31) & ~31u) == ((it->second.pitch + 31) & ~31u) &&                               // D4470
            t.width <= it->second.pitch && t.height <= it->second.height)                                // D4471
        {
            MtlResolved& r = it->second;                                                                 // D4473
            m.Dirty().Sync(t.base, r.size);                                                              // D4474
            if (m.Dirty().Newest(t.base, r.size) <= r.seq)                                               // D4475
            {
                ++m.stats.depthHits;                                                                         // D4477
                return Make(t, r.tex, t.swizzle, 3u, t.width, t.height,
                            float(t.width) / float(r.pitch), float(t.height) / float(r.height));         // D4479-4480
            }
            r.valid = false;                                                                             // D4482
        }
    }
    // Formats the interpreter cannot decode sample as unswizzled (0, 0, 0, 1) on the CPU: same answer, no guest
    // bytes needed, never write back or upload for them.
    if (!FormatBpp(t.format) && !IsBlockCompressed(t.format))                                            // D4488
        return Make(t, DefaultTexture(m, array), 0x688, 0, t.width, t.height, 1.0f, 1.0f);               // D4492
    const uint32_t slices = array && t.dimension == 3 ? 6u : 1u;                                         // D4494
    const uint32_t bytes = TextureGuestBytes(t, slices);                                                 // D4495
    if (!bytes || uint64_t(t.base) + bytes > (1ull << 32))                                               // D4496
        return Make(t, DefaultTexture(m, array), 0x688, 0, t.width, t.height, 1.0f, 1.0f);               // D4500
    const uint32_t levels = MipLevelsFor(t, array);                                                      // D4502
    uint32_t mipStart = 0, mipSize = 0;                                                                  // D4503
    MipRange(t, levels, mipStart, mipSize);                                                              // D4504

    // GPU-resolved data at this address?
    for (auto it = m.resolved.begin(); it != m.resolved.end(); ++it)                                     // D4508
    {
        MtlResolved& r = it->second;                                                                     // D4510
        if (!r.valid) continue;                                                                          // D4511
        const uint32_t rs = it->first;                                                                   // D4512
        if (!(uint64_t(t.base) < uint64_t(rs) + r.size && uint64_t(rs) < uint64_t(t.base) + bytes) &&    // D4513
            !(mipSize && uint64_t(mipStart) < uint64_t(rs) + r.size && uint64_t(rs) < uint64_t(mipStart) + mipSize)) continue;   // D4514
        m.Dirty().Sync(rs, r.size);                                                                      // D4515
        if (m.Dirty().Newest(rs, r.size) > r.seq) { r.valid = false; continue; }   // the CPU overwrote it  // D4516
        if (!array && t.base == rs && (r.rg16f ? t.format == FMT_16_16_FLOAT : t.format == FMT_8_8_8_8) && t.tiled &&   // D4517
            ((t.pitch + 31) & ~31u) == ((r.pitch + 31) & ~31u) && t.width <= r.pitch && t.height <= r.height)            // D4518
        {
            ++m.stats.resolvedHits;                                                                      // D4520
            return Make(t, r.tex, t.swizzle, (r.swapRB && !r.rg16f) ? 2u : 0u, t.width, t.height,
                        float(t.width) / float(r.pitch), float(t.height) / float(r.height));             // D4522-4523
        }
        // Any other view of resolved bytes reads guest memory: make it coherent once.
        if (!r.cpuCoherent) m.WriteBack(rs, r);                                                          // D4527-4534
    }

    const bool fast = t.format == FMT_8_8_8_8 || t.format == FMT_8 ||                                    // D4538
                      t.format == FMT_DXT1 || t.format == FMT_DXT2_3 || t.format == FMT_DXT4_5;          // D4539
    const TexKey key{ t.base, t.format, t.width, t.height, t.pitch, t.tiled ? 1u : 0u, t.endian, fast ? 0xFFFFFFFFu : t.swizzle,   // D4540
                      slices | (array ? 0x100u : 0u), levels > 1 ? t.mipBase : 0u, levels | (levels > 1 && t.packedMips ? 0x100u : 0u) };   // D4541
    m.Dirty().Sync(t.base, bytes);                                                                       // D4542
    if (mipSize) m.Dirty().Sync(mipStart, mipSize);                                                      // D4543
    auto it = s.textures.find(key);                                                                      // D4548
    if (it != s.textures.end() && m.Dirty().Newest(t.base, bytes) <= it->second.seq &&                   // D4549
        (!mipSize || m.Dirty().Newest(mipStart, mipSize) <= it->second.seq))                             // D4550
    {
        const CachedTexture& c = it->second;                                                             // D4552
        return Make(t, c.tex, c.genericSwizzled ? 0x688u : t.swizzle, c.mode, t.width, t.height,
                    float(t.width) / float(c.allocW), float(t.height) / float(c.allocH), slices, c.levels);   // D4553-4554
    }
    if (s.textures.size() > 4096 || s.textureBytes > TextureBudgetBytes)                                 // D4556
    {
        std::fprintf(stderr, "[gpu-metal] texture cache flushed: %zu entries, %.1f MB\n",
                     s.textures.size(), double(s.textureBytes) / 1048576.0);                             // D4558-4559
        s.textures.clear();                                                                              // D4560
        s.textureBytes = 0;                                                                              // D4561
    }
    CachedTexture& c = s.textures[key];                                                                  // D4564
    UploadTexture(m, t, c, slices, array, levels);                                                       // D4565
    m.statTextures = s.textures.size();                                                                  // D5086 textures= (cache size)
    c.rangeStart = t.base; c.rangeSize = bytes; c.mipStart = mipStart; c.mipSize = mipSize; c.seq = m.Dirty().Current();   // D4566
    return Make(t, c.tex, c.genericSwizzled ? 0x688u : t.swizzle, c.mode, t.width, t.height,
                float(t.width) / float(c.allocW), float(t.height) / float(c.allocH), slices, c.levels);  // D4567-4568
}

MTLSamplerAddressMode AddressMode(uint32_t m)                                                            // D213-222
{
    switch (m)
    {
    case 0: return MTLSamplerAddressModeRepeat;                                                          // D216 WRAP
    case 1: return MTLSamplerAddressModeMirrorRepeat;                                                    // D217 MIRROR
    case 3: case 5: case 7: return MTLSamplerAddressModeMirrorClampToEdge;                               // D218 MIRROR_ONCE
    case 6: return MTLSamplerAddressModeClampToBorderColor;                                              // D219 BORDER
    default: return MTLSamplerAddressModeClampToEdge;                                                    // D220 CLAMP
    }
}

} // namespace

// ================================================================ contract candidate
// D4431-4440 DropTexturesOverlapping: Resolve (surfaces, slice 4) evicts guest-decoded textures over the bytes
// it just wrote (D1066). Not in the frozen Impl: surfaces declares this signature itself
//   void MetalDropTexturesOverlapping(Backend::Impl&, uint32_t start, uint32_t size);
// Contract candidate: Impl::DropTexturesOverlapping in the "textures" section.
void MetalDropTexturesOverlapping(Backend::Impl&, uint32_t start, uint32_t size)
{
    auto& textures = TexState().textures;
    for (auto it = textures.begin(); it != textures.end();)                                              // D4433
    {
        const CachedTexture& c = it->second;                                                             // D4435
        const bool overlap = (uint64_t(c.rangeStart) < uint64_t(start) + size && uint64_t(start) < uint64_t(c.rangeStart) + c.rangeSize) ||   // D4436
                             (c.mipSize && uint64_t(c.mipStart) < uint64_t(start) + size && uint64_t(start) < uint64_t(c.mipStart) + c.mipSize);   // D4437
        it = overlap ? textures.erase(it) : std::next(it);                                               // D4438
    }
}

// ================================================================ Backend::Impl (textures section)
Backend::Impl::Bound Backend::Impl::BindTexture(const uint32_t* regs, uint32_t slot, bool array)
{
    @autoreleasepool {
        return BindTextureFrom(*this, regs, slot, array);                                                // D4443-4445
    }
}

// D2673-2685 Sampler + D209-239 SamplerDesc. `slot` is only the slice-2 harness hook (impl.h); the cache key is
// the D3D11 one. cube = the same sampler with clamp addressing (dxmt_sampler.cpp:28-44, BINDINGS.md cube_sN:
// clamp-to-border when min and mag filter are both linear, else clamp-to-edge). lodBias = MipLODBias (D235):
// Metal samplers have no LOD bias, airconv applies meta_sN.
Backend::Impl::SamplerSet Backend::Impl::Sampler(uint32_t slot, const uint32_t info[4], uint64_t mip64)
{
    (void)slot;
    static const uint32_t anisoFor[5] = { 0, 2, 4, 8, 16 };                                              // D2675
    const uint32_t userAniso = anisoFor[std::clamp(display_settings::GetTextureFilter(), 0, 4)];        // D2676
    const uint32_t mip = uint32_t(mip64);   // MipSampling is 24 bits; D225's sign extension needs the u32 shift
    const uint64_t key = info[0] | (info[1] << 3) | (info[2] << 6) | (info[3] << 7) | (uint64_t(mip) << 9) |
                         (uint64_t(userAniso) << 33);                                                    // D2677
    SamplerSet& out = TexState().samplers[key];                                                                 // D2678
    if (out.s) return out;
    @autoreleasepool {
        const bool minLinear = mip & 1;                                                                  // D224
        const uint32_t mipMode = (mip >> 1) & 3, aniso = (mip >> 3) & 7, minLevel = (mip >> 6) & 0xF, maxLevel = (mip >> 10) & 0xF;   // D225
        const int32_t bias = int32_t(mip << 8) >> 22;                                                    // D226
        MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
        d.minFilter = minLinear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;          // D227
        d.magFilter = info[2] ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;            // D228
        d.mipFilter = mipMode == 1 ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;             // D229
        d.maxAnisotropy = 1;
        const uint32_t maxAniso = std::max<uint32_t>(aniso ? 1u << (aniso - 1) : 1u, userAniso);         // D231
        if (maxAniso > 1 && maxLevel > 0 && minLinear)                                                   // D232
        {
            // D3D11_FILTER_ANISOTROPIC: min, mag and mip linear, plus the anisotropy (DXMT maps it the same way).
            d.minFilter = d.magFilter = MTLSamplerMinMagFilterLinear;
            d.mipFilter = MTLSamplerMipFilterLinear;
            d.maxAnisotropy = std::min<uint32_t>(maxAniso, 16);
        }
        d.sAddressMode = AddressMode(info[0]); d.tAddressMode = AddressMode(info[1]);                    // D233
        d.rAddressMode = MTLSamplerAddressModeClampToEdge;                                               // D233 (W clamp)
        d.lodMinClamp = float(std::min(minLevel, maxLevel)); d.lodMaxClamp = float(maxLevel);            // D234
        d.compareFunction = MTLCompareFunctionNever;                                                     // D236
        d.borderColor = info[3] == 1 ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorTransparentBlack;   // D237
        d.normalizedCoordinates = YES;
        d.supportArgumentBuffers = YES;                                                                  // BINDINGS.md sN
        id<MTLSamplerState> s = [device newSamplerStateWithDescriptor:d];
        if (!s) throw Failure("metal: sampler state");
        const bool linear = d.minFilter == MTLSamplerMinMagFilterLinear && d.magFilter == MTLSamplerMinMagFilterLinear;
        const MTLSamplerAddressMode cubeMode = linear ? MTLSamplerAddressModeClampToBorderColor : MTLSamplerAddressModeClampToEdge;
        d.sAddressMode = d.tAddressMode = d.rAddressMode = cubeMode;
        // DXMT's border colour is opaque white unless the D3D11 sampler addresses BORDER (d3d11_state_object.cpp:
        // 758-772); its cube variant inherits it, and draw's D711 clamp (2, 2) never asks for BORDER.
        if (info[0] != 6 && info[1] != 6) d.borderColor = MTLSamplerBorderColorOpaqueWhite;
        id<MTLSamplerState> cube = [device newSamplerStateWithDescriptor:d];
        if (!cube) throw Failure("metal: cube sampler state");
        out.s = s;
        out.cube = cube;
        out.lodBias = std::clamp(float(bias) / 32.0f, -16.0f, 15.99f);                                  // D235
    }
    return out;
}

} // namespace ttt2_hw
