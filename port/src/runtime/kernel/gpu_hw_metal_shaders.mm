// R4 Metal backend, shaders (gpu_hw_metal_impl.h "shaders" section, M1 slice 2): the pixel-shader cache
// (microcode -> generic HLSL text key -> airconv ps_<KEY>.metallib + ps_<KEY>.layout.tsv), the fixed CPU-vertex
// vertex shader, render pipelines and depth/stencil states.
// Every line ported from gpu_hw_d3d11.h carries a "// D<line>" marker (the D3D11 backend stays the reference).
//
// No shader is compiled from HLSL here: the metallib cache is converted offline (m1/convertir_cache.sh); a key
// missing from it is a cache miss (logged once, counted, the draw is skipped as D3D11 skips a PS not ready, D390).
//
// gpu_hw_translate.h needs the interpreter's microcode decoders (Cf / Alu / VFetch / TFetch, the CF_ / V_ / S_ /
// FMT_ enums, DecodeCfPair), which live in gpu_render.cpp's anonymous namespace. The block below is a VERBATIM
// copy of gpu_render.cpp from "enum CfOp" up to "// Bits per texel" - the same range
// port/tools/test_vertex_index_guard.py compiles the header with. Keep it identical to gpu_render.cpp.
// It and the translator come BEFORE the Metal / Foundation headers, so no Apple macro can touch the names.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- BEGIN verbatim copy of gpu_render.cpp ("enum CfOp" .. "// Bits per texel") ----
enum CfOp
{
    CF_NOP = 0, CF_EXEC = 1, CF_EXEC_END = 2, CF_COND_EXEC = 3, CF_COND_EXEC_END = 4,
    CF_COND_EXEC_PRED = 5, CF_COND_EXEC_PRED_END = 6, CF_LOOP_START = 7, CF_LOOP_END = 8,
    CF_COND_CALL = 9, CF_RETURN = 10, CF_COND_JMP = 11, CF_ALLOC = 12,
    CF_COND_EXEC_PRED_CLEAN = 13, CF_COND_EXEC_PRED_CLEAN_END = 14, CF_MARK_VS_FETCH_DONE = 15
};

// A control-flow instruction is 48 bits: `lo` is the low dword, `hi` the high 16 bits.
struct Cf
{
    uint32_t lo = 0, hi = 0;
    uint32_t op()    const { return (hi >> 12) & 0xF; }
    uint32_t addr()  const { return lo & 0xFFF; }
    uint32_t count() const { return (lo >> 12) & 0x7; }
    uint32_t seq()   const { return (lo >> 16) & 0xFFF; }
    uint32_t boolAddr() const { return (hi >> 2) & 0xFF; }     // CondExec
    uint32_t cond()  const { return (hi >> 10) & 0x1; }
    bool isExec() const
    {
        uint32_t o = op();
        return o == CF_EXEC || o == CF_EXEC_END || o == CF_COND_EXEC || o == CF_COND_EXEC_END ||
               o == CF_COND_EXEC_PRED || o == CF_COND_EXEC_PRED_END ||
               o == CF_COND_EXEC_PRED_CLEAN || o == CF_COND_EXEC_PRED_CLEAN_END;
    }
    bool isEnd() const
    {
        uint32_t o = op();
        return o == CF_EXEC_END || o == CF_COND_EXEC_END || o == CF_COND_EXEC_PRED_END ||
               o == CF_COND_EXEC_PRED_CLEAN_END;
    }
};

void DecodeCfPair(const uint32_t* g, Cf& a, Cf& b)
{
    a.lo = g[0];
    a.hi = g[1] & 0xFFFF;
    b.lo = (g[1] >> 16) | (g[2] << 16);
    b.hi = g[2] >> 16;
}

enum AluVec
{
    V_ADD = 0, V_MUL, V_MAX, V_MIN, V_SEQ, V_SGT, V_SGE, V_SNE, V_FRC, V_TRUNC, V_FLOOR,
    V_MAD, V_CNDEQ, V_CNDGE, V_CNDGT, V_DP4, V_DP3, V_DP2ADD, V_CUBE, V_MAX4,
    V_SETP_EQ_PUSH, V_SETP_NE_PUSH, V_SETP_GT_PUSH, V_SETP_GE_PUSH,
    V_KILL_EQ, V_KILL_GT, V_KILL_GE, V_KILL_NE, V_DST, V_MAXA
};

enum AluScalar
{
    S_ADDS = 0, S_ADDS_PREV, S_MULS, S_MULS_PREV, S_MULS_PREV2, S_MAXS, S_MINS, S_SEQS,
    S_SGTS, S_SGES, S_SNES, S_FRCS, S_TRUNCS, S_FLOORS, S_EXP, S_LOGC, S_LOG, S_RCPC,
    S_RCPF, S_RCP, S_RSQC, S_RSQF, S_RSQ, S_MAXAS, S_MAXASF, S_SUBS, S_SUBS_PREV,
    S_SETP_EQ, S_SETP_NE, S_SETP_GT, S_SETP_GE, S_SETP_INV, S_SETP_POP, S_SETP_CLR,
    S_SETP_RSTR, S_KILLS_EQ, S_KILLS_GT, S_KILLS_GE, S_KILLS_NE, S_KILLS_ONE, S_SQRT,
    S_MULSC0 = 42, S_MULSC1, S_ADDSC0, S_ADDSC1, S_SUBSC0, S_SUBSC1, S_SIN, S_COS,
    S_RETAIN_PREV
};

struct Alu
{
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    uint32_t vectorDest()       const { return w0 & 0x3F; }
    bool     absConstants()     const { return (w0 >> 7) & 1; }
    uint32_t scalarDest()       const { return (w0 >> 8) & 0x3F; }
    bool     scalarDestRelative() const { return (w0 >> 14) & 1; }
    bool     exportData()       const { return (w0 >> 15) & 1; }
    uint32_t vectorWriteMask()  const { return (w0 >> 16) & 0xF; }
    uint32_t scalarWriteMask()  const { return (w0 >> 20) & 0xF; }
    bool     vectorSaturate()   const { return (w0 >> 24) & 1; }
    bool     scalarSaturate()   const { return (w0 >> 25) & 1; }
    uint32_t scalarOpcode()     const { return (w0 >> 26) & 0x3F; }

    uint32_t srcSwizzle(int n)  const { return (w1 >> ((3 - n) * 8)) & 0xFF; }   // n = 1..3
    bool     srcNegate(int n)   const { return (w1 >> (24 + (3 - n))) & 1; }
    bool     predCondition()    const { return (w1 >> 27) & 1; }
    bool     isPredicated()     const { return (w1 >> 28) & 1; }
    bool     carRelative()      const { return (w1 >> 29) & 1; }
    bool     const1Relative()   const { return (w1 >> 30) & 1; }
    bool     const0Relative()   const { return (w1 >> 31) & 1; }

    uint32_t srcRegister(int n) const { return (w2 >> ((3 - n) * 8)) & 0xFF; }   // n = 1..3
    uint32_t vectorOpcode()     const { return (w2 >> 24) & 0x1F; }
    bool     srcSelect(int n)   const { return (w2 >> (29 + (3 - n))) & 1; }
};

struct VFetch
{
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    uint32_t opcode()        const { return w0 & 0x1F; }
    uint32_t srcRegister()   const { return (w0 >> 5) & 0x3F; }
    uint32_t dstRegister()   const { return (w0 >> 12) & 0x3F; }
    uint32_t constIndex()    const { return (w0 >> 20) & 0x1F; }
    uint32_t constIndexSel() const { return (w0 >> 25) & 0x3; }
    uint32_t srcSwizzle()    const { return (w0 >> 30) & 0x3; }
    uint32_t dstSwizzle()    const { return w1 & 0xFFF; }
    uint32_t format()        const { return (w1 >> 16) & 0x3F; }
    bool     numFormatAll()  const { return (w1 >> 13) & 1; }   // 1 = integer, 0 = normalized
    bool     signedRfMode()  const { return (w1 >> 14) & 1; }
    int32_t  expAdjust()     const { return (int32_t)((w1 << 2) & 0xFC000000) >> 26; }
    bool     isMiniFetch()   const { return (w1 >> 30) & 1; }
    uint32_t stride()        const { return w2 & 0xFF; }
    int32_t  offset()        const { return (int32_t)((w2 << 1) & 0xFFFFFF00) >> 9; }
};

struct TFetch
{
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    uint32_t opcode()      const { return w0 & 0x1F; }
    uint32_t srcRegister() const { return (w0 >> 5) & 0x3F; }
    uint32_t dstRegister() const { return (w0 >> 12) & 0x3F; }
    uint32_t constIndex()  const { return (w0 >> 20) & 0x1F; }
    bool     coordDenorm() const { return (w0 >> 25) & 1; }
    uint32_t srcSwizzle()  const { return (w0 >> 26) & 0x3F; }
    uint32_t dstSwizzle()  const { return w1 & 0xFFF; }
    uint32_t magFilter()   const { return (w1 >> 12) & 0x3; }   // 0 point, 1 linear, 3 = fetch constant's
    // dword 1 bit 28 use_computed_lod: the hardware picks the mip level from the coordinate's screen-space
    // derivatives (every TTT2 tfetch but 121 of 23,718; the rest sample level 0).
    bool     useComputedLod() const { return (w1 >> 28) & 1; }
    // FetchOpDimension, dword 2 bits 14-15 (Xenia ucode.h): 0 1D, 1 2D, 2 3D/stacked, 3 cube.
    // TTT2's pixel shaders use 2D (w2 = 0x4000) and cube (0xC000) fetches.
    uint32_t dimension()   const { return (w2 >> 14) & 0x3; }
};

// --------------------------------------------------------------------------------------
// texture / vertex format unpacking
// --------------------------------------------------------------------------------------
enum Fmt
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
// ---- END verbatim copy of gpu_render.cpp ----

} // namespace

#include "gpu_hw_translate.h"
#include "gpu_shader_key_cache.h"
#include "gpu_hw_metal_impl.h"

#import <Foundation/Foundation.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace ttt2_hw {

namespace {

using Impl = Backend::Impl;

// State the frozen Impl has no fields for: the PS cache (D2907 pixelShaders_), its key memo (psKeys_), the
// fixed VS (D2111 passthrough_), pipelines and depth states (D3D11: blendStates_ / depthStates_ maps).
// One Backend exists (Get()), so one instance, as core.mm's CoreState.
struct ShaderState {
    ShaderKeyCache psKeys;                                       // D3047
    std::unordered_map<uint64_t, Impl::Ps> ps;                   // D2907, key = PsKey(microcode)
    std::unordered_set<uint64_t> missLogged;                     // text keys logged as cache misses
    uint64_t cacheMisses = 0;                                    // distinct text keys missing from the cache
    bool dirReady = false;
    std::filesystem::path dir;                                   // the metallib cache directory
    id<MTLLibrary> vsLibrary = nil;
    id<MTLFunction> vs[17] = {};                                 // D2111: one per interpolator count 0..16
    std::map<std::array<uint64_t, 3>, id<MTLRenderPipelineState>> pipelines;
    std::unordered_map<uint64_t, id<MTLDepthStencilState>> depthStates;   // D2905-ish depthStates_
};
ShaderState& S() { static ShaderState* s = new ShaderState; return *s; }   // leaked with the Backend (D5140)

uint64_t PsKey(const std::vector<uint32_t>& code) { return Fnv(code.data(), code.size() * 4, 0xC0FFEEull + code.size()); }   // D3341

std::string Describe(NSError* e)
{
    if (!e) return "unknown error";
    const char* s = e.localizedDescription.UTF8String;
    return std::string(s ? s : "unknown error") + " (code " + std::to_string(long(e.code)) + ")";
}

std::string Hex16(uint64_t v)
{
    char b[24];
    std::snprintf(b, sizeof(b), "%016llX", (unsigned long long)v);
    return b;
}

// D2974-3001 ShaderCacheDirectory (the _WIN32 branch dropped: the Objective-C++ files build only on Apple).
std::filesystem::path ShaderCacheDirectory()
{
    std::filesystem::path p;
#if defined(__APPLE__)
    // macOS: ~/Library/Caches/TTT2Recomp/shadercache (the system may clear it; it is rebuilt).
    if (const char* override = std::getenv("TTT2_SHADER_CACHE")) p = override;                              // D2979
    else if (const char* home = std::getenv("HOME")) p = std::filesystem::path(home) / "Library" / "Caches" / "TTT2Recomp" / "shadercache";   // D2980
    else return {};                                                                                          // D2981
#else
    if (const char* override = std::getenv("TTT2_SHADER_CACHE")) p = override;                              // D2983
    else if (const char* cache = std::getenv("XDG_CACHE_HOME")) p = std::filesystem::path(cache) / "TTT2Recomp" / "shadercache";   // D2984
    else if (const char* home = std::getenv("HOME")) p = std::filesystem::path(home) / ".cache" / "TTT2Recomp" / "shadercache";    // D2985
    else return {};                                                                                          // D2986
#endif
    std::error_code ec;                                                                                      // D2998
    std::filesystem::create_directories(p, ec);                                                              // D2999
    return ec ? std::filesystem::path{} : p;                                                                 // D3000
}

// gpu_hw_metal_impl.h: $TTT2_METAL_CACHE, else <shader cache dir>/metal. Read once, at the first pixel shader.
const std::filesystem::path& MetalCacheDirectory()
{
    ShaderState& s = S();
    if (!s.dirReady)
    {
        s.dirReady = true;
        if (const char* e = std::getenv("TTT2_METAL_CACHE"); e && *e) s.dir = e;
        else if (std::filesystem::path base = ShaderCacheDirectory(); !base.empty()) s.dir = base / "metal";
        std::fprintf(stderr, "[gpu-metal] pixel shader cache: %s\n", s.dir.empty() ? "(none: every shader misses)" : s.dir.string().c_str());
    }
    return s.dir;
}

// One row of ps_<KEY>.layout.tsv (m0/duelo/layout.py): table, buffer index, byte offset, type, name; plus
// "#size <table> <bytes>". Same parser as readLayout in m0/duelo/duelo.mm.
struct LayoutRow { std::string table, type, name; int index = 0, offset = 0; };

bool ReadLayout(const std::filesystem::path& path, std::vector<LayoutRow>& rows, std::map<std::string, int>& sizes)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string t = ss.str();
    size_t a = 0;
    while (a < t.size())
    {
        size_t b = t.find('\n', a);
        if (b == std::string::npos) b = t.size();
        std::string line = t.substr(a, b - a);
        a = b + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> c;
        size_t s = 0, e;
        while ((e = line.find('\t', s)) != std::string::npos) { c.push_back(line.substr(s, e - s)); s = e + 1; }
        c.push_back(line.substr(s));
        if (c.size() == 3 && c[0] == "#size") sizes[c[1]] = std::atoi(c[2].c_str());
        else if (c.size() == 5) rows.push_back({c[0], c[3], c[4], std::atoi(c[1].c_str()), std::atoi(c[2].c_str())});
    }
    return !rows.empty();
}

// Fills e.fn / e.args / e.argBytes / e.cbCount from <dir>/ps_<KEY>.metallib + .layout.tsv. False + why on failure;
// missing = the files are not there (a cache miss, not an error).
bool LoadMetallib(id<MTLDevice> device, const std::filesystem::path& dir, Impl::Ps& e, bool& missing, std::string& why)
{
    missing = false;
    const std::string base = "ps_" + Hex16(e.textKey);
    const std::filesystem::path lib = dir / (base + ".metallib"), layout = dir / (base + ".layout.tsv");
    std::error_code ec;
    if (dir.empty() || !std::filesystem::exists(lib, ec) || !std::filesystem::exists(layout, ec)) { missing = true; return false; }
    std::vector<LayoutRow> rows;
    std::map<std::string, int> sizes;
    if (!ReadLayout(layout, rows, sizes)) { why = "unreadable " + layout.string(); return false; }
    // BINDINGS.md: cbuffer_table = fragment buffer(0), one gpuAddress per declared cb in ascending register order;
    // binding_table = buffer(1). The generic variant always reads cb1 (the in-shader alpha test, PsStateCB), so the
    // only valid cb lists are {cb0, cb1} and {cb1}: cbCount 2 or 1 then names them without ambiguity for draw.
    std::vector<std::pair<int, std::string>> cbs;
    std::vector<Impl::ArgField> args;
    for (const LayoutRow& r : rows)
    {
        if (r.table == "cbuffer_table")
        {
            if (r.index != 0) { why = "cbuffer_table at buffer(" + std::to_string(r.index) + "), expected 0"; return false; }
            cbs.push_back({r.offset, r.name});
        }
        else if (r.table == "binding_table")
        {
            if (r.index != 1) { why = "binding_table at buffer(" + std::to_string(r.index) + "), expected 1"; return false; }
            args.push_back({r.type, r.name, uint32_t(r.offset)});
        }
        else { why = "unknown table " + r.table; return false; }
    }
    std::sort(cbs.begin(), cbs.end());
    const bool both = cbs.size() == 2 && cbs[0] == std::make_pair(0, std::string("cb0")) && cbs[1] == std::make_pair(8, std::string("cb1"));
    const bool only1 = cbs.size() == 1 && cbs[0] == std::make_pair(0, std::string("cb1"));
    if (!both && !only1)
    {
        why = "cbuffer_table is not {cb0, cb1} or {cb1}:";
        for (auto& [off, name] : cbs) why += " " + name + "@" + std::to_string(off);
        return false;
    }
    std::sort(args.begin(), args.end(), [](const Impl::ArgField& a, const Impl::ArgField& b) { return a.offset < b.offset; });
    uint32_t bytes = 0;
    for (const Impl::ArgField& a : args) bytes = std::max(bytes, a.offset + 8);
    if (auto it = sizes.find("binding_table"); it != sizes.end() && it->second > 0) bytes = std::max(bytes, uint32_t(it->second));

    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:lib.c_str()]];
    if (!data) { why = "unreadable " + lib.string(); return false; }
    dispatch_data_t dd = dispatch_data_create(data.bytes, data.length, nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithData:dd error:&err];
    if (!library) { why = "newLibraryWithData: " + Describe(err); return false; }
    id<MTLFunction> fn = [library newFunctionWithName:@"shader_main"];
    if (!fn) { why = "no shader_main in " + lib.string(); return false; }
    e.fn = fn;
    e.args = std::move(args);
    e.argBytes = bytes;
    e.cbCount = uint32_t(cbs.size());
    return true;
}

// D2112-2130 Passthrough(n), in MSL: the CPU already produced clip-space vertices; POSITION + n TEXCOORDs in, the
// pixel shader always receives 16 interpolators (unsent = 0, D2107-2109). PS input TEXCOORDk = user(reg{k+1}_0)
// (gpu_hw_metal.h). One function per interpolator count, ttt2_cpu_vs_<n>, all in one library compiled at run time.
std::string CpuVsMsl()
{
    std::string s = "#include <metal_stdlib>\nusing namespace metal;\n\nstruct CpuVsOut {\n    float4 p [[position]];\n";
    for (int r = 1; r <= 16; ++r)
        s += "    float4 r" + std::to_string(r) + " [[user(reg" + std::to_string(r) + "_0)]];\n";        // D2118 (a[16] : TEXCOORD0)
    s += "};\n";
    for (uint32_t n = 0; n <= 16; ++n)
    {
        const std::string N = std::to_string(n);
        s += "\nstruct CpuVsIn" + N + " {\n    float4 p [[attribute(0)]];\n";                            // D2116, D2125
        for (uint32_t k = 0; k < n; ++k)
            s += "    float4 a" + std::to_string(k) + " [[attribute(" + std::to_string(k + 1) + ")]];\n";   // D2117, D2127
        s += "};\nvertex CpuVsOut ttt2_cpu_vs_" + N + "(CpuVsIn" + N + " v [[stage_in]])\n{\n    CpuVsOut o;\n    o.p = v.p;\n";   // D2119
        for (uint32_t r = 1; r <= 16; ++r)                                                               // D2119-2120
            s += "    o.r" + std::to_string(r) + " = " + (r - 1 < n ? "v.a" + std::to_string(r - 1) : std::string("float4(0.0)")) + ";\n";
        s += "    return o;\n}\n";                                                                       // D2121
    }
    return s;
}

id<MTLFunction> CpuVs(Impl& m, uint32_t interps)
{
    ShaderState& s = S();
    if (!s.vsLibrary)
    {
        NSError* err = nil;
        id<MTLLibrary> lib = [m.device newLibraryWithSource:[NSString stringWithUTF8String:CpuVsMsl().c_str()]
                                                    options:[MTLCompileOptions new] error:&err];
        if (!lib) throw Failure("metal: fixed CPU-vertex VS did not compile: " + Describe(err));
        for (uint32_t n = 0; n <= 16; ++n)
        {
            s.vs[n] = [lib newFunctionWithName:[NSString stringWithFormat:@"ttt2_cpu_vs_%u", n]];
            if (!s.vs[n]) throw Failure("metal: fixed CPU-vertex VS function " + std::to_string(n) + " missing");
        }
        s.vsLibrary = lib;
    }
    return s.vs[std::min<uint32_t>(interps, 16)];
}

// D4296-4300 StencilRefMask: the back-face register only when nothing but back faces can show.
uint32_t StencilRefMask(const DrawState& st)
{
    const bool backOnly = ((st.depthCtl >> 7) & 1) && (st.cull & 3) == 1;                                // D4298
    return backOnly ? st.stencilRefMaskBf : st.stencilRefMask;                                           // D4299
}

// D4235-4250: Xenos blend factor -> Metal. 12/13 = constant colour, 14/15 = constant alpha: Metal has both
// natively (blendColor / blendAlpha read one blend colour), so D4224-4228's alpha-as-factor trick is not needed.
MTLBlendFactor BlendFactor(uint32_t f, bool alpha)
{
    switch (f)
    {
    case 0: return MTLBlendFactorZero;                                                                   // D4236
    case 1: return MTLBlendFactorOne;                                                                    // D4237
    case 4: return alpha ? MTLBlendFactorSourceAlpha : MTLBlendFactorSourceColor;                        // D4238
    case 5: return alpha ? MTLBlendFactorOneMinusSourceAlpha : MTLBlendFactorOneMinusSourceColor;        // D4239
    case 6: return MTLBlendFactorSourceAlpha;                                                            // D4240
    case 7: return MTLBlendFactorOneMinusSourceAlpha;                                                    // D4241
    case 8: return alpha ? MTLBlendFactorDestinationAlpha : MTLBlendFactorDestinationColor;              // D4242
    case 9: return alpha ? MTLBlendFactorOneMinusDestinationAlpha : MTLBlendFactorOneMinusDestinationColor;   // D4243
    case 10: return MTLBlendFactorDestinationAlpha;                                                      // D4244
    case 11: return MTLBlendFactorOneMinusDestinationAlpha;                                              // D4245
    case 12: return MTLBlendFactorBlendColor;                                                            // D4246
    case 14: return MTLBlendFactorBlendAlpha;                                                            // D4246 (native constant alpha)
    case 13: return MTLBlendFactorOneMinusBlendColor;                                                    // D4247
    case 15: return MTLBlendFactorOneMinusBlendAlpha;                                                    // D4247 (native constant alpha)
    case 16: return alpha ? MTLBlendFactorOne : MTLBlendFactorSourceAlphaSaturated;                      // D4248
    default: return MTLBlendFactorOne;                                                                   // D4249: CPU BlendFactor default
    }
}

MTLBlendOperation BlendOp(uint32_t o)                                                                   // D4252-4259
{
    switch (o)
    {
    case 1: return MTLBlendOperationSubtract;
    case 2: return MTLBlendOperationMin;
    case 3: return MTLBlendOperationMax;
    case 4: return MTLBlendOperationReverseSubtract;
    default: return MTLBlendOperationAdd;
    }
}

} // namespace

// ================================================================ Backend::Impl (shaders section)
const Backend::Impl::Ps* Backend::Impl::PixelShader(const std::vector<uint32_t>& code)
{
    ShaderState& s = S();
    const uint64_t key = s.psKeys.Get(code, PsKey);                                                      // D3047
    if (auto it = s.ps.find(key); it != s.ps.end())                                                      // D3050-3056
        return it->second.failed ? nullptr : &it->second;
    Initialize();
    Ps& e = s.ps[key];
    // D3384-3393 BuildPixelShader, generic variant, alpha test on (TranslatePixelShader(code, false, nullptr)).
    const PsTranslation t = TranslatePixelShader(code);                                                  // D3389
    if (!t.unsupported.empty())                                                                          // D3390
    {
        std::fprintf(stderr, "[gpu-metal] pixel shader %016llX (%zu dwords) not translatable: %s - its draws are skipped\n",
                     (unsigned long long)key, code.size(), t.unsupported.c_str());                       // D3077-3079
        e.failed = true;                                                                                 // D3080
        return nullptr;
    }
    e.slots = t.textureSlots; e.cubeSlots = t.cubeSlots; e.constantRows = t.constantRows;               // D3391
    const std::string hlsl = PixelShaderHlsl(t, TextureStateVariant::Generic, true);                     // D3394
    e.textKey = Fnv(hlsl.data(), hlsl.size());                                                           // D3417
    bool missing = false;
    std::string why;
    if (!LoadMetallib(device, MetalCacheDirectory(), e, missing, why))
    {
        e.failed = true;
        if (missing)
        {
            ++s.cacheMisses;
            if (s.missLogged.insert(e.textKey).second)
                std::fprintf(stderr, "[gpu-metal] ps cache miss %016llX\n", (unsigned long long)e.textKey);
        }
        else
            std::fprintf(stderr, "[gpu-metal] pixel shader %016llX (ps_%016llX) failed to load - its draws are skipped: %s\n",
                         (unsigned long long)key, (unsigned long long)e.textKey, why.c_str());
        return nullptr;
    }
    return &e;
}

id<MTLRenderPipelineState> Backend::Impl::Pipeline(const Ps& ps, uint32_t interps, MTLPixelFormat color, MTLPixelFormat depth,
                                                   uint32_t blend, uint32_t writeMask)
{
    interps = std::min<uint32_t>(interps, 16);
    const uint32_t b = blend;                                                                            // D4218
    const uint32_t srcF = b & 0x1F, op = (b >> 5) & 7, dstF = (b >> 8) & 0x1F;                          // D4219
    const uint32_t srcFA = (b >> 16) & 0x1F, opA = (b >> 21) & 7, dstFA = (b >> 24) & 0x1F;             // D4220
    const bool passthrough = srcF == 1 && dstF == 0 && srcFA == 1 && dstFA == 0 && op == 0 && opA == 0;  // D4221
    const bool hasColor = color != MTLPixelFormatInvalid;
    // D4229 key: blend word (0 when passthrough) | write mask; plus what Metal bakes into the pipeline.
    const std::array<uint64_t, 3> key = {
        ps.textKey,
        (uint64_t(passthrough || !hasColor ? 0 : b) << 32) | (uint64_t(hasColor ? writeMask & 15 : 0) << 8) | interps,
        (uint64_t(color) << 32) | uint64_t(depth) };
    ShaderState& s = S();
    if (auto it = s.pipelines.find(key); it != s.pipelines.end()) return it->second;
    @autoreleasepool {
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = CpuVs(*this, interps);                                                        // D450
        d.fragmentFunction = ps.fn;
        // D2124-2127 input layout: POSITION float4 @0, TEXCOORDi float4 @16+16i, one buffer, per vertex; D394 stride.
        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        for (uint32_t a = 0; a <= interps; ++a)
        {
            vd.attributes[a].format = MTLVertexFormatFloat4;
            vd.attributes[a].offset = 16 * a;
            vd.attributes[a].bufferIndex = 0;
        }
        vd.layouts[0].stride = 16 + 16 * interps;                                                        // D394-395
        vd.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
        vd.layouts[0].stepRate = 1;
        d.vertexDescriptor = vd;
        if (hasColor)
        {
            MTLRenderPipelineColorAttachmentDescriptor* ca = d.colorAttachments[0];
            ca.pixelFormat = color;
            ca.blendingEnabled = passthrough ? NO : YES;                                                 // D4263
            ca.sourceRGBBlendFactor = BlendFactor(srcF, false);                                          // D4264
            ca.destinationRGBBlendFactor = BlendFactor(dstF, false);
            ca.rgbBlendOperation = BlendOp(op);
            ca.sourceAlphaBlendFactor = BlendFactor(srcFA, true);                                        // D4265
            ca.destinationAlphaBlendFactor = BlendFactor(dstFA, true);
            ca.alphaBlendOperation = BlendOp(opA);
            const uint32_t wm = writeMask & 15;                                                          // D4266: bit0 R .. bit3 A
            ca.writeMask = ((wm & 1) ? MTLColorWriteMaskRed : 0) | ((wm & 2) ? MTLColorWriteMaskGreen : 0) |
                           ((wm & 4) ? MTLColorWriteMaskBlue : 0) | ((wm & 8) ? MTLColorWriteMaskAlpha : 0);
        }
        d.depthAttachmentPixelFormat = depth;
        if (depth == MTLPixelFormatDepth32Float_Stencil8) d.stencilAttachmentPixelFormat = depth;
        NSError* err = nil;
        id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&err];
        if (!p)
            throw Failure("metal: pipeline for ps_" + Hex16(ps.textKey) + " (interps " + std::to_string(interps) +
                          ", colour " + std::to_string(long(color)) + ", depth " + std::to_string(long(depth)) + "): " + Describe(err));
        return s.pipelines[key] = p;
    }
}

id<MTLDepthStencilState> Backend::Impl::DepthState(const DrawState& st)
{
    uint64_t key = (st.zEnable ? 1u : 0u) | (st.zWrite ? 2u : 0u) | (st.zFunc << 2);                     // D4307
    if (st.stencil)                                                                                      // D4308
    {
        const uint32_t masks = (StencilRefMask(st) >> 8) & 0xFFFF;                                       // D4310
        key |= (1ull << 5) | (uint64_t(st.depthCtl >> 7) << 6) | (uint64_t(masks) << 31);               // D4311
    }
    ShaderState& s = S();
    if (auto it = s.depthStates.find(key); it != s.depthStates.end()) return it->second;                 // D4313-4314
    @autoreleasepool {
        MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
        // D4316-4318. Metal has no depth enable: off = always pass, never write (as DepthEnable = FALSE).
        // Xenos compare functions are in D3D11's order (NEVER + n), which is also Metal's (Never = 0 .. Always = 7).
        d.depthCompareFunction = st.zEnable ? MTLCompareFunction(st.zFunc & 7) : MTLCompareFunctionAlways;
        d.depthWriteEnabled = (st.zEnable && st.zWrite) ? YES : NO;
        if (st.stencil)                                                                                  // D4319
        {
            const uint32_t c = st.depthCtl, rm = StencilRefMask(st);                                     // D4321
            // D4325-4332: face(bits) = func 0-2, fail 3-5, pass 6-8, depth-fail 9-11. Ops keep, zero, replace,
            // incr-clamp, decr-clamp, invert, incr-wrap, decr-wrap: D3D11's order and Metal's (Keep = 0 ..).
            auto face = [rm](uint32_t bits) {
                MTLStencilDescriptor* f = [MTLStencilDescriptor new];
                f.stencilCompareFunction = MTLCompareFunction(bits & 7);                                 // D4328
                f.stencilFailureOperation = MTLStencilOperation((bits >> 3) & 7);                        // D4329
                f.depthStencilPassOperation = MTLStencilOperation((bits >> 6) & 7);                      // D4330
                f.depthFailureOperation = MTLStencilOperation((bits >> 9) & 7);                          // D4331
                f.readMask = (rm >> 8) & 0xFF;                                                           // D4323
                f.writeMask = (rm >> 16) & 0xFF;                                                         // D4324
                return f;
            };
            d.frontFaceStencil = face(c >> 8);                                                           // D4333
            d.backFaceStencil = ((c >> 7) & 1) ? face(c >> 20) : face(c >> 8);                           // D4334
        }
        id<MTLDepthStencilState> state = [device newDepthStencilStateWithDescriptor:d];
        if (!state) throw Failure("metal: newDepthStencilStateWithDescriptor");                         // D4336
        return s.depthStates[key] = state;                                                               // D4337
    }
}

} // namespace ttt2_hw
