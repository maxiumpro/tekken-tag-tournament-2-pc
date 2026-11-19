// R4 Metal backend, M1 slice 1: PROVISIONAL bodies for everything public that is not core, plus a provisional
// Impl::FrontTexture. Each later slice moves its part into its own file (gpu_hw_metal_draw.mm, _shaders.mm,
// _surfaces.mm) and deletes it here. Unsupported paths fail by name (gpu_hw_metal_impl.h porting rule).
//
// Game symbols: g_base (kernel_shared.h; declared here directly so this file does not pull in the PPC
// headers) and DecodeFrontBufferUploadSurface (gpu_hw_texture_upload.h, body in gpu_render.cpp).

#include "gpu_hw_metal_impl.h"
#include "gpu_hw_texture_upload.h"

#import <Foundation/Foundation.h>

#include <cstdio>
#include <cstring>

extern uint8_t* g_base;   // kernel_shared.h: global guest memory base

namespace ttt2_hw {

void Backend::Resolve(uint32_t, uint32_t, uint32_t, uint32_t, bool, uint32_t, bool, uint32_t, uint32_t, bool, uint32_t)
{
    throw Failure("metal: Resolve not implemented (M1 slice 4)");
}

void Backend::CompareResolve(uint32_t, const uint32_t*, uint32_t, uint32_t)
{
    throw Failure("metal: CompareResolve not implemented (M1 slice 4)");
}

bool Backend::DumpResolved(uint32_t, const char*)
{
    throw Failure("metal: DumpResolved not implemented (M1 slice 4)");
}

bool Backend::WriteBackRange(uint32_t, uint32_t, bool)
{
    // Nothing was resolved on the GPU (Resolve throws until slice 4), so guest memory already holds everything.
    if (impl_->resolved.empty()) return false;
    throw Failure("metal: WriteBackRange not implemented (M1 slice 4)");
}

std::function<bool()> Backend::PendingCopyWaiter() const
{
    return {};
}

int Backend::PrebuildShaders(const std::vector<std::filesystem::path>&, unsigned)
{
    std::fprintf(stderr, "[gpu-metal] PrebuildShaders not implemented (M1 slice 3)\n");
    return 3;
}

bool Backend::PrebuildCurrent()
{
    return true;
}

namespace {
// D4927-4933: the CPU-decoded front buffer (cpuFront_), 1280x720 BGRA8. Updated through a blit in the frame's
// command buffer, so a frame still on the GPU never sees the next frame's CPU write.
id<MTLTexture> g_cpuFront = nil;
} // namespace

// logicalW / logicalH: the guest size the returned texture holds (scaled or not). D4906-4939.
id<MTLTexture> Backend::Impl::FrontTexture(uint32_t fb, bool& storedSwap, uint32_t& logicalW, uint32_t& logicalH)
{
    storedSwap = false;                                                          // D4908
    if (!fb) return nil;                                                         // D4909
    auto it = resolved.find(fb);                                                 // D4910
    if (it != resolved.end() && it->second.valid && it->second.tex && it->second.pitch >= 1280 && it->second.height >= 720)   // D4911
    {
        // D4913-4915 (dirty-tracker staleness check) arrives with the surfaces slice; nothing resolves before it.
        storedSwap = it->second.swapRB;                                          // D4916
        logicalW = it->second.pitch; logicalH = it->second.height;               // D4917
        return it->second.tex;                                                   // D4918
    }
    logicalW = 1280; logicalH = 720;                                             // D4922
    // Not GPU-resolved: decode guest memory like the GDI path. The B8G8R8A8 view reads the word as ARGB, red
    // and blue exchanged against the fetch order (low byte = x), the same as a copy_dest_swap resolve.
    storedSwap = true;                                                           // D4926
    Initialize();
    if (!g_cpuFront)                                                             // D4927-4933
    {
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                     width:1280 height:720 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModePrivate;   // written by blit only
        g_cpuFront = [device newTextureWithDescriptor:d];
        if (!g_cpuFront) throw Failure("metal: CPU front-buffer texture");
    }
    static TextureUploadSurface decoded;                                         // D4934
    if (!DecodeFrontBufferUploadSurface(g_base, fb, 1280, 720, decoded))         // D4935-4936
        throw Failure("metal: invalid guest front-buffer upload");
    // D4937 (UpdateSubresource): staging buffer -> texture, ordered in the frame's command buffer.
    id<MTLBuffer> staging = [device newBufferWithBytes:decoded.bytes.data() length:size_t(decoded.rowPitch) * 720
                                               options:MTLResourceStorageModeShared];
    if (!staging) throw Failure("metal: front-buffer staging buffer");
    id<MTLBlitCommandEncoder> blit = [Cmd() blitCommandEncoder];
    if (!blit) throw Failure("metal: blit command encoder");
    [blit copyFromBuffer:staging sourceOffset:0 sourceBytesPerRow:decoded.rowPitch sourceBytesPerImage:size_t(decoded.rowPitch) * 720
              sourceSize:MTLSizeMake(1280, 720, 1) toTexture:g_cpuFront destinationSlice:0 destinationLevel:0
       destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    return g_cpuFront;                                                           // D4938
}

} // namespace ttt2_hw
