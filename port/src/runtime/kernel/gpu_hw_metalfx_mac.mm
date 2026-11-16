// MetalFX spatial upscaling at presentation: see gpu_hw_metalfx.h.
//
// Objective-C++ with ARC (port/CMakeLists.txt, ttt2_mac_metalfx). DXVK Native's Windows headers
// stay out of this file: their BOOL and `interface` clash with Objective-C. The few COM calls go
// through local declarations of IUnknown (include/native/windows/unknwn.h: no virtual destructor)
// and of DXVK's interop interfaces, in the vtable order and with the GUIDs of DXVK Native 1.9.2b's
// src/dxgi/dxgi_interfaces.h, which DXVK does not install. Vulkan types come from the bundle's
// Vulkan-Headers 1.4.363 (mac/build_libs.sh), which have VK_EXT_metal_objects.
//
// Per frame, on the presenting thread:
//   1. the exports (VkImage -> MTLTexture) and every check, nothing submitted yet;
//   2. FlushRenderingCommands (the immediate context's Flush, then the CS thread drained) and
//      LockSubmissionQueue (DXVK's submit thread idle: every command list is past vkQueueSubmit,
//      which with MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS=1 has committed its MTLCommandBuffers);
//   3. one MTLCommandBuffer on MoltenVK's MTLCommandQueue: the scaler (and a blit when the output
//      lacks the scaler's usages), committed after DXVK's work and before the pass that reads it;
//   4. ReleaseSubmissionQueue on every path that locked (Unlock below).
// TTT2_METALFX_SYNC=1: PresentScaled waits for the GPU (D3D11_QUERY_EVENT) before the call, and
// waitUntilCompleted follows step 4, to tell ordering faults from image faults.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#define VK_USE_PLATFORM_METAL_EXT 1
#include <vulkan/vulkan.h>

#include "gpu_hw_metalfx.h"

#include <dlfcn.h>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct ComGuid { uint32_t a; uint16_t b, c; uint8_t d[8]; };
struct ComUnknown
{
    virtual int32_t QueryInterface(const ComGuid& riid, void** object) = 0;
    virtual uint32_t AddRef() = 0;
    virtual uint32_t Release() = 0;
};
struct VkInteropDevice;
// IDXGIVkInteropSurface (dxgi_interfaces.h:164-208).
struct VkInteropSurface : ComUnknown
{
    virtual int32_t GetDevice(VkInteropDevice** device) = 0;
    virtual int32_t GetVulkanImageInfo(VkImage* handle, VkImageLayout* layout, VkImageCreateInfo* info) = 0;
};
// IDXGIVkInteropDevice (dxgi_interfaces.h:220-292).
struct VkInteropDevice : ComUnknown
{
    virtual void GetVulkanHandles(VkInstance* instance, VkPhysicalDevice* physical, VkDevice* device) = 0;
    virtual void GetSubmissionQueue(VkQueue* queue, uint32_t* family) = 0;
    virtual void TransitionSurfaceLayout(VkInteropSurface* surface, const VkImageSubresourceRange* subresources,
                                         VkImageLayout from, VkImageLayout to) = 0;
    virtual void FlushRenderingCommands() = 0;
    virtual void LockSubmissionQueue() = 0;
    virtual void ReleaseSubmissionQueue() = 0;
};
// dxgi_interfaces.h:373 and :375.
constexpr ComGuid kInteropDeviceIid{ 0xe2ef5fa5, 0xdc21, 0x4af7, { 0x90, 0xc4, 0xf6, 0x7e, 0xf6, 0xa0, 0x93, 0x23 } };
constexpr ComGuid kInteropSurfaceIid{ 0x5546cf8c, 0x77e7, 0x4341, { 0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d } };

template <typename T>
T* Query(void* object, const ComGuid& iid)
{
    void* out = nullptr;
    if (!object || static_cast<ComUnknown*>(object)->QueryInterface(iid, &out) < 0) return nullptr;
    return static_cast<T*>(out);
}

struct State
{
    bool configured = false, sync = false;
    std::string dead;       // a reason that holds for the rest of the run
    std::string logged;     // the last state line, so each change is logged once
    // Cached per D3D11 device (the pointer only identifies it).
    const void* device = nullptr;
    VkDevice vkDevice = VK_NULL_HANDLE;
    VkQueue vkQueue = VK_NULL_HANDLE;
    PFN_vkExportMetalObjectsEXT exportObjects = nullptr;
    id<MTLDevice> mtlDevice = nil;
    id<MTLCommandQueue> mtlQueue = nil;
    float maxScale = 2.0f;
    // The scaler for sw x sh -> vw x vh, or the sizes it refused.
    id<MTLFXSpatialScaler> scaler = nil;
    uint32_t sw = 0, sh = 0, vw = 0, vh = 0;
    uint32_t refused[4] = {};
    id<MTLTexture> own = nil;   // the scaler's output when DXVK's texture lacks its usages
};

State& S()
{
    static State s;
    return s;
}

void Log(const std::string& line)
{
    State& s = S();
    if (s.logged == line) return;
    s.logged = line;
    std::fprintf(stderr, "[metalfx] %s\n", line.c_str());
}

bool Off(const std::string& reason)
{
    Log("off: " + reason);
    return false;
}

bool Dead(const std::string& reason)
{
    S().dead = reason;
    return Off(reason);
}

std::string Format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string Format(const char* fmt, ...)
{
    char buffer[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    return buffer;
}

void DropScaler(State& s)
{
    s.scaler = nil;
    s.own = nil;
    s.sw = s.sh = s.vw = s.vh = 0;
    std::fill(std::begin(s.refused), std::end(s.refused), 0u);
}

void DropDevice(State& s)
{
    DropScaler(s);
    s.device = nullptr;
    s.vkDevice = VK_NULL_HANDLE;
    s.vkQueue = VK_NULL_HANDLE;
    s.exportObjects = nullptr;
    s.mtlDevice = nil;
    s.mtlQueue = nil;
}

// The Vulkan handles, the export entry point, the MTLDevice and MoltenVK's MTLCommandQueue.
bool InitDevice(State& s, VkInteropDevice* interop)
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = 0;
    interop->GetVulkanHandles(&instance, &physical, &device);
    interop->GetSubmissionQueue(&queue, &family);
    if (!instance || !physical || !device || !queue) return Dead("DXVK's interop returned no Vulkan handles");
    // DXVK links the Vulkan loader (libvulkan.1.dylib); its symbols are global.
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(RTLD_DEFAULT, "vkGetInstanceProcAddr"));
    if (!gipa) return Dead("vkGetInstanceProcAddr not found");
    auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa(instance, "vkEnumerateDeviceExtensionProperties"));
    if (!gdpa || !enumerate) return Dead("Vulkan entry points missing");
    uint32_t count = 0;
    enumerate(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    enumerate(physical, nullptr, &count, extensions.data());
    const bool offered = std::any_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& e) {
        return std::string(e.extensionName) == VK_EXT_METAL_OBJECTS_EXTENSION_NAME;
    });
    if (!offered) return Dead("the driver does not offer " VK_EXT_METAL_OBJECTS_EXTENSION_NAME);
    auto exportObjects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(gdpa(device, "vkExportMetalObjectsEXT"));
    if (!exportObjects) return Dead("vkExportMetalObjectsEXT missing (DXVK without mac/patches/dxvk-1.9.2b-metal-objects-macos.patch?)");
    VkExportMetalCommandQueueInfoEXT queueInfo{ VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT };
    queueInfo.queue = queue;
    VkExportMetalDeviceInfoEXT deviceInfo{ VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT, &queueInfo };
    VkExportMetalObjectsInfoEXT objects{ VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT, &deviceInfo };
    exportObjects(device, &objects);
    id<MTLDevice> mtlDevice = deviceInfo.mtlDevice;
    id<MTLCommandQueue> mtlQueue = queueInfo.mtlCommandQueue;
    if (!mtlDevice || !mtlQueue) return Dead("MoltenVK exported no MTLDevice or MTLCommandQueue");
    if (![MTLFXSpatialScalerDescriptor supportsDevice:mtlDevice])
        return Dead(Format("MetalFX spatial scaling is not supported on %s", [[mtlDevice name] UTF8String]));
    // No scale query exists for the spatial scaler; the temporal one's (macOS 14) bounds the device,
    // MetalFX's original 2x without it. A size the scaler refuses still returns nil below.
    float maxScale = 2.0f;
    if (@available(macOS 14.0, *))
        maxScale = std::max(1.0f, [MTLFXTemporalScalerDescriptor supportedInputContentMaxScaleForDevice:mtlDevice]);
    s.vkDevice = device;
    s.vkQueue = queue;
    s.exportObjects = exportObjects;
    s.mtlDevice = mtlDevice;
    s.mtlQueue = mtlQueue;
    s.maxScale = maxScale;
    return true;
}

struct Image { VkImage image = VK_NULL_HANDLE; VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO }; id<MTLTexture> texture = nil; };

bool Export(State& s, VkInteropSurface* surface, Image& out)
{
    if (surface->GetVulkanImageInfo(&out.image, nullptr, &out.info) < 0 || !out.image) return false;
    VkExportMetalTextureInfoEXT textureInfo{ VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT };
    textureInfo.image = out.image;
    textureInfo.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VkExportMetalObjectsInfoEXT objects{ VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT, &textureInfo };
    s.exportObjects(s.vkDevice, &objects);
    out.texture = textureInfo.mtlTexture;   // a strong reference from here on
    return out.texture != nil;
}

// Everything but the COM references, which the caller releases after the queue is unlocked.
bool Run(State& s, const void* device, VkInteropDevice* interop, VkInteropSurface* inSurface, VkInteropSurface* outSurface)
{
    if (s.device != device)
    {
        DropDevice(s);
        if (!InitDevice(s, interop)) return false;
        s.device = device;
    }
    Image in, out;
    if (!Export(s, inSurface, in) || !Export(s, outSurface, out)) return Dead("a texture export failed");
    const uint32_t sw = in.info.extent.width, sh = in.info.extent.height;
    const uint32_t vw = out.info.extent.width, vh = out.info.extent.height;
    if (in.texture.pixelFormat != MTLPixelFormatBGRA8Unorm || out.texture.pixelFormat != MTLPixelFormatBGRA8Unorm)
        return Dead(Format("texture formats %lu and %lu, not BGRA8Unorm", (unsigned long)in.texture.pixelFormat,
                           (unsigned long)out.texture.pixelFormat));
    if (vw <= sw || vh <= sh) return Off(Format("%ux%u -> %ux%u does not enlarge", sw, sh, vw, vh));
    const float scale = std::max(float(vw) / float(sw), float(vh) / float(sh));
    if (scale > s.maxScale)
        return Off(Format("%ux%u -> %ux%u is %.2fx, above the device's %.2fx", sw, sh, vw, vh, scale, s.maxScale));
    if (!s.scaler || s.sw != sw || s.sh != sh || s.vw != vw || s.vh != vh)
    {
        if (s.refused[0] == sw && s.refused[1] == sh && s.refused[2] == vw && s.refused[3] == vh) return false;
        DropScaler(s);
        MTLFXSpatialScalerDescriptor* descriptor = [[MTLFXSpatialScalerDescriptor alloc] init];
        descriptor.inputWidth = sw;
        descriptor.inputHeight = sh;
        descriptor.outputWidth = vw;
        descriptor.outputHeight = vh;
        descriptor.colorTextureFormat = MTLPixelFormatBGRA8Unorm;
        descriptor.outputTextureFormat = MTLPixelFormatBGRA8Unorm;
        // The image already carries the game's gamma.
        descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        id<MTLFXSpatialScaler> scaler = [descriptor newSpatialScalerWithDevice:s.mtlDevice];
        if (!scaler)
        {
            s.refused[0] = sw; s.refused[1] = sh; s.refused[2] = vw; s.refused[3] = vh;
            return Off(Format("the scaler refused %ux%u -> %ux%u BGRA8Unorm", sw, sh, vw, vh));
        }
        s.scaler = scaler;
        s.sw = sw; s.sh = sh; s.vw = vw; s.vh = vh;
    }
    const MTLTextureUsage inNeeds = s.scaler.colorTextureUsage, outNeeds = s.scaler.outputTextureUsage;
    if ((in.texture.usage & inNeeds) != inNeeds)
        return Dead(Format("the input's usage 0x%lx lacks the scaler's 0x%lx", (unsigned long)in.texture.usage,
                           (unsigned long)inNeeds));
    const bool blit = (out.texture.usage & outNeeds) != outNeeds;
    if (!blit) s.own = nil;
    else if (!s.own || s.own.width != vw || s.own.height != vh)
    {
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                                                              width:vw
                                                                                             height:vh
                                                                                          mipmapped:NO];
        descriptor.usage = outNeeds;
        descriptor.storageMode = MTLStorageModePrivate;
        s.own = [s.mtlDevice newTextureWithDescriptor:descriptor];
        if (!s.own) return Off(Format("no %ux%u texture for the scaler's output", vw, vh));
        s.own.label = @"TTT2 MetalFX output";
    }
    Log(Format("on: %ux%u -> %ux%u, output usage 0x%lx (scaler needs 0x%lx), blit %s%s", sw, sh, vw, vh,
               (unsigned long)out.texture.usage, (unsigned long)outNeeds, blit ? "yes" : "no", s.sync ? ", sync" : ""));

    id<MTLCommandBuffer> commandBuffer = nil;
    {
        interop->FlushRenderingCommands();
        interop->LockSubmissionQueue();
        struct Unlock
        {
            VkInteropDevice* interop;
            ~Unlock() { interop->ReleaseSubmissionQueue(); }
        } unlock{ interop };
        commandBuffer = [s.mtlQueue commandBuffer];
        if (!commandBuffer) return Off("no Metal command buffer");
        commandBuffer.label = @"TTT2 MetalFX";
        id<MTLTexture> target = blit ? s.own : out.texture;
        @try
        {
            s.scaler.colorTexture = in.texture;
            s.scaler.outputTexture = target;
            s.scaler.inputContentWidth = sw;
            s.scaler.inputContentHeight = sh;
            [s.scaler encodeToCommandBuffer:commandBuffer];
            if (blit)
            {
                id<MTLBlitCommandEncoder> copy = [commandBuffer blitCommandEncoder];
                [copy copyFromTexture:target
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(vw, vh, 1)
                            toTexture:out.texture
                     destinationSlice:0
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                [copy endEncoding];
            }
        }
        @catch (NSException* exception)
        {
            // Never committed: the command buffer goes away with its commands.
            const char* what = [[exception description] UTF8String];
            return Dead(std::string("encoding failed: ") + (what ? what : "an Objective-C exception"));
        }
        [commandBuffer commit];
    }
    if (s.sync) [commandBuffer waitUntilCompleted];
    return true;
}

} // namespace

namespace metalfx {

bool Available(const char** reason)
{
    State& s = S();
    if (!s.configured)
    {
        s.configured = true;
        const char* on = std::getenv("TTT2_METALFX");
        const char* sync = std::getenv("TTT2_METALFX_SYNC");
        const char* submits = std::getenv("MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS");
        s.sync = sync && *sync == '1';
        @autoreleasepool
        {
            if (!on || *on != '1') Dead("TTT2_METALFX is not 1");
            else if (!NSClassFromString(@"MTLFXSpatialScalerDescriptor")) Dead("the MetalFX framework is missing");
            else if (submits && *submits == '0') Dead("MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS=0 breaks the queue order");
        }
    }
    if (reason) *reason = s.dead.empty() ? nullptr : s.dead.c_str();
    return s.dead.empty();
}

bool Upscale(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* in, ID3D11Texture2D* out)
{
    // The context is flushed by FlushRenderingCommands (D3D11VkInterop: Flush, then the CS thread).
    (void)context;
    if (!Available() || !device || !in || !out) return false;
    @autoreleasepool
    {
        State& s = S();
        auto* interop = Query<VkInteropDevice>(device, kInteropDeviceIid);
        auto* inSurface = Query<VkInteropSurface>(in, kInteropSurfaceIid);
        auto* outSurface = Query<VkInteropSurface>(out, kInteropSurfaceIid);
        const bool ok = interop && inSurface && outSurface ? Run(s, device, interop, inSurface, outSurface)
                                                           : Dead("DXVK's Vulkan interop interfaces are missing");
        if (outSurface) outSurface->Release();
        if (inSurface) inSurface->Release();
        if (interop) interop->Release();
        return ok;
    }
}

void Release()
{
    @autoreleasepool
    {
        DropDevice(S());
    }
}

} // namespace metalfx
