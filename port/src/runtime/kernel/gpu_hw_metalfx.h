#pragma once
// MetalFX spatial upscaling at presentation (macOS, DXVK Native on MoltenVK).
//
// Replaces only the enlarging scale pass of Backend::PresentScaled (gpu_hw_d3d11.h): the image
// at render resolution (after the display pass and FXAA) goes through Apple's spatial scaler
// into a D3D11 texture of the window's 16:9 area; CAS or a texel-for-texel copy then writes the
// back buffer as before. Both textures stay DXVK's; their MTLTextures come from
// VK_EXT_metal_objects (mac/patches/dxvk-1.9.2b-metal-objects-macos.patch), and the Metal work
// is committed on MoltenVK's own MTLCommandQueue right after DXVK's submissions, so queue order
// is the only synchronisation (MVK_CONFIG_SYNCHRONOUS_QUEUE_SUBMITS=1, host_runtime_mac.cpp).
//
//   TTT2_METALFX=1        on (default off)
//   TTT2_METALFX_SYNC=1   waits for the GPU before and after the upscale (debugging)
//
// Any missing piece leaves the bicubic path with one "[metalfx] off: <reason>" line.
// No Metal types here; outside Apple every function does nothing and returns false.

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;

namespace metalfx {

#if defined(__APPLE__)
// TTT2_METALFX=1, the framework present and nothing failed for good. Cheap, every frame.
bool Available(const char** reason = nullptr);
// in: BGRA8 render-size image; out: BGRA8 target of the output size (both RENDER_TARGET |
// SHADER_RESOURCE). True when out now holds the upscaled image; false leaves both untouched.
// out must not be cleared or drawn by D3D11 in this frame before it is read.
bool Upscale(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* in, ID3D11Texture2D* out);
// Drops the scaler, its own output texture and the cached device handles.
void Release();
#else
inline bool Available(const char** reason = nullptr) { if (reason) *reason = "not Apple"; return false; }
inline bool Upscale(ID3D11Device*, ID3D11DeviceContext*, ID3D11Texture2D*, ID3D11Texture2D*) { return false; }
inline void Release() {}
#endif

} // namespace metalfx
