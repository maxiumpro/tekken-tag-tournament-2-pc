// R4 Metal backend, M1: PROVISIONAL bodies for what no slice owns yet. Slice 4 moved FrontTexture, Resolve,
// CompareResolve, DumpResolved, WriteBackRange and PendingCopyWaiter to gpu_hw_metal_surfaces.mm; what remains
// is the shader prebuild (main.cpp --build-shaders and the start-up check). Unsupported paths fail by name
// (gpu_hw_metal_impl.h porting rule).

#include "gpu_hw_metal_impl.h"

#include <cstdio>

namespace ttt2_hw {

int Backend::PrebuildShaders(const std::vector<std::filesystem::path>&, unsigned)
{
    std::fprintf(stderr, "[gpu-metal] PrebuildShaders not implemented (M1 slice 3)\n");
    return 3;
}

bool Backend::PrebuildCurrent()
{
    return true;
}

} // namespace ttt2_hw
