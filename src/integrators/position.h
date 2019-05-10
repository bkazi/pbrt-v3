//
// Created by Bilal Kazi on 2019-04-03.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_POSITION_H
#define PBRT_INTEGRATORS_POSITION_H

#include "core/camera.h"
#include "integrator.h"
#include "integrators/directlighting.h"
#include "pbrt.h"
#include "scene.h"

namespace pbrt {

// DirectLightingIntegrator Declarations
class PositionIntegrator : public SamplerIntegrator {
  public:
    // DirectLightingIntegrator Public Methods
    PositionIntegrator(LightStrategy strategy, int maxDepth,
                             std::shared_ptr<const Camera> camera,
                             std::shared_ptr<Sampler> sampler,
                             const Bounds2i &pixelBounds)
        : SamplerIntegrator(camera, sampler, pixelBounds),
          strategy(strategy),
          maxDepth(maxDepth),
          camera(camera) {}
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;
    void Preprocess(const Scene &scene, Sampler &sampler);

  private:
    // DirectLightingIntegrator Private Data
    const LightStrategy strategy;
    const int maxDepth;
    std::vector<int> nLightSamples;
    std::shared_ptr<const Camera> camera;
    float maxDist;
    float maxX, minX, maxY, minY, maxZ, minZ;
};

PositionIntegrator *CreatePositionIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_POSITION_H
