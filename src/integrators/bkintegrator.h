//
// Created by Bilal Kazi on 2019-04-01.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_BK_H
#define PBRT_INTEGRATORS_BK_H

#include "tensorflow/core/public/session.h"
#include <fstream>
#include <mutex>
#include <ostream>
#include <vector>
#include "integrator.h"
#include "lightdistrib.h"
#include "pbrt.h"

namespace pbrt {

class BkIntegrator : public SamplerIntegrator {
  public:
    BkIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                   std::shared_ptr<Sampler> sampler,
                   const Bounds2i &pixelBounds, Float rrThreshold = 1,
                   const std::string &lightSampleStrategy = "spatial");

    void Preprocess(const Scene &scene, Sampler &sampler);
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

  private:
    const int maxDepth;
    const Float rrThreshold;
    const std::string lightSampleStrategy;
    std::unique_ptr<LightDistribution> lightDistribution;
    std::unique_ptr<tensorflow::Session> sess;

};

BkIntegrator *CreateBkIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_BK_H
