//
// Created by Bilal Kazi on 2019-04-01.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_BK_TRAIN_H
#define PBRT_INTEGRATORS_BK_TRAIN_H

#include <fstream>
#include <ostream>
#include <mutex>
#include "pbrt.h"
#include "integrator.h"
#include "lightdistrib.h"

namespace pbrt {

class BkTrainIntegrator : public SamplerIntegrator {
  public:
    BkTrainIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
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
    std::mutex *m;
    std::ofstream *outfile;
};

BkTrainIntegrator *CreateBkTrainIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_BK_TRAIN_H
