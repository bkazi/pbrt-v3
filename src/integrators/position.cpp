//
// Created by Bilal Kazi on 2019-04-03.
//

#include "integrators/position.h"
#include "interaction.h"
#include "paramset.h"
#include "camera.h"
#include "film.h"
#include "stats.h"

namespace pbrt {

// DirectLightingIntegrator Method Definitions
void PositionIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    if (strategy == LightStrategy::UniformSampleAll) {
        // Compute number of samples to use for each light
        for (const auto &light : scene.lights)
            nLightSamples.push_back(sampler.RoundCount(light->nSamples));

        // Request samples for sampling all lights
        for (int i = 0; i < maxDepth; ++i) {
            for (size_t j = 0; j < scene.lights.size(); ++j) {
                sampler.Request2DArray(nLightSamples[j]);
                sampler.Request2DArray(nLightSamples[j]);
            }
        }

        Point3f camerasPos = camera->CameraToWorld(0, Point3f(0.f, 0.f, 0.f));
        maxDist = maxX = maxY = maxZ = -std::numeric_limits<Float>::infinity();
        minX = minY = minZ = std::numeric_limits<Float>::infinity();
        for (int i = 0; i < 8; ++i) {
            maxDist = std::max(
                maxDist, (camerasPos - scene.WorldBound().Corner(i)).Length());
            maxX = std::max(maxX, scene.WorldBound().Corner(i).x);
            minX = std::min(minX, scene.WorldBound().Corner(i).x);
            maxY = std::max(maxY, scene.WorldBound().Corner(i).y);
            minY = std::min(minY, scene.WorldBound().Corner(i).y);
            maxZ = std::max(maxZ, scene.WorldBound().Corner(i).z);
            minZ = std::min(minZ, scene.WorldBound().Corner(i).z);
        }
    }
}

Spectrum PositionIntegrator::Li(const RayDifferential &ray,
                                      const Scene &scene, Sampler &sampler,
                                      MemoryArena &arena, int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L += light->Le(ray);
        return L;
    }
    auto *rgb = new Float[3];
    auto point = isect.p;

    // Integrator for the supplementary renders

//    For the depth image uncomment the following
//    float distance = ray.tMax;
//    rgb[0] = 1.f;
//    rgb[1] = 1.f;
//    rgb[2] = 1.f;
//    return Spectrum::FromRGB(rgb) * ((1 - distance/maxDist) + 1e-3f);

//    For the position image uncomment following
//    rgb[0] = ((point.x - minX) / (maxX - minX)) + 1e-3f;
//    rgb[1] = ((point.y - minY) / (maxY - minY)) + 1e-3f;
//    rgb[2] = ((point.z - minZ) / (maxZ - minZ)) + 1e-3f;
//    return Spectrum::FromRGB(rgb);

//    Following computes the RGB image
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf)
        return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth);
    Vector3f wo = isect.wo;
    // Compute emitted light if ray hit an area light source
    L += isect.Le(wo);
    auto sample = sampler.Get2D();
    return isect.bsdf->rho(wo, 1, &sample, BxDFType(BSDF_ALL & ~BSDF_SPECULAR));
}

PositionIntegrator *CreatePositionIntegrator(
    const ParamSet &params, std::shared_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
    LightStrategy strategy;
    std::string st = params.FindOneString("strategy", "all");
    if (st == "one")
        strategy = LightStrategy::UniformSampleOne;
    else if (st == "all")
        strategy = LightStrategy::UniformSampleAll;
    else {
        Warning(
            "Strategy \"%s\" for direct lighting unknown. "
            "Using \"all\".",
            st.c_str());
        strategy = LightStrategy::UniformSampleAll;
    }
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  np);
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    return new PositionIntegrator(strategy, maxDepth, camera, sampler,
                                        pixelBounds);
}

}  // namespace pbrt