//
// Created by Bilal Kazi on 2019-04-01.
//

#include "integrators/bkintegrator.h"
#include <core/interaction.h>
#include <iostream>
#include <sstream>
#include <string>
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "integrators/bkintegrator.h"
#include "interaction.h"
#include "parallel.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
tensorflow::Status LoadGraph(const std::string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return tensorflow::Status::OK();
}

namespace pbrt {

STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
BkIntegrator::BkIntegrator(int maxDepth,
                           std::shared_ptr<const Camera> camera,
                           std::shared_ptr<Sampler> sampler,
                           const Bounds2i &pixelBounds, Float rrThreshold,
                           const std::string &lightSampleStrategy)
    : SamplerIntegrator(camera, sampler, pixelBounds),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy) {
        const std::string export_dir = "./model/frozen_graph.pb";
        TF_CHECK_OK(LoadGraph(export_dir, &sess));
      }

void BkIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

Spectrum BkIntegrator::Li(const RayDifferential &r, const Scene &scene,
                          Sampler &sampler, MemoryArena &arena,
                          int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);

    // Intersect _ray_ with scene and store intersection in _isect_
    SurfaceInteraction isect;
    bool foundIntersection = scene.Intersect(ray, &isect);

    if (!foundIntersection) {
        return L;
    }

    // Compute scattering functions and skip over medium boundaries
    isect.ComputeScatteringFunctions(ray, arena, true);
    if (!isect.bsdf) {
        VLOG(2) << "Skipping intersection due to null bsdf";
        ray = isect.SpawnRay(ray.d);
        return L;
    }

    // Sample BSDF to get new path direction
    Vector3f wo = -ray.d;
    tensorflow::TensorShape originShape({1, 3});
    tensorflow::TensorShape incidentShape({1, 3});
    tensorflow::TensorShape normalShape({1, 3});
    tensorflow::Tensor origin(tensorflow::DT_FLOAT, originShape);
    tensorflow::Tensor incident(tensorflow::DT_FLOAT, incidentShape);
    tensorflow::Tensor normal(tensorflow::DT_FLOAT, normalShape);
    tensorflow::Tensor image_rgb(tensorflow::DT_STRING, tensorflow::TensorShape());
    tensorflow::Tensor image_depth(tensorflow::DT_STRING, tensorflow::TensorShape());
    tensorflow::Tensor image_position(tensorflow::DT_STRING, tensorflow::TensorShape());

    auto originData = origin.flat<float>().data();
    for (int i = 0; i < 3; i++) originData[i] = r.o[i];
    auto incidentData = incident.flat<float>().data();
    for (int i = 0; i < 3; i++) incidentData[i] = wo[i];
    auto normalData = normal.flat<float>().data();
    for (int i = 0; i < 3; i++) normalData[i] = isect.shading.n[i];
    image_rgb.scalar<std::string>()() = "./images/cornell-box-color.png";
    image_depth.scalar<std::string>()() = "./images/cornell-box-depth.png";
    image_position.scalar<std::string>()() = "./images/cornell-box-position.png";

    tensor_dict feed_dict = {
        {"origin", origin},
        {"incident", incident},
        {"normal", normal},
        {"image_rgb", image_rgb},
        {"image_depth", image_depth},
        {"image_position", image_position}
    };

    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(sess->Run(feed_dict, {"out"}, {}, &outputs));
    auto rgb = new Float[3];
    for (int i = 0; i < 3; i++) rgb[i] = outputs[0].flat<float>()(i);
    return RGBSpectrum::FromRGB(rgb);
}

BkIntegrator *CreateBkIntegrator(const ParamSet &params,
                                 std::shared_ptr<Sampler> sampler,
                                 std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 5);
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
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new BkIntegrator(maxDepth, camera, sampler, pixelBounds,
                            rrThreshold, lightStrategy);
}

}  // namespace pbrt