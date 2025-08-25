/*

/   Author: Muhammad Gaffar
*/

#pragma once

#include <model_base.h>
#include <stackngf.h>
#include <utility.h>

namespace update {
    void local_update(utility::random& rng, ModelBase& model, std::vector<GF>& GF, int time_slice, double& acc_rate);
}