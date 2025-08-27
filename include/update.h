#pragma once

#include <model.h>
#include <field.h>
#include <stackngf.h>
#include <utility.h>

namespace update {
    void local_update(utility::random& rng, AttractiveHubbard& model, std::vector<GF>& GF, int l, double& acc_rate) {
        int accepted = 0;
        const int nv = model.ns();
        GHQField& field = model.fields();

        std::vector<int> field_order(nv);
        for (int i = 0; i < nv; ++i) {
            field_order[i] = i;
        }
        std::shuffle(field_order.begin(), field_order.end(), rng.get_generator());

        for (int idx = 0; idx < nv; ++idx) {
            int i = field_order[idx];

            int old_field = field.single_val(l, i);
            int new_field = field.propose_new_field(old_field);

            auto [R, delta] = model.local_update_ratio(GF, l, i, new_field);

            double metropolis_p = std::min(1.0, std::abs(R));
            if (rng.bernoulli(metropolis_p)) {

                accepted += 1;
                
                model.update_greens_local(GF, delta, l, i);
                
                field.set_single_field(l, i, new_field);
            }
        }

        acc_rate = static_cast<double>(accepted) / nv;
    }
}