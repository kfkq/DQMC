#include <update.h>

namespace update {
    void local_update(utility::random& rng, ModelBase& model, std::vector<GF>& GF, int time_slice, double& acc_rate) {
        int accepted = 0;
        const int nv = model.nv();

        std::vector<int> field_order(nv);
        for (int i = 0; i < nv; ++i) {
            field_order[i] = i;
        }
        std::shuffle(field_order.begin(), field_order.end(), rng.get_generator());

        for (int idx = 0; idx < nv; ++idx) {
            int i = field_order[idx];

            auto new_field = model.propose_field(time_slice, i);

            double ratio = model.local_update_ratio(GF, time_slice, i, new_field);

            double metropolis_p = std::min(1.0, std::abs(ratio));
            if (rng.bernoulli(metropolis_p)) {

                accepted += 1;
                
                model.update_greens_local(GF, time_slice, i);
                
                model.set_field_value(time_slice, i, new_field);
            }
        }

        acc_rate = static_cast<double>(accepted) / nv;
    }
}