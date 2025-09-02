#pragma once

#include <model.h>
#include <dqmc.h>
#include <field.h>
#include <stackngf.h>
#include <utility.h>

namespace update {
    // Declaration for local_update
    void local_update(utility::random& rng, AttractiveHubbard& model, std::vector<GF>& GF, int l, double& acc_rate);

    // Declaration for partner_rank
    int partner_rank(const int rank, const int world_size, const int exchange_attempt);

    // Declaration for replica_exchange
    void replica_exchange(int rank, int world_size, utility::random& rng,
                        int& exchange_attempt, int& exchange_accepted,
                        AttractiveHubbard& model, DQMC& sim, 
                        std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);

} // namespace update