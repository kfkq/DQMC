#ifndef UPDATES_HPP
#define UPDATES_HPP

#include "dqmc.hpp"
#include "model.hpp"
#include "utility.hpp"
#include <vector>

namespace updates {

void replica_exchange(
    int sweep,
    int exchange_freq,
    int rank,
    int world_size,
    DQMC& sim,
    model::HubbardAttractiveU& hubbard,
    std::vector<GF>& greens,
    std::vector<linalg::LDRStack>& propagation_stacks,
    utility::random& rng,
    int& exchange_attempts,
    int& exchange_accepts
);

} // namespace updates

#endif // UPDATES_HPP