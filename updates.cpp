#include "updates.hpp"
#include <mpi.h>
#include <iostream>

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
    int& exchange_accepts)
{
    if (sweep % exchange_freq != 0) {
        return;
    }

    // Alternate between odd-even and even-odd pairs for swapping
    int pairing = (sweep / exchange_freq) % 2;

    for (int p_rank = pairing; p_rank < world_size - 1; p_rank += 2) {
        int partner = p_rank + 1;
        if (rank == p_rank || rank == partner) {
            int receiver = (rank == p_rank) ? partner : p_rank;

            // 1. Save original state and calculate action before swap
            arma::imat original_fields = hubbard.get_fields();
            double action_before = sim.calculate_action(greens);

            // 2. Exchange fields and pre-swap actions with partner
            arma::imat partner_fields(hubbard.nt(), hubbard.ns());
            double partner_action_before;
            MPI_Sendrecv(original_fields.memptr(), original_fields.n_elem, MPI_INT, receiver, 0,
                         partner_fields.memptr(), partner_fields.n_elem, MPI_INT, receiver, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&action_before, 1, MPI_DOUBLE, receiver, 1,
                         &partner_action_before, 1, MPI_DOUBLE, receiver, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 3. "What-if" calculation with partner's fields
            hubbard.set_fields(partner_fields);
            int n_flavor = hubbard.n_flavor();
            std::vector<linalg::LDRStack> temp_stacks(n_flavor);
            std::vector<GF> temp_greens(n_flavor);
            for (int nfl = 0; nfl < n_flavor; ++nfl) {
                temp_stacks[nfl] = sim.init_stacks(nfl);
                temp_greens[nfl] = sim.init_greenfunctions(temp_stacks[nfl]);
            }
            double action_after = sim.calculate_action(temp_greens);

            // 4. Exchange "after" actions and lower rank makes the decision
            int accepted = 0;
            double partner_action_after;
            MPI_Sendrecv(&action_after, 1, MPI_DOUBLE, receiver, 2,
                         &partner_action_after, 1, MPI_DOUBLE, receiver, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (rank == p_rank) {
                exchange_attempts++;
                double delta = (action_before + partner_action_before) - (action_after + partner_action_after);
                if (delta > 0 || (delta > -30 && rng.bernoulli(exp(delta)))) {
                    accepted = 1;
                    exchange_accepts++;
                }
            }

            // 5. Broadcast decision and update state
            MPI_Bcast(&accepted, 1, MPI_INT, p_rank, MPI_COMM_WORLD);

            if (accepted) { // Commit the swap
                propagation_stacks = std::move(temp_stacks);
                greens = std::move(temp_greens);
            } else { // Reject and revert
                hubbard.set_fields(original_fields);
            }
        }
    }
}

} // namespace updates