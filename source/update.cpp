#include <update.h>
#include <mpi.h>

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
            int new_field = field.propose_new_field(old_field, rng);

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

    int partner_rank(int& rank, int& world_size, int& exchange_attempt) {
        int p = exchange_attempt % 2;
        int partner;
        if ( (rank+p)%2 == 0 ) {
            partner = (rank+1);
        } else {
            partner = (rank-1); 
        }
        return partner;
    }

    void replica_exchange(int rank, int world_size, utility::random& rng,
                        int& exchange_attempt, int& exchange_accepted,
                        AttractiveHubbard& model, DQMC& sim, 
                        std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks) 
    {
        exchange_attempt++;
        int partner = partner_rank(rank, world_size, exchange_attempt);        

        if (partner < 0 || partner >= world_size) {
            return; // Partner is out of bounds, do nothing
        }

        // --- MPI FIELD EXCHANGE ---
        arma::imat my_fields = model.fields().fields();
        std::vector<long long> send_buffer(my_fields.memptr(), my_fields.memptr() + my_fields.n_elem);
        std::vector<long long> recv_buffer(my_fields.n_elem);

        MPI_Sendrecv(send_buffer.data(), send_buffer.size(), MPI_LONG_LONG, partner, 0,
                     recv_buffer.data(), recv_buffer.size(), MPI_LONG_LONG, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        arma::imat partner_fields(my_fields.n_rows, my_fields.n_cols);
        std::memcpy(partner_fields.memptr(), recv_buffer.data(), recv_buffer.size() * sizeof(long long));
        // --- END EXCHANGE ---

        // Calculate the action for both fields
        double SC = model.global_action(greens); // S_r({s}_r)
        
        // Temporarily set partner's fields to calculate the "cross" action term
        model.fields().set_fields(partner_fields);
        for (int flv = 0; flv < model.n_flavor(); ++flv) {
            propagation_stacks[flv] = sim.init_stacks(flv);
            greens[flv] = sim.init_greenfunctions(propagation_stacks[flv]);
        }
        double SC_prime = model.global_action(greens); // S_r({s}_{r+1})

        double SC_partner, SC_prime_partner;
        MPI_Sendrecv(&SC_prime, 1, MPI_DOUBLE, partner, 1,
                     &SC_prime_partner, 1, MPI_DOUBLE, partner, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Sendrecv(&SC, 1, MPI_DOUBLE, partner, 2,
                     &SC_partner, 1, MPI_DOUBLE, partner, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        bool accept = false; // Initialize
        if (rank < partner) { // Lower rank is the decider
            double deltaS = (SC_prime + SC_prime_partner) - (SC + SC_partner);
            double metropolis_p = std::min(1.0, std::exp(-deltaS));
            accept = rng.bernoulli(metropolis_p);
            
            // Only rank 0 tracks total acceptance
            if (rank == 0) {
                exchange_accepted += accept;
            }
            MPI_Send(&accept, 1, MPI_C_BOOL, partner, 3, MPI_COMM_WORLD);
        } else { // Higher rank receives the decision
            MPI_Recv(&accept, 1, MPI_C_BOOL, partner, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }      
        
        // If the swap was NOT accepted, we must restore the original state,
        // because the trial calculation above overwrote it.
        if (!accept) {
            model.fields().set_fields(my_fields);
            for (int flv = 0; flv < model.n_flavor(); ++flv) {
                propagation_stacks[flv] = sim.init_stacks(flv);
                greens[flv] = sim.init_greenfunctions(propagation_stacks[flv]);
            }
        }
        // If the swap WAS accepted, the state is already correct from the trial calculation.
    }
}