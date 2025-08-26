/*
/   This module provides an implementation of attractive Hubbard model.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>

#include <stackngf.h>
#include <stablelinalg.h>
#include <lattice.h>
#include <utility.h>

namespace model {

    // Type aliases for better readability
    using Matrix = arma::mat;
    using Vector = arma::vec;
    using IMatrix = arma::imat;
    using GreenFunc = arma::mat;

    class HubbardAttractiveU {
        private:
            // Model parameters
            double t_;        // hopping parameter
            double U_;        // interaction strength
            double mu_;       // chemical potential
            double dtau_;     // imaginary time step
            double alpha_;    // Hubbard-Stratonovich coupling
            double n_flavor_; // Number of factorized DQMC product (spin)
            
            // Lattice info
            const int ns_;     // number of lattice sites
            const int nt_;       // number of time slices
            
            // Matrices
            Matrix expK_; 
            Matrix invexpK_; 
            Vector expV_;

            //GHQField
            arma::vec gamma_;
            arma::vec eta_;
            IMatrix fields_;
            
            // Random number generator
            utility::random& rng_;

            void init_expK(const Lattice& lat);
            void init_GHQfields();

        public:
            HubbardAttractiveU(const Lattice& lat, 
                            double t, double U, double mu, 
                            double dtau, int nt,
                            utility::random& rng);

            // Getters
            const Matrix& expK() const { return expK_; }
            const IMatrix& fields() const { return fields_; }
            int nt() const { return nt_; }
            int ns() const { return ns_; }
            int n_flavor() const { return n_flavor_; }

            // Functions that will be used in the simulation
            Matrix calc_B(int t, int nfl);
            Matrix calc_invB(int t, int nfl);

            double acceptance_ratio(GreenFunc& Gtt, double delta, int i);
            void update_greens(GreenFunc& gtt, double delta, int i);
            double update_time_slice(std::vector<GF>& greens, int l);
    };

} // namespace model

namespace Observables {
    double calculate_density(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_doubleOccupancy(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_swavePairing(const std::vector<GF>& greens, const Lattice& lat);
    arma::mat calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_doublonTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_currxxTau(const std::vector<GF>& greens, const Lattice& lat);
} // namespace observables