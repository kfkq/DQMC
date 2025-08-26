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

class AttractiveHubbard {
    private:
        // Model parameters
        double t_;        // hopping parameter
        double mu_;       // chemical potential
        double g_;        // g = sqrt(dtau * U / 2)
        double alpha_;    // Hubbard-Stratonovich coupling
        
        // Lattice info
        int ns_;     // number of lattice sites
        int nt_;       // number of time slices
        
        // Matrices
        arma::mat expK_; 
        arma::mat invexpK_; 

        //GHQField
        arma::vec gamma_;
        arma::vec eta_;
        arma::imat fields_;
        
        // Random number generator
        utility::random& rng_;

        arma::mat build_K_matrix(const Lattice& lat);
        void init_GHQfields();

    public:
        AttractiveHubbard(const utility::parameters& params, const Lattice& lat, utility::random& rng);

        // Getters
        const arma::mat& expK(int flv) const { return expK_; }
        const arma::mat& invexpK(int flv) const { return invexpK_; }
        arma::vec expV(int l, int flv);
        arma::vec invexpV(int l, int flv);

        const arma::imat& fields() const { return fields_; }

        int nt() const { return nt_; }
        int ns() const { return ns_; }
        int n_flavor() const { return 1; }

        double acceptance_ratio(arma::mat& Gtt, double delta, int i);
        void update_greens(arma::mat& gtt, double delta, int i);
        double update_time_slice(std::vector<GF>& greens, int l);
};

namespace Observables {
    double calculate_density(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_doubleOccupancy(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_swavePairing(const std::vector<GF>& greens, const Lattice& lat);
    arma::mat calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_doublonTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_currxxTau(const std::vector<GF>& greens, const Lattice& lat);
} // namespace observables