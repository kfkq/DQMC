/*
/   This module provides an implementation of attractive Hubbard model.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>

#include <stackngf.h>
#include <stablelinalg.h>
#include <field.h>
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

        GHQField fields_;
        
        // Random number generator
        utility::random& rng_;

        arma::mat build_K_matrix(const Lattice& lat);
    public:
        AttractiveHubbard(const utility::parameters& params, const Lattice& lat, utility::random& rng);

        // Getters
        const arma::mat& expK(int flv) const { return expK_; }
        const arma::mat& invexpK(int flv) const { return invexpK_; }
        arma::vec expV(int l, int flv);
        arma::vec invexpV(int l, int flv);

        GHQField& fields() { return fields_; }

        int nt() const { return nt_; }
        int ns() const { return ns_; }
        int n_flavor() const { return 1; }

        double det_ratio(arma::mat& gtt, double delta, int i);
        std::pair<double, double> bosonic_ratio(int new_field, int old_field);
        std::pair<double, double> local_update_ratio(std::vector<GF>& GF, int l, int field_idx, int new_field);
        void update_greens_local(std::vector<GF>& GF, double delta, int l, int field_idx);
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