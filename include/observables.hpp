#ifndef OBSERVABLES_HPP
#define OBSERVABLES_HPP

#include "dqmc.hpp"
#include <armadillo>

namespace Observables {
    double calculate_density(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_doubleOccupancy(const std::vector<GF>& greens, const Lattice& lat);
    double calculate_swavePairing(const std::vector<GF>& greens, const Lattice& lat);
    arma::mat calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_doublonTau(const std::vector<GF>& greens, const Lattice& lat);
    arma::cube calculate_currxxTau(const std::vector<GF>& greens, const Lattice& lat);
}

#endif
