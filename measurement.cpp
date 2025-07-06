#include "measurement.hpp"

using GreenFunc = arma::mat;

namespace Observables {

double calculate_density(DQMC& sim) 
{
    GreenFunc gttup = sim.get_Gttup();
    GreenFunc gttdn = sim.get_Gttdn();

    double density_up = 1.0 - arma::trace(gttup) / gttup.n_rows;
    double density_dn = 1.0 - arma::trace(gttdn) / gttdn.n_rows;
    
    return density_up + density_dn;  // Total density
}

} //end of namespace

