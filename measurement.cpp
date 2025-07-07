#include "measurement.hpp"

using GreenFunc = arma::mat;

namespace Observables {

double calculate_density(std::vector<GF>&  greens) 
{

    double density_tot = 0.0;
    const int n_flavor = greens.size();
    const int ns = greens[0].Gtt.n_rows;

    for (int nfl = 0; nfl < n_flavor; nfl++) {
        density_tot += 1.0 - arma::trace(greens[nfl].Gtt) / ns;
    }    
    
    return density_tot;
}

} //end of namespace

