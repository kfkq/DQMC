#include "measurement.hpp"

using GreenFunc = arma::mat;

namespace Observables {

double calculate_density(const std::vector<GF>&  greens) 
{

    double density_tot = 0.0;
    const int n_flavor = greens.size();
    const int ns = greens[0].Gtt.n_rows;

    for (int nfl = 0; nfl < n_flavor; nfl++) {
        density_tot += 1.0 - arma::trace(greens[nfl].Gtt) / ns;
    }    
    
    return 2.0 * density_tot;
}

double calculate_doubleOccupancy(const std::vector<GF>&  greens) 
{
    const int ns = greens[0].Gtt.n_rows;
    
    double d_occ = 0.0;
    for(int i = 0; i < ns; i++) {
        double n_up = 1.0 - greens[0].Gtt(i,i);
        double n_dn = n_up; 

        d_occ += n_up * n_dn;
    }
    d_occ = (d_occ/ns);

    return d_occ;
}

} //end of namespace

