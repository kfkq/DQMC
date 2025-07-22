#include "measurement.hpp"

using GreenFunc = arma::mat;
using Matrix = arma::mat;

namespace Observables {

double calculate_density(const std::vector<GF>&  greens, const Lattice& lat) 
{
    /*
    * calculate_density (scalar)
    * -----------------
    * Computes the average particle density per site,
    *   <n> = (1/N) Σ_i <n_i> = (1/N) Σ_i <n_{i↑} + n_{i↓}>.
    * 
    * For the attractive Hubbard model the spin-up and spin-down Green's
    * functions are identical, so we need measure only one spin component.
    * Using the relation
    *   <n_{iσ}> = 1 - <c_{iσ} c_{iσ}^†> = 1 - G_{iσ,iσ}(τ,τ),
    * the total density becomes
    *   <n> = 2 ( 1 - (1/N) Tr G↑ )
    */

    double density_tot = 0.0;
    const int n_sites = greens[0].Gtt.n_rows;

    density_tot += 1.0 - arma::trace(greens[0].Gtt) / n_sites;
    
    return 2.0 * density_tot;
}


double calculate_doubleOccupancy(const std::vector<GF>&  greens, const Lattice& lat) 
{
    /*
    * calculate_doubleOccupancy (scalar)
    * -------------------------
    * Computes the average double occupancy per site,
    *   <D> = (1/N) Σ_i <n_{i↑} n_{i↓}>,
    * where n_{iσ} is the density of spin-σ electrons on site i.
    * 
    * For the attractive Hubbard model with spin symmetry:
    *   n_{i↑} = 1 - <c_{i↑} c_{i↑}^†> = 1 - G_{i,i}
    *   n_{i↓} = n_{i↑}  (up-down symmetry)
    * Therefore the local double occupancy on site i is (1 - G↑_{i,i})².
    */

    const int n_sites = greens[0].Gtt.n_rows;
    
    double d_occ = 0.0;
    for(int i = 0; i < n_sites; i++) {
        double n_up = 1.0 - greens[0].Gtt(i,i);
        d_occ += std::pow(n_up,2);
    }

    return d_occ / n_sites;
}

double calculate_swavePairing(const std::vector<GF>&  greens, const Lattice& lat) {
    /*
    * calculate_swavePairing (scalar)
    * ----------------------
    * Computes the static s-wave pairing structure factor at zero momentum,
    *   χ_{s-wave}(q=0) = (1/N) Σ_{i,j} <Δ_i^† Δ_j>,
    * where Δ_i^† = c_{i↑}^† c_{i↓}^† is the on-site s-wave pair creation operator.
    * 
    * In terms of Green's functions and wick decomposition:
    *   <Δ_i^† Δ_j> = <c_{i↑}^† c_{i↓}^† c_{j↓} c_{j↑}>
                    = <c_{i↑}^† c_{j↑}><c_{i↓}^† c_{j↓}> - <c_{i↑}^† c_{j↓}><c_{i↓}^† c_{j↑}>
    *   With up-down factorization and symmetry, the second term vanishes and the first simplifies to
    *   <Δ_i^† Δ_j> = (δ_{ji} - G↑_{j,i}) (δ_{ji} - G↑_{j,i})
    */
    
    const int n_sites = greens[0].Gtt.n_rows;
    GreenFunc gtt_conj = arma::eye<GreenFunc>(n_sites, n_sites) - greens[0].Gtt;

    // calculate static swave pairing, q = 0
    double swave = 0.0;
    for(int i = 0; i < n_sites; i++) {
        for(int j = 0; j < n_sites; j++) {
            swave += std::pow(gtt_conj(j, i),2);
        }
    }
    return swave / n_sites;
}

Matrix calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat) {
    
    const int n_sites = greens[0].Gtt.n_rows;

    // Compute average density
    double n_avg = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        n_avg += 2.0 * (1.0 - greens[0].Gtt(i,i));
    }
    n_avg /= n_sites;

    // Initialize connected density-density correlation matrix
    arma::mat ninj_conn(n_sites, n_sites);
    ninj_conn.zeros();

    // Compute <n_i n_j> - <n_i><n_j>
    for (int i = 0; i < n_sites; ++i) {
        double n_i = 2.0 * (1.0 - greens[0].Gtt(i,i));
        for (int j = 0; j < n_sites; ++j) {
            double n_j = 2.0 * (1.0 - greens[0].Gtt(j,j));
            
            // Connected correlation: <n_i n_j> - <n_i><n_j>
            double density_product = n_i * n_j;
            double exchange_term = 2.0 * (1.0 - greens[0].Gtt(j,i)) * greens[0].Gtt(i,j);
            
            ninj_conn(i,j) = density_product + exchange_term - n_avg * n_avg;
        }
    }

    return ninj_conn;
}

} //end of Observables namespace

