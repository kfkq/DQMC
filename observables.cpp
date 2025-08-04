#include "dqmc.hpp"

using GreenFunc = arma::mat;
using Matrix = arma::mat;

namespace Observables {

double calculate_density(const std::vector<GF>&  greens, const Lattice& lat) 
{
    /*
    * calculate_density (scalar)
    * -----------------
    *   <n> = (1/N) Σ_i <n_i> = (1/N) Σ_i <n_{i↑} + n_{i↓}>.
    *   <n_{iσ}> = 1 - <c_{iσ} c_{iσ}^†> = 1 - G_{iσ,iσ},
    */

    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    double density = 0.0;
    for (int i = 0; i < n_sites; i++) {
        density += Gup_c(i,i) + Gdn_c(i,i);
    }
    density = density / n_sites;

    return density;
}


double calculate_doubleOccupancy(const std::vector<GF>&  greens, const Lattice& lat) 
{
    /*
    * calculate_doubleOccupancy (scalar)
    * -------------------------
    *   <D> = (1/N) Σ_i <n_{i↑} n_{i↓}>,
    */

    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    double d_occ = 0.0;
    for (int i = 0; i < n_sites; i++) {
        d_occ += Gup_c(i,i) * Gdn_c(i,i);
    }
    d_occ = d_occ / n_sites;

    return d_occ;
}

double calculate_swavePairing(const std::vector<GF>&  greens, const Lattice& lat) {
    /*
    * calculate_swavePairing (scalar)
    * ----------------------
    *   χ_{s-wave}(q=0) = (1/N) Σ_{i,j} <Δ_i^† Δ_j>,
    *   Δ_i^† = c_{i↑}^† c_{i↓}^†
    * 
    * wick decomposition:
    *   <Δ_i^† Δ_j> = <c_{i↑}^† c_{i↓}^† c_{j↓} c_{j↑}>
                    = <c_{i↑}^† c_{j↑}><c_{i↓}^† c_{j↓}> - <c_{i↑}^† c_{j↓}><c_{i↓}^† c_{j↑}>
    *   With up-down factorization and symmetry, the second term vanishes and the first simplifies to
    *   <Δ_i^† Δ_j> = (δ_{ji} - G↑_{j,i}) (δ_{ji} - G↑_{j,i})
    */

    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    double swave = 0.0;
    for(int i = 0; i < n_sites; i++) {
        for(int j = 0; j < n_sites; j++) {
            swave += Gup_c(j,i) * Gdn_c(j,i);
        }
    }
    swave = swave / n_sites;

    return swave;
}

Matrix calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat) {
    
    const int n_sites = greens[0].G00.n_rows;

    // Compute average density
    double n_avg = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        n_avg += 2.0 * (1.0 - greens[0].G00(i,i));
    }
    n_avg /= n_sites;

    // Initialize connected density-density correlation matrix
    arma::mat ninj_conn(n_sites, n_sites);
    ninj_conn.zeros();

    // Compute <n_i n_j> - <n_i><n_j>
    for (int i = 0; i < n_sites; ++i) {
        double n_i = 2.0 * (1.0 - greens[0].G00(i,i));
        for (int j = 0; j < n_sites; ++j) {
            double n_j = 2.0 * (1.0 - greens[0].G00(j,j));
            
            // Connected correlation: <n_i n_j> - <n_i><n_j>
            double density_product = n_i * n_j;
            double exchange_term = 2.0 * (1.0 - greens[0].G00(j,i)) * greens[0].G00(i,j);
            
            ninj_conn(i,j) = density_product + exchange_term - n_avg * n_avg;
        }
    }

    return ninj_conn;
}

std::vector<Matrix> calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    std::vector<GreenFunc> greenTau(n_tau, GreenFunc(n_sites,n_sites));
    for (int tau = 0; tau < n_tau; ++tau) {
        GreenFunc Gttup = greens[0].Gtt[tau];
        GreenFunc Gt0up = greens[0].Gt0[tau];
        GreenFunc G0tup = greens[0].G0t[tau];
        GreenFunc Gttdn = greens[0].Gtt[tau];
        GreenFunc Gt0dn = greens[0].Gt0[tau];
        GreenFunc G0tdn = greens[0].G0t[tau];

        greenTau[tau] = Gt0up + Gt0dn;
    }
    return greenTau;
}

} //end of Observables namespace

