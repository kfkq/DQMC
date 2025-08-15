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

arma::cube calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    arma::cube greenTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        GreenFunc Gttup = greens[0].Gtt[tau];
        GreenFunc Gt0up = greens[0].Gt0[tau];
        GreenFunc G0tup = greens[0].G0t[tau];
        GreenFunc Gttdn = greens[0].Gtt[tau];
        GreenFunc Gt0dn = greens[0].Gt0[tau];
        GreenFunc G0tdn = greens[0].G0t[tau];

        greenTau.slice(tau) = Gt0up + Gt0dn;
    }
    return greenTau;
}

arma::cube calculate_doublonTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    GreenFunc Gup = greens[0].G00;
    GreenFunc Gdn = greens[0].G00; 
    GreenFunc Gup_c = arma::eye(n_sites, n_sites) - Gup;
    GreenFunc Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    arma::cube doublonTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        GreenFunc Gttup = greens[0].Gtt[tau];
        GreenFunc Gt0up = greens[0].Gt0[tau];
        GreenFunc G0tup = greens[0].G0t[tau];
        GreenFunc Gttdn = greens[0].Gtt[tau];
        GreenFunc Gt0dn = greens[0].Gt0[tau];
        GreenFunc G0tdn = greens[0].G0t[tau];

        for (int i = 0; i < n_sites; ++i) {
            for (int j = 0; j < n_sites; ++j) {
                doublonTau(i,j,tau) = Gt0up(i,j) * Gt0dn(i,j);
            }
        }
    }
    return doublonTau;
}

arma::cube calculate_currxxTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.Lx();
    int Ly = lat.Ly();
    int lat_size = lat.size(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    GreenFunc G00up = greens[0].G00;
    GreenFunc G00dn = greens[0].G00; 
    GreenFunc G00up_c = arma::eye(n_sites, n_sites) - G00up;
    GreenFunc G00dn_c = arma::eye(n_sites, n_sites) - G00dn;

    arma::cube currxxTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        GreenFunc Gttup = greens[0].Gtt[tau];
        GreenFunc Gt0up = greens[0].Gt0[tau];
        GreenFunc G0tup = greens[0].G0t[tau];
        GreenFunc Gttdn = greens[0].Gtt[tau];
        GreenFunc Gt0dn = greens[0].Gt0[tau];
        GreenFunc G0tdn = greens[0].G0t[tau];

        for (int i = 0; i < lat_size; ++i) {
            int ix = lat.site_neighbors(i, {1, 0}, 0);
            double dc_term1_i = Gttup(ix, i ) + Gttdn(ix, i );
            double dc_term2_i = Gttup(i , ix) + Gttdn(i , ix);

            for (int j = 0; j < lat_size; ++j) {
                int jx = lat.site_neighbors(j, {1, 0}, 0);
                double dc_term1_j = G00up(jx, j ) + G00dn(jx, j );
                double dc_term2_j = G00up(j , jx) + G00dn(j , jx);

                double c_term1 = G0tup(jx,i ) * Gt0up(ix,j ) + G0tdn(jx,i ) * Gt0dn(ix,j );
                double c_term2 = G0tup(j ,i ) * Gt0up(ix,jx) + G0tdn(j ,i ) * Gt0dn(ix,jx);
                double c_term3 = G0tup(jx,ix) * Gt0up(i ,j ) + G0tdn(jx,ix) * Gt0dn(i ,j );
                double c_term4 = G0tup(j ,ix) * Gt0up(i ,jx) + G0tdn(j ,ix) * Gt0dn(i ,jx);

                double term_1 = dc_term1_i * dc_term1_j - c_term1;
                double term_2 = dc_term1_i * dc_term2_j - c_term2;
                double term_3 = dc_term2_i * dc_term1_j - c_term3;
                double term_4 = dc_term2_i * dc_term2_j - c_term4;

                currxxTau(i,j, tau) = - (term_1 - term_2 - term_3 + term_4);
            }
        }
    }
    return currxxTau;
}

} //end of Observables namespace
