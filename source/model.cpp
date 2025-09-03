#include <model.h>

AttractiveHubbard::AttractiveHubbard(
    /* Constructor for Attractive Hubbard Model
    / 
    /     H = - t \sum_{<i,j>} c_i^dagger c_j - \mu \sum_i n_i 
    /         - U \sum_i n_{i\uparrow} n_{i\downarrow}
    */

    const utility::parameters& params,
    const Lattice& lat,
    utility::random& rng,
    double replica_beta
) : rng_(rng)
{    
    //
    t_  = params.getDouble("hubbard", "t");
    mu_ = params.getDouble("hubbard", "mu");
    ns_ = lat.n_cells();
    nt_ = params.getDouble("simulation", "nt");

    const double U  = params.getDouble("hubbard", "U");
    const double dtau = replica_beta / nt_;    

    fields_ = GHQField(nt_, ns_, rng);

    g_ = sqrt(0.5 * std::abs(U) * dtau);
    alpha_ = -1.0;
    
    // compute and store the necessary constant and matrices for model and simulation
    arma::mat K = build_K_matrix(lat);
    expK_ = arma::expmat(-dtau * K);
    invexpK_ = arma::expmat(dtau * K);
    expKhalf_ = arma::expmat(-0.5 * dtau * K);
    invexpKhalf_ = arma::expmat(0.5 * dtau * K);
} 

// Helper function to build the kinetic matrix K for a square lattice
arma::mat AttractiveHubbard::build_K_matrix(const Lattice& lat) {
    int n_sites = lat.n_sites();
    arma::mat K(n_sites, n_sites, arma::fill::zeros);

    for (int i = 0; i < n_sites; ++i) {
        K(i, i) = -mu_;
        
        // single orbital for neighbor finding
        int orb = 0; 
        
        // Hopping in +x direction
        int neighbor_x = lat.site_neighbors(i, {1, 0}, orb);
        K(i, neighbor_x) = -t_;
        K(neighbor_x, i) = -t_;
        
        // Hopping in +y direction
        int neighbor_y = lat.site_neighbors(i, {0, 1}, orb);
        K(i, neighbor_y) = -t_;
        K(neighbor_y, i) = -t_;
    }
    return K;
}

arma::vec AttractiveHubbard::expV(int l, int flv) {
    // For the attractive model, expV is the same for both flavors.
    const int nv = fields_.nv();
    arma::vec expV(nv);

    for (int i = 0; i < nv; ++i) {
        int f = fields_.single_val(l, i);
        expV(i) = std::exp(g_ * fields_.eta(f));
    }
    return expV;
}

arma::vec AttractiveHubbard::invexpV(int l, int flv) {
    // For the attractive model, expV is the same for both flavors.
    const int nv = fields_.nv();
    arma::vec invexpV(nv);

    for (int i = 0; i < nv; ++i) {
        int f = fields_.single_val(l, i);
        invexpV(i) = std::exp(-g_ * fields_.eta(f));
    }
    return invexpV;
}

/* --------------------------------------------------------------------------------------------- 
/ model updates' functions
--------------------------------------------------------------------------------------------- */

double AttractiveHubbard::det_ratio(arma::mat& G00, double delta, int i) {
    /*
    / fermion det ratio
    / this model has equal factorization, so just power to 2
    */
    double detR_flv = 1.0 + (1.0 - G00(i, i)) * delta; 
    return std::pow(detR_flv, 2);
}

std::pair<double, double> AttractiveHubbard::bosonic_ratio(int new_field, int old_field) {
    /*
        bosonic term 
    */
   double d_eta = fields_.eta(new_field) - fields_.eta(old_field);
   double bosonic_ratio = std::exp(alpha_ * g_ * d_eta);
   double delta = (1.0 / bosonic_ratio) - 1.0;
   return {bosonic_ratio, delta};
}

std::pair<double, double> AttractiveHubbard::local_update_ratio(std::vector<GF>& GF, int l, int field_idx, int new_field) {
    /*
        total local update ratio
    */

    int flv = 0;

    int old_field = fields_.single_val(l, field_idx);
    double gammaR = fields_.gamma(new_field) / fields_.gamma(old_field);
    auto [bosonR, delta] = bosonic_ratio(new_field, old_field);
    double detR = det_ratio(GF[flv].Gtt[l+1], delta, field_idx);

    return {gammaR * bosonR * detR, delta};
}

void AttractiveHubbard::update_greens_local(std::vector<GF>& GF, double delta, int l, int i) {
    /*
    / update green's function locally using shermann morrison
    /   G'_{jk} = G_{jk} - Δ/Rσ * G_{ji} * (δ_{ik} - G_{ik})
    */

    int flv = 0;

    double prefactor = delta / (1.0 + (1.0 - GF[flv].Gtt[l+1](i, i)) * delta);
    arma::vec    U = GF[flv].Gtt[l+1].col(i);
    arma::rowvec V = GF[flv].Gtt[l+1].row(i);
    V(i) = V(i) - 1.0;

    GF[flv].Gtt[l+1] += prefactor * U * V;
}

double AttractiveHubbard::global_action(const std::vector<GF>& greens) {
    /*
        S = -log(Weight)
    */
    int flv = 0;
    double S = -2.0 * greens[flv].log_det_M;

    double logbosonR = 0.0;
    double loggammaR = 0.0;

    arma::imat fs = fields_.fields();
    for (int i = 0; i < fs.n_elem; ++i) {
        int f = fs(i);
        logbosonR += alpha_ * g_ * fields_.eta(f);
        loggammaR += std::log(fields_.gamma(f));
    }

    S -= logbosonR + loggammaR;
    return S;
}

/* --------------------------------------------------------------------------------------------- 
/ List of observables
--------------------------------------------------------------------------------------------- */

namespace Observables {

double calculate_density(const std::vector<GF>&  greens, const Lattice& lat) 
{
    /*
    * calculate_density (scalar)
    * -----------------
    *   <n> = (1/N) Σ_i <n_i> = (1/N) Σ_i <n_{i↑} + n_{i↓}>.
    *   <n_{iσ}> = 1 - <c_{iσ} c_{iσ}^†> = 1 - G_{iσ,iσ},
    */

    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();

    arma::mat Gup = greens[0].Gtt[0];
    arma::mat Gdn = greens[0].Gtt[0]; 
    arma::mat Gup_c = arma::eye(n_sites, n_sites) - Gup;
    arma::mat Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

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

    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();

    arma::mat Gup = greens[0].Gtt[0];
    arma::mat Gdn = greens[0].Gtt[0]; 
    arma::mat Gup_c = arma::eye(n_sites, n_sites) - Gup;
    arma::mat Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

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

    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();

    arma::mat Gup = greens[0].Gtt[0];
    arma::mat Gdn = greens[0].Gtt[0]; 
    arma::mat Gup_c = arma::eye(n_sites, n_sites) - Gup;
    arma::mat Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    double swave = 0.0;
    for(int i = 0; i < n_sites; i++) {
        for(int j = 0; j < n_sites; j++) {
            swave += Gup_c(j,i) * Gdn_c(j,i);
        }
    }
    swave = swave / n_sites;

    return swave;
}

arma::mat calculate_densityCorr(const std::vector<GF>& greens, const Lattice& lat) {
    
    const int n_sites = greens[0].Gtt[0].n_rows;

    // Compute average density
    double n_avg = 0.0;
    for (int i = 0; i < n_sites; ++i) {
        n_avg += 2.0 * (1.0 - greens[0].Gtt[0](i,i));
    }
    n_avg /= n_sites;

    // Initialize connected density-density correlation matrix
    arma::mat ninj_conn(n_sites, n_sites);
    ninj_conn.zeros();

    // Compute <n_i n_j> - <n_i><n_j>
    for (int i = 0; i < n_sites; ++i) {
        double n_i = 2.0 * (1.0 - greens[0].Gtt[0](i,i));
        for (int j = 0; j < n_sites; ++j) {
            double n_j = 2.0 * (1.0 - greens[0].Gtt[0](j,j));
            
            // Connected correlation: <n_i n_j> - <n_i><n_j>
            double density_product = n_i * n_j;
            double exchange_term = 2.0 * (1.0 - greens[0].Gtt[0](j,i)) * greens[0].Gtt[0](i,j);
            
            ninj_conn(i,j) = density_product + exchange_term - n_avg * n_avg;
        }
    }

    return ninj_conn;
}

arma::cube calculate_greenTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    arma::mat Gup = greens[0].Gtt[0];
    arma::mat Gdn = greens[0].Gtt[0]; 
    arma::mat Gup_c = arma::eye(n_sites, n_sites) - Gup;
    arma::mat Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    arma::cube greenTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        arma::mat Gttup = greens[0].Gtt[tau];
        arma::mat Gt0up = greens[0].Gt0[tau];
        arma::mat G0tup = greens[0].G0t[tau];
        arma::mat Gttdn = greens[0].Gtt[tau];
        arma::mat Gt0dn = greens[0].Gt0[tau];
        arma::mat G0tdn = greens[0].G0t[tau];

        greenTau.slice(tau) = Gt0up + Gt0dn;
    }
    return greenTau;
}

arma::cube calculate_doublonTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    arma::mat Gup = greens[0].Gtt[0];
    arma::mat Gdn = greens[0].Gtt[0]; 
    arma::mat Gup_c = arma::eye(n_sites, n_sites) - Gup;
    arma::mat Gdn_c = arma::eye(n_sites, n_sites) - Gdn;

    arma::cube doublonTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        arma::mat Gttup = greens[0].Gtt[tau];
        arma::mat Gt0up = greens[0].Gt0[tau];
        arma::mat G0tup = greens[0].G0t[tau];
        arma::mat Gttdn = greens[0].Gtt[tau];
        arma::mat Gt0dn = greens[0].Gt0[tau];
        arma::mat G0tdn = greens[0].G0t[tau];

        for (int i = 0; i < n_sites; ++i) {
            for (int j = 0; j < n_sites; ++j) {
                doublonTau(i,j,tau) = Gt0up(i,j) * Gt0dn(i,j);
            }
        }
    }
    return doublonTau;
}

arma::cube calculate_currxxTau(const std::vector<GF>& greens, const Lattice& lat) {
    int Lx = lat.L1();
    int Ly = lat.L2();
    int lat_size = lat.n_cells(); 
    int n_sites  = lat.n_sites();
    int n_tau = greens[0].Gtt.size();

    arma::mat G00up = greens[0].Gtt[0];
    arma::mat G00dn = greens[0].Gtt[0]; 
    arma::mat G00up_c = arma::eye(n_sites, n_sites) - G00up;
    arma::mat G00dn_c = arma::eye(n_sites, n_sites) - G00dn;

    arma::cube currxxTau(n_sites, n_sites, n_tau);
    for (int tau = 0; tau < n_tau; ++tau) {
        arma::mat Gttup = greens[0].Gtt[tau];
        arma::mat Gt0up = greens[0].Gt0[tau];
        arma::mat G0tup = greens[0].G0t[tau];
        arma::mat Gttdn = greens[0].Gtt[tau];
        arma::mat Gt0dn = greens[0].Gt0[tau];
        arma::mat G0tdn = greens[0].G0t[tau];

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