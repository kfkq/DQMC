#include <model/ahubbard.h>
#include <utility.h> 
#include <lattice.h>
#include <cmath>

// Helper function to build the kinetic matrix K for a square lattice
arma::mat build_K_matrix(const Lattice& lat, double t, double mu) {
    int n_sites = lat.n_sites();
    arma::mat K(n_sites, n_sites, arma::fill::zeros);

    for (int i = 0; i < n_sites; ++i) {
        K(i, i) = -mu;
        
        // single orbital for neighbor finding
        int orb = 0; 
        
        // Hopping in +x direction
        int neighbor_x = lat.site_to_neighbor(i, {1, 0}, orb);
        K(i, neighbor_x) = -t;
        K(neighbor_x, i) = -t;
        
        // Hopping in +y direction
        int neighbor_y = lat.site_to_neighbor(i, {0, 1}, orb);
        K(i, neighbor_y) = -t;
        K(neighbor_y, i) = -t;
    }
    return K;
}

// --- Constructor ---
HubbardAttractiveU::HubbardAttractiveU(
    const utility::parameters& params, 
    const Lattice& lat, utility::random& rng)
    : t_(params.getDouble("hubbard", "t")),
      mu_(params.getDouble("hubbard", "mu")),
      field_(params.getInt("simulation", "nt"), lat.n_sites(), rng)
{
    // --- Pre-calculate values needed for the member initializer list ---
    const double U = params.getDouble("hubbard", "U");
    const double beta = params.getDouble("simulation", "beta");
    const int nt = params.getInt("simulation", "nt");
    const double dtau = beta / nt;

    // g = sqrt(U/2 * dtau)
    g_ = std::sqrt(0.5 * std::abs(U) * dtau);
    alpha_ = -1.0;

    // --- Finish setup in the constructor body ---
    arma::mat K = build_K_matrix(lat, t_, mu_);
    expK_ = arma::expmat(-dtau * K);
    invexpK_ = arma::expmat(dtau * K);
}


// --- Implementations of the ModelBase virtual functions ---

int HubbardAttractiveU::n_size() const {
    return expK_.n_rows;
}

int HubbardAttractiveU::nv() const {
    return field_.get_nv();
}

int HubbardAttractiveU::n_timesteps() const {
    return field_.get_fields().n_rows;
}

int HubbardAttractiveU::n_flavors() const {
    return 1;
}

void HubbardAttractiveU::set_fields(const arma::imat& new_fields) {
    field_.set_fields(new_fields);
}

void HubbardAttractiveU::set_field_value(int time_slice, int field_idx, int new_field) {
    field_.set_field_value(time_slice, field_idx, new_field);
}

int HubbardAttractiveU::propose_field(int time_slice, int field_idx) {
    int old_field = field_.get_field_value(time_slice, field_idx);
    return field_.propose_new_field(old_field);
}

const arma::mat& HubbardAttractiveU::get_expK() const {
    return expK_;
}

const arma::mat& HubbardAttractiveU::get_invexpK() const {
    return invexpK_;
}

arma::vec HubbardAttractiveU::get_expV(int time_slice, int flavor) {
    // For the attractive model, expV is the same for both flavors.
    const int n_v = field_.get_fields().n_cols;
    arma::vec expV(n_v);

    for (int i = 0; i < n_v; ++i) {
        // The coupling g = sqrt(U*dtau) is already stored in the field object
        int f = field_.get_field_value(time_slice, i);
        expV(i) = std::exp(g_ * field_.get_eta(f));
    }
    return expV;
}

arma::vec HubbardAttractiveU::get_invexpV(int time_slice, int flavor) {
    const int n_v = field_.get_fields().n_cols;
    arma::vec invexpV(n_v);

    for (int i = 0; i < n_v; ++i) {
        int f = field_.get_field_value(time_slice, i);
        invexpV(i) = std::exp(-g_ * field_.get_eta(f));
    }
    return invexpV;
}

double HubbardAttractiveU::det_ratio(arma::mat& gtt, double delta, int site_idx) {
    return 1.0 + (1.0 - gtt(site_idx, site_idx)) * delta;
}

double HubbardAttractiveU::local_update_ratio(
    std::vector<GF>& GF,
    int time_slice,
    int site_idx,
    int new_field_value)
{
    int old_field_value = field_.get_field_value(time_slice, site_idx);

    // --- Calculate Ratios ---
    double gamma_ratio = field_.get_gamma(new_field_value) / field_.get_gamma(old_field_value);
    
    double delta_eta = field_.get_eta(new_field_value) - field_.get_eta(old_field_value);
    double bosonic_ratio = std::exp(alpha_ * g_ * delta_eta);
    delta_ = (1.0 / bosonic_ratio) - 1.0;
    
    double fermionic_ratio = det_ratio(GF[0].Gtt[time_slice + 1], delta_, site_idx);
    
    return bosonic_ratio * gamma_ratio * std::pow(fermionic_ratio, 2);
}

void HubbardAttractiveU::update_greens_local(std::vector<GF>& GF, int time_slice, int site_idx) {
    /*
        using sherman-morrison formula
    */

    double prefactor = delta_ / (1.0 + (1.0 - GF[0].Gtt[time_slice + 1](site_idx, site_idx)) * delta_);
    arma::vec    U = GF[0].Gtt[time_slice + 1].col(site_idx);
    arma::rowvec V = GF[0].Gtt[time_slice + 1].row(site_idx);
    V(site_idx) = V(site_idx) - 1.0;

    // rank-1 update
    GF[0].Gtt[time_slice + 1] += prefactor * U * V;
}

double HubbardAttractiveU::calculate_global_action(const std::vector<GF>& GF) {

    // log det ratio of fermion. times two for two spin flavor degeneracy
    double s_fermionic = 2.0 * GF[0].log_det_M;

    double s_bosonic = 0.0;
    double s_gamma = 0.0;

    int nv = field_.get_nv();
    int nt = n_timesteps();
    for (int l = 0; l < nt; ++l) {
        for (int i = 0; i < nv; ++i) {
            int f = field_.get_field_value(l,i);
            s_bosonic += alpha_ * g_ * field_.get_eta(f);
            s_gamma   += std::log(field_.get_gamma(f));
        }
    }

    return s_gamma + s_bosonic + s_fermionic;

}