#include "dqmc.hpp"
#include "model.hpp"

namespace model {

    HubbardAttractiveU::HubbardAttractiveU(
        /* Constructor for Attractive Hubbard Model
        /    Used for initializing the model
        / 
        /     model H = - t \sum_{<i,j>} c_i^dagger c_j - \mu \sum_i n_i 
        /               - U \sum_i n_{i\uparrow} n_{i\downarrow}
        */

        const Lattice& lat,  
        double t,                    
        double U,              
        double mu,               
        double dtau,        
        int nt,
        utility::random& rng
    ) : 
        // Initialize model parameters
        t_(t),
        U_(U),
        mu_(mu),
        dtau_(dtau),
        n_flavor_(1),
        
        ns_(lat.size()),
        nt_(nt),
        rng_(rng)
    {    
        // compute and store the necessary constant and matrices for model and simulation
        init_expK(lat);
        expV_.set_size(ns_);
        init_GHQfields();

        reverse_sweep_ = false;
    } 

    
    void HubbardAttractiveU::init_expK(const Lattice& lat) {
        /*  Initialize the one-body operator H_0 in matrix form 
        /       K = - t \sum_{<i,j>} c^†_i c_j - \mu \sum_i c^†_i c_i
        */


        Matrix K(ns_,ns_);
        for (int i = 0; i < ns_; i++) {
            int ix = lat.site_neighbors(i, {1, 0}, 0);
            K(i, ix) = -t_;
            K(ix, i) = -t_;

            int iy = lat.site_neighbors(i, {0, 1}, 0);
            K(i, iy) = -t_;
            K(iy, i) = -t_;
            
            K(i, i) = -mu_;
        }
        
        expK_ = arma::expmat(-dtau_ * K);
        invexpK_ = arma::expmat(dtau_ * K);
    }

    void HubbardAttractiveU::init_GHQfields() {
        alpha_ = std::sqrt(0.5 * std::abs(U_) * dtau_); // Note the sqrt!
        
        gamma_.set_size(4);
        eta_.set_size(4);
        
        double s6 = std::sqrt(6.0);

        // Mapping l={-2,-1,1,2} to indices {0,1,2,3}
        gamma_(0) = 1.0 - s6 / 3.0;
        gamma_(1) = 1.0 + s6 / 3.0;
        gamma_(2) = 1.0 + s6 / 3.0;
        gamma_(3) = 1.0 - s6 / 3.0;

        eta_(0) = -std::sqrt(2.0 * (3.0 + s6));
        eta_(1) = -std::sqrt(2.0 * (3.0 - s6));
        eta_(2) =  std::sqrt(2.0 * (3.0 - s6));
        eta_(3) =  std::sqrt(2.0 * (3.0 + s6));

        fields_.set_size(nt_, ns_);
        // Fill with random {0, 1, 2, 3}
        for(int t = 0; t < nt_; ++t) {
            for(int i = 0; i < ns_; ++i) {
                fields_(t, i) = rng_.rand_GHQField();
            }
        }
    }

    /* --------------------------------------------------------------------------------------------- 
    /   Functions for calculation of B matrix = Bup = Bdn
    /       B = expV * expK = exp(-dtau *V) * exp(-dtau * K)
    /       exp(-dtau *V) is in the form of HS decomposition
    --------------------------------------------------------------------------------------------- */

    Matrix HubbardAttractiveU::calc_B(int t, int nfl) {
        /* 
        / Calculate B matrix at time slice t (same for both spins in attractive Hubbard)
        /       B = \gamma(t,i) exp( \sqrt(\delta\tau U) * \eta(t,i)) * exp(-dtau * K)
        */
        for (int i = 0; i < ns_; i++) {
            expV_(i) = std::exp( alpha_ * eta_(fields_(t, i)) );
        }

        return linalg::diag_mul_mat(expV_, expK_); 
    }
    
    Matrix HubbardAttractiveU::calc_invB(int t, int nfl) {
        for (int i = 0; i < ns_; i++) {
            expV_(i) = std::exp( -alpha_ * eta_(fields_(t, i)) );
        }

        return linalg::mat_mul_diag(invexpK_, expV_);
    }

    /* --------------------------------------------------------------------------------------------- 
    / Functions for MC sweeping and updates
    --------------------------------------------------------------------------------------------- */

    double HubbardAttractiveU::acceptance_ratio(GreenFunc& G00, double delta, int i) {
        /*
        / calculate the probability weight if fields updated
        /   Rσ = 1 + (1 - G_{ii}) * \Delta
        */
        return 1.0 + (1.0 - G00(i, i)) * delta; 
    }

    void HubbardAttractiveU::update_greens(GreenFunc& g00, double delta, int i) {
        /*
        / update green's function locally
        /   G'_{jk} = G_{jk} - Δ/Rσ * G_{ji} * (δ_{ik} - G_{ik})
        */
        double prefactor = delta / (1.0 + (1.0 - g00(i, i)) * delta);
        arma::vec    U = g00.col(i);
        arma::rowvec V = g00.row(i);
        V(i) = V(i) - 1.0;

        g00 += prefactor * U * V;
    }

    double HubbardAttractiveU::update_time_slice(std::vector<GF>& greens, int l) {
        /*
        / Update Green's function and fields in one time slice by going through all space indices
        */
        assert(l >= 0 && l < nt_);
        assert(greens.size() == 1);

        int accepted_ns = 0;

        int step = reverse_sweep_ ? -1 : 1;
        int start = reverse_sweep_ ? ns_ - 1 : 0;
        int end = reverse_sweep_ ? -1 : ns_;

        for (int i = start; i != end; i += step) {
            // 1. Propose a new state
            int old_field = fields_(l, i);
            int new_field;
            do {
                new_field = rng_.rand_GHQField();
            } while (new_field == old_field); // make sure new field != old field

            // 2. total acc_ratio = gamma_R * bosonic_R * fermionic_R
            double gamma_ratio = gamma_(new_field) / gamma_(old_field);

            double delta_eta = eta_(new_field) - eta_(old_field);
            double bosonic_ratio = std::exp(-1.0 * alpha_ * delta_eta);
            double delta = (1.0 / bosonic_ratio) - 1.0;
            
            double fermionic_ratio  = acceptance_ratio(greens[0].G00, delta, i);

            double acc_ratio        = bosonic_ratio * gamma_ratio * std::pow(fermionic_ratio, 2);
            double metropolis_p = std::min(1.0, std::abs(acc_ratio));
            if (rng_.bernoulli(metropolis_p)) {
                accepted_ns += 1;
                update_greens(greens[0].G00, delta, i);
                fields_(l, i) = new_field;
            }
        }

        reverse_sweep_ = !reverse_sweep_;
        return static_cast<double>(accepted_ns) / ns_;
    }

} // end model namespace