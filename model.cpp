#include "dqmc.hpp"
#include "model.hpp"

namespace model {

    HubbardAttractiveU::HubbardAttractiveU(
        /* Constructor for Attractive Hubbard Model
        /    Used for initializing the model
        / 
        /     model H = - t \sum_{<i,j>} c_i^dagger c_j - \mu \sum_i n_i 
        /               - U \sum_i n_{i\uparrow} n_{i\downarrow}
        /    
        /     input parameters:
        /         lat: lattice object containing lattice information to build hopping term
        /         t: hopping parameter
        /         U: interaction strength
        /         mu: chemical potential
        /         dtau: imaginary time step
        /         nt: number of time slices
        / 
        /     output: stored in member variables
        /         expK: matrix exponential of the kinetic term
        /         fields: auxiliary fields matrix (N_tau x N_sites)
        /         alpha: coupling constant for the Hubbard-Stratonovich transformation
        */

        const Lattice& lat,  
        double t,                    
        double U,              
        double mu,               
        double dtau,        
        int nt    
    ) : 
        // Initialize model parameters
        t_(t),
        U_(U),
        mu_(mu),
        dtau_(dtau),
        n_flavor_(1),
        
        ns_(lat.size()),
        nt_(nt)
    {    
        // compute and store the necessary constant and matrices for model and simulation
        init_expK(lat);

        expV_.set_size(ns_);
        
        //compute_alpha();
        
        //init_fields();

        init_GHQfields();
    } // End of model initialization

    
    void HubbardAttractiveU::init_expK(const Lattice& lat) {
        /*  Initialize the one-body operator H_0 in matrix form 
        /   Kinetic Matrix [K] and Chemical Potential and compute its exponential
        /
        /       K = - t \sum_{<i,j>} c^†_i c_j - \mu \sum_i c^†_i c_i
        /
        /   input parameters:
        /       lat: lattice object containing lattice information to build hopping term
        /       
        /   return:
        /       expK = exp(-dtau * K)
        /
        /   The resulted expK is stored in member variable
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

    void HubbardAttractiveU::compute_alpha() {
        /* Compute the coupling constant of hubbard stratonovich transformation \alpha
        /
        /   return:
        /        \alpha = acosh(exp(abs(U) * dtau / 2))
        /
        /   the resulted coupling constant is stored in member variable
        */

        alpha_ = std::acosh(std::exp(std::abs(U_) * dtau_ / 2.0));
    }

    
    void HubbardAttractiveU::init_fields() {
        /* Initialize the auxiliary fields matrix s
        /  attractive Hubbard model hubbard stratonovich fields is diagonal in real space
        /  we store in matrix s_{t,i} the value of the field at time slice t and site i
        /  the fields are initialized to random values of ±1
        /
        /  return:
        /       s = rand(-1, +1) with size (N_tau x N_sites)
        / 
        /   The resulted fields matrix is stored in member variable
        */

        // Initialize fields matrix (N_tau x N_sites)
        fields_ = IMatrix(nt_, ns_);
        
        // Fill with random ±1 using bernoulli(0.5)
        for(int t = 0; t < nt_; ++t) {
            for(int i = 0; i < ns_; ++i) {
               fields_(t, i) = utility::random::bernoulli(0.5) ? 1 : -1;
            }
        }
    }

    void HubbardAttractiveU::init_GHQfields() {
        alpha_ = std::sqrt(0.5 * std::abs(U_) * dtau_); // Note the sqrt!
        
        gamma_.set_size(4);
        eta_.set_size(4);
        
        double s6 = std::sqrt(6.0);
        // Mapping l={-2,-1,1,2} to indices {0,1,2,3}
        // l = -2 -> index 0
        // l = -1 -> index 1
        // l = +1 -> index 2
        // l = +2 -> index 3
        gamma_(0) = 1.0 - s6 / 3.0; // l=-2
        gamma_(1) = 1.0 + s6 / 3.0; // l=-1
        gamma_(2) = 1.0 + s6 / 3.0; // l=+1
        gamma_(3) = 1.0 - s6 / 3.0; // l=+2

        eta_(0) = -std::sqrt(2.0 * (3.0 + s6)); // l=-2
        eta_(1) = -std::sqrt(2.0 * (3.0 - s6)); // l=-1
        eta_(2) =  std::sqrt(2.0 * (3.0 - s6)); // l=+1
        eta_(3) =  std::sqrt(2.0 * (3.0 + s6)); // l=+2

        // Precompute choices for updates to avoid resampling
        choices_.set_size(4, 3);
        choices_ = {{1, 2, 3},   // if current field = 0, choose new = 1,2,3
                    {0, 2, 3},   // if current field = 1, choose new = 0,2,3
                    {0, 1, 3},   // if current field = 2, choose new = 0,1,3
                    {0, 1, 2}}; // if current field = 3, choose new = 0,1,2

        fields_.set_size(nt_, ns_);
        // Fill with random {0, 1, 2, 3}
        for(int t = 0; t < nt_; ++t) {
            for(int i = 0; i < ns_; ++i) {
                fields_(t, i) = utility::random::uniform_int(0, 3);
            }
        }
    }

    /* --------------------------------------------------------------------------------------------- 
    /   Functions for calculation of B matrix
    /   These functions are model dependent so that the functions must defined in model.cpp
    / 
    /       B = expV * expK = exp(-dtau *V) * exp(-dtau * K)
    /
    /       exp(-dtau * V) -> H.S. decoupling
    /           -> expVup = exp(\alpha * s_{t,i})
    /           -> expVdn = exp(\alpha * s_{t,i})
    /
    /       *we have time-reversal symmetry for attractive decoupling, 
    /           but for generality we still kept both spin
    /
    /       B matrices nor expV are not stored as member variable. 
    /       because what's matter is propagation stack.
    /
    /       **for attractive U, we have additional term after decoupling, the term that is not 
    /           related to fermionic operator n_{i\sigma}, this is bosonic term.
    /           -> exp_boson = exp(-\alpha * s_{t,i}) = W_{boson}
    /           this will not be included in B matrix calculation, it will be calculated separately
    /           from fermionic weight probability, a bosonic weight
    /
    --------------------------------------------------------------------------------------------- */

    Matrix HubbardAttractiveU::calc_B(int t, int nfl) {
        /* 
        / Calculate B matrix at time slice t (same for both spins in attractive Hubbard)
        /       B = exp(alpha * s_{t,i}) * exp(-dtau * K)
        /
        / input parameters:
        /       t: Global time index
        /     nfl: spin index (doesn't matter since n_flavor = 1 in this model)
        /
        / return:
        /       B matrix at time t
        */
        for (int i = 0; i < ns_; i++) {
            expV_(i) = std::exp( alpha_ * eta_(fields_(t, i)) );
        }

        return linalg::diag_mul_mat(expV_, expK_); 
    }
    
    Matrix HubbardAttractiveU::calc_invB(int t, int nfl) {
        /* 
        / Calculate inverse of B matrix at time slice t
        /       B^{-1} = exp(dtau * K) * exp(-alpha * s_{t,i})
        /
        / input parameters:
        /       t: Global time index
        /     nfl: spin index (doesn't matter since it is exactly same for this model)
        /
        / return:
        /       B^{-1} matrix at time t
        */
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
        /   Rσ = 1 + (1 - G_{ii})(exp(-2 * α * s(l, i)) - 1)
        */
        return 1.0 + (1.0 - G00(i, i)) * delta; 
    }

    void HubbardAttractiveU::update_fields(int l, int i) {
        /*
        / HS field update only by flippling it
        */
        fields_(l, i) = - fields_(l, i);
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
        /
        / Args:
        /    greens: Vector of Green's functions (expected size = 1 for attractive Hubbard model)
        /    l: Time slice index to update
        /    
        / Returns:
        /    Acceptance rate (fraction of accepted updates)
        */
        assert(l >= 0 && l < nt_);
        assert(greens.size() == 1);

        int accepted_ns = 0;
        for (int i = 0; i < ns_; ++i) {
            // 1. Propose a new state
            int old_field_idx = fields_(l, i);
            int proposal_idx = utility::random::uniform_int(0, 2);
            int new_field_idx = choices_(old_field_idx, proposal_idx);

            // 2. Calculate the weight ratio R
            double delta_eta = eta_(new_field_idx) - eta_(old_field_idx);

            double bosonic_ratio = std::exp(-alpha_ * delta_eta);

            double delta = (1.0 / bosonic_ratio) - 1.0;
            
            // calculate total weight ratio at (l, i)
            double fermionic_ratio  = acceptance_ratio(greens[0].G00, delta, i);
            double gamma_ratio      = gamma_(new_field_idx) / gamma_(old_field_idx);
            double acc_ratio        = bosonic_ratio * gamma_ratio * ::pow(fermionic_ratio, 2);

            // propose an metropolis update if we accept within probability p = `acc_ratio`
            double metropolis_p = std::min(1.0, std::abs(acc_ratio));
            if (utility::random::bernoulli(metropolis_p)) {
                accepted_ns += 1;
                update_greens(greens[0].G00, delta, i);
                fields_(l, i) = new_field_idx;
            }
        }
        return static_cast<double>(accepted_ns) / ns_;
    }

} // end namespace