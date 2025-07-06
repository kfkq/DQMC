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

        const lattice::Lattice& lat,  
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
        
        ns_(lat.N_sites),
        nt_(nt)
    {    
        // compute and store the necessary constant and matrices for model and simulation
        init_expK(lat);
        
        compute_alpha();
        
        init_fields();
    } // End of model initialization

    
    void HubbardAttractiveU::init_expK(const lattice::Lattice& lat) {
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

        Matrix K = -t_ * lattice::nn_matrix(lat);
        
        K.diag() -= mu_;
        
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

    Matrix HubbardAttractiveU::calc_B(int t) {
        /* 
        / Calculate B matrix at time slice t (same for both spins in attractive Hubbard)
        /       B = exp(alpha * s_{t,i}) * exp(-dtau * K)
        /
        / input parameters:
        /       t: Global time index
        /
        / return:
        /       B matrix at time t
        */
        Vector expV(ns_);
        for (int i = 0; i < ns_; i++) {
            expV(i) = std::exp(alpha_ * fields_(t, i));
        }

        return linalg::diag_mul_mat(expV, expK_); 
    }

    Matrix HubbardAttractiveU::calc_Bup(int t) {
        return calc_B(t);
    }

    Matrix HubbardAttractiveU::calc_Bdn(int t) {
        return calc_B(t);
    }
    
    Matrix HubbardAttractiveU::calc_invB(int t) {
        /* 
        / Calculate inverse of B matrix at time slice t
        /       B^{-1} = exp(dtau * K) * exp(-alpha * s_{t,i})
        /
        / input parameters:
        /       t: Global time index
        /
        / return:
        /       B^{-1} matrix at time t
        */
        Vector expV(ns_);
        for (int i = 0; i < ns_; i++) {
            expV(i) = std::exp(-alpha_ * fields_(t, i));
        }

        return linalg::mat_mul_diag(invexpK_, expV);
    }

    Matrix HubbardAttractiveU::calc_invBup(int t) {
        return calc_invB(t);
    }
    
    Matrix HubbardAttractiveU::calc_invBdn(int t) {
        return calc_invB(t);
    }

    /* --------------------------------------------------------------------------------------------- 
    / Functions for MC sweeping and updates
    --------------------------------------------------------------------------------------------- */

    double HubbardAttractiveU::acceptance_ratio(GreenFunc& Gtt, double delta, int i) {
        /*
        / calculate the probability weight if fields updated
        /   Rσ = 1 + (1 - G_{ii})(exp(-2 * α * s(l, i)) - 1)
        */
        return 1.0 + (1.0 - Gtt(i, i)) * delta; 
    }

    void HubbardAttractiveU::update_fields(int l, int i) {
        /*
        / HS field update only by flippling it
        */
        fields_(l, i) = - fields_(l, i);
    }

    void HubbardAttractiveU::update_greens(GreenFunc& gtt, double delta, int i) {
        /*
        / update green's function locally
        /   G'_{jk} = G_{jk} - Δ/Rσ * G_{ji} * (δ_{ik} - G_{ik})
        */
        double prefactor = delta / (1.0 + (1.0 - gtt(i, i)) * delta);
        arma::vec    U = gtt.col(i);
        arma::rowvec V = gtt.row(i);
        V(i) = V(i) - 1.0;

        gtt += prefactor * U * V;
    }

    double HubbardAttractiveU::update_time_slice(GreenFunc& Gttup, GreenFunc& Gttdn, int l) {
        /*
        / update Green's function and fields in one time slice by go through all space index
        */
        assert(l >= 0 && l < nt_);

        int accepted_ns = 0;
        for (int i = 0; i < ns_; ++i) {
            // calculate Δ(l,i) = exp(-2 * a * s(l, i)) - 1.0
            double delta = std::exp(-2.0 * alpha_ * fields_(l, i)) - 1.0;
            
            // calculate acceptance ratio at (l, i)
            double acc_ratio_up = acceptance_ratio(Gttup, delta, i);
            double acc_ratio_dn = acceptance_ratio(Gttdn, delta, i);

            double bosonic_ratio = 1.0 / (delta + 1.0); // non zero, since HS decomposition gives (n_up + n_dn - 1)^2

            double acc_ratio    = bosonic_ratio * acc_ratio_up * acc_ratio_dn;

            // propose an metropolis update if we accept within probability p = `acc_ratio`
            double metropolis_p = std::min(1.0, std::abs(acc_ratio));
            if (utility::random::bernoulli(metropolis_p)) {
                accepted_ns += 1;
                update_greens(Gttup, delta, i);
                update_greens(Gttdn, delta, i);
                update_fields(l, i);
            }
        }
        return accepted_ns / ns_;
    }

} // end namespace