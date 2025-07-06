/*
/   This is module for DQMC simulation
/
/   The center of this library is propagation stack and Green's function object
/   From those object we will do DQMC thermalization, sweeping, and measurement
/
/   Author: Muhammad Gaffar
*/

#ifndef DQMC_HPP
#define DQMC_HPP

#include "model.hpp"
#include "linalg.hpp"
#include <armadillo>

// Type aliases for better readability
using Matrix = arma::mat;
using Vector = arma::vec;
using GreenFunc = arma::mat; // in general should be complex, but current attractive hubbard only need real valued matrix.

class DQMC {
private:
    model::HubbardAttractiveU& model_;  
    linalg::LDRStack propagation_up_stack_;        
    linalg::LDRStack propagation_dn_stack_; 

    // Green's function matrices
    GreenFunc Gttup_;
    GreenFunc Gttdn_;
    GreenFunc Gt0up_;
    GreenFunc Gt0dn_;
    GreenFunc G0tup_;
    GreenFunc G0tdn_;

    int n_stab_;
    
    double acc_rate_;
    double avg_sgn_;
    
    // Helper functions
    void init_stacks();
    void init_greenfunctions();

    inline int global_l(int stack_idx, int loc_l) const { return stack_idx * n_stab_ + loc_l;  }
    
    inline int stack_idx(int l) const { return l / n_stab_; }
    
    inline int local_l(int l) const { return l % n_stab_; }

    void calculate_Bproduct_up_naive(Matrix& Bprod_up, int stack_idx);
    void calculate_Bproduct_dn_naive(Matrix& Bprod_dn, int stack_idx);

    void propagate_equaltime_GF_up(int l);
    void propagate_equaltime_GF_dn(int l);

    void update_stack_up_forward(Matrix& Bprod_up, int i_stack);
    void update_stack_dn_forward(Matrix& Bprod_dn, int i_stack);

    void update_equaltime_GF_up_at_stabilization_forward(int l);
    void update_equaltime_GF_dn_at_stabilization_forward(int l);

    void propagate_equaltime_GF_up_reverse(int l);
    void propagate_equaltime_GF_dn_reverse(int l);

    void update_stack_up_backward(Matrix& Bprod_up, int i_stack);
    void update_stack_dn_backward(Matrix& Bprod_dn, int i_stack);

    void update_equaltime_GF_up_at_stabilization_backward(int l);
    void update_equaltime_GF_dn_at_stabilization_backward(int l);

    double check_error(const Matrix& Gtt_temp, const Matrix& Gtt);
public:

    // Constructor
    DQMC(model::HubbardAttractiveU& model, int n_stab)
        : model_(model), n_stab_(n_stab),
          propagation_up_stack_(model.nt() / n_stab),
          propagation_dn_stack_(model.nt() / n_stab) 
    {
        // Initialize the stacks
        init_stacks();

        // Initialize Green's functions
        init_greenfunctions();

    }

    // Getters
    const linalg::LDRStack& get_propagation_up_stack() const { return propagation_up_stack_; }
    const linalg::LDRStack& get_propagation_dn_stack() const { return propagation_dn_stack_; }
    const GreenFunc& get_Gttup() const { return Gttup_; }
    const GreenFunc& get_Gttdn() const { return Gttdn_; }

    // sweep
    void sweep_0_to_beta();
    void sweep_beta_to_0();
};

#endif