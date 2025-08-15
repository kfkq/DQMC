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

#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include "model.hpp"
#include "linalg.hpp"
#include <armadillo>

// Type aliases for better readability
using Matrix = arma::mat;
using Vector = arma::vec;
using GreenFunc = arma::mat; // in general should be complex, but current attractive hubbard only need real valued matrix.

struct GF {
    std::vector<GreenFunc> Gtt;
    std::vector<GreenFunc> Gt0;  // G(τ,0)
    std::vector<GreenFunc> G0t;  // G(0,τ)
};

class DQMC {
private:
    model::HubbardAttractiveU& model_;

    int n_stab_;
    int n_stack_;
    
    double acc_rate_;
    double avg_sgn_;
    
    // Helper functions
    inline int global_l(int stack_idx, int loc_l) const { return stack_idx * n_stab_ + loc_l;  }
    inline int stack_idx(int l) const { return l / n_stab_; }
    inline int local_l(int l) const { return l % n_stab_; }

    void calculate_Bproduct(Matrix& Bprod, int stack_idx, int nfl);

    void propagate_GF_forward(GF& greens, int l, int nfl);
    void update_stack_forward(linalg::LDRStack& propagation_stack, Matrix& Bprod, int i_stack);
    void stabilize_GF_forward(GF& greens, linalg::LDRStack& propagation_stack, int l);

    void propagate_GF_backward(GF& greens, int l, int nfl);
    void update_stack_backward(linalg::LDRStack& propagation_stack, Matrix& Bprod, int i_stack);
    void stabilize_GF_backward(GF& greens, linalg::LDRStack& propagation_stack, int l);

    void propagate_unequalTime_GF_forward(GF& greens, int l, int nfl);
    void propagate_Bt0_Bbt(linalg::LDR& Bt0, linalg::LDR& Bbt, 
                            linalg::LDRStack& propagation_stack,
                            Matrix& Bprod, int i_stack);
    void stabilize_unequalTime(GF& greens, linalg::LDR& Bt0, linalg::LDR& Bbt, int l);

    double check_error(const Matrix& Gtt_temp, const Matrix& Gtt);
public:

    // Constructor
    DQMC(model::HubbardAttractiveU& model, int n_stab)
        : model_(model), n_stab_(n_stab),
          n_stack_(model.nt() / n_stab), acc_rate_(0.0), avg_sgn_(1.0) {}

    // Getters
    double acc_rate() { return acc_rate_; }

    // most important initialization before sweeps
    linalg::LDRStack init_stacks(int nfl);
    GF init_greenfunctions(linalg::LDRStack& propagation_stack);

    // sweep
    void sweep_0_to_beta(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks);
    void sweep_beta_to_0(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks);

    void sweep_unequalTime(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks);
};

#endif