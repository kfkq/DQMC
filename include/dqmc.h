/*
/   This module is the core DQMC linear algebra engine.
/
/   It is responsible for numerically stable propagation through imaginary
/   time by managing the LDR matrix stacks and calculating Green's functions.
/
/   It is a generic engine that operates on the ModelBase interface.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>

#include <stackngf.h>
#include <stablelinalg.h>
#include <model.h>

class DQMC {
private:
    model::HubbardAttractiveU& model_;

    int n_stab_;
    int n_stack_;

    std::vector<arma::mat> B_;
    std::vector<arma::mat> invB_;    
    
    double acc_rate_;
    double avg_sgn_;
    
    // Helper functions
    inline int global_l(int stack_idx, int loc_l) const { return stack_idx * n_stab_ + loc_l;  }
    inline int stack_idx(int l) const { return l / n_stab_; }
    inline int local_l(int l) const { return l % n_stab_; }

    void calculate_Bproduct(arma::mat& Bprod, int stack_idx, int nfl, bool recalculate_cache = true);

    void propagate_GF_forward(GF& greens, int l, int nfl);
    void update_stack_forward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_GF_forward(GF& greens, LDRStack& propagation_stack, int l);

    void propagate_GF_backward(GF& greens, int l, int nfl);
    void update_stack_backward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_GF_backward(GF& greens, LDRStack& propagation_stack, int l);

    void propagate_unequalTime_GF_forward(GF& greens, int l, int nfl);
    void propagate_Bt0_Bbt(stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, 
                            LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_unequalTime(GF& greens, stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, int l);

    double check_error(const arma::mat& Gtt_temp, const arma::mat& Gtt);
public:

    // Constructor
    DQMC(model::HubbardAttractiveU& model, int n_stab)
        : model_(model), n_stab_(n_stab),
          n_stack_(model.nt() / n_stab), acc_rate_(0.0), avg_sgn_(1.0) {}

    // Getters
    double acc_rate() { return acc_rate_; }

    // most important initialization before sweeps
    LDRStack init_stacks(int nfl);
    GF init_greenfunctions(LDRStack& propagation_stack);

    // sweep
    void sweep_0_to_beta(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
    void sweep_beta_to_0(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);

    void sweep_unequalTime(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
};