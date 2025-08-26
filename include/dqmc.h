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

#include <utility.h>
#include <stackngf.h>
#include <stablelinalg.h>
#include <model.h>

class DQMC {
private:
    AttractiveHubbard& model_;

    bool isUnequalTime_;

    int n_stab_;
    int n_stack_;

     // --- Caching for B-matrices ---
    // To avoid recomputing B when no update field update.
    std::vector<std::vector<arma::mat>> B_; // B_[flavor][time_slice]
    std::vector<std::vector<arma::mat>> invB_;    
    
    double acc_rate_;
    double avg_sgn_;

    // for precision tracking
    double max_precision_error_;
    double total_precision_error_;
    double num_cumulated_precision_error_;
    
    // Helper functions
    inline int global_l(int stack_idx, int loc_l) const { return stack_idx * n_stab_ + loc_l;  }
    inline int stack_idx(int l) const { return l / n_stab_; }
    inline int local_l(int l) const { return l % n_stab_; }

    // --- Core Propagation Logic ---
    arma::mat calculate_B(arma::mat& expK, arma::mat& expV);
    arma::mat calculate_B(arma::mat& expK, arma::vec& expV);
    arma::mat calculate_invB(arma::mat& expK, arma::mat& expV);
    arma::mat calculate_invB(arma::mat& expK, arma::vec& expV);

    arma::mat calculate_Bbar(int stack_idx, int flv, bool recalculate_cache = true);

    void propagate_GF_forward(std::vector<GF>& greens, int l);
    void update_stack_forward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_GF_forward(GF& greens, LDRStack& propagation_stack, int l);

    void propagate_GF_backward(std::vector<GF>& greens, int l);
    void update_stack_backward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_GF_backward(GF& greens, LDRStack& propagation_stack, int l);

    void propagate_unequalTime_GF_forward(std::vector<GF>& greens, int l);
    void propagate_Bt0_Bbt(stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, 
                            LDRStack& propagation_stack, arma::mat& Bprod, int i_stack);
    void stabilize_unequalTime(GF& greens, stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, int l);

    double check_error(const arma::mat& Gtt_temp, const arma::mat& Gtt);
public:

    // Constructor
    DQMC(const utility::parameters& params, AttractiveHubbard& model);

    // Getters
    double acc_rate() { return acc_rate_; }
    double max_err() { return max_precision_error_; }
    double mean_err() { return total_precision_error_ / num_cumulated_precision_error_; }

    // most important initialization before sweeps
    LDRStack init_stacks(int flv);
    GF init_greenfunctions(LDRStack& propagation_stack);

    // sweep
    void sweep_0_to_beta(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
    void sweep_beta_to_0(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);

    void sweep_unequalTime(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
};