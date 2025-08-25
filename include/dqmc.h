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

#include <model_base.h>
#include <stackngf.h>
#include <stablelinalg.h>
#include <update.h>
#include <utility.h> 

class DQMC {
    private:
        ModelBase& model_;

        utility::random& rng_;

        int n_stab_;
        int n_stack_;
        
        double acc_rate_; // Tracks the acceptance rate from local updates
        
        // --- Helper functions for index mapping ---
        int global_l(int stack_idx, int loc_l) const;
        int stack_idx(int l) const;
        int local_l(int l) const;

        // --- Core Propagation Logic ---
        arma::mat calc_B(arma::mat& expK, arma::mat& expV);
        arma::mat calc_B(arma::mat& expK, arma::vec& expV);
        arma::mat calc_invB(arma::mat& expK, arma::mat& expV);
        arma::mat calc_invB(arma::mat& expK, arma::vec& expV);

        arma::mat calc_Bbar(int i_stack, int flv, bool recalculate_cache = true);

        void propagate_GF_forward(GF& greens, int time_slice, int flv);
        void update_stack_forward(LDRStack& propagation_stack, const arma::mat& Bbar, int i_stack);
        void stabilize_GF_forward(GF& greens, LDRStack& propagation_stack, int time_slice);

        void propagate_GF_backward(GF& greens, int time_slice, int flv);
        void update_stack_backward(LDRStack& propagation_stack, const arma::mat& Bbar, int i_stack);
        void stabilize_GF_backward(GF& greens, LDRStack& propagation_stack, int time_slice);

        void propagate_unequalTime_GF_forward(GF& greens, int time_slice, int flv);
        void propagate_Bt0_Bbt(stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, 
                            LDRStack& propagation_stack, arma::mat& Bbar, int i_stack);
        void stabilize_unequalTime(GF& greens, stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, int time_slice);

        double check_error(const arma::mat& Gtt_temp, const arma::mat& Gtt);

        // --- Caching for B-matrices ---
        // To avoid recomputing exp(V) multiple times
        std::vector<std::vector<arma::mat>> B_cache_; // B_cache_[flavor][time_slice]
        std::vector<std::vector<arma::mat>> invB_cache_;

    public:
        // Constructor
        DQMC(const utility::parameters& params, ModelBase& model, utility::random& rng);

        // --- Getters ---
        double acceptance_rate() const { return acc_rate_; }

        // --- Initialization ---
        LDRStack init_stacks(int flavor);
        GF init_greenfunctions(LDRStack& propagation_stack);

        // --- Main Sweep Routines ---
        void sweep_0_to_beta(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
        void sweep_beta_to_0(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
        void sweep_unequal_time(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks);
};