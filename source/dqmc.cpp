#include <dqmc.h>


/* --------------------------------------------------------------------------------
/
/   Initialization of stacks and Green's function
/
-------------------------------------------------------------------------------- */
LDRStack DQMC::init_stacks(int flv) {
    LDRStack propagation_stack(n_stack_);

    B_.resize(model_.nt(), arma::mat(model_.ns(), model_.ns()));
    invB_.resize(model_.nt(), arma::mat(model_.ns(), model_.ns()));

    arma::mat Bprod(model_.ns(), model_.ns());

    for (int i_stack = n_stack_ - 1; i_stack >= 0; i_stack--) {
        Bprod.eye();

        calculate_Bproduct(Bprod, i_stack, flv);

        stablelinalg::LDR Bbar = stablelinalg::to_LDR(Bprod);
        
        if (i_stack == n_stack_ - 1) {
            propagation_stack[i_stack] = Bbar;
        } else {
            propagation_stack[i_stack] = stablelinalg::ldr_mul_ldr(propagation_stack[i_stack + 1], Bbar);
        }
    }

    return propagation_stack;
}

GF DQMC::init_greenfunctions(LDRStack& propagation_stack) {
    int nt = model_.nt();
    
    GF greens;
    greens.Gtt.resize(nt+1);
    greens.Gt0.resize(nt+1);
    greens.G0t.resize(nt+1);

    greens.Gtt[0] = stablelinalg::inv_I_plus_ldr(propagation_stack[0], greens.log_det_M);

    return greens;
}

void DQMC::calculate_Bproduct(arma::mat& Bprod, int i_stack, int flv, bool recalculate_cache) {
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);
        if (recalculate_cache) {
            B_[l] = model_.calc_B(l, flv);
        }
        Bprod = B_[l] * Bprod;
    }
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Greens Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_forward(GF& greens, int l, int flv) {
    /*
    /   Gtt = B_l * Gtt * B_l^{-1}
    */

    B_[l] = model_.calc_B(l, flv);
    invB_[l] = model_.calc_invB(l, flv);

    greens.Gtt[l+1] = B_[l] * greens.Gtt[l] * invB_[l];
}

void DQMC::update_stack_forward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack) {
    /*
    / stack[0] = B(τ', 0)
    / stack[1] = B(τ'', 0) = B(τ'', τ')B(τ', 0)
    / and so on
    */
    
    if (i_stack == 0) {
        propagation_stack[i_stack] = stablelinalg::to_LDR(Bprod);
    } else {
        propagation_stack[i_stack] = stablelinalg::mat_mul_ldr(Bprod, propagation_stack[i_stack - 1]);
    }
}

void DQMC::stabilize_GF_forward(GF& greens, LDRStack& propagation_stack, int l) {
    int i_stack = stack_idx(l);
    
    if (l == model_.nt() - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l+1] = stablelinalg::inv_I_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // G(τ,τ) =  [I + B(τ,0)B(β,τ)]^{-1}
        greens.Gtt[l+1] = stablelinalg::inv_I_plus_ldr_mul_ldr(
                    propagation_stack[i_stack], 
                    propagation_stack[i_stack + 1]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Backward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_backward(GF& greens, int l, int flv) {
    /*
    /       Gtt = B_l^{-1} * Gttup * B_l
    */

    B_[l] = model_.calc_B(l, flv);
    invB_[l] = model_.calc_invB(l, flv);

    greens.Gtt[l] = invB_[l] * greens.Gtt[l+1] * B_[l];
}

void DQMC::update_stack_backward(LDRStack& propagation_stack, arma::mat& Bprod, int i_stack) {
    /*
    / stack[end  ] = B(β, τ')
    / stack[end-1] = B(β, τ'') = B(β, τ') * B(τ', τi'')
    / and so on
    */

    if (i_stack == n_stack_-1) {
        propagation_stack[i_stack] = stablelinalg::to_LDR(Bprod);
    } else {
        propagation_stack[i_stack] = stablelinalg::ldr_mul_mat(propagation_stack[i_stack + 1], Bprod);
    }
}

void DQMC::stabilize_GF_backward(GF& greens, LDRStack& propagation_stack, int l) {
    int i_stack = stack_idx(l);
    
    if (l == 0) { // at beginning of propagation
        //  G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l] = stablelinalg::inv_I_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // G(τ,τ) =  [I + B(τ,0)B(β,τ)]^{-1}
        greens.Gtt[l] = stablelinalg::inv_I_plus_ldr_mul_ldr(
                        propagation_stack[i_stack-1], 
                        propagation_stack[i_stack]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Unequal Time propagation
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_unequalTime_GF_forward(GF& greens, int l, int flv) {
    /*
    /       Gtt = B_l * Gtt * B_l^{-1}
    /       Gt0 = B_l * Gt0
    /       G0t = G0t * B_l^{-1}
    */

    if (l == 0) { // at τ = 0
        // G(τ,0) = G(0,0)
        greens.Gt0[0] = greens.Gtt[0];
        // G(0,τ) = - [ I - G(0,0) ]
        greens.G0t[0] = greens.Gtt[0] - arma::eye(model_.ns(),model_.ns());       
    }
    greens.Gtt[l+1] = B_[l] * greens.Gtt[l] * invB_[l];
    greens.Gt0[l+1] = B_[l] * greens.Gt0[l];
    greens.G0t[l+1] = greens.G0t[l] * invB_[l];
}

void DQMC::propagate_Bt0_Bbt(stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, 
                            LDRStack& propagation_stack, arma::mat& Bprod, int i_stack) 
{


    if (i_stack == 0) {
        Bt0 = stablelinalg::to_LDR(Bprod);
    } else {
        Bt0 = stablelinalg::mat_mul_ldr(Bprod, Bt0);
    }

    if (i_stack < propagation_stack.size() - 1) {  
        Bbt = propagation_stack[i_stack + 1];
    }
}

void DQMC::stabilize_unequalTime(GF& greens, stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, int l) {
    if (l == model_.nt() - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l+1] = stablelinalg::inv_I_plus_ldr(Bt0, greens.log_det_M);

        //  G(β,0) = I - G(0,0)
        greens.Gt0[l+1] = stablelinalg::I_minus_mat(greens.Gtt[l+1]);

        //  G(0,β) = -G(0,0)
        greens.G0t[l+1] = -greens.Gtt[l+1];
    } 
    else {
        greens.Gtt[l+1] = stablelinalg::inv_I_plus_ldr_mul_ldr(Bt0, Bbt);
        greens.Gt0[l+1] = stablelinalg::inv_invldr_plus_ldr(Bt0, Bbt);
        greens.G0t[l+1] = -stablelinalg::inv_invldr_plus_ldr(Bbt, Bt0);
    }
}

/* --------------------------------------------------------------------------------
/
/   Utilities
/
-------------------------------------------------------------------------------- */


double DQMC::check_error(const arma::mat& Gtt_temp, const arma::mat& Gtt) {
    /*
    / Check the error between the two Green's function matrices
    /   return: max|Gtt_temp - Gtt| (element-wise maximum absolute difference)
    */
    return arma::max(arma::max(arma::abs(Gtt_temp - Gtt)));
}

/* --------------------------------------------------------------------------------
/
/   Main routines
/
-------------------------------------------------------------------------------- */

void DQMC::sweep_0_to_beta(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks) {
    /*
    /   do monte carlo sweep Δτ -> 2Δτ -> ... -> nt * Δτ
    /   for each time slice t, we propagate our equaltime Green's functions
    /   and update the propagation stack.
    /   when we reach our local_time loc_t = n_stab-1, we do stabilization
    */
    
    int loc_l;
    int i_stack;
    double acc_l;

    int n_flavor = static_cast<int>(greens.size());
    const int nt = model_.nt();

    std::vector<arma::mat> Bprods(n_flavor, arma::mat(model_.ns(), model_.ns()));

    // Loop over time slices forward
    // We propagate our GF, from 0 to β
    // the iteration then start processing from 0 + Δτ, 0 + 2Δτ, ..., 0 + nt*Δτ (β)
    for (int l = 0; l < nt; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_GF_forward(greens[flv], l, flv);
        }

        // update HS field over space given time slice
        acc_l = model_.update_time_slice(greens, l);
        acc_rate_ += acc_l / nt;

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {
                Bprods[flv].eye();

                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l+1];

                // Calculate Bprod
                calculate_Bproduct(Bprods[flv], i_stack, flv);

                // Update stacks
                update_stack_forward(propagation_stacks[flv], Bprods[flv], i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_GF_forward(greens[flv], propagation_stacks[flv], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[l+1]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF forward precision reach beyond threshold > 1e-6. Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void DQMC::sweep_beta_to_0(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks) {
    /*
    /   Do monte carlo sweep t = β-Δτ, ..., 0
    /   For each time slice t, we propagate our Green's functions
    /   then update fields over space
    /   when we reach our local_time loc_t = 0, we do stabilization
    /
    */
    
    int loc_l;
    int i_stack;
    double acc_l;

    int n_flavor = static_cast<int>(greens.size());
    const int nt = model_.nt();

    std::vector<arma::mat> Bprods(n_flavor, arma::mat(model_.ns(), model_.ns()));

    // Loop over time slices in reverse
    // the iteration is not starting from β, but β - Δτ
    // β - Δτ, β - 2Δτ, ..., 0
    for (int l = model_.nt() - 1; l >= 0; --l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        // update HS field over space given time slice
        acc_l = model_.update_time_slice(greens, l);
        acc_rate_ += acc_l / nt;

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_GF_backward(greens[flv], l, flv);
        }

        // Do the stabilization every interval time, the beginning of stack
        if (loc_l == 0) {
            double max_error = 0.0;     

            for (int flv = 0; flv < n_flavor; flv++) {
                Bprods[flv].eye();

                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l];

                // Calculate Bprod
                calculate_Bproduct(Bprods[flv], i_stack, flv);

                // Update stacks
                update_stack_backward(propagation_stacks[flv], Bprods[flv], i_stack);

                // Calculate Green's functions at the beginning of local time within stack
                stabilize_GF_backward(greens[flv], propagation_stacks[flv], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[l]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF backward precision reach beyond threshold > 1e-6. Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void DQMC::sweep_unequalTime(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks) {

    // do nothing if unequalTime is not active
    if (!isUnequalTime_) {
        return ;
    }

    int loc_l;
    int i_stack;

    int n_flavor = static_cast<int>(greens.size());
    const int nt = model_.nt();

    std::vector<arma::mat> Bprods(n_flavor, arma::mat(model_.ns(), model_.ns()));
    stablelinalg::LDR Bt0;
    stablelinalg::LDR Bbt;

    for (int l = 0; l < nt; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_unequalTime_GF_forward(greens[flv], l, flv);
        }

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {
                Bprods[flv].eye();

                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l+1];
                arma::mat Gt0_temp = greens[flv].Gt0[l+1];
                arma::mat G0t_temp = greens[flv].G0t[l+1];

                // Calculate Bprod
                calculate_Bproduct(Bprods[flv], i_stack, flv, false);

                // Update stacks
                propagate_Bt0_Bbt(Bt0, Bbt, propagation_stacks[flv], Bprods[flv], i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_unequalTime(greens[flv], Bt0, Bbt, l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[l+1]);
                max_error = std::max(max_error, error);
                error = check_error(Gt0_temp, greens[flv].Gt0[l+1]);
                max_error = std::max(max_error, error);
                error = check_error(G0t_temp, greens[flv].G0t[l+1]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF unequaltime precision reach beyond threshold > 1e-6. Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

}