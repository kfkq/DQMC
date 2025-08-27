#include <dqmc.h>
#include <update.h>

// --- Constructor ---
DQMC::DQMC(const utility::parameters& params, AttractiveHubbard& model)
        : model_(model), acc_rate_(0.0), avg_sgn_(1.0) 
{
    nt_ = params.getInt("simulation", "nt");
    n_stab_ = params.getInt("simulation", "n_stab");
    n_stack_ = std::ceil(static_cast<double>(nt_) / n_stab_);
    isUnequalTime_ = params.getBool("simulation", "isMeasureUnequalTime");

    // at the final stack, the number of time slice can be less n_stab_
    loc_l_end_.resize(n_stack_);
    for (int i_stack = 0; i_stack < n_stack_; ++i_stack) {
        loc_l_end_[i_stack] = n_stab_ - 1;
        if (i_stack == n_stack_ - 1 && nt_ % n_stab_ != 0) loc_l_end_[i_stack] = nt_ % n_stab_ - 1;
    }

    // Initialize the cache for B-matrices
    int nfl = model_.n_flavor();
    int ns = model_.ns();
    B_.resize(nfl);
    invB_.resize(nfl);
    for (auto& flavor_cache : B_) {
        flavor_cache.resize(nt_, arma::mat(ns, ns, arma::fill::zeros));
    }
    for (auto& flavor_cache : invB_) {
        flavor_cache.resize(nt_, arma::mat(ns, ns, arma::fill::zeros));
    }

    max_precision_error_ = 0.0;
    total_precision_error_ = 0.0;
    num_cumulated_precision_error_ = 0.0;

}

/* --------------------------------------------------------------------------------
/
/   Initialization of stacks and Green's function
/
-------------------------------------------------------------------------------- */
LDRStack DQMC::init_stacks(int flv) {
    LDRStack propagation_stack(n_stack_);

    for (int i_stack = n_stack_ - 1; i_stack >= 0; i_stack--) {
        auto Bbar = calculate_Bbar(i_stack, flv);

        stablelinalg::LDR Bbar_ldr = stablelinalg::to_LDR(Bbar);
        
        if (i_stack == n_stack_ - 1) {
            propagation_stack[i_stack] = Bbar_ldr;
        } else {
            propagation_stack[i_stack] = stablelinalg::ldr_mul_ldr(propagation_stack[i_stack + 1], Bbar_ldr);
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

// --- B matrices calculation ---
arma::mat DQMC::calculate_B(arma::mat& expK, arma::mat& expV) {
    return expV * expK;
}
arma::mat DQMC::calculate_B(arma::mat& expK, arma::vec& expV) {
    return stablelinalg::diag_mul_mat(expV, expK);
}
arma::mat DQMC::calculate_invB(arma::mat& invexpK, arma::mat& invexpV) {
    return invexpK * invexpV;
}
arma::mat DQMC::calculate_invB(arma::mat& invexpK, arma::vec& invexpV) {
    return stablelinalg::mat_mul_diag(invexpK, invexpV);
}

arma::mat DQMC::calculate_Bbar(int i_stack, int flv, bool recalculate_cache) {
    int ns = model_.ns();

    arma::mat Bbar(ns, ns, arma::fill::eye);

    for (int loc_l = 0; loc_l <= loc_l_end_[i_stack]; loc_l++) {
        int l = global_l(i_stack, loc_l);
        if (recalculate_cache) {
            auto expK = model_.expK(flv);
            auto expV = model_.expV(l, flv);
            
            B_[flv][l] = calculate_B(expK, expV);
            
        }
        Bbar = B_[flv][l] * Bbar;
    }
    return Bbar;
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Greens Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_forward(std::vector<GF>& greens, int l) {
    /*
    /   Gtt = B_l * Gtt * B_l^{-1}
    */

    int n_flavor = model_.n_flavor();

    for (int flv = 0; flv < n_flavor; flv++)
    {
        auto expK = model_.expK(flv);
        auto expV = model_.expV(l, flv);
        B_[flv][l] = calculate_B(expK, expV);

        auto invexpK = model_.invexpK(flv);
        auto invexpV = model_.invexpV(l, flv);
        invB_[flv][l] = calculate_invB(invexpK, invexpV);

        greens[flv].Gtt[l+1] = B_[flv][l] * greens[flv].Gtt[l] * invB_[flv][l];
    }
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
    const int nt = model_.nt();
    int i_stack = stack_idx(l);
    
    if (l == nt - 1) { // at last propagation
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

void DQMC::propagate_GF_backward(std::vector<GF>& greens, int l) {
    /*
    /       Gtt = B_l^{-1} * Gttup * B_l
    */

    int n_flavor = model_.n_flavor();
    for (int flv = 0; flv < n_flavor; flv++)
    {   
        auto expK = model_.expK(flv);
        auto expV = model_.expV(l, flv);
        B_[flv][l] = calculate_B(expK, expV);

        auto invexpK = model_.invexpK(flv);
        auto invexpV = model_.invexpV(l, flv);
        invB_[flv][l] = calculate_invB(invexpK, invexpV);

        greens[flv].Gtt[l] = invB_[flv][l] * greens[flv].Gtt[l+1] * B_[flv][l];
    }
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

void DQMC::propagate_unequalTime_GF_forward(std::vector<GF>& greens, int l) {
    /*
    /       Gtt = B_l * Gtt * B_l^{-1}
    /       Gt0 = B_l * Gt0
    /       G0t = G0t * B_l^{-1}
    */

    const int ns = model_.ns();
    int n_flavor = model_.n_flavor();

    for (int flv = 0; flv < n_flavor; flv++)
    {
        if (l == 0) { // at τ = 0
            // G(τ,0) = G(0,0)
            greens[flv].Gt0[0] = greens[flv].Gtt[0];
            // G(0,τ) = - [ I - G(0,0) ]
            greens[flv].G0t[0] = greens[flv].Gtt[0] - arma::eye(ns, ns);       
        }
        greens[flv].Gtt[l+1] = B_[flv][l] * greens[flv].Gtt[l] * invB_[flv][l];
        greens[flv].Gt0[l+1] = B_[flv][l] * greens[flv].Gt0[l];
        greens[flv].G0t[l+1] = greens[flv].G0t[l] * invB_[flv][l];
    }
    
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
    double error = arma::max(arma::max(arma::abs(Gtt_temp - Gtt)));
    
    if (error > max_precision_error_) max_precision_error_ = error;
    total_precision_error_ += error;
    num_cumulated_precision_error_++;

    return error;
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

    int loc_l_end = n_stab_ - 1;
    if (i_stack == n_stack_ - 1 && nt_ % n_stab_ != 0) loc_l_end = nt_ % n_stab_;

    // Loop over time slices forward
    // We propagate our GF, from 0 to β
    // the iteration then start processing from 0 + Δτ, 0 + 2Δτ, ..., 0 + nt*Δτ (β)
    for (int l = 0; l < nt_; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        propagate_GF_forward(greens, l);

        // update HS field over space given time slice
        update::local_update(model_.rng(), model_, greens, l, acc_l);
        acc_rate_ += acc_l / nt_;

        // Do the stabilization at interval time
        if (loc_l  == loc_l_end_[i_stack]) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {
                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l+1];

                // Calculate Bprod
                auto Bbar = calculate_Bbar(i_stack, flv);

                // Update stacks
                update_stack_forward(propagation_stacks[flv], Bbar, i_stack);

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

    // Loop over time slices in reverse
    // the iteration is not starting from β, but β - Δτ
    // β - Δτ, β - 2Δτ, ..., 0
    for (int l = model_.nt() - 1; l >= 0; --l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        // update HS field over space given time slice
        update::local_update(model_.rng(), model_, greens, l, acc_l);
        acc_rate_ += acc_l / nt;

        propagate_GF_backward(greens, l);

        // Do the stabilization every interval time, the beginning of stack
        if (loc_l == 0) {
            double max_error = 0.0;     

            for (int flv = 0; flv < n_flavor; flv++) {
                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l];

                // Calculate Bprod
                auto Bbar = calculate_Bbar(i_stack, flv);

                // Update stacks
                update_stack_backward(propagation_stacks[flv], Bbar, i_stack);

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

    stablelinalg::LDR Bt0;
    stablelinalg::LDR Bbt;

    for (int l = 0; l < nt_; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        propagate_unequalTime_GF_forward(greens, l);

        // Do the stabilization at interval time
        if (loc_l  == loc_l_end_[i_stack]) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {
                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[l+1];
                arma::mat Gt0_temp = greens[flv].Gt0[l+1];
                arma::mat G0t_temp = greens[flv].G0t[l+1];

                // Calculate Bprod
                auto Bbar = calculate_Bbar(i_stack, flv, false);

                // Update stacks
                propagate_Bt0_Bbt(Bt0, Bbt, propagation_stacks[flv], Bbar, i_stack);

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