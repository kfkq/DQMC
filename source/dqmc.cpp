#include <dqmc.h>
#include <stdexcept>

// --- Constructor ---
DQMC::DQMC(const utility::parameters& params, ModelBase& model, utility::random& rng) 
    : model_(model), rng_(rng)
{
    int nt = params.getInt("simulation", "nt");
    n_stab_ = params.getInt("simulation", "n_stab");
    n_stack_ = nt / n_stab_;
    acc_rate_ = 0.0;

    // Initialize the cache for B-matrices
    int nfl = model_.n_flavors();
    int ns = model_.n_size();
    B_cache_.resize(nfl);
    invB_cache_.resize(nfl);
    for (auto& flavor_cache : B_cache_) {
        flavor_cache.resize(nt, arma::mat(ns, ns, arma::fill::zeros));
    }
    for (auto& flavor_cache : invB_cache_) {
        flavor_cache.resize(nt, arma::mat(ns, ns, arma::fill::zeros));
    }
}

// --- Helper Functions ---
int DQMC::global_l(int stack_idx, int loc_l) const { return stack_idx * n_stab_ + loc_l; }
int DQMC::stack_idx(int l) const { return l / n_stab_; }
int DQMC::local_l(int l) const { return l % n_stab_; }

// --- B matrices calculation ---
arma::mat DQMC::calc_B(arma::mat& expK, arma::mat& expV) {
    return expV * expK;
}
arma::mat DQMC::calc_B(arma::mat& expK, arma::vec& expV) {
    return stablelinalg::diag_mul_mat(expV, expK);
}
arma::mat DQMC::calc_invB(arma::mat& invexpK, arma::mat& invexpV) {
    return invexpK * invexpV;
}
arma::mat DQMC::calc_invB(arma::mat& invexpK, arma::vec& invexpV) {
    return stablelinalg::mat_mul_diag(invexpK, invexpV);
}

arma::mat DQMC::calc_Bbar(int i_stack, int flv, bool recalculate_cache) {
    int nsize = model_.n_size();
    arma::mat Bbar(nsize, nsize, arma::fill::eye);
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);

        if (recalculate_cache) {
            auto expK = model_.get_expK();
            auto expV = model_.get_expV(l, flv);
            B_cache_[flv][l] = calc_B(expK, expV);
        }

        Bbar = B_cache_[flv][l] * Bbar;
    }

    return Bbar;
}

/* --------------------------------------------------------------------------------
/
/   Initialization of stacks and Green's function
/
-------------------------------------------------------------------------------- */
LDRStack DQMC::init_stacks(int flv) {
    LDRStack propagation_stack(n_stack_);
    
    for (int i_stack = n_stack_ - 1; i_stack >= 0; --i_stack) {
        
        arma::mat Bbar = calc_Bbar(i_stack, flv);
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
    
    int nt = model_.n_timesteps();
    
    GF greens;
    greens.Gtt.resize(nt+1);
    greens.Gt0.resize(nt+1);
    greens.G0t.resize(nt+1);
    greens.log_det_M = 0.0;

    greens.Gtt[0] = stablelinalg::inv_I_plus_ldr(propagation_stack[0], greens.log_det_M);

    return greens;
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Greens Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_forward(GF& greens, int time_slice, int flv) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gtt = B_l * Gtt * B_l^{-1}
    */

    auto expK = model_.get_expK();
    auto expV = model_.get_expV(time_slice, flv);
    B_cache_[flv][time_slice] = calc_B(expK, expV);

    auto invexpK = model_.get_invexpK();
    auto invexpV = model_.get_invexpV(time_slice, flv);
    invB_cache_[flv][time_slice] = calc_invB(invexpK, invexpV);

    greens.Gtt[time_slice+1] = B_cache_[flv][time_slice] * greens.Gtt[time_slice] * invB_cache_[flv][time_slice];
}

void DQMC::update_stack_forward(LDRStack& propagation_stack, const arma::mat& Bbar, int i_stack) {
    if (i_stack == 0) {
        propagation_stack[i_stack] = stablelinalg::to_LDR(Bbar);
    } else {
        propagation_stack[i_stack] = stablelinalg::mat_mul_ldr(Bbar, propagation_stack[i_stack - 1]);
    }
}

void DQMC::stabilize_GF_forward(GF& greens, LDRStack& propagation_stack, int time_slice) {
    int i_stack = stack_idx(time_slice);
    int nt = model_.n_timesteps();
    if (time_slice == nt - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[time_slice+1] = stablelinalg::inv_I_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // calculate [I + B(τ,0)B(β,τ)]^{-1}
        greens.Gtt[time_slice+1] = stablelinalg::inv_I_plus_ldr_mul_ldr(
                    propagation_stack[i_stack], 
                    propagation_stack[i_stack + 1]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Backward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_backward(GF& greens, int time_slice, int flv) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ - dτ
    /       Gttup = B_l^{-1} * Gttup * B_l
    */

    auto expK = model_.get_expK();
    auto expV = model_.get_expV(time_slice, flv);
    B_cache_[flv][time_slice] = calc_B(expK, expV);

    auto invexpK = model_.get_invexpK();
    auto invexpV = model_.get_invexpV(time_slice, flv);
    invB_cache_[flv][time_slice] = calc_invB(invexpK, invexpV);

    greens.Gtt[time_slice] = invB_cache_[flv][time_slice] * greens.Gtt[time_slice+1] * B_cache_[flv][time_slice];
}

void DQMC::update_stack_backward(LDRStack& propagation_stack, const arma::mat& Bbar, int i_stack) {
    if (i_stack == n_stack_-1) {
        propagation_stack[i_stack] = stablelinalg::to_LDR(Bbar);
    } else {
        propagation_stack[i_stack] = stablelinalg::ldr_mul_mat(propagation_stack[i_stack + 1], Bbar);
    }
}

void DQMC::stabilize_GF_backward(GF& greens, LDRStack& propagation_stack, int time_slice) {
    int i_stack = stack_idx(time_slice);
    int nt = model_.n_timesteps();
    if (time_slice == 0) { // at beginning of propagation
        //  G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[time_slice] = stablelinalg::inv_I_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // calculate [I + B(τ,0)B(β,τ)]^{-1}
        greens.Gtt[time_slice] = stablelinalg::inv_I_plus_ldr_mul_ldr(
                    propagation_stack[i_stack - 1], 
                    propagation_stack[i_stack]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Unequal Time propagation
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_unequalTime_GF_forward(GF& greens, int time_slice, int flv) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gtt = B_l * Gtt * B_l^{-1}
    */

    int nsize = model_.n_size();

    if (time_slice == 0) {
        greens.Gt0[0] = greens.Gtt[0];
        greens.G0t[0] = greens.Gtt[0] - arma::eye(nsize, nsize);       
    }

    auto expK = model_.get_expK();
    auto expV = model_.get_expV(time_slice, flv);
    B_cache_[flv][time_slice] = calc_B(expK, expV);

    auto invexpK = model_.get_invexpK();
    auto invexpV = model_.get_invexpV(time_slice, flv);
    invB_cache_[flv][time_slice] = calc_invB(invexpK, invexpV);

    greens.Gtt[time_slice+1] = B_cache_[flv][time_slice] * greens.Gtt[time_slice] * invB_cache_[flv][time_slice];
    greens.Gt0[time_slice+1] = B_cache_[flv][time_slice] * greens.Gt0[time_slice];
    greens.G0t[time_slice+1] = greens.G0t[time_slice] * invB_cache_[flv][time_slice];
}

void DQMC::propagate_Bt0_Bbt(stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, 
                            LDRStack& propagation_stack, arma::mat& Bbar, int i_stack) 
{

    if (i_stack == 0) {
        Bt0 = stablelinalg::to_LDR(Bbar);
    } else {
        Bt0 = stablelinalg::mat_mul_ldr(Bbar, Bt0);
    }

    if (i_stack < propagation_stack.size() - 1) {  
        Bbt = propagation_stack[i_stack + 1];
    }
}

void DQMC::stabilize_unequalTime(GF& greens, stablelinalg::LDR& Bt0, stablelinalg::LDR& Bbt, int time_slice) {
    int nt = model_.n_timesteps();
    if (time_slice == nt - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[time_slice+1] = stablelinalg::inv_I_plus_ldr(Bt0, greens.log_det_M);

        //  G(β,0) = I - G(0,0)
        greens.Gt0[time_slice+1] = stablelinalg::I_minus_mat(greens.Gtt[time_slice+1]);

        //  G(0,β) = -G(0,0)
        greens.G0t[time_slice+1] = -greens.Gtt[time_slice+1];
    } 
    else {
        greens.Gtt[time_slice+1] = stablelinalg::inv_I_plus_ldr_mul_ldr(Bt0, Bbt);
        greens.Gt0[time_slice+1] = stablelinalg::inv_invldr_plus_ldr(Bt0, Bbt);
        greens.G0t[time_slice+1] = -stablelinalg::inv_invldr_plus_ldr(Bbt, Bt0);
    }
}

/* --------------------------------------------------------------------------------
/
/   Utilities
/
-------------------------------------------------------------------------------- */
double DQMC::check_error(const arma::mat& Gtt_temp, const arma::mat& Gtt) {

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
    const int nt = model_.n_timesteps();

    // Loop over time slices forward from 0 to β
    for (int time_slice = 0; time_slice < nt; ++time_slice) {
        // Get local time and stack index
        loc_l = local_l(time_slice);
        i_stack = stack_idx(time_slice);

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_GF_forward(greens[flv], time_slice, flv);
        }

        // update HS field over space given time slice
        update::local_update(rng_, model_, greens, time_slice, acc_l);
        acc_rate_ += acc_l / nt;

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {

                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[time_slice+1];

                // Calculate Bprod
                arma::mat Bbar = calc_Bbar(i_stack, flv);

                // Update stacks
                update_stack_forward(propagation_stacks[flv], Bbar, i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_GF_forward(greens[flv], propagation_stacks[flv], time_slice);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[time_slice+1]);
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
    const int nt = model_.n_timesteps();

    // Loop over time slices forward from 0 to β
    for (int time_slice = nt - 1; time_slice >= 0; --time_slice) {
        // Get local time and stack index
        loc_l = local_l(time_slice);
        i_stack = stack_idx(time_slice);

        // update HS field over space given time slice
        update::local_update(rng_, model_, greens, time_slice, acc_l);
        acc_rate_ += acc_l / nt;

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_GF_forward(greens[flv], time_slice, flv);
        }

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {

                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[time_slice];

                // Calculate Bprod
                arma::mat Bbar = calc_Bbar(i_stack, flv);

                // Update stacks
                update_stack_backward(propagation_stacks[flv], Bbar, i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_GF_backward(greens[flv], propagation_stacks[flv], time_slice);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[time_slice]);
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

void DQMC::sweep_unequal_time(std::vector<GF>& greens, std::vector<LDRStack>& propagation_stacks) {
    int loc_l;
    int i_stack;

    int n_flavor = static_cast<int>(greens.size());
    const int nt = model_.n_timesteps();

    stablelinalg::LDR Bt0;
    stablelinalg::LDR Bbt;

    for (int time_slice = 0; time_slice < nt; ++time_slice) {
        // Get local time and stack index
        loc_l = local_l(time_slice);
        i_stack = stack_idx(time_slice);

        for (int flv = 0; flv < n_flavor; flv++) {
            propagate_unequalTime_GF_forward(greens[flv], time_slice, flv);
        }

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int flv = 0; flv < n_flavor; flv++) {
                // save naive propagation equal time Green's function
                arma::mat Gtt_temp = greens[flv].Gtt[time_slice + 1];
                arma::mat Gt0_temp = greens[flv].Gt0[time_slice + 1];
                arma::mat G0t_temp = greens[flv].G0t[time_slice + 1];

                // Calculate Bprod
                arma::mat Bbar = calc_Bbar(i_stack, flv, false);

                // Update stacks
                propagate_Bt0_Bbt(Bt0, Bbt, propagation_stacks[flv], Bbar, i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_unequalTime(greens[flv], Bt0, Bbt, time_slice);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[flv].Gtt[time_slice + 1]);
                max_error = std::max(max_error, error);
                error = check_error(Gt0_temp, greens[flv].Gt0[time_slice + 1]);
                max_error = std::max(max_error, error);
                error = check_error(G0t_temp, greens[flv].G0t[time_slice + 1]);
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