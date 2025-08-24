#include "dqmc.hpp"

linalg::LDRStack DQMC::init_stacks(int nfl) {
    /*
    / Initialize propagation stacks for up and down spins, stored as member variable
    / We initialize the forward stacks
    / F_stacks = [F_1, F_2, ..., F_end]; with size = n_stack = nt / n_stab
    /
    /   where F_end = B_{nt-1} * B_{nt-2} * ... * B_{nt-n_stab}
    /         F_{end-1} = [F_end] * [B_{nt-n_stab-1} * B_{nt-n_stab-2} * ... * B_{nt-2*n_stab}]
    /         ...
    /         F_1 = [F_2] * [B_{n_stab-1} * ... * B_1 * B_0]
    /
    */

    linalg::LDRStack propagation_stack(n_stack_);

    B_.resize(model_.nt(), Matrix(model_.ns(), model_.ns()));
    invB_.resize(model_.nt(), Matrix(model_.ns(), model_.ns()));

    Matrix Bprod(model_.ns(), model_.ns());
    
    // Loop over stack, calculate F[end] to F[1]
    for (int i_stack = n_stack_ - 1; i_stack >= 0; i_stack--) {
        Bprod.eye();

        calculate_Bproduct(Bprod, i_stack, nfl, false);

        // once B product is calculated, we transform into LDR form for stable propagation
        linalg::LDR Bbar = linalg::LDR::from_qr(Bprod);
        
        // first filling of the stack no need to propagate, just simply product of B matrices
        // next stack will be propagated using prev calculated stack and calculated product of B matrices
        if (i_stack == n_stack_ - 1) {
            propagation_stack[i_stack] = Bbar;
        } else {
            propagation_stack[i_stack] = linalg::LDR::ldr_mul_ldr(propagation_stack[i_stack + 1], Bbar);
        }
    }

    return propagation_stack;
}

GF DQMC::init_greenfunctions(linalg::LDRStack& propagation_stack) {
    /*
    / Initialize equal time Green's functions
    /      G(τ = 0, τ = 0) = [1 + B(β,0)]^{-1}
    /
    / Dynamical Green's function will be calculated when sweep over time slice starts.
    */
    
    int ns = model_.ns();
    int nt = model_.nt();
    
    GF greens;
    greens.Gtt.resize(nt+1);
    greens.Gt0.resize(nt+1);
    greens.G0t.resize(nt+1);
    greens.log_det_M = 0.0;

    greens.Gtt[0] = linalg::LDR::inv_eye_plus_ldr(propagation_stack[0], greens.log_det_M);

    return greens;
}

void DQMC::calculate_Bproduct(Matrix& Bprod, int i_stack, int nfl, bool cache) {
    /*
    /   Build product of Bup matrices for every time interval n_stab
    /    Loop over time slice within stack
    /    calculate naively: Prod = B * B * ... * B
    /      loc_t = [0, ..., n_stab-1] is a relative time slice within stack
    /      t = [0, ..., nt-1] is a global time slice in which we convert t->global_t(loc_t)
    */
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);
        if (!cache) {
            B_[l] = model_.calc_B(l, nfl);
        }
        Bprod = B_[l] * Bprod;
    }
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Greens Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_forward(GF& greens, int l, int nfl) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gtt = B_l * Gtt * B_l^{-1}
    */

    B_[l] = model_.calc_B(l, nfl);
    invB_[l] = model_.calc_invB(l, nfl);

    greens.Gtt[l+1] = B_[l] * greens.Gtt[l] * invB_[l];
}

void DQMC::update_stack_forward(linalg::LDRStack& propagation_stack, Matrix& Bprod, int i_stack) {
    /*
    / Update the propagation stack when we do forward sweep
    /   if i_stack == 0, stack[0] = B(τ0, 0)
    /   else,            stack[i] = B(τi, 0) = B(τi, τi') * stack[i-1]
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    
    if (i_stack == 0) {
        propagation_stack[i_stack] = linalg::LDR::from_qr(Bprod);
    } else {
        propagation_stack[i_stack] = linalg::LDR::mat_mul_ldr(Bprod, propagation_stack[i_stack - 1]);
    }
}

void DQMC::stabilize_GF_forward(GF& greens, linalg::LDRStack& propagation_stack, int l) {
    /*
    /  Update Green's functions at stabilization time
    /     We call after we update our current stack at i 
    /     (replacing reverse to forward stack), so we have
    /       B(τi, 0) = stack[i]
    /     Since stack[i+1] have not been replaced, we still retain old reverse stack
    /       B(β, τi) = stack[i+1]
    /
    /     Then update Green's functions
    /       Gttup = G(τ,τ) =  [I + B(τi,0)B(β,τi)]^{-1}
    /       Gt0up = G(τ,0) = [B(τi, 0)^{-1} + B(β,τi)]^{-1}
    /       G0tup = G(0,τ) = -[B(β,τi)^{-1} + B(τi, 0)]^{-1}
    /    If we reach final stack,
    /       Gttup = G(β,β) = G(0,0) = [I + B(β,0)]^{-1}
    /       Gt0up = G(β,0) = [I - G(0,0)]
    /       G0tup = G(0,β) = -G(0,0)
    /  
    */

    int i_stack = stack_idx(l);
    
    if (l == model_.nt() - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l+1] = linalg::LDR::inv_eye_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // calculate 
        greens.Gtt[l+1] = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_stack[i_stack], 
                    propagation_stack[i_stack + 1]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Backward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_GF_backward(GF& greens, int l, int nfl) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ - dτ
    /       Gttup = B_l^{-1} * Gttup * B_l
    */

    B_[l] = model_.calc_B(l, nfl);
    invB_[l] = model_.calc_invB(l, nfl);

    greens.Gtt[l] = invB_[l] * greens.Gtt[l+1] * B_[l];
}

void DQMC::update_stack_backward(linalg::LDRStack& propagation_stack, Matrix& Bprod, int i_stack) {
    /*
    / Update the propagation stack when we do backward sweep
    /   if i_stack == end, stack[end] = B(β, β - nstab*Δτ)
    /   else,              stack[i] = B(β, τi') = stack[i+1] * B(τi, τi')
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    if (i_stack == n_stack_-1) {
        propagation_stack[i_stack] = linalg::LDR::from_qr(Bprod);
    } else {
        propagation_stack[i_stack] = linalg::LDR::ldr_mul_mat(propagation_stack[i_stack + 1], Bprod);
    }
}

void DQMC::stabilize_GF_backward(GF& greens, linalg::LDRStack& propagation_stack, int l) {
    /*
    /  Update equal time Green's functions at stabilization time
    /     We call after we update our current stack at i 
    /     (replacing forward to backward stack), so we have
    /       B(β, τi) = stack[i]
    /     Since stack[i-1] have not been replaced, we still retain old forward stack
    /       B(τi, 0) = stack[i-1]
    /
    /   Update:
    /       Gttup = G(τ,τ) =  [I + B(τi,0)B(β,τi)]^{-1}
    /       If we reach final stack,
    /       Gttup = G(β,β) = G(0,0) = [I + B(β,0)]^{-1}
    /  
    */

    int i_stack = stack_idx(l);
    
    if (l == 0) { // at beginning of propagation
        //  G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l] = linalg::LDR::inv_eye_plus_ldr(propagation_stack[i_stack], greens.log_det_M);
    } else {
        // calculate 
        greens.Gtt[l] = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                        propagation_stack[i_stack-1], 
                        propagation_stack[i_stack]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Unequal Time propagation
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_unequalTime_GF_forward(GF& greens, int l, int nfl) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gtt = B_l * Gtt * B_l^{-1}
    */

    if (l == 0) {
        greens.Gt0[0] = greens.Gtt[0];
        greens.G0t[0] = greens.Gtt[0] - arma::eye(model_.ns(),model_.ns());       
    }
    greens.Gtt[l+1] = B_[l] * greens.Gtt[l] * invB_[l];
    greens.Gt0[l+1] = B_[l] * greens.Gt0[l];
    greens.G0t[l+1] = greens.G0t[l] * invB_[l];
}

void DQMC::propagate_Bt0_Bbt(linalg::LDR& Bt0, linalg::LDR& Bbt, 
                            linalg::LDRStack& propagation_stack,
                            Matrix& Bprod, int i_stack) 
{

    if (i_stack == 0) {
        Bt0 = linalg::LDR::from_qr(Bprod);
    } else {
        Bt0 = linalg::LDR::mat_mul_ldr(Bprod, Bt0);
    }

    if (i_stack < propagation_stack.size() - 1) {  
        Bbt = propagation_stack[i_stack + 1];
    }
}

void DQMC::stabilize_unequalTime(GF& greens, linalg::LDR& Bt0, linalg::LDR& Bbt, int l) {
    if (l == model_.nt() - 1) { // at last propagation
        //  G(β, β) = G(0,0) = [I + B(β,0)]^{-1}
        greens.Gtt[l+1] = linalg::LDR::inv_eye_plus_ldr(Bt0, greens.log_det_M);

        //  G(β,0) = I - G(0,0)
        greens.Gt0[l+1] = linalg::eye_minus_mat(greens.Gtt[l+1]);

        //  G(0,β) = -G(0,0)
        greens.G0t[l+1] = -greens.Gtt[l+1];
    } 
    else {
        greens.Gtt[l+1] = linalg::LDR::inv_eye_plus_ldr_mul_ldr(Bt0, Bbt);
        greens.Gt0[l+1] = linalg::LDR::inv_invldr_plus_ldr(Bt0, Bbt);
        greens.G0t[l+1] = -linalg::LDR::inv_invldr_plus_ldr(Bbt, Bt0);
    }
}

/* --------------------------------------------------------------------------------
/
/   Utilities
/
-------------------------------------------------------------------------------- */

double DQMC::calculate_action(const std::vector<GF>& greens) {
    /*
    / Calculates the total action S = -log(W) for the current HS field configuration.
    / The weight W is given by:
    /   W = (Π_{l,i} γ) * exp(-S_bosonic) * Π_flavor det(M_flavor)
    /
    / The total action S is therefore the sum of three components:
    /   S = S_gamma + S_bosonic + S_fermionic
    /
    / 1. S_fermionic = -Σ_flavor log|det(M_flavor)|
    /    We sum the stored log_det_M from each GF object.
    /
    / 2. S_bosonic = Σ_{l,i} α * η(s_{l,i})
    /    This is the exponential part of the HS transform.
    /
    / 3. S_gamma = -Σ_{l,i} log(γ(s_{l,i}))
    */

    // 1. Fermionic Action
    double s_fermionic = 0.0;
    for (const auto& gf : greens) {
        s_fermionic -= gf.log_det_M;
    }

    // this shouldn't be here, but for the moment only for attractive hubbard
    if (greens.size() == 1) {
        s_fermionic *= 2.0;
    }

    // 2. Bosonic and Gamma Action
    const auto& fields = model_.get_fields();
    const auto& gamma = model_.get_gamma();
    const auto& eta = model_.get_eta();
    const double alpha = model_.get_alpha();

    double s_bosonic = 0.0;
    double s_gamma = 0.0;

    for (arma::uword i = 0; i < fields.n_elem; ++i) {
        int field_val = fields(i);
        s_bosonic += alpha * eta(field_val);
        s_gamma   -= std::log(gamma(field_val));
    }

    return s_fermionic + s_bosonic + s_gamma;
}

double DQMC::check_error(const Matrix& Gtt_temp, const Matrix& Gtt) {
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

void DQMC::sweep_0_to_beta(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks) {
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

    std::vector<Matrix> Bprods(n_flavor, Matrix(model_.ns(), model_.ns()));

    // Loop over time slices forward
    // We propagate our GF, from 0 to β
    // the iteration then start processing from 0 + Δτ, 0 + 2Δτ, ..., 0 + nt*Δτ (β)
    for (int l = 0; l < nt; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        for (int nfl = 0; nfl < n_flavor; nfl++) {
            propagate_GF_forward(greens[nfl], l, nfl);
        }

        // update HS field over space given time slice
        acc_l = model_.update_time_slice(greens, l);
        acc_rate_ += acc_l / nt;

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int nfl = 0; nfl < n_flavor; nfl++) {
                Bprods[nfl].eye();

                // save naive propagation equal time Green's function
                GreenFunc G00_temp = greens[nfl].Gtt[l+1];

                // Calculate Bprod
                calculate_Bproduct(Bprods[nfl], i_stack, nfl, false);

                // Update stacks
                update_stack_forward(propagation_stacks[nfl], Bprods[nfl], i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_GF_forward(greens[nfl], propagation_stacks[nfl], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(G00_temp, greens[nfl].Gtt[l+1]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF forward error beyond threshold > 1e-6 at l=" << l 
                          << ". Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void DQMC::sweep_beta_to_0(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks) {
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

    std::vector<Matrix> Bprods(n_flavor, Matrix(model_.ns(), model_.ns()));

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

        for (int nfl = 0; nfl < n_flavor; nfl++) {
            propagate_GF_backward(greens[nfl], l, nfl);
        }

        // Do the stabilization every interval time, the beginning of stack
        if (loc_l == 0) {
            double max_error = 0.0;     

            for (int nfl = 0; nfl < n_flavor; nfl++) {
                Bprods[nfl].eye();

                // save naive propagation equal time Green's function
                GreenFunc G00_temp = greens[nfl].Gtt[l];

                // Calculate Bprod
                calculate_Bproduct(Bprods[nfl], i_stack, nfl, false);

                // Update stacks
                update_stack_backward(propagation_stacks[nfl], Bprods[nfl], i_stack);

                // Calculate Green's functions at the beginning of local time within stack
                stabilize_GF_backward(greens[nfl], propagation_stacks[nfl], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(G00_temp, greens[nfl].Gtt[l]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF backward error beyond threshold > 1e-6 at l=" << l 
                          << ". Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}

void DQMC::sweep_unequalTime(std::vector<GF>& greens, std::vector<linalg::LDRStack>& propagation_stacks) {
    int loc_l;
    int i_stack;

    int n_flavor = static_cast<int>(greens.size());
    const int nt = model_.nt();

    std::vector<Matrix> Bprods(n_flavor, Matrix(model_.ns(), model_.ns()));
    linalg::LDR Bt0;
    linalg::LDR Bbt;

    for (int l = 0; l < nt; ++l) {
        // Get local time and stack index
        loc_l = local_l(l);
        i_stack = stack_idx(l);

        for (int nfl = 0; nfl < n_flavor; nfl++) {
            propagate_unequalTime_GF_forward(greens[nfl], l, nfl);
        }

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;  

            for (int nfl = 0; nfl < n_flavor; nfl++) {
                Bprods[nfl].eye();

                // save naive propagation equal time Green's function
                GreenFunc Gtt_temp = greens[nfl].Gtt[l+1];
                GreenFunc Gt0_temp = greens[nfl].Gt0[l+1];
                GreenFunc G0t_temp = greens[nfl].G0t[l+1];

                // Calculate Bprod
                calculate_Bproduct(Bprods[nfl], i_stack, nfl, true);

                // Update stacks
                propagate_Bt0_Bbt(Bt0, Bbt, propagation_stacks[nfl], Bprods[nfl], i_stack);

                // Calculated Green's function at the end of local time within stack
                stabilize_unequalTime(greens[nfl], Bt0, Bbt, l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[nfl].Gtt[l+1]);
                max_error = std::max(max_error, error);
                error = check_error(Gt0_temp, greens[nfl].Gt0[l+1]);
                max_error = std::max(max_error, error);
                error = check_error(G0t_temp, greens[nfl].G0t[l+1]);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "unequal GF error beyond threshold > 1e-6 at l=" << l 
                          << ". Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

}