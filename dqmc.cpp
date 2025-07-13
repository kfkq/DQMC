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

    Matrix Bprod(model_.ns(), model_.ns());
    
    // Loop over stack, calculate F[end] to F[1]
    for (int i_stack = n_stack_ - 1; i_stack >= 0; i_stack--) {
        Bprod.eye();

        calculate_Bproduct(Bprod, i_stack, nfl);

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
    
    // Initialize Green's function matrices
    // B(β,0) = propagation_stack_[0]
    Matrix Gtt = linalg::LDR::inv_eye_plus_ldr(propagation_stack[0]);
    Matrix Gt0 = Matrix(ns, ns, arma::fill::eye);
    Matrix G0t = Matrix(ns, ns, arma::fill::eye);

    return GF{Gtt, Gt0, G0t};
}

void DQMC::calculate_Bproduct(Matrix& Bprod, int i_stack, int nfl) {
    /*
    /   Build product of Bup matrices for every time interval n_stab
    /    Loop over time slice within stack
    /    calculate naively: Prod = B * B * ... * B
    /      loc_t = [0, ..., n_stab-1] is a relative time slice within stack
    /      t = [0, ..., nt-1] is a global time slice in which we convert t->global_t(loc_t)
    */
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);
        Bprod = model_.calc_B(l, nfl) * Bprod;
    }
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Greens Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_equaltime_GF(GF& greens, int l, int nfl) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gtt = B_l * Gtt * B_l^{-1}
    */

    Matrix B_l = model_.calc_B(l, nfl);
    Matrix invB_l = model_.calc_invB(l, nfl);

    greens.Gtt = B_l * greens.Gtt * invB_l;
    //Gt0up_ = B_t * Gt0up_
    //G0tup_ =       G0tup_ * invB_t;
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

void DQMC::update_equaltime_GF_at_stabilization_forward(GF& greens, linalg::LDRStack& propagation_stack, int l) {
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
        greens.Gtt = linalg::LDR::inv_eye_plus_ldr(propagation_stack[i_stack]);

        //  G(β,0) = I - G(0,0)
        //Gt0up_ = linalg::LDR::eye_minus_mat(Gttup_);

        //  G(0,β) = -G(0,0)
        //G0tup_ = -Gttup_;
    } else {
        // calculate 
        greens.Gtt = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_stack[i_stack], 
                    propagation_stack[i_stack + 1]);

        //Gt0up_ = linalg::LDR::inv_invldr_plus_ldr(
        //            propagation_up_stack_[i_stack], 
        //            propagation_up_stack_[i_stack + 1]);
        
        //G0tup_ = -linalg::LDR::inv_invldr_plus_ldr(
        //            propagation_up_stack_[i_stack + 1], 
        //            propagation_up_stack_[i_stack]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Backward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_equaltime_GF_reverse(GF& greens, int l, int nfl) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ - dτ
    /       Gttup = B_l^{-1} * Gttup * B_l
    */

    Matrix B_l = model_.calc_B(l, nfl);
    Matrix invB_l = model_.calc_invB(l, nfl);

    greens.Gtt = invB_l * greens.Gtt * B_l;
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

void DQMC::update_equaltime_GF_at_stabilization_backward(GF& greens, linalg::LDRStack& propagation_stack, int l) {
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
        greens.Gtt = linalg::LDR::inv_eye_plus_ldr(propagation_stack[i_stack]);
    } else {
        // calculate 
        greens.Gtt = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                        propagation_stack[i_stack-1], 
                        propagation_stack[i_stack]);
    }
}


double DQMC::check_error(const Matrix& Gtt_temp, const Matrix& Gtt) {
    /*
    / Check the error between the two Green's function matrices
    /   Gtt_temp = temporary matrix of naive product of B matrices 
    /   Gtt = re calculated Green's function by stabilization
    /   return: max|Gtt_temp - Gtt| (element-wise maximum absolute difference)
    */
    
    // Calculate element-wise absolute difference and find maximum
    return arma::max(arma::max(arma::abs(Gtt_temp - Gtt)));
}

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
            propagate_equaltime_GF(greens[nfl], l, nfl);
        }

        // update HS field over space given time slice
        // update HS field over space given time slice
        acc_l = model_.update_time_slice(greens, l);
        acc_rate_ += acc_l / nt;

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {      
            double max_error = 0.0;     

            for (int nfl = 0; nfl < n_flavor; nfl++) {
                Bprods[nfl].eye();

                // save naive propagation equal time Green's function
                GreenFunc Gtt_temp = greens[nfl].Gtt;

                // Calculate Bprod
                calculate_Bproduct(Bprods[nfl], i_stack, nfl);

                // Update stacks
                update_stack_forward(propagation_stacks[nfl], Bprods[nfl], i_stack);

                // Calculated Green's function at the end of local time within stack
                update_equaltime_GF_at_stabilization_forward(greens[nfl], propagation_stacks[nfl], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[nfl].Gtt);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF error beyond threshold > 1e-6 at l=" << l 
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
            propagate_equaltime_GF_reverse(greens[nfl], l, nfl);
        }

        // Do the stabilization every interval time, the beginning of stack
        if (loc_l == 0) {
            double max_error = 0.0;     

            for (int nfl = 0; nfl < n_flavor; nfl++) {
                Bprods[nfl].eye();

                // save naive propagation equal time Green's function
                GreenFunc Gtt_temp = greens[nfl].Gtt;

                // Calculate Bprod
                calculate_Bproduct(Bprods[nfl], i_stack, nfl);

                // Update stacks
                update_stack_backward(propagation_stacks[nfl], Bprods[nfl], i_stack);

                // Calculate Green's functions at the beginning of local time within stack
                update_equaltime_GF_at_stabilization_backward(greens[nfl], propagation_stacks[nfl], l);

                // Check error in Green's function calculated by stabilization and naive product
                double error = check_error(Gtt_temp, greens[nfl].Gtt);
                max_error = std::max(max_error, error);
            }

            if (max_error > 1e-6) {
                std::cerr << "GF error beyond threshold > 1e-6 at l=" << l 
                          << ". Try reducing n_stab or increasing nt. Error: " 
                          << max_error << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
}