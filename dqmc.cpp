#include "dqmc.hpp"

void DQMC::init_stacks() {
    /*
    / Initialize propagation stacks for up and down spins, stored as member variable
    / We initialze the forward stacks
    / F_stacks = [F_1, F_2, ..., F_end]; with size = n_stack = nt / n_stab
    /
    /   where F_end = B_{nt-1} * B_{nt-2} * ... * B_{nt-n_stab}
    /         F_{end-1} = [F_end] * [B_{nt-n_stab-1} * B_{nt-n_stab-2} * ... * B_{nt-2*n_stab}]
    /         ...
    /         F_1 = [F_2] * [B_{n_stab-1} * ... * B_1 * B_0]
    /
    */

    Matrix Bprod_up = Matrix(model_.ns(), model_.ns());
    Matrix Bprod_dn = Matrix(model_.ns(), model_.ns());

    int n_stack = propagation_up_stack_.size();

    // temporary matrices for LDR
    linalg::LDR temp_ldr;
    
    // Loop over stack, calculate F[end] to F[1]
    for (int i_stack = n_stack - 1; i_stack >= 0; i_stack--) {
        Bprod_up.eye();
        Bprod_dn.eye();

        calculate_Bproduct_up_naive(Bprod_up, i_stack);
        calculate_Bproduct_dn_naive(Bprod_dn, i_stack);

        // once B product is calculated, we transform into LDR form for stable propagation
        linalg::LDR Bbar_up = linalg::LDR::from_qr(Bprod_up);
        linalg::LDR Bbar_dn = linalg::LDR::from_qr(Bprod_dn);
        
        // first filling of the stack no need to propagate, just simply product of B matrices
        // next stack will be propagated using prev calculated stack and calculated product of B matrices
        if (i_stack == n_stack - 1) {
            propagation_up_stack_[i_stack] = Bbar_up;
            propagation_dn_stack_[i_stack] = Bbar_dn;
        } else {
            propagation_up_stack_[i_stack] = linalg::LDR::ldr_mul_ldr(propagation_up_stack_[i_stack + 1], Bbar_up);
            propagation_dn_stack_[i_stack] = linalg::LDR::ldr_mul_ldr(propagation_dn_stack_[i_stack + 1], Bbar_dn);
        }
        
    }
}

void DQMC::init_greenfunctions() {
    /*
    / Initialize equal time Green's functions
    /       G(τ = 0, τ = 0) =  [1 + B(β,0)]^{-1}
    /
    /    *dynamical Green's function will be calculated when sweep over time slice starts
    /     no need to calculate now. Now, we initialize using identity matrix.
    */
    
    int ns = model_.ns();
    
    // Initialize Green's function matrices to identity
    Gttup_ = Matrix(ns, ns, arma::fill::eye);
    Gttdn_ = Matrix(ns, ns, arma::fill::eye);

    Gt0up_ = Matrix(ns, ns, arma::fill::eye);
    Gt0dn_ = Matrix(ns, ns, arma::fill::eye);

    G0tup_ = Matrix(ns, ns, arma::fill::eye);
    G0tdn_ = Matrix(ns, ns, arma::fill::eye);

    // Calculate the initial equal time Green's function
    // B(β,0) already initialized in our propagation stack
    // B(β,0) = propagation_stack_[0]
    Gttup_ = linalg::LDR::inv_eye_plus_ldr(propagation_up_stack_[0]);
    Gttdn_ = linalg::LDR::inv_eye_plus_ldr(propagation_dn_stack_[0]);
}

void DQMC::calculate_Bproduct_up_naive(Matrix& Bprod_up, int i_stack) {
    /*
    /   Build product of Bup matrices for every time interval n_stab
    /    Loop over time slice within stack
    /    calculate naively: Prod = B * B * ... * B
    /      loc_t = [0, ..., n_stab-1] is a relative time slice within stack
    /      t = [0, ..., nt-1] is a global time slice in which we convert t->global_t(loc_t)
    */
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);
        Bprod_up = model_.calc_Bup(l) * Bprod_up;
    }
}

void DQMC::calculate_Bproduct_dn_naive(Matrix& Bprod_dn, int i_stack) {
    /*
    /   Build product of Bdn matrices for every time interval n_stab
    /    Loop over time slice within stack
    /    calculate naively: Prod = B * B * ... * B
    /      loc_t = [0, ..., n_stab-1] is a relative time slice within stack
    /      t = [0, ..., nt-1] is a global time slice in which we convert t->global_t(loc_t)
    */
    
    for (int loc_l = 0; loc_l < n_stab_; loc_l++) {
        int l = global_l(i_stack, loc_l);
        Bprod_dn = model_.calc_Bdn(l) * Bprod_dn;
    }
}

/* --------------------------------------------------------------------------------
/
/   Forward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_equaltime_GF_up(int l) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gttup = B_l * Gttup * B_l^{-1}
    */

    Matrix B_l = model_.calc_Bup(l);
    Matrix invB_l = model_.calc_invBup(l);

    Gttup_ = B_l * Gttup_ * invB_l;
    //Gt0up_ = B_t * Gt0up_
    //G0tup_ =       G0tup_ * invB_t;
}

void DQMC::propagate_equaltime_GF_dn(int l) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ + dτ
    /       Gttdn = B_l * Gttdn * B_l^{-1}
    */

    Matrix B_l = model_.calc_Bdn(l);
    Matrix invB_l = model_.calc_invBdn(l);

    Gttdn_ = B_l * Gttdn_ * invB_l;
    //Gt0dn_ = B_t * Gt0dn_;
    //G0tdn_ =       G0tdn_ * invB_t;
}

void DQMC::update_stack_up_forward(Matrix& Bprod_up, int i_stack) {
    /*
    / Update the propagation stack when we do forward sweep
    /   if i_stack == 0, stack[0] = B(τ0, 0)
    /   else,            stack[i] = B(τi, 0) = B(τi, τi') * stack[i-1]
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    
    if (i_stack == 0) {
        propagation_up_stack_[i_stack] = linalg::LDR::from_qr(Bprod_up);
    } else {
        propagation_up_stack_[i_stack] = linalg::LDR::mat_mul_ldr(Bprod_up, propagation_up_stack_[i_stack - 1]);
    }
}

void DQMC::update_stack_dn_forward(Matrix& Bprod_dn, int i_stack) {
    /*
    / Update the propagation stack when we do forward sweep
    /   if i_stack == 0, stack[0] = B(τ0, 0)
    /   else,            stack[i] = B(τi, 0) = B(τi, τi') * stack[i-1]
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    
    if (i_stack == 0) {
        propagation_dn_stack_[i_stack] = linalg::LDR::from_qr(Bprod_dn);
    } else {
        propagation_dn_stack_[i_stack] = linalg::LDR::mat_mul_ldr(Bprod_dn, propagation_dn_stack_[i_stack - 1]);
    }
}

void DQMC::update_equaltime_GF_up_at_stabilization_forward(int l) {
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
        Gttup_ = linalg::LDR::inv_eye_plus_ldr(propagation_up_stack_[i_stack]);

        //  G(β,0) = I - G(0,0)
        //Gt0up_ = linalg::LDR::eye_minus_mat(Gttup_);

        //  G(0,β) = -G(0,0)
        //G0tup_ = -Gttup_;
    } else {
        // calculate 
        Gttup_ = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_up_stack_[i_stack], 
                    propagation_up_stack_[i_stack + 1]);

        //Gt0up_ = linalg::LDR::inv_invldr_plus_ldr(
        //            propagation_up_stack_[i_stack], 
        //            propagation_up_stack_[i_stack + 1]);
        
        //G0tup_ = -linalg::LDR::inv_invldr_plus_ldr(
        //            propagation_up_stack_[i_stack + 1], 
        //            propagation_up_stack_[i_stack]);
    }
}

void DQMC::update_equaltime_GF_dn_at_stabilization_forward(int l) {
    /*
    /  Update Green's functions at stabilization time
    /     We call after we update our current stack at i
    /     (replacing reverse to forward stack), so we have
    /       B(τi, 0) = stack[i]
    /     Since stack[i+1] have not been replaced, we still retain old reverse stack
    /       B(β, τi) = stack[i+1]
    /
    /     Then update Green's functions
    /       Gttdn = G(τ,τ) =  [I + B(τi,0)B(β,τi)]^{-1}
    /       Gt0dn = G(τ,0) =  [B(τi, 0)^{-1} + B(β,τi)]^{-1}
    /       G0tdn = G(0,τ) = -[B(β,τi)^{-1} + B(τi, 0)]^{-1}
    /    If we reach final stack,
    /       Gttdn = G(β,β) = G(0,0) = [I + B(β,0)]^{-1}
    /       Gt0dn = G(β,0) = [I - G(0,0)]
    /       G0tdn = G(0,β) = -G(0,0)
    */

    int i_stack = stack_idx(l);
    
    if (l == model_.nt() - 1) { // at last propagation
        Gttdn_ = linalg::LDR::inv_eye_plus_ldr(propagation_dn_stack_[i_stack]);

        //Gt0dn_ = linalg::LDR::eye_minus_mat(Gttdn_);

        //G0tdn_ = -Gttdn_;
    } else {
        Gttdn_ = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_dn_stack_[i_stack], 
                    propagation_dn_stack_[i_stack + 1]);

        // Gt0dn_ = linalg::LDR::inv_invldr_plus_ldr(
        //             propagation_dn_stack_[i_stack], 
        //             propagation_dn_stack_[i_stack + 1]);
        
        // G0tdn_ = -linalg::LDR::inv_invldr_plus_ldr(
        //             propagation_dn_stack_[i_stack + 1], 
        //             propagation_dn_stack_[i_stack]);
    }
}

/* --------------------------------------------------------------------------------
/
/   Backward propagation Functions
/
-------------------------------------------------------------------------------- */

void DQMC::propagate_equaltime_GF_up_reverse(int l) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ - dτ
    /       Gttup = B_l^{-1} * Gttup * B_l
    */

    Matrix B_l = model_.calc_Bup(l);
    Matrix invB_l = model_.calc_invBup(l);

    Gttup_ = invB_l * Gttup_ * B_l;
}

void DQMC::propagate_equaltime_GF_dn_reverse(int l) {
    /*
    / Propagate Green's function to current imaginary time t, τ -> τ - dτ
    /       Gttdn = B_l^{-1} * Gttdn * B_l
    */

    Matrix B_l = model_.calc_Bdn(l);
    Matrix invB_l = model_.calc_invBdn(l);

    Gttdn_ = invB_l * Gttdn_ * B_l;
}

void DQMC::update_stack_up_backward(Matrix& Bprod_up, int i_stack) {
    /*
    / Update the propagation stack when we do backward sweep
    /   if i_stack == end, stack[end] = B(β, β - nstab*Δτ)
    /   else,              stack[i] = B(β, τi') = stack[i+1] * B(τi, τi')
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    int n_stack = propagation_up_stack_.size();
    if (i_stack == n_stack-1) {
        propagation_up_stack_[i_stack] = linalg::LDR::from_qr(Bprod_up);
    } else {
        propagation_up_stack_[i_stack] = linalg::LDR::ldr_mul_mat(propagation_up_stack_[i_stack + 1], Bprod_up);
    }
}

void DQMC::update_stack_dn_backward(Matrix& Bprod_dn, int i_stack) {
    /*
    / Update the propagation stack when we do backward sweep
    /   if i_stack == end, stack[end] = B(β, β - nstab*Δτ)
    /   else,              stack[i] = B(β, τi') = stack[i+1] * B(τi, τi')
    /  Here τi means time at the end of stack, τi' means time at the beginning of stack
    */
    int n_stack = propagation_dn_stack_.size();
    if (i_stack == n_stack-1) {
        propagation_dn_stack_[i_stack] = linalg::LDR::from_qr(Bprod_dn);
    } else {
        propagation_dn_stack_[i_stack] = linalg::LDR::ldr_mul_mat(propagation_dn_stack_[i_stack + 1], Bprod_dn);
    }
}

void DQMC::update_equaltime_GF_up_at_stabilization_backward(int l) {
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
        Gttup_ = linalg::LDR::inv_eye_plus_ldr(propagation_up_stack_[i_stack]);
    } else {
        // calculate 
        Gttup_ = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_up_stack_[i_stack-1], 
                    propagation_up_stack_[i_stack]);
    }
}

void DQMC::update_equaltime_GF_dn_at_stabilization_backward(int l) {
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
        Gttdn_ = linalg::LDR::inv_eye_plus_ldr(propagation_dn_stack_[i_stack]);
    } else {
        // calculate 
        Gttdn_ = linalg::LDR::inv_eye_plus_ldr_mul_ldr(
                    propagation_dn_stack_[i_stack-1], 
                    propagation_dn_stack_[i_stack]);
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

void DQMC::sweep_0_to_beta() {
    /*
    /   do monte carlo sweep Δτ -> 2Δτ -> ... -> nt * Δτ
    /   for each time slice t, we propagate our equaltime Green's functions
    /   and update the propagation stack.
    /   when we reach our local_time loc_t = n_stab-1, we do stabilization
    */
    
    int loc_l;
    double acc_l;

    Matrix Bprod_up = Matrix(model_.ns(), model_.ns());
    Matrix Bprod_dn = Matrix(model_.ns(), model_.ns());

    GreenFunc Gtt_up_temp = Matrix(model_.ns(), model_.ns());
    GreenFunc Gtt_dn_temp = Matrix(model_.ns(), model_.ns());

    // Loop over time slices forward
    // We propagate our GF, from 0 to β
    // the iteration then start processing from 0 + Δτ, 0 + 2Δτ, ..., 0 + nt*Δτ (β)
    for (int l = 0; l < model_.nt(); ++l) {
        // Get the stack index and local time
        loc_l = local_l(l);

        // Propagate Green's functions to current time
        propagate_equaltime_GF_up(l);
        propagate_equaltime_GF_dn(l);

        // update ising field over space given time slice
        acc_l = model_.update_time_slice(Gttup_, Gttdn_, l);

        acc_rate_ += acc_l / model_.nt();

        // Do the stabilization at interval time
        if (loc_l  == n_stab_ - 1) {
            Bprod_up.eye();
            Bprod_dn.eye();

            // Check which stack we are in
            int i_stack = stack_idx(l);

            // save naive propagation equal time Green's function
            Gtt_up_temp = Gttup_;
            Gtt_dn_temp = Gttdn_;

            // Calculate Bprod
            calculate_Bproduct_up_naive(Bprod_up, i_stack);
            calculate_Bproduct_dn_naive(Bprod_dn, i_stack);

            // Update stacks
            update_stack_up_forward(Bprod_up, i_stack);
            update_stack_dn_forward(Bprod_dn, i_stack);

            // Calculated Green's function at the end of local time within stack
            update_equaltime_GF_up_at_stabilization_forward(l);
            update_equaltime_GF_dn_at_stabilization_forward(l);

            // Check error in Green's function calculated by stabilization and naive product
            double error_up = check_error(Gtt_up_temp, Gttup_);
            double error_dn = check_error(Gtt_dn_temp, Gttdn_);
            double error = std::max(error_up, error_dn);
            if (error > 1e-6) {
                std::cerr << "GF error beyond threshold > 1e-6, try to reduce nstab or bigger nt" << ": " << error << std::endl;
                std::exit(1);
            }
        }
    }
}

void DQMC::sweep_beta_to_0() {
    /*
    /   Do monte carlo sweep t = β-Δτ, ..., 0
    /   For each time slice t, we propagate our Green's functions
    /   then update fields over space
    /   when we reach our local_time loc_t = 0, we do stabilization
    /
    */
    
    int loc_l;
    double acc_l;

    Matrix Bprod_up = Matrix(model_.ns(), model_.ns());
    Matrix Bprod_dn = Matrix(model_.ns(), model_.ns());

    GreenFunc Gtt_up_temp = Matrix(model_.ns(), model_.ns());
    GreenFunc Gtt_dn_temp = Matrix(model_.ns(), model_.ns());

    // Loop over time slices in reverse
    // the iteration is not starting from β, but β - Δτ
    // β - Δτ, β - 2Δτ, ..., 0
    for (int l = model_.nt() - 1; l >= 0; --l) {
        loc_l = local_l(l);

        // update ising field over space given time slice
        acc_l = model_.update_time_slice(Gttup_, Gttdn_, l);

        acc_rate_ += acc_l / model_.nt();

        propagate_equaltime_GF_up_reverse(l);
        propagate_equaltime_GF_dn_reverse(l);

        // Do the stabilization every interval time, the beginning of stack
        if (loc_l == 0) {
            Bprod_up.eye();
            Bprod_dn.eye();

            int i_stack = stack_idx(l);

            // Save Green's functions before stabilization for error check
            Gtt_up_temp = Gttup_;
            Gtt_dn_temp = Gttdn_;

            // Calculate Bprod for current stack
            calculate_Bproduct_up_naive(Bprod_up, i_stack);
            calculate_Bproduct_dn_naive(Bprod_dn, i_stack);

            // Update propagation stacks
            update_stack_up_backward(Bprod_up, i_stack);
            update_stack_dn_backward(Bprod_dn, i_stack);

            // Calculate Green's functions at the beginning of local time within stack
            update_equaltime_GF_up_at_stabilization_backward(l);
            update_equaltime_GF_dn_at_stabilization_backward(l);

            // Check numerical stability
            double error_up = check_error(Gtt_up_temp, Gttup_);
            double error_dn = check_error(Gtt_dn_temp, Gttdn_);
            double error = std::max(error_up, error_dn);
            if (error > 1e-6) {
                std::cerr << "GF error beyond threshold > 1e-6, try to reduce nstab or bigger nt" << ": " << error << std::endl;
                std::exit(1);
            }
        }
        
    }
}