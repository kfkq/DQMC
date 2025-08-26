#include <stablelinalg.h>

namespace stablelinalg {

// ----------------------------------------------------------------------------
// basic matrix operations
// ----------------------------------------------------------------------------

arma::mat diag_mul_mat(const arma::vec& diag, const arma::mat& mat) {
    /*
        D * M
    */
    return arma::diagmat(diag) * mat;
}

arma::mat mat_mul_diag(const arma::mat& mat, const arma::vec& diag) {
    /*
        M * D
    */
    return mat * arma::diagmat(diag);
}

arma::mat I_minus_mat(const arma::mat& mat) {
    /*
        I - M
    */
    const arma::uword ns = mat.n_rows;
    return arma::eye(ns, ns) - mat;
}

// ----------------------------------------------------------------------------
// stable linear algebra operations
// ----------------------------------------------------------------------------

LDR to_LDR(const arma::mat& M) {
    /*
        M -> F = L * d * R using QR decomposition
    */
    arma::mat Q_mat, R_mat;
    arma::uvec P_vec;
    bool success = arma::qr(Q_mat, R_mat, P_vec, M, "vector");
    
    if (!success) {
        throw std::runtime_error("QR decomposition failed in to_LDR");
    }
    
    arma::vec D_vec = arma::abs(R_mat.diag());
    arma::vec D_inv_vec = 1.0 / D_vec;
    arma::mat R_normalized = diag_mul_mat(D_inv_vec, R_mat);
    
    arma::uvec P_inv_vec = arma::sort_index(P_vec);
    arma::mat R_final = R_normalized.cols(P_inv_vec);
    
    return LDR(Q_mat, D_vec, R_final);
}

LDR ldr_mul_mat(const LDR& ldr, const arma::mat& M) {
    /*
        F'  = F * M
    */
    arma::mat M_prime = ldr.get_R() * M;
    M_prime = diag_mul_mat(ldr.get_d(), M_prime);
    
    LDR qr_decomp = to_LDR(M_prime);
    
    return LDR(ldr.get_L() * qr_decomp.get_L(), qr_decomp.get_d(), qr_decomp.get_R());
}

LDR mat_mul_ldr(const arma::mat& M, const LDR& ldr) {
    /*
        F' = M * F
    */
    arma::mat M_prime = M * ldr.get_L();
    M_prime = mat_mul_diag(M_prime, ldr.get_d());

    LDR qr_decomp = to_LDR(M_prime);

    return LDR(qr_decomp.get_L(), qr_decomp.get_d(), qr_decomp.get_R() * ldr.get_R());
}

LDR ldr_mul_ldr(const LDR& F1, const LDR& F2) {
    /*
        F' = F1 * F2
    */
    arma::mat M_prime = F1.get_R() * F2.get_L();
    M_prime = diag_mul_mat(F1.get_d(), M_prime);
    M_prime = mat_mul_diag(M_prime, F2.get_d());
    
    LDR qr_decomp = to_LDR(M_prime);

    return LDR(F1.get_L() * qr_decomp.get_L(), qr_decomp.get_d(), qr_decomp.get_R() * F2.get_R());
}

arma::mat inv_I_plus_ldr(const LDR& F, double& logdetM) {
    /*
        G = [1 + F]^-1
    */
    const int n_sites = F.n_rows();
    
    arma::vec D_large = arma::ones(n_sites);
    arma::vec D_small = arma::ones(n_sites);
    for (arma::uword i = 0; i < n_sites; ++i) {
        if (F.get_d()(i) >= 1.0) {
            D_large(i) = F.get_d()(i);
        } else {
            D_small(i) = F.get_d()(i);
        }
    }
    
    arma::vec D_large_inv = 1.0 / D_large;
    arma::mat R_inv_D_large_inv;
    arma::solve(R_inv_D_large_inv, F.get_R(), arma::diagmat(D_large_inv));
    
    arma::mat L_D_small = mat_mul_diag(F.get_L(), D_small);
    
    arma::mat M = R_inv_D_large_inv + L_D_small;

    double log_det_D_large = arma::sum(arma::log(D_large));
    arma::cx_double log_det_M_complex = arma::log_det(M);
    logdetM = log_det_D_large + log_det_M_complex.real();
    
    arma::mat G_transpose;
    arma::solve(G_transpose, M.t(), R_inv_D_large_inv.t());
    
    return G_transpose.t();
}

arma::mat inv_I_plus_ldr_mul_ldr(const LDR& F1, const LDR& F2) {
    /*
        G = [I + F1*F2]^-1
    */
    const int n_sites = F1.n_rows();

    arma::vec D1_large = arma::ones(n_sites);
    arma::vec D1_small = arma::ones(n_sites);
    for (arma::uword i = 0; i < n_sites; ++i) {
        if (F1.get_d()(i) >= 1.0) { D1_large(i) = F1.get_d()(i); } else { D1_small(i) = F1.get_d()(i); }
    }

    arma::vec D2_large = arma::ones(n_sites);
    arma::vec D2_small = arma::ones(n_sites);
    for (arma::uword i = 0; i < n_sites; ++i) {
        if (F2.get_d()(i) >= 1.0) { D2_large(i) = F2.get_d()(i); } else { D2_small(i) = F2.get_d()(i); }
    }

    arma::mat R2_inv_D2_large_inv;
    arma::solve(R2_inv_D2_large_inv, F2.get_R(), arma::diagmat(1.0 / D2_large));

    arma::mat TermA = diag_mul_mat(1.0 / D1_large, F1.get_L().t() * R2_inv_D2_large_inv);
    arma::mat TermB = diag_mul_mat(D1_small, F1.get_R() * mat_mul_diag(F2.get_L(), D2_small));
    arma::mat M = TermA + TermB;

    arma::mat RHS_for_M_inv_solve = diag_mul_mat(1.0 / D1_large, F1.get_L().t());
    arma::mat Y;
    arma::solve(Y, M, RHS_for_M_inv_solve);

    return R2_inv_D2_large_inv * Y;
}

arma::mat inv_invldr_plus_ldr(const LDR& F1, const LDR& F2) {
    /*
        G = [F1^-1 + F2]^-1
    */
    const int n_sites = F1.n_rows();

    arma::vec D1_large = arma::ones(n_sites);
    arma::vec D1_small = arma::ones(n_sites);
    for (arma::uword i = 0; i < n_sites; ++i) {
        if (F1.get_d()(i) >= 1.0) { D1_large(i) = F1.get_d()(i); } else { D1_small(i) = F1.get_d()(i); }
    }

    arma::vec D2_large = arma::ones(n_sites);
    arma::vec D2_small = arma::ones(n_sites);
    for (arma::uword i = 0; i < n_sites; ++i) {
        if (F2.get_d()(i) >= 1.0) { D2_large(i) = F2.get_d()(i); } else { D2_small(i) = F2.get_d()(i); }
    }

    arma::mat R2_inv_D2_large_inv;
    arma::solve(R2_inv_D2_large_inv, F2.get_R(), arma::diagmat(1.0 / D2_large));

    arma::mat TermA = diag_mul_mat(1.0 / D1_large, F1.get_L().t() * R2_inv_D2_large_inv);
    arma::mat TermB = diag_mul_mat(D1_small, F1.get_R() * mat_mul_diag(F2.get_L(), D2_small));
    arma::mat M = TermA + TermB;

    arma::mat RHS_for_M_inv_solve = diag_mul_mat(D1_small, F1.get_R());
    arma::mat Y;
    arma::solve(Y, M, RHS_for_M_inv_solve);

    return R2_inv_D2_large_inv * Y;
}

} // namespace