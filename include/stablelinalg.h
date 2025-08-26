/*
/   Stable linear algebra library for DQMC, based on M = L.D.R decomposition
/   to avoid numerical instability when multiplying matrices.
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <armadillo>
#include <vector>

namespace stablelinalg {

    // --- LDR Class Definition ---
    class LDR {
    private:
        arma::mat L_;
        arma::vec d_;
        arma::mat R_;
        
    public:
        LDR() = default;
        LDR(const arma::mat& L, const arma::vec& d, const arma::mat& R): L_(L), d_(d), R_(R) {}
        
        const arma::mat& get_L() const { return L_; }
        const arma::vec& get_d() const { return d_; }
        const arma::mat& get_R() const { return R_; }
        
        arma::mat to_matrix() const { return L_ * arma::diagmat(d_) * R_; }
        int n_rows() const { return L_.n_rows; }
        int n_cols() const { return R_.n_cols; }
    };

    // --- Free Functions for regular Matrix Operations ---
    arma::mat diag_mul_mat(const arma::vec& diag, const arma::mat& mat);
    arma::mat mat_mul_diag(const arma::mat& mat, const arma::vec& diag);
    arma::mat I_minus_mat(const arma::mat& mat);

    // --- Free Functions for LDR Algebra ---
    LDR to_LDR(const arma::mat& M);
    LDR ldr_mul_mat(const LDR& ldr, const arma::mat& M);
    LDR mat_mul_ldr(const arma::mat& M, const LDR& ldr);
    LDR ldr_mul_ldr(const LDR& F1, const LDR& F2);
    
    arma::mat inv_I_plus_ldr(const LDR& F, double& log_det_M_out);
    arma::mat inv_I_plus_ldr_mul_ldr(const LDR& F1, const LDR& F2);
    arma::mat inv_invldr_plus_ldr(const LDR& F1, const LDR& F2);

} // namespace stablelinalg