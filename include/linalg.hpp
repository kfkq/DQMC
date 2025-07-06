/*
/   This is stable linear algebra library for DQMC Hubbard model.
/
/   The center of this library is LDR object, which is decomposition of matrix
/   using QR decomposition.
/
/   its based on treating various matrix multiplication as LDR object.
/   to prevent numerical instability when multiplying long product of matrices.
/
/   Author: Muhammad Gaffar
*/


#ifndef LINALG_HPP
#define LINALG_HPP

#include <armadillo>

namespace linalg {

    // Type aliases for better readability
    using Matrix = arma::mat;
    using Vector = arma::vec;
    using GreenFunc = arma::mat;

    // Multiply diagonal matrix (represented by vector) with matrix from left: D * M
    inline Matrix diag_mul_mat(const Vector& diag, const Matrix& mat) {
        Matrix result = mat;
        for (arma::uword i = 0; i < result.n_cols; ++i) {
            result.col(i) %= diag;  // element-wise multiplication
        }
        return result;
    }

    // Multiply matrix with diagonal matrix (represented by vector) from right: M * D
    inline Matrix mat_mul_diag(const Matrix& mat, const Vector& diag) {
        Matrix result = mat;
        for (arma::uword i = 0; i < result.n_rows; ++i) {
            result.row(i) %= diag.t();  // element-wise multiplication with transposed diag
        }
        return result;
    }

    class LDR {
    private:
        Matrix L_;    // Left matrix
        Vector d_;    // Diagonal elements
        Matrix R_;    // Right matrix
        
    public:
        // Constructors
        LDR() = default;
        LDR(const Matrix& L, const Vector& d, const Matrix& R) 
            : L_(L), d_(d), R_(R) {}
        
        // Getters
        const Matrix& L() const { return L_; }
        const Vector& d() const { return d_; }
        const Matrix& R() const { return R_; }
        
        // Convert back to matrix
        inline Matrix to_matrix() const {
            return L_ * arma::diagmat(d_) * R_;
        }
        
        // Matrix dimensions
        int n_rows() const { return L_.n_rows; }
        int n_cols() const { return R_.n_cols; }

        /* ----------------------------------------------------------------------------
        /                       stable linear algebra operation
        ---------------------------------------------------------------------------- */ 

        static inline LDR from_qr(const Matrix& M) {
            /*
            / QR decomposition of a Matrix
            /   Input parameters:
            /       Matrix
            / 
            /   Return:
            /       LDR object
            */


            // QR decomposition with column pivoting
            Matrix Q, R;
            arma::uvec P;
            bool success = arma::qr(Q, R, P, M, "vector");
            
            if (!success) {
                throw std::runtime_error("QR decomposition failed");
            }
            
            // Extract diagonal elements
            Vector d = arma::abs(R.diag());
            
            // Normalize R by its diagonal
            Vector d_inv = 1.0 / d;
            Matrix R_norm = diag_mul_mat(d_inv, R);
            
            // Restore original column order
            arma::uvec P_inv = arma::sort_index(P);
            R_norm = R_norm.cols(P_inv);
            
            return LDR(Q, d, R_norm);
        }

        static LDR ldr_mul_mat(const LDR& ldr, const Matrix& M) {
            /*
            / Multiply LDR decomposition by a matrix from right: (LDR)M = L'D'R'
            /   ldr' = L * D * (R * M)
            /        = L * (D * M0) = L * M0
            /        = (L * L0) * (D0) * (R0)
            /        = L' * D' * R'
            /   
            /   Input:
            /       LDR ldr
            /       Matrix M
            /
            /   Return:
            /       LDR ldr'       
            */

            Matrix M0 = ldr.R_ * M;
            M0 = diag_mul_mat(ldr.d_, M0);
            
            LDR qr = LDR::from_qr(M0);
            
            return LDR(ldr.L_ * qr.L(), qr.d(), qr.R());
        }

        static LDR mat_mul_ldr(Matrix& M, LDR& ldr) {
            /*
            / Multiply matrix by LDR decomposition: M(LDR) = L'D'R'
            / ldr' = (M * L * D) * R = M0 * R
            /      = L0 * D0 * R0 * R
            /      = (L0) * (D0) * (R0*R) = L' * D' * R'
            /
            /   Input:
            /       Matrix M
            /       LDR ldr
            /
            /   Return:
            /       LDR ldr'
            */
            Matrix M0 = M * ldr.L_;
            M0 = mat_mul_diag(M0, ldr.d_);

            LDR qr = LDR::from_qr(M0);

            return LDR(qr.L(), qr.d(), qr.R() * ldr.R_);
                                  
        }

        static LDR ldr_mul_ldr(const LDR& ldr1, const LDR& ldr2) {
            /*
            / Multiply LDR decomposition by LDR: (L1D1R1)(L2D2R2) = L'D'R'
            /   ldr' = L1 * D1 * (R1 * L2) * D2 * R2
            /        = L1 * (D1 * M0 * D2) * R2
            /        = L1 * M0 * R2
            /        = L1 * L0 * D0 * R0 * R2
            /        = (L1 * L0) * D0 * (R0 * R2)
            /        = L' * D' * R'
            /   
            /   Input:
            /       LDR ldr1
            /       LDR ldr2
            /
            /   Return:
            /       LDR ldr'       
            */

            Matrix M0 = ldr1.R_ * ldr2.L_;
            M0 = arma::diagmat(ldr1.d_) * M0;
            M0 = mat_mul_diag(M0, ldr2.d_);
            
            LDR qr = LDR::from_qr(M0);

            return LDR(ldr1.L_ * qr.L(), qr.d(), qr.R_ * ldr2.R_);
        }

        static GreenFunc inv_eye_plus_ldr(const LDR& ldr) {
            /*
            / Compute (I + LDR)^-1 numerically stable using:
            /   G = [I + LDR]^{-1} = R^{-1} D_max^{-1} M^{-1}
            /   where M = R^{-1} D_max^{-1} + L D_min
            /
            / Input:
            /   LDR ldr
            /
            / Return:
            /   GreenFunc G
            */
            
            const int n = ldr.n_rows();
            
            // Step 1: Split D into D_min and D_max for numerical stability
            Vector d_max = ldr.d();
            Vector d_min = arma::ones(n);
            for (arma::uword i = 0; i < n; ++i) {
                if (d_max(i) < 1.0) {
                    d_min(i) = d_max(i);
                    d_max(i) = 1.0;
                }
            }
            
            // Step 2: Compute R^{-1}
            Matrix R_inv = arma::inv(ldr.R());
            
            // Step 3: Compute R^{-1} D_max^{-1}
            Vector d_max_inv = 1.0 / d_max;
            Matrix RD = mat_mul_diag(R_inv, d_max_inv);
            
            // Step 4: Compute L D_min
            Matrix LD = mat_mul_diag(ldr.L(), d_min);
            
            // Step 5: Form M = R^{-1} D_max^{-1} + L D_min
            Matrix M = RD + LD;
            
            // Step 6: Solve M X = I for X = M^{-1}
            Matrix M_inv = arma::inv(M);
            
            // Step 7: Return G = R^{-1} D_max^{-1} M^{-1}
            return RD * M_inv;
        }

        static GreenFunc inv_eye_plus_ldr_mul_ldr(const LDR& ldr1, const LDR& ldr2) {
            /*
            /  compute G = (I + F1F2)^{-1}, where F is LDR matrix
            /       G = R2^{-1} * D2_max^{-1} * M^{-1} * D1_max^{-1} * L1
            /       M = D1_max^{-1} * L1 * R2^{-1} * D2_max^{-1} + D1_min * R1 * L2 * D2_min
            */

            // Step 1: Split D into D_min and D_max for numerical stability
            int n = ldr1.n_rows();
            Vector d1_max = ldr1.d();
            Vector d1_min = arma::ones(n);
            for (arma::uword i = 0; i < n; ++i) {
                if (d1_max(i) < 1.0) {
                    d1_min(i) = d1_max(i);
                    d1_max(i) = 1.0;
                }
            }

            n = ldr2.n_rows();
            Vector d2_max = ldr2.d();
            Vector d2_min = arma::ones(n);
            for (arma::uword i = 0; i < n; ++i) {
                if (d2_max(i) < 1.0) {
                    d2_min(i) = d2_max(i);
                    d2_max(i) = 1.0;
                }
            }

            // Step 2: calculate R2^{-1} * D2_max^{-1}
            Matrix R2_inv = arma::inv(ldr2.R());
            Vector d2_max_inv = 1.0 / d2_max;
            Matrix RD2 = mat_mul_diag(R2_inv, d2_max_inv);

            // Step 3: calculate D1_max^{-1} * L1^†
            Vector d1_max_inv = 1.0 / d1_max;
            Matrix DL1 = diag_mul_mat(d1_max_inv, ldr1.L().t());

            // Step 4: calculate D1_min * R1
            Matrix DR1 = diag_mul_mat(d1_min, ldr1.R());

            // Step 5: calculate L2 * D2_min
            Matrix LD2 = mat_mul_diag(ldr2.L(), d2_min);

            // Step 6: calculate M^{-1} = (D1_max^{-1} * L1^† * R2^{-1} * D2_max^{-1} + D1_min * R1 * L2 * D2_min)^{-1}
            Matrix M = DL1 * RD2 + DR1 * LD2;
            Matrix M_inv = arma::inv(M);

            // Step 7: return G = R2^{-1} * D2_max^{-1} * M^{-1} * D1_max^{-1} * L1^†
            return RD2 * M_inv * DL1;
        }
    };

    class LDRStack {
    private:
        const size_t n_stack_;      // Number of stacks (nt/n_stab)
        std::vector<LDR> stack_;

    public:
        // Constructor takes actual stack size
        explicit LDRStack(size_t n_stack) 
            : n_stack_(n_stack), stack_(n_stack) {}
        
        // Direct access by index
        LDR& operator[](size_t idx) { 
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            return stack_[idx]; 
        }
        const LDR& operator[](size_t idx) const { 
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            return stack_[idx]; 
        }
        
        // Stack properties
        constexpr size_t size() const { return n_stack_; }
        
        // Direct assignment
        void set(size_t idx, const LDR& ldr) {
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            stack_[idx] = ldr;
        }
    };

} // namespace linalg

#endif