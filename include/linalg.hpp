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
        // Matrix result = mat;
        // for (arma::uword i = 0; i < result.n_cols; ++i) {
        //     result.col(i) %= diag;  // element-wise multiplication
        // }
        // return result;
        return arma::diagmat(diag) * mat;
    }

    // Multiply matrix with diagonal matrix (represented by vector) from right: M * D
    inline Matrix mat_mul_diag(const Matrix& mat, const Vector& diag) {
        // Matrix result = mat;
        // for (arma::uword i = 0; i < result.n_rows; ++i) {
        //     result.row(i) %= diag.t();  // element-wise multiplication with transposed diag
        // }
        // return result;
        return mat * arma::diagmat(diag);
    }

    inline GreenFunc eye_minus_mat(const Matrix mat) {
        /*
        / Compute (I - G)
        /
        */

        const int ns = mat.n_rows;

        return arma::eye(ns, ns) - mat;
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
            / Decomposes a general matrix M into the LDR format (M = LDR) using a pivoted QR decomposition.
            / This is the fundamental building block for stable matrix products.
            /
            /   1. Perform pivoted QR decomposition: M * P = Q * R, where P is a permutation matrix.
            /   2. Extract the diagonal scales: D = |diag(R)|.
            /   3. Normalize R: R_norm = D⁻¹ * R.
            /   4. Un-apply the pivoting from R_norm to restore the original column order.
            /   5. The final decomposition is M = Q * D * (R_norm * P⁻¹).
            /      We define L = Q and the new R = R_norm * P⁻¹.
            /
            / Input:
            /   M: The matrix to be decomposed.
            /
            / Return:
            /   An LDR object representing M.
            */

            // Step 1: Perform QR decomposition with column pivoting.
            // M * P = Q * R, where P is a permutation matrix that improves numerical stability
            // by moving large columns to the left. `P_vec` stores the permutation indices.
            Matrix Q_mat, R_mat;
            arma::uvec P_vec; // Stores the column permutation vector
            bool success = arma::qr(Q_mat, R_mat, P_vec, M, "vector");
            
            if (!success) {
                throw std::runtime_error("QR decomposition failed in from_qr");
            }
            
            // Step 2: Extract the diagonal elements of R as the scaling matrix D.
            // We take the absolute value as the sign is absorbed into the R matrix.
            Vector D_vec = arma::abs(R_mat.diag());
            
            // Step 3: Normalize the R matrix by the diagonal scales.
            // This makes the diagonal of the new R matrix consist of +1 or -1.
            Vector D_inv_vec = 1.0 / D_vec;
            Matrix R_normalized = diag_mul_mat(D_inv_vec, R_mat);
            
            // Step 4: Undo the column pivoting to restore the original matrix structure.
            // We need to apply the inverse permutation P⁻¹ to the columns of R_normalized.
            arma::uvec P_inv_vec = arma::sort_index(P_vec);
            Matrix R_final = R_normalized.cols(P_inv_vec);
            
            // Return the LDR object where L=Q, D=d, and R is the normalized, un-pivoted R matrix.
            return LDR(Q_mat, D_vec, R_final);
        }

        static LDR ldr_mul_mat(const LDR& ldr, const Matrix& M) {
            /*
            / Stably multiplies an LDR matrix by a standard matrix from the right: (LDR) * M.
            /
            / The operation is grouped as L * (D*R*M). The inner part is re-decomposed
            / into a new L'D'R' to maintain stability, yielding (L*L') * D' * R'.
            */

            // Step 1: Calculate the inner part of the product: M_prime = D*R*M
            Matrix M_prime = ldr.R() * M;
            M_prime = diag_mul_mat(ldr.d(), M_prime);
            
            // Step 2: Re-decompose the intermediate matrix M_prime into a stable L'D'R' form.
            LDR qr_decomp = LDR::from_qr(M_prime);
            
            // Step 3: Combine to form the final LDR object: (L * L') * D' * R'
            return LDR(ldr.L() * qr_decomp.L(), qr_decomp.d(), qr_decomp.R());
        }

        static LDR mat_mul_ldr(const Matrix& M, const LDR& ldr) {
            /*
            / Stably multiplies a standard matrix by an LDR matrix from the left: M * (LDR).
            /
            / The operation is grouped as (M*L*D) * R. The inner part is re-decomposed
            / into a new L'D'R' to maintain stability, yielding L' * D' * (R'*R).
            */

            // Step 1: Calculate the inner part of the product: M_prime = M*L*D
            Matrix M_prime = M * ldr.L();
            M_prime = mat_mul_diag(M_prime, ldr.d());

            // Step 2: Re-decompose the intermediate matrix M_prime into a stable L'D'R' form.
            LDR qr_decomp = LDR::from_qr(M_prime);

            // Step 3: Combine to form the final LDR object: L' * D' * (R' * R)
            return LDR(qr_decomp.L(), qr_decomp.d(), qr_decomp.R() * ldr.R());
        }

        static LDR ldr_mul_ldr(const LDR& F1, const LDR& F2) {
            /*
            / Stably multiplies two LDR matrices: F₁ * F₂.
            /
            / The operation is grouped as L₁ * (D₁*R₁*L₂*D₂) * R₂. The inner part is
            / re-decomposed to maintain stability, yielding (L₁*L') * D' * (R'*R₂).
            */

            // Step 1: Calculate the inner part of the product: M_prime = D₁*R₁*L₂*D₂
            Matrix M_prime = F1.R() * F2.L();
            M_prime = arma::diagmat(F1.d()) * M_prime;
            M_prime = mat_mul_diag(M_prime, F2.d());
            
            // Step 2: Re-decompose the intermediate matrix M_prime into a stable L'D'R' form.
            LDR qr_decomp = LDR::from_qr(M_prime);

            // Step 3: Combine to form the final LDR object: (L₁*L') * D' * (R'*R₂)
            return LDR(F1.L() * qr_decomp.L(), qr_decomp.d(), qr_decomp.R() * F2.R());
        }

        static GreenFunc inv_eye_plus_ldr(const LDR& F) {
            /*
            / Computes G = [I + F]⁻¹ in a numerically stable way, where F = LDR.
            / This is used for calculating the equal-time Green's function G(0,0) or G(β,β).
            /
            / The stable formula, inspired by the ALF library, is:
            /   G = (R⁻¹ D_large⁻¹) * M⁻¹
            /   where M = (R⁻¹ D_large⁻¹) + (L * D_small)
            */
            
            const int n_sites = F.n_rows();
            
            // --- Step 1: Separate the diagonal matrix D into large (>=1) and small (<1) scales ---
            // This is the core of the stabilization. We handle numbers with large and small
            // magnitudes separately to avoid floating-point precision loss.
            Vector D_large = arma::ones(n_sites);
            Vector D_small = arma::ones(n_sites);
            for (arma::uword i = 0; i < n_sites; ++i) {
                if (F.d()(i) >= 1.0) {
                    D_large(i) = F.d()(i);
                } else {
                    D_small(i) = F.d()(i);
                }
            }
            
            // --- Step 2: Compute the term involving large scales: R⁻¹ * D_large⁻¹ ---
            // We solve the linear system R * X = D_large⁻¹ for X, which is more stable
            // than explicitly calculating the inverse of R.
            Vector D_large_inv = 1.0 / D_large;
            Matrix R_inv_D_large_inv;
            arma::solve(R_inv_D_large_inv, F.R(), arma::diagmat(D_large_inv));
            
            // --- Step 3: Compute the term involving small scales: L * D_small ---
            Matrix L_D_small = mat_mul_diag(F.L(), D_small);
            
            // --- Step 4: Form the intermediate matrix M ---
            // M is constructed by adding the two well-conditioned terms. This matrix M
            // is much more numerically stable to invert than the original (I + LDR).
            Matrix M = R_inv_D_large_inv + L_D_small;
            
            // --- Step 5: Solve for the final Green's function G ---
            // We want to compute G = (R⁻¹ * D_large⁻¹) * M⁻¹.
            // This is equivalent to solving the system G * M = (R⁻¹ * D_large⁻¹).
            // To use arma::solve (which solves A*X=B), we transpose the system:
            // Mᵀ * Gᵀ = (R⁻¹ * D_large⁻¹)ᵀ
            GreenFunc G_transpose;
            arma::solve(G_transpose, M.t(), R_inv_D_large_inv.t());
            
            // Return the transpose of the solution to get the final G.
            return G_transpose.t();
        }

        static GreenFunc inv_eye_plus_ldr_mul_ldr(const LDR& F1, const LDR& F2) {
            /*
            / Computes G = [I + F₁F₂]⁻¹ in a numerically stable way, where F₁ and F₂ are LDR matrices.
            / This is used for calculating the equal-time Green's function G(τ,τ).
            /
            / The stable formula, inspired by the ALF library, is:
            /   G = (R₂⁻¹ D₂,max⁻¹) * M⁻¹ * (D₁,max⁻¹ L₁ᵀ)
            /   M = (D₁,max⁻¹ L₁ᵀ R₂⁻¹ D₂,max⁻¹) + (D₁,min R₁ L₂ D₂,min)
            */

            const int n_sites = F1.n_rows();

            // --- Step 1: Separate diagonal matrices D₁ and D₂ into large and small scales ---
            // This is the core of the stabilization. We handle numbers >= 1.0 (large scales)
            // and numbers < 1.0 (small scales) separately to avoid precision loss.

            // Separate scales for F₁ = L₁D₁R₁
            Vector D1_large = arma::ones(n_sites);
            Vector D1_small = arma::ones(n_sites);
            for (arma::uword i = 0; i < n_sites; ++i) {
                if (F1.d()(i) >= 1.0) {
                    D1_large(i) = F1.d()(i);
                } else {
                    D1_small(i) = F1.d()(i);
                }
            }

            // Separate scales for F₂ = L₂D₂R₂
            Vector D2_large = arma::ones(n_sites);
            Vector D2_small = arma::ones(n_sites);
            for (arma::uword i = 0; i < n_sites; ++i) {
                if (F2.d()(i) >= 1.0) {
                    D2_large(i) = F2.d()(i);
                } else {
                    D2_small(i) = F2.d()(i);
                }
            }

            // --- Step 2: Calculate the two terms that form the intermediate matrix M ---

            // Pre-calculate the common term (R₂⁻¹ * D₂,max⁻¹) by solving the linear system
            // R₂ * X = D₂,max⁻¹. This is more stable than explicitly inverting R₂.
            Matrix R2_inv_D2_large_inv;
            arma::solve(R2_inv_D2_large_inv, F2.R(), arma::diagmat(1.0 / D2_large));

            // Calculate Term A of M: (D₁,max⁻¹ * L₁ᵀ) * (R₂⁻¹ * D₂,max⁻¹)
            Matrix TermA = diag_mul_mat(1.0 / D1_large, F1.L().t() * R2_inv_D2_large_inv);

            // Calculate Term B of M: (D₁,min * R₁) * (L₂ * D₂,min)
            Matrix TermB = diag_mul_mat(D1_small, F1.R() * mat_mul_diag(F2.L(), D2_small));

            // Form the intermediate matrix M
            Matrix M = TermA + TermB;

            // --- Step 3: Solve for the final Green's function G ---

            // Define the right-hand side for the final solve: RHS = D₁,max⁻¹ * L₁ᵀ
            Matrix RHS_for_M_inv_solve = diag_mul_mat(1.0 / D1_large, F1.L().t());

            // Solve the system M * Y = RHS to get Y = M⁻¹ * RHS
            Matrix Y;
            arma::solve(Y, M, RHS_for_M_inv_solve);

            // The final result is G = (R₂⁻¹ * D₂,max⁻¹) * Y
            return R2_inv_D2_large_inv * Y;
        }

        static GreenFunc inv_invldr_plus_ldr(const LDR& F1, const LDR& F2) {
            /*
            / Computes G = [F₁⁻¹ + F₂]⁻¹ in a numerically stable way.
            / This is typically used to calculate the time-displaced Green's function G(τ,0).
            /
            / The stable formula is:
            /   G = (R₂⁻¹ D₂,max⁻¹) * M⁻¹ * (D₁,min R₁)
            /   M = (D₁,max⁻¹ L₁ᵀ R₂⁻¹ D₂,max⁻¹) + (D₁,min R₁ L₂ D₂,min)
            */

            const int n_sites = F1.n_rows();

            // --- Step 1: Separate diagonal matrices D₁ and D₂ into large and small scales ---
            // This part is identical to the function above.

            // Separate scales for F₁ = L₁D₁R₁
            Vector D1_large = arma::ones(n_sites);
            Vector D1_small = arma::ones(n_sites);
            for (arma::uword i = 0; i < n_sites; ++i) {
                if (F1.d()(i) >= 1.0) {
                    D1_large(i) = F1.d()(i);
                } else {
                    D1_small(i) = F1.d()(i);
                }
            }

            // Separate scales for F₂ = L₂D₂R₂
            Vector D2_large = arma::ones(n_sites);
            Vector D2_small = arma::ones(n_sites);
            for (arma::uword i = 0; i < n_sites; ++i) {
                if (F2.d()(i) >= 1.0) {
                    D2_large(i) = F2.d()(i);
                } else {
                    D2_small(i) = F2.d()(i);
                }
            }

            // --- Step 2: Calculate the two terms that form the intermediate matrix M ---
            // This part is also identical to the function above.

            // Pre-calculate the common term (R₂⁻¹ * D₂,max⁻¹)
            Matrix R2_inv_D2_large_inv;
            arma::solve(R2_inv_D2_large_inv, F2.R(), arma::diagmat(1.0 / D2_large));

            // Calculate Term A of M: (D₁,max⁻¹ * L₁ᵀ) * (R₂⁻¹ * D₂,max⁻¹)
            Matrix TermA = diag_mul_mat(1.0 / D1_large, F1.L().t() * R2_inv_D2_large_inv);

            // Calculate Term B of M: (D₁,min * R₁) * (L₂ * D₂,min)
            Matrix TermB = diag_mul_mat(D1_small, F1.R() * mat_mul_diag(F2.L(), D2_small));

            // Form the intermediate matrix M
            Matrix M = TermA + TermB;

            // --- Step 3: Solve for the final Green's function G ---
            // This is the only part that differs from the previous function.

            // Define the right-hand side for the final solve: RHS = D₁,min * R₁
            Matrix RHS_for_M_inv_solve = diag_mul_mat(D1_small, F1.R());

            // Solve the system M * Y = RHS to get Y = M⁻¹ * RHS
            Matrix Y;
            arma::solve(Y, M, RHS_for_M_inv_solve);

            // The final result is G = (R₂⁻¹ * D₂,max⁻¹) * Y
            return R2_inv_D2_large_inv * Y;
        }
    };

    class LDRStack {
    private:
        size_t n_stack_ = 0;                // Changed from const to mutable
        std::vector<LDR> stack_;

    public:
        // default constructor
        LDRStack() = default;

        // Constructor
        explicit LDRStack(size_t n_stack) 
            : n_stack_(n_stack), stack_(n_stack) {}
        
        // move operations
        LDRStack(LDRStack&& other) noexcept
            : n_stack_(other.n_stack_),
            stack_(std::move(other.stack_)) {}
        
        LDRStack& operator=(LDRStack&& other) noexcept {
            if (this != &other) {
                n_stack_ = other.n_stack_;
                stack_ = std::move(other.stack_);
            }
            return *this;
        }
        
        // Disallow copying (if not needed)
        LDRStack(const LDRStack&) = delete;
        LDRStack& operator=(const LDRStack&) = delete;

        // Element access
        LDR& operator[](size_t idx) { 
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            return stack_[idx]; 
        }
        
        const LDR& operator[](size_t idx) const { 
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            return stack_[idx]; 
        }
        
        // Capacity
        constexpr size_t size() const noexcept { return n_stack_; }
        
        // Modification
        void set(size_t idx, const LDR& ldr) {
            if (idx >= n_stack_) throw std::out_of_range("LDR Stack index out of bounds");
            stack_[idx] = ldr;
        }
    };

} // namespace linalg

#endif