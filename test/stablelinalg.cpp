#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <stablelinalg.h>
#include <armadillo>

// Helper function to check if two matrices are approximately equal
bool approx_equal(const arma::mat& A, const arma::mat& B, double tol = 1e-12) {
    return arma::approx_equal(A, B, "absdiff", tol);
}

TEST_CASE("LDR Decomposition and Reconstruction") {
    // Create a random, non-trivial matrix
    arma::mat original_matrix = arma::randu<arma::mat>(10, 10);
    original_matrix(1, 2) += 5.0; // Make it less uniform
    original_matrix(3, 1) -= 3.0;

    SUBCASE("to_LDR correctly decomposes and to_matrix reconstructs") {
        // Decompose the matrix into LDR form
        stablelinalg::LDR ldr_form = stablelinalg::to_LDR(original_matrix);

        // Reconstruct the matrix from the LDR form
        arma::mat reconstructed_matrix = ldr_form.to_matrix();

        // Check if the reconstructed matrix is the same as the original
        CHECK(approx_equal(original_matrix, reconstructed_matrix));
    }
}

TEST_CASE("LDR Multiplication") {
    arma::mat A = arma::randu<arma::mat>(8, 8);
    arma::mat B = arma::randu<arma::mat>(8, 8);

    stablelinalg::LDR ldr_A = stablelinalg::to_LDR(A);
    stablelinalg::LDR ldr_B = stablelinalg::to_LDR(B);

    SUBCASE("ldr_mul_mat") {
        stablelinalg::LDR ldr_AB = stablelinalg::ldr_mul_mat(ldr_A, B);
        arma::mat mat_AB = A * B;
        CHECK(approx_equal(ldr_AB.to_matrix(), mat_AB));
    }

    SUBCASE("mat_mul_ldr") {
        stablelinalg::LDR ldr_AB = stablelinalg::mat_mul_ldr(A, ldr_B);
        arma::mat mat_AB = A * B;
        CHECK(approx_equal(ldr_AB.to_matrix(), mat_AB));
    }

    SUBCASE("ldr_mul_ldr") {
        stablelinalg::LDR ldr_AB = stablelinalg::ldr_mul_ldr(ldr_A, ldr_B);
        arma::mat mat_AB = A * B;
        CHECK(approx_equal(ldr_AB.to_matrix(), mat_AB));
    }
}

TEST_CASE("Inverse and Log-Determinant") {
    arma::mat A = arma::randu<arma::mat>(6, 6);
    A = A + A.t(); // Make it symmetric to ensure it's well-behaved
    stablelinalg::LDR ldr_A = stablelinalg::to_LDR(A);

    arma::mat I = arma::eye(6, 6);
    arma::mat M = I + A;

    double log_det_M_calc;
    arma::mat G_calc = stablelinalg::inv_I_plus_ldr(ldr_A, log_det_M_calc);
    
    SUBCASE("inv_I_plus_ldr calculates the inverse correctly") {
        arma::mat G_expected = arma::inv(M);
        CHECK(approx_equal(G_calc, G_expected, 1e-10));
    }

    SUBCASE("inv_I_plus_ldr calculates the log-determinant correctly") {
        arma::cx_double log_det_M_expected_complex = arma::log_det(M);
        double log_det_M_expected = log_det_M_expected_complex.real();
        CHECK(doctest::Approx(log_det_M_calc).epsilon(1e-10) == log_det_M_expected);
    }
}