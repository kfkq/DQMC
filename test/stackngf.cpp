#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "stackngf.h"
#include "stablelinalg.h" // For stablelinalg::LDR
#include <armadillo>
#include <vector>

// Helper to check for approximate equality of matrices
bool approx_equal_mat(const arma::mat& A, const arma::mat& B, double tol = 1e-12) {
    return arma::approx_equal(A, B, "absdiff", tol);
}

TEST_CASE("GF Struct Functionality") {
    SUBCASE("Default constructor initializes log_det_M") {
        GF gf;
        CHECK(gf.log_det_M == 0.0);
        CHECK(gf.Gtt.empty());
        CHECK(gf.Gt0.empty());
        CHECK(gf.G0t.empty());
    }

    SUBCASE("Resizing Gtt, Gt0, G0t works correctly") {
        GF gf;
        int nt = 5;
        int ns = 3;

        gf.Gtt.resize(nt + 1, arma::mat(ns, ns, arma::fill::zeros));
        gf.Gt0.resize(nt + 1, arma::mat(ns, ns, arma::fill::zeros));
        gf.G0t.resize(nt + 1, arma::mat(ns, ns, arma::fill::zeros));

        CHECK(gf.Gtt.size() == nt + 1);
        CHECK(gf.Gt0.size() == nt + 1);
        CHECK(gf.G0t.size() == nt + 1);

        CHECK(gf.Gtt[0].n_rows == ns);
        CHECK(gf.Gtt[0].n_cols == ns);
    }
}

TEST_CASE("LDRStack Class Functionality") {
    const size_t n_stack_size = 5;
    const int matrix_dim = 3;

    // Create some dummy LDR objects for testing
    stablelinalg::LDR ldr1(arma::eye(matrix_dim, matrix_dim), arma::ones<arma::vec>(matrix_dim), arma::eye(matrix_dim, matrix_dim));
    stablelinalg::LDR ldr2(arma::randu<arma::mat>(matrix_dim, matrix_dim), arma::randu<arma::vec>(matrix_dim), arma::randu<arma::mat>(matrix_dim, matrix_dim));

    SUBCASE("Constructor initializes with correct size") {
        LDRStack stack(n_stack_size);
        CHECK(stack.size() == n_stack_size);
    }

    SUBCASE("Element access (operator[]) works correctly") {
        LDRStack stack(n_stack_size);
        stack.set(0, ldr1);
        stack.set(1, ldr2);

        CHECK(approx_equal_mat(stack[0].to_matrix(), ldr1.to_matrix()));
        CHECK(approx_equal_mat(stack[1].to_matrix(), ldr2.to_matrix()));

        // Check const access
        const LDRStack& const_stack = stack;
        CHECK(approx_equal_mat(const_stack[0].to_matrix(), ldr1.to_matrix()));
    }

    SUBCASE("set() method updates elements") {
        LDRStack stack(n_stack_size);
        stack.set(2, ldr1);
        CHECK(approx_equal_mat(stack[2].to_matrix(), ldr1.to_matrix()));

        stack.set(2, ldr2); // Overwrite
        CHECK(approx_equal_mat(stack[2].to_matrix(), ldr2.to_matrix()));
    }

    SUBCASE("Move constructor transfers ownership") {
        LDRStack original_stack(n_stack_size);
        original_stack.set(0, ldr1);
        original_stack.set(1, ldr2);

        LDRStack moved_stack = std::move(original_stack);

        // CHECK 1: The new object has the correct state.
        CHECK(moved_stack.size() == n_stack_size);
        CHECK(approx_equal_mat(moved_stack[0].to_matrix(), ldr1.to_matrix()));
        CHECK(approx_equal_mat(moved_stack[1].to_matrix(), ldr2.to_matrix()));

        // CHECK 2 (Optional but good): The original object is in a valid state.
        // We don't check its size or contents, but we can test that we can,
        // for example, assign a new object to it without crashing.
        LDRStack another_stack(1);
        original_stack = std::move(another_stack);
        CHECK(original_stack.size() == 1);
    }

    SUBCASE("Move assignment operator transfers ownership") {
        LDRStack original_stack(n_stack_size);
        original_stack.set(0, ldr1);

        LDRStack target_stack(n_stack_size + 1); // Different size to ensure old data is cleared
        target_stack.set(0, ldr2);

        target_stack = std::move(original_stack);

        // CHECK 1: The target object has the correct state.
        CHECK(target_stack.size() == n_stack_size);
        CHECK(approx_equal_mat(target_stack[0].to_matrix(), ldr1.to_matrix()));

        // CHECK 2 (Optional but good): The original object is in a valid state.
        LDRStack another_stack(2);
        original_stack = std::move(another_stack);
        CHECK(original_stack.size() == 2);
    }

    SUBCASE("Out of range access throws exception") {
        LDRStack stack(n_stack_size);
        CHECK_THROWS_AS(stack[n_stack_size], std::out_of_range);
        CHECK_THROWS_AS(stack.set(n_stack_size, ldr1), std::out_of_range);
    }
}