#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "model/ahubbard.h"
#include "utility.h"
#include "lattice.h"
#include <vector>
#include <cmath>

// Helper to check for approximate equality
bool approx_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

TEST_CASE("HubbardAttractiveU Model Initialization and Properties") {
    // --- 1. Setup ---
    std::ofstream pfile("test_params_model.ini");
    pfile << "[lattice]\nLx = 4\nLy = 6\n[hubbard]\nU = 4.0\nt = 1.0\nmu = -0.5\n"
          << "[simulation]\nbeta = 2.0\nnt = 20\nn_stab = 5\n";
    pfile.close();

    utility::parameters params("test_params_model.ini");
    utility::random rng;
    rng.set_seed(123);

    std::array<double, 2> a1 = {1.0, 0.0};
    std::array<double, 2> a2 = {0.0, 1.0};
    std::vector<std::array<double, 2>> orbs = {{0.0, 0.0}};
    Lattice lat(a1, a2, orbs, params.getInt("lattice", "Lx"), params.getInt("lattice", "Ly"));

    HubbardAttractiveU model(params, lat, rng);

    SUBCASE("Model correctly reports its basic properties") {
        CHECK(model.n_timesteps() == 20);
        CHECK(model.n_flavors() == 1); // Attractive Hubbard model has 2 degenerate flavors
    }

    SUBCASE("expK and invexpK matrices are correctly sized and inverses of each other") {
        const arma::mat& expK = model.get_expK();
        const arma::mat& invexpK = model.get_invexpK();
        int n_sites = lat.n_sites();

        CHECK(expK.n_rows == n_sites);
        CHECK(expK.n_cols == n_sites);
        CHECK(invexpK.n_rows == n_sites);
        CHECK(invexpK.n_cols == n_sites);

        // Check if their product is the identity matrix
        arma::mat product = expK * invexpK;
        arma::mat identity = arma::eye(n_sites, n_sites);
        CHECK(arma::approx_equal(product, identity, "absdiff", 1e-12));
    }

    SUBCASE("get_expV returns a vector of the correct size") {
        int time_slice = 5;
        int flavor = 0;
        arma::vec expV = model.get_expV(time_slice, flavor);
        CHECK(expV.n_elem == lat.n_sites());
    }

    // Cleanup
    std::remove("test_params_model.ini");
}