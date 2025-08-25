#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "dqmc.h"
#include "model/ahubbard.h"
#include "utility.h"
#include "lattice.h"
#include <vector>
#include <cmath>
#include <fstream>

// A test fixture to create a standard simulation setup for DQMC tests.
// This avoids code duplication and ensures a consistent state for each test case.
struct DQMC_Test_Fixture {
    utility::parameters params;
    utility::random rng;
    Lattice lat;
    HubbardAttractiveU model;
    DQMC sim;

    DQMC_Test_Fixture()
        // 1. Create a temporary parameters file with a small, fast-to-test configuration.
        : params([]() {
              std::ofstream pfile("test_params_dqmc.ini");
              pfile << "[lattice]\nLx = 2\nLy = 2\n"
                    << "[hubbard]\nU = 4.0\nt = 1.0\nmu = 0.0\n"
                    << "[simulation]\nbeta = 1.0\nnt = 10\nn_stab = 5\n";
              pfile.close();
              return utility::parameters("test_params_dqmc.ini");
          }()),
          // 2. Initialize all necessary simulation components.
          rng(),
          lat({1.0, 0.0}, {0.0, 1.0}, {{0.0, 0.0}},
              params.getInt("lattice", "Lx"), params.getInt("lattice", "Ly")),
          model(params, lat, rng),
          sim(params, model, rng)
    {
        // 3. Use a fixed seed for reproducible test results.
        rng.set_seed(42);
    }

    ~DQMC_Test_Fixture() {
        // 4. Clean up the temporary file after the test is done.
        std::remove("test_params_dqmc.ini");
    }
};

TEST_CASE("DQMC Engine Initialization") {
    DQMC_Test_Fixture fix;

    SUBCASE("init_stacks creates a stack of the correct size") {
        int n_flavors = fix.model.n_flavors();
        for (int flv = 0; flv < n_flavors; ++flv) {
            LDRStack stack = fix.sim.init_stacks(flv);
            int expected_n_stack = fix.params.getInt("simulation", "nt") / fix.params.getInt("simulation", "n_stab");
            CHECK(stack.size() == expected_n_stack);
        }
    }

    SUBCASE("init_greenfunctions calculates G(0,0) correctly") {
        // --- Setup ---
        int n_flavors = fix.model.n_flavors();
        int nt = fix.model.n_timesteps();
        int ns = fix.model.n_size();
        arma::mat B_beta_0 = arma::eye(ns, ns);

        // --- Naive (unstable) calculation of B(beta, 0) ---
        // This is safe only for the very small 'nt' used in this test.
        for (int l = nt - 1; l >= 0; --l) {
            arma::mat expK = fix.model.get_expK();
            arma::vec expV = fix.model.get_expV(l, 0); // flavor 0
            arma::mat B_l = stablelinalg::diag_mul_mat(expV, expK);
            B_beta_0 = B_beta_0 * B_l;
        }

        // Expected G(0,0) = [I + B(beta,0)]^-1
        arma::mat G00_expected = arma::inv(arma::eye(ns, ns) + B_beta_0);

        // --- DQMC (stable) calculation ---
        LDRStack stack = fix.sim.init_stacks(0); // flavor 0
        GF greens = fix.sim.init_greenfunctions(stack);
        arma::mat G00_calculated = greens.Gtt[0];

        // --- Comparison ---
        // Check that the stable and naive methods produce the same result.
        CHECK(arma::approx_equal(G00_calculated, G00_expected, "absdiff", 1e-9));
        CHECK(greens.Gtt.size() == nt + 1);
    }
}