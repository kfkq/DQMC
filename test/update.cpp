#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <model/ahubbard.h> // Concrete model for the test
#include <update.h>          // The functions we are testing
#include <utility.h>         // For parameters and random
#include <lattice.h>         // For creating the lattice
#include <stackngf.h>       // For GF struct
#include <dqmc.h>            // For DQMC engine to initialize state

#include <vector>
#include <cmath>
#include <iomanip>

// Helper to check for approximate equality
bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

TEST_CASE("Update Module: Local Ratio EQUAL Global Ratio for single flip") {
    // --- 1. Setup a complete simulation state ---
    // Create a dummy parameters file content
    std::ofstream pfile("test_params_update.in");
    pfile << "[lattice]\nLx = 4\nLy = 4\n[hubbard]\nU = 4.0\nt = 1.0\nmu = 0.0\n"
          << "[simulation]\nbeta = 2.0\nnt = 50\nn_stab = 10\n";
    pfile.close();

    utility::parameters params("test_params_update.in");
    utility::random rng;
    rng.set_seed(1337);

    std::array<double, 2> a1 = {1.0, 0.0};
    std::array<double, 2> a2 = {0.0, 1.0};
    std::vector<std::array<double, 2>> orbs = {{0.0, 0.0}};
    int Lx = params.getInt("lattice", "Lx");
    int Ly = params.getInt("lattice", "Ly");

    Lattice lat(a1, a2, orbs, Lx, Ly);

    HubbardAttractiveU model(params, lat, rng);
    DQMC sim(params, model, rng);

    // Initialize the state (stacks and Green's functions)
    int n_flavors = model.n_flavors();
    std::vector<LDRStack> stacks(n_flavors);
    std::vector<GF> greens(n_flavors);
    for (int i = 0; i < n_flavors; ++i) {
        stacks[i] = sim.init_stacks(i);
        greens[i] = sim.init_greenfunctions(stacks[i]);
    }

    // --- 2. Calculate the action of the original state ---
    double S_old = model.calculate_global_action(greens);
    INFO("S_old = ", S_old);

    // --- 3. Propose a single, specific local move ---
    int time_slice = 10;
    int site_idx = 3;
    int new_field = model.propose_field(time_slice, site_idx);

    // --- 4. Calculate the ratio using the FAST LOCAL update formula ---
    double R_local = model.local_update_ratio(greens, time_slice, site_idx, new_field);
    INFO("R_local = ", R_local);

    // --- 5. Calculate the ratio using the SLOW GLOBAL action method ---
    // a) Apply the proposed change to the model's fields
    model.set_field_value(time_slice, site_idx, new_field);

    // b) Re-initialize the entire simulation state from scratch to get the new G(0,0) and log_det_M
    std::vector<LDRStack> new_stacks(n_flavors);
    std::vector<GF> new_greens(n_flavors);
    for (int i = 0; i < n_flavors; ++i) {
        new_stacks[i] = sim.init_stacks(i);
        new_greens[i] = sim.init_greenfunctions(new_stacks[i]);
    }

    // c) Calculate the new total action
    double S_new = model.calculate_global_action(new_greens);
    INFO("S_new = ", S_new);

    // d) The ratio of Boltzmann weights is exp((S_new - S_old))
    double R_global = std::exp(S_new - S_old);
    INFO("R_global = ", R_global);

    // --- 6. Compare the results ---
    CHECK(approx_equal(R_local, R_global));

    // Cleanup the dummy file
    std::remove("test_params_update.in");
}