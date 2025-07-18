#include "dqmc.hpp"
#include "linalg.hpp"
#include "lattice.hpp"
#include "model.hpp" 
#include "measurement.hpp"

#include <toml.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

#include <mpi.h>

int main(int argc, char** argv) {
    // ----------------------------------------------------------------- 
    //                         MPI Initialization
    // -----------------------------------------------------------------

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size; 
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // identify main processor
    int master = 0;

    // ----------------------------------------------------------------- 
    //                       DQMC Initialization
    // -----------------------------------------------------------------

    // Initialize the random number generator with a seed
    utility::random::set_seed(std::time(nullptr) + rank);

    // parse parameters file
    toml::value params = toml::parse("parameters.toml");

    // lattice parameters
    std::string latt_type = toml::find<std::string>(params, "lattice", "type");
    int Lx = toml::find<int>(params, "lattice" ,"Lx");
    int Ly = toml::find<int>(params, "lattice", "Ly");

    // hubbard model parameters
    double t = toml::find<double>(params, "hubbard", "t");
    double U = toml::find<double>(params, "hubbard", "U");
    double mu = toml::find<double>(params, "hubbard", "mu");

    // dqmc simulation parameters
    double beta = toml::find<double>(params, "simulation", "beta");
    int nt = toml::find<int>(params, "simulation", "nt");
    double dtau = beta / nt;

    int n_stab = toml::find<int>(params, "simulation", "n_stab");

    int n_sweeps = toml::find<int>(params, "simulation", "n_sweeps");
    int n_therms = toml::find<int>(params, "simulation", "n_therms");
    int n_bins = toml::find<int>(params, "simulation", "n_bins");

    // Lattice initialization
    auto lat = lattice::create_lattice(latt_type, Lx, Ly);

    // Model initialization
    auto hubbard = model::HubbardAttractiveU(lat, t, U, mu, dtau, nt);
    
    // model dependent DQMC factorization. Hubbard model are factorized by spin index.
    int n_flavor = hubbard.n_flavor(); 

    // DQMC simulation initialization
    auto sim = DQMC(hubbard, n_stab);

    // propagation stacks and greens initialization
    std::vector<linalg::LDRStack> propagation_stacks(n_flavor);
    std::vector<GF>               greens(n_flavor);
    for (int nfl = 0; nfl < n_flavor; nfl++) {
        propagation_stacks[nfl] = sim.init_stacks(nfl);
        greens[nfl]             = sim.init_greenfunctions(propagation_stacks[nfl]);
    }

    if (rank == 0) {
        std::cout << 
        "=== DQMC Attractive Hubbard ===\n"
        "Lattice        : " + latt_type + "  " + std::to_string(Lx) + "×" + std::to_string(Ly) + "\n" +
        "t              : " + std::to_string(t) + "\n" +
        "U              : " + std::to_string(U) + "\n" +
        "mu             : " + std::to_string(mu) + "\n" +
        "β              : " + std::to_string(std::round(beta*1000.0)/1000.0) + "\n" +
        "Nthermal       : " + std::to_string(n_therms) + "\n" + 
        "Nsweep per bin : " + std::to_string(n_sweeps) + "\n" + 
        "Nbin           : " + std::to_string(n_bins) + "\n\n" 
        << std::flush;
    }

    // measurement container
    MeasurementManager measurements(MPI_COMM_WORLD, rank);
    measurements.addScalar("density", Observables::calculate_density);
    measurements.addScalar("doubleOcc", Observables::calculate_doubleOccupancy);
    measurements.addScalar("swave", Observables::calculate_swavePairing);
    measurements.addEqualTime("densityCorr", Observables::calculate_densityCorr);

    // ----------------------------------------------------------------- 
    //                     Start of DQMC simulation
    // -----------------------------------------------------------------

    if (rank == 0) {
        std::cout << "Start of thermalization \n" << std::flush;
    }

    // thermalization
    const auto t0_therm = std::chrono::steady_clock::now();
    for (int i = 0; i < n_therms; ++i) {
        sim.sweep_0_to_beta(greens, propagation_stacks);
        sim.sweep_beta_to_0(greens, propagation_stacks);
    }
    const auto dt_therm = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_therm).count();
    
    if (rank == 0) {
        std::cout << "Thermalization done in " + std::to_string(dt_therm) + " s\n" << std::flush;
    }

    if (rank == 0) {
        std::cout << "Start of DQMC measurement sweeps \n" << std::flush;
    }

    // measurement sweeps
    double local_time = 0.0;
    for (int ibin = 0; ibin < n_bins; ++ibin) {
        const auto t0_bin = std::chrono::steady_clock::now();   // start timer for this bin
        for (int isweep = 0; isweep < n_sweeps; ++isweep) {
            sim.sweep_0_to_beta(greens, propagation_stacks);
            measurements.measure(greens);

            sim.sweep_beta_to_0(greens, propagation_stacks);
            measurements.measure(greens);
        }
        local_time += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_bin).count();

        measurements.accumulate();
    }
    
    // ----------------------------------------------------------------- 
    //                         Finalization
    // -----------------------------------------------------------------
    
    // Final analysis
    if (rank == 0) {
        std::cout << "Final analysis: Jacknife resampling of data files \n" << std::flush;
    }
    measurements.jacknifeAnalysis();

    double total_time = 0.0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double avg_per_sweep = total_time / (n_bins * n_sweeps * world_size);

        // elapsed wall-time for the whole measurement phase
        int total_sec = static_cast<int>(local_time);   // local_time already holds the **master** bin-loop time
        int h = total_sec / 3600;
        int m = (total_sec % 3600) / 60;
        int s = total_sec % 60;

        std::cout << "DQMC measurement sweeps are finished in "
                << h << " hours "
                << m << " minutes "
                << s << " seconds.\n";

        std::cout << "Average time each sweep = "
                << std::fixed << std::setprecision(3) << avg_per_sweep << " s\n";

        // average acceptance rate (global)
        std::cout << "Average acceptance rate = "
                << std::fixed << std::setprecision(4) << sim.acc_rate() / (2.0 * (n_bins * n_sweeps + n_therms)) << "\n";
    }

    MPI_Finalize();

    return 0;
}
