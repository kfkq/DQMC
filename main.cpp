#include "dqmc.hpp"
#include "linalg.hpp"
#include "lattice.hpp"
#include "model.hpp" 
#include "measurement.hpp"
#include "observables.hpp"

#include "utility.hpp"
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
    utility::random rng;
    rng.set_seed(std::time(nullptr) + rank);

    // parse parameters file
    utility::parameters params("parameters.in");

    // lattice parameters
    std::string latt_type = params.getString("lattice", "type");
    int Lx = params.getInt("lattice", "Lx");
    int Ly = params.getInt("lattice", "Ly");

    // hubbard model parameters
    double t = params.getDouble("hubbard", "t");
    double U = params.getDouble("hubbard", "U");
    double mu = params.getDouble("hubbard", "mu");

    // dqmc simulation parameters
    double beta = params.getDouble("simulation", "beta");
    int nt = params.getInt("simulation", "nt");
    double dtau = beta / nt;

    int n_stab = params.getInt("simulation", "n_stab");

    int n_sweeps = params.getInt("simulation", "n_sweeps");
    int n_therms = params.getInt("simulation", "n_therms");
    int n_bins = params.getInt("simulation", "n_bins");

    bool isUnequalTime = params.getBool("simulation", "isMeasureUnequalTime", false);

    // Lattice creation
    std::array<double,2> a1{{1.0, 0.0}};
    std::array<double,2> a2{{0.0, 1.0}};
    std::vector<std::array<double,2>> orbs{{{0.0, 0.0}}};
    Lattice lat = Lattice::create_lattice(a1, a2, orbs, Lx, Ly);

    // Model initialization
    auto hubbard = model::HubbardAttractiveU(lat, t, U, mu, dtau, nt, rng);
    
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

    utility::io::print_info(
        "-------- Hamiltonian Parameters -------- \n"
        "Hamiltonian name       : Attractive Hubbard \n",
        "Lattice                : ", latt_type, " ", Lx, "×", Ly, '\n',
        "t                      : ", t, '\n',
        "U                      : ", U, '\n',
        "mu                     : ", mu, '\n',
        "β                      : ", beta, '\n',
        "------ Numerical & QMC Parameters ------ \n"
        "Trotter Discretization : ", dtau, '\n',
        "N of Imaginary Time    : ", nt, '\n',
        "Stabilization interval : ", n_stab, '\n',
        "N thermalization sweeps: ", n_therms, '\n',
        "N sweeps per bin       : ", n_sweeps, '\n',
        "N bins                 : ", n_bins, "\n"
        "Measure UnequalTime ?  : ", isUnequalTime, "\n\n"
    );

    // measurement container
    MeasurementManager measurements(MPI_COMM_WORLD, rank);
    measurements.addScalar("density", Observables::calculate_density);
    measurements.addScalar("doubleOcc", Observables::calculate_doubleOccupancy);
    measurements.addScalar("swave", Observables::calculate_swavePairing);
    measurements.addEqualTime("densityCorr", Observables::calculate_densityCorr);
    if (isUnequalTime) {
        measurements.addUnequalTime("greenTau", Observables::calculate_greenTau);
        measurements.addUnequalTime("currxxTau", Observables::calculate_currxxTau);
    }

    // ----------------------------------------------------------------- 
    //                     Start of DQMC simulation
    // -----------------------------------------------------------------

    // thermalization
    const auto t0_therm = std::chrono::steady_clock::now();
    for (int i = 0; i < n_therms; ++i) {
        sim.sweep_0_to_beta(greens, propagation_stacks);
        sim.sweep_beta_to_0(greens, propagation_stacks);
    }
    const auto dt_therm = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_therm).count();
    
    utility::io::print_info("Thermalization done in ", dt_therm, " seconds\n");

    // measurement sweeps
    double local_time = 0.0;
    for (int ibin = 0; ibin < n_bins; ++ibin) {
        const auto t0_bin = std::chrono::steady_clock::now();   // start timer for this bin

        for (int isweep = 0; isweep < n_sweeps; ++isweep) {
            sim.sweep_0_to_beta(greens, propagation_stacks);
            measurements.measure(greens, lat);

            sim.sweep_beta_to_0(greens, propagation_stacks);
            measurements.measure(greens, lat);

            if (isUnequalTime) {
                // do sweep without updating HS field for unequal time measurements.
                sim.sweep_unequalTime(greens, propagation_stacks);
                measurements.measure_unequalTime(greens, lat);
            }
        }
        measurements.accumulate(lat);

        local_time += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_bin).count();
    }
    
    // ----------------------------------------------------------------- 
    //                         Finalization
    // -----------------------------------------------------------------
    
    // Final analysis
    utility::io::print_info("Final Data Analysis: Fourier Transform + Jacknife Analysis \n");
    measurements.fourierTransform(lat);
    measurements.jacknifeAnalysis();

    // Computational time details
    double total_time = 0.0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    {
        const double avg_per_sweep = total_time / (n_bins * n_sweeps * world_size);
        const int total_sec = static_cast<int>(local_time);
        const int h = total_sec / 3600;
        const int m = (total_sec % 3600) / 60;
        const int s = total_sec % 60;

        utility::io::print_info(
            "DQMC measurement sweeps are finished in ",
            h, " hours ", m, " minutes ", s, " seconds.\n"
            "Average time each sweep = ",
            std::fixed, std::setprecision(3), avg_per_sweep, " s\n"
            "Average acceptance rate = ",
            std::fixed, std::setprecision(4),
            sim.acc_rate() / (2.0 * (n_bins * n_sweeps + n_therms)), '\n'
        );
    }

    MPI_Finalize();

    return 0;
}