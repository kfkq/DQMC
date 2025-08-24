#include "dqmc.hpp"
#include "linalg.hpp"
#include "lattice.hpp"
#include "model.hpp" 
#include "measurementh5.hpp"
#include "observables.hpp"
#include "updates.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <numeric>
#include "utility.hpp" // Moved for clarity, and needed for parameter loading

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

    // --- Parameter Loading Logic ---
    // All ranks load the main parameters.in file.
    utility::parameters params("parameters.in");

    // If PT is enabled, ranks > 0 must load an additional override file.
    bool use_pt = params.getBool("parallel_tempering", "enabled", false);
    if (use_pt && rank > 0) {
        std::string override_file = "parallel_tempering/parameters_" + std::to_string(rank) + ".in";
        if (utility::io::file_exists(override_file)) {
            utility::parameters override_params(override_file);
            params.override_with(override_params);
        } else {
            std::cerr << "FATAL ERROR: Parallel tempering is enabled, but the override file for rank "
                << rank << " (" << override_file << ") was not found." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

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
    int n_bins = params.getInt("simulation", "n_bins");
    int total_sweeps = n_bins * n_sweeps;

    bool isUnequalTime = params.getBool("simulation", "isMeasureUnequalTime", false);

    // Lattice creation
    std::array<double,2> a1{{1.0, 0.0}};
    std::array<double,2> a2{{0.0, 1.0}};
    std::vector<std::array<double,2>> orbs{{{0.0, 0.0}}};
    Lattice lat = Lattice::create_lattice(a1, a2, orbs, Lx, Ly);

    // Save lattice information for analysis
    if (rank == master) {
        // Create results directory if it doesn't exist
        struct stat info;
        if (stat("results", &info) != 0) {
            #if defined(_WIN32)
            _mkdir("results");
            #else
            mkdir("results", 0755);
            #endif
        }
        
        // Write lattice info to file
        std::string info_file = "results/info";
        std::ofstream info_out(info_file);
        if (info_out.is_open()) {
            info_out << "Lx " << Lx << "\n";
            info_out << "Ly " << Ly << "\n";
            info_out << "a1_x " << a1[0] << "\n";
            info_out << "a1_y " << a1[1] << "\n";
            info_out << "a2_x " << a2[0] << "\n";
            info_out << "a2_y " << a2[1] << "\n";
            info_out << "n_orb " << lat.n_orb() << "\n";
            info_out.close();
        }
    }

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
        measurements.addUnequalTime("doublonTau", Observables::calculate_doublonTau);
        measurements.addUnequalTime("currxxTau", Observables::calculate_currxxTau);
    }

    // ----------------------------------------------------------------- 
    //                     Start of DQMC simulation
    // -----------------------------------------------------------------

    // --- Measurement Sweeps ---
    double local_time = 0.0;
    int exchange_attempts = 0;
    int exchange_accepts = 0;
    const int pt_exchange_freq = use_pt ? params.getInt("parallel_tempering", "exchange_freq") : 0;

    utility::io::print_info("DQMC Sweeps starts ...\n");
    const auto t0_measure = std::chrono::steady_clock::now();
    for (int isweep = 1; isweep <= total_sweeps; ++isweep) {
        sim.sweep_0_to_beta(greens, propagation_stacks);
        sim.sweep_beta_to_0(greens, propagation_stacks);
        
        measurements.measure(greens, lat);
        if (isUnequalTime) {
            sim.sweep_unequalTime(greens, propagation_stacks);
            measurements.measure_unequalTime(greens, lat);
        }

        if (use_pt) {
            updates::replica_exchange(isweep, pt_exchange_freq, rank, world_size,
                                      sim, hubbard, greens, propagation_stacks,
                                      rng, exchange_attempts, exchange_accepts);
        }

        if (isweep % n_sweeps == 0) {
            measurements.accumulate(lat);
        }
    }
    local_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_measure).count();
    
    // ----------------------------------------------------------------- 
    //                         Finalization
    // -----------------------------------------------------------------

    // Computational time details
    double total_time = 0.0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int total_exchange_attempts=0, total_exchange_accepts=0;
    if(use_pt){
        MPI_Reduce(&exchange_attempts, &total_exchange_attempts, 1, MPI_INT, MPI_SUM, master, MPI_COMM_WORLD);
        MPI_Reduce(&exchange_accepts, &total_exchange_accepts, 1, MPI_INT, MPI_SUM, master, MPI_COMM_WORLD);
    }

    if (rank == master) {
        const int total_sec = static_cast<int>(total_time/world_size);
        const int h = total_sec / 3600;
        const int m = (total_sec % 3600) / 60;
        const int s = total_sec % 60;

        utility::io::print_info(
            "DQMC measurement sweeps are finished in ",
            h, " hours ", m, " minutes ", s, " seconds.\n",
            "Local update acceptance rate = ",
            std::fixed, std::setprecision(4),
            sim.acc_rate() / (2.0 * (total_sweeps)), '\n'
        );

        if(use_pt && total_exchange_attempts > 0){
            utility::io::print_info(
                "PT exchange acceptance rate = ",
                std::fixed, std::setprecision(4),
                static_cast<double>(total_exchange_accepts) / total_exchange_attempts, '\n'
            );
        }
    }

    MPI_Finalize();

    return 0;
}
