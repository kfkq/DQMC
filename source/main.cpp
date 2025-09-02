#include <mpi.h>

#include <stablelinalg.h>
#include <lattice.h>
#include <model.h>
#include <dqmc.h>
#include <update.h>
#include <measurementh5.h>
#include <utility.h>

#include <iomanip>
#include <chrono>

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
    //                 Parameter Handling & State Management
    // -----------------------------------------------------------------
    utility::parameters params("parameters.in");
    utility::random rng(std::time(nullptr) + rank);

    bool pt_enabled = params.getBool("ParallelTempering", "enabled", false);
    double my_beta;
    int exchange_step;
    //for replica moves tracker
    int exchange_attempt = 0;
    int exchange_accepted = 0; // only track master
    int n_replicas = 1;

    if (pt_enabled) {
        std::vector<double> betas = params.getDoubleVector("ParallelTempering", "betas");
        n_replicas = betas.size();

        if (rank == master) {
            utility::io::print_info("Parallel Tempering enabled.\n");
            if (n_replicas != world_size) {
                std::cerr << "ERROR: The number of betas (" << n_replicas 
                          << ") in parameters.in must match the number of MPI processes (" 
                          << world_size << ")." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (world_size % 2 != 0) {
                 std::cerr << "ERROR: currently number of processor ( nprocs = " << world_size 
                           << ") need to be even for replica exchange" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        my_beta = betas[rank];
        exchange_step = params.getInt("ParallelTempering", "sweep_steps");
    } else {
        if (rank == master) {
            utility::io::print_info("Standard DQMC run (Parallel Tempering disabled).\n");
        }
        my_beta = params.getDouble("simulation", "beta");
    }

    // ----------------------------------------------------------------- 
    //                       DQMC Initialization
    // -----------------------------------------------------------------
    int n_sweeps = params.getInt("simulation", "n_sweeps");
    int n_therms = params.getInt("simulation", "n_therms");
    int n_bins = params.getInt("simulation", "n_bins");

    // Lattice creation
    std::array<double,2> a1{{1.0, 0.0}};
    std::array<double,2> a2{{0.0, 1.0}};
    std::vector<std::array<double,2>> orbs{{{0.0, 0.0}}};
    Lattice lat(params, a1, a2, orbs);

    // Save lattice information for analysis scripts
    if (rank == master) lat.save_info("results/info");

    // Model initialization
    AttractiveHubbard model(params, lat, rng, my_beta);
    int n_flavor = model.n_flavor();

    // DQMC simulation initialization
    DQMC sim(params, model);

    // propagation stacks and greens initialization
    std::vector<LDRStack> propagation_stacks(n_flavor);
    std::vector<GF>       greens(n_flavor);
    for (int flv = 0; flv < n_flavor; flv++) {
        propagation_stacks[flv] = sim.init_stacks(flv);
        greens[flv]             = sim.init_greenfunctions(propagation_stacks[flv]);
    }

    // measurement container
    MeasurementManager measurements(params, MPI_COMM_WORLD, rank);
    measurements.addScalar("density", Observables::calculate_density);
    measurements.addScalar("doubleOcc", Observables::calculate_doubleOccupancy);
    measurements.addScalar("swave", Observables::calculate_swavePairing);
    measurements.addEqualTime("densityCorr", Observables::calculate_densityCorr);
    measurements.addUnequalTime("greenTau", Observables::calculate_greenTau);
    measurements.addUnequalTime("doublonTau", Observables::calculate_doublonTau);
    measurements.addUnequalTime("currxxTau", Observables::calculate_currxxTau);

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
    int bin_sweeps = n_bins * n_sweeps;
    double local_time = 0.0;
    const auto t0_bin = std::chrono::steady_clock::now(); 

    for (size_t isweep = 1; isweep <= bin_sweeps; ++ isweep) {

        // replica exchange
        if (pt_enabled && (isweep % exchange_step == 0)) {
            MPI_Barrier(MPI_COMM_WORLD); // make sure all procs sync at this point
            update::replica_exchange(rank, world_size, rng,
                                    exchange_attempt, exchange_accepted, 
                                    model, sim,
                                    greens, propagation_stacks);
        }

        // basic sweep
        sim.sweep_0_to_beta(greens, propagation_stacks);
        sim.sweep_beta_to_0(greens, propagation_stacks);
        sim.sweep_unequalTime(greens, propagation_stacks);
        measurements.measure(greens, lat);

        // measurement
        if (isweep % n_sweeps == 0) {
            measurements.accumulate(lat);
        }
    }

    local_time = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_bin).count();
    
    // ----------------------------------------------------------------- 
    //                         Finalization
    // -----------------------------------------------------------------

    // Computational time details
    double total_time = 0.0;

    double local_acc_rate = sim.acc_rate() / (n_bins * 2.0 * n_sweeps + 2.0 * n_therms); // two comes from sweep back and forth in timeslice
    double total_acc_rate = 0.0;

    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_acc_rate, &total_acc_rate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    {
        const double avg_per_sweep = total_time / (n_bins * n_sweeps * world_size);
        const int total_sec = static_cast<int>(local_time);
        const int h = total_sec / 3600;
        const int m = (total_sec % 3600) / 60;
        const int s = total_sec % 60;

        utility::io::print_info(
            "DQMC measurement sweeps are finished in ",
            h, " hours ", m, " minutes ", s, " seconds.\n"
            "Average acceptance rate = ", std::fixed, std::setprecision(4), total_acc_rate, '\n',
            "Max, Mean Precision Error = ", std::scientific, std::setprecision(4), sim.max_err(), ", ", sim.mean_err(), '\n'
        );

        if (pt_enabled) {
            const double exchange_rate = static_cast<double>(exchange_accepted) / exchange_attempt;
            utility::io::print_info(
                "Parallel tempering exchange rate = ", std::fixed, std::setprecision(4), exchange_rate, '\n'
            );
        }
    }

    MPI_Finalize();

    return 0;
}
