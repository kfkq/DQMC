#include "dqmc.hpp"
#include "linalg.hpp"
#include "lattice.hpp"
#include "model.hpp" 
#include "measurementh5.hpp"
#include "observables.hpp"
#include "mutuner.hpp"

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

    // annealing parameters
    bool simulated_annealing = params.getBool("annealing", "enabled", false);
    double hot_beta = params.getDouble("annealing", "hot_beta", 1.0);
    double annealing_maxsweeps = params.getInt("annealing", "max_sweeps", 5000);

    // mutuner parameters
    bool tuner_enabled = params.getBool("mutuner", "enabled", false);
    double target_density = params.getDouble("mutuner", "target_density", 1.0);
    double conv_tolerance = params.getDouble("mutuner", "convergence_tolerance", 1e-4);
    int tuner_sweeps = params.getInt("mutuner", "max_tuner_sweeps", 1000);
    double memory_fraction = params.getDouble("mutuner", "memory_fraction", 0.5);
    double energy_scale = params.getDouble("mutuner", "energy_scale", t);
    int min_sweeps_conv = params.getInt("mutuner", "min_sweeps", 100);
    int fixed_window_size = params.getInt("mutuner", "fixed_window_size", 0);
    bool metastable = params.getBool("mutuner", "metastable", false);

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
    if (simulated_annealing) {
        hubbard.set_dtau(hot_beta/nt, lat);
    }
    
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
        measurements.addUnequalTime("doublonTau", Observables::calculate_doublonTau);
        measurements.addUnequalTime("currxxTau", Observables::calculate_currxxTau);
    }

    // ----------------------------------------------------------------- 
    //                     Start of DQMC simulation
    // -----------------------------------------------------------------

    // thermalization
    const auto t0_therm = std::chrono::steady_clock::now();

    utility::io::print_info("Starting thermalization...\n");
    if (simulated_annealing) {
        double R = std::exp(std::log(beta/hot_beta) / annealing_maxsweeps);

        double curr_beta = hubbard.dtau() * nt;
        
        int sweep_beta = 0;
        bool need_tuner = true;
        while (sweep_beta < annealing_maxsweeps) {
            sim.sweep_0_to_beta(greens, propagation_stacks);
            sim.sweep_beta_to_0(greens, propagation_stacks);

            double density_old  = Observables::calculate_N(greens, lat) / lat.n_sites();
            bool tuner_converged = false;
            if (tuner_enabled && need_tuner) {
                MuTuner tuner(target_density, curr_beta, lat.n_sites(), energy_scale, 
                        mu, memory_fraction, fixed_window_size, conv_tolerance, min_sweeps_conv);
                int current_sweep = 0;
                while (!tuner_converged && current_sweep < tuner_sweeps) {

                    // Perform one full DQMC sweep (forward and backward)
                    for (int i = 0; i < 50; ++i) {
                        sim.sweep_0_to_beta(greens, propagation_stacks);
                        sim.sweep_beta_to_0(greens, propagation_stacks);
                    }

                    // Measure the current density
                    double current_N  = Observables::calculate_N(greens, lat);
                    double current_N2 = Observables::calculate_N2(greens, lat);

                    // Update the tuner and get the new chemical potential
                    double new_mu = tuner.update(current_N, current_N2);

                    // Update the chemical potential in the model
                    hubbard.set_mu(new_mu, lat);

                    // reinitialize green's function and stacks due to change of parameters
                    for (int nfl = 0; nfl < n_flavor; nfl++) {
                        propagation_stacks[nfl] = sim.init_stacks(nfl);
                        greens[nfl]             = sim.init_greenfunctions(propagation_stacks[nfl]);
                    }

                    // Print status (optional but highly recommended)
                    //if ((current_sweep + 1) % 10 == 0) { // Print every 10 sweeps
                    //    tuner.print_status();
                    //}
                    tuner_converged = tuner.is_converged();
                    current_sweep++;
                }

                if (tuner_converged) {
                    utility::io::print_info("MuTuner converged after ", current_sweep, " sweeps.\n");
                } else {
                    utility::io::print_info("MuTuner did not converge after max ", tuner_sweeps, " sweeps.\n");
                }
            }
            tuner_converged = false;

            double density_new  = Observables::calculate_N(greens, lat) / lat.n_sites();
            utility::io::print_info("density = ", density_new, "\n");
            

            double dtau_new = 1.0 * std::pow(R, (sweep_beta+1)) / nt;
            hubbard.set_dtau(dtau_new, lat);

            // reinitialize green's function and stacks due to change of parameters
            for (int nfl = 0; nfl < n_flavor; nfl++) {
                propagation_stacks[nfl] = sim.init_stacks(nfl);
                greens[nfl]             = sim.init_greenfunctions(propagation_stacks[nfl]);
            }

            sweep_beta++;
        }
    } else if (tuner_enabled) {
        MuTuner tuner(target_density, beta, lat.n_sites(), energy_scale, 
                mu, memory_fraction, fixed_window_size, conv_tolerance, min_sweeps_conv);

        int current_sweep = 0;
        while (!tuner.is_converged() && current_sweep < tuner_sweeps) {
            // Perform 200 DQMC sweep (forward and backward)
            for (int i = 0; i < 50; ++i) {
                sim.sweep_0_to_beta(greens, propagation_stacks);
                sim.sweep_beta_to_0(greens, propagation_stacks);
            }

            // Measure the current density
            double current_N  = Observables::calculate_N(greens, lat);
            double current_N2 = Observables::calculate_N2(greens, lat);

            // Update the tuner and get the new chemical potential
            double new_mu = tuner.update(current_N, current_N2);

            // Update the chemical potential in the model
            hubbard.set_mu(new_mu, lat);

            //reinitialize green's function and stacks due to change of parameters
            for (int nfl = 0; nfl < n_flavor; nfl++) {
               propagation_stacks[nfl] = sim.init_stacks(nfl);
               greens[nfl]             = sim.init_greenfunctions(propagation_stacks[nfl]);
            }

            // Print status (optional but highly recommended)
            if ((current_sweep + 1) % 10 == 0) { // Print every 10 sweeps
               tuner.print_status();
            }
            
            current_sweep++;
        }

        if (tuner.is_converged()) {
            utility::io::print_info("MuTuner converged after ", current_sweep, " sweeps.\n");
            if (metastable) {
                hubbard.set_mu(0.0, lat);
            }

            //reinitialize green's function and stacks due to change of parameters
            for (int nfl = 0; nfl < n_flavor; nfl++) {
               propagation_stacks[nfl] = sim.init_stacks(nfl);
               greens[nfl]             = sim.init_greenfunctions(propagation_stacks[nfl]);
            }
        } else {
            utility::io::print_info("MuTuner did not converge after max ", tuner_sweeps, " sweeps.\n");
        }
    } else {
        utility::io::print_info("Starting fixed thermalization...\n");
        for (int i = 0; i < n_therms; ++i) {
            sim.sweep_0_to_beta(greens, propagation_stacks);
            sim.sweep_beta_to_0(greens, propagation_stacks);
        }
    }

    // --- SYNCHRONIZATION POINT ---
    MPI_Barrier(MPI_COMM_WORLD);
    // ---------------------------

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
            "Average acceptance rate = ",
            std::fixed, std::setprecision(4),
            sim.acc_rate() / (2.0 * (n_bins * n_sweeps + n_therms)), '\n'
        );
    }

    MPI_Finalize();

    return 0;
}