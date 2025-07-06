#include "lattice.hpp"
#include "model.hpp" 
#include "dqmc.hpp"
#include "measurement.hpp"

#include <toml.hpp>
#include <iostream>
#include <ctime>
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

    // DQMC simulation initialization
    auto sim = DQMC(hubbard, n_stab);

    // measurement container
    scalarObservable density("density", rank);

    // ----------------------------------------------------------------- 
    //                     Start of DQMC simulation
    // -----------------------------------------------------------------

    // thermalization
    for (int i = 0; i < n_therms; ++i) {
        sim.sweep_0_to_beta();
        sim.sweep_beta_to_0();
    }

    // measurement sweeps
    for (int ibin = 0; ibin < n_bins; ++ibin) {
        for (int isweep = 0; isweep < n_sweeps; ++isweep) {
            sim.sweep_0_to_beta();

            density += Observables::calculate_density(sim);

            sim.sweep_beta_to_0();

            density += Observables::calculate_density(sim);
        }

        density.accumulate();
        density.reset();
    }
    
    // ----------------------------------------------------------------- 
    //                         MPI Finalization
    // -----------------------------------------------------------------
    MPI_Finalize();

    return 0;
}