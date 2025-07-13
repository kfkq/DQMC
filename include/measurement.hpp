#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP


#include "dqmc.hpp"

#include <mpi.h>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <sys/stat.h> 


class scalarObservable {
private:
    std::string filename_;
    
    // accumulation
    double local_sum_ = 0.0;
    double global_sum_ = 0.0;

    int local_count_ = 0;
    int global_count_ = 0;

    bool ensure_results_dir(int rank) {
        if (rank == 0) {
            struct stat info;
            if (stat("results", &info) != 0) {
                #ifdef _WIN32
                    int status = _mkdir("results");
                #else
                    int status = mkdir("results", 0755);
                #endif
                if (status != 0) return false;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }

public:
    scalarObservable(const std::string& filename, int rank) : filename_("results/" + filename) {
        if (!ensure_results_dir(rank)) {
            throw std::runtime_error("Could not create results directory");
        }
        
        if (rank == 0) {
            std::ofstream out(filename_);
            out << "#" << std::setw(20) << "value\n" ;
        }
    }
    
    void operator+=(double value) {
        local_sum_ += value;
        local_count_++;
    }
    
    void accumulate(MPI_Comm comm) {
        MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count_, 1, MPI_INT, MPI_SUM, comm);
        
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            double mean = global_sum_ / global_count_;
            
            std::ofstream out(filename_, std::ios::app);
            out << std::fixed << std::setprecision(10)
                << std::setw(20) << mean << "\n";
        }
    }
    
    void reset() {
        local_sum_ = 0.0;
        local_count_ = 0;
    }
};

class MeasurementManager {
private:
    std::vector<scalarObservable> scalarObservables_;
    std::vector<std::function<double(const std::vector<GF>&)>> calculators_;
    MPI_Comm comm_;
    int rank_;
    
public:
    MeasurementManager(MPI_Comm comm, int rank) : comm_(comm), rank_(rank) {}

    void addScalar(const std::string& name, 
             std::function<double(const std::vector<GF>&)> calculator) {
        scalarObservables_.emplace_back(name + ".dat", rank_);
        calculators_.push_back(calculator);
    }

    void measure(const std::vector<GF>& greens) {
        for (size_t i = 0; i < calculators_.size(); ++i) {
            scalarObservables_[i] += calculators_[i](greens);
        }
    }

    void accumulate() {
        for (auto& obs : scalarObservables_) {
            obs.accumulate(comm_);
        }
    }

    void reset() {
        for (auto& obs : scalarObservables_) {
            obs.reset();
        }
    }
};

namespace Observables {
    double calculate_density(const std::vector<GF>& greens);
    double calculate_doubleOccupancy(const std::vector<GF>& greens);
    double calculate_swavePairing(const std::vector<GF>& greens);
}

#endif