#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>  // For directory creation

#include "dqmc.hpp"

class scalarObservable {
private:
    std::string filename_;
    const int precision_ = 10;
    const int mean_width_ = 20;
    const int var_width_ = 20;
    
    // Local accumulation
    double local_sum_ = 0.0;
    double local_sum_sq_ = 0.0;
    int local_count_ = 0;
    
    // Global accumulation
    double global_sum_ = 0.0;
    double global_sum_sq_ = 0.0;
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
            out << "#" << std::setw(mean_width_-1) << "mean" 
                << std::setw(var_width_) << "std_err\n"; 
        }
    }
    
    void operator+=(double value) {
        local_sum_ += value;
        local_sum_sq_ += value * value;
        local_count_++;
    }
    
    void accumulate(MPI_Comm comm) {
        MPI_Allreduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_sum_sq_, &global_sum_sq_, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&local_count_, &global_count_, 1, MPI_INT, MPI_SUM, comm);
        
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            double mean = global_sum_ / global_count_;
            double variance = (global_sum_sq_ / global_count_) - (mean * mean);
            double std_err = 0.0;                                                                                                                                                           
            if (global_count_ > 1) {                                                                                                                                                        
                // Standard error of the mean                                                                                                                                               
                std_err = std::sqrt(variance / (global_count_ - 1));                                                                                                         
            }
            
            std::ofstream out(filename_, std::ios::app);
            out << std::fixed << std::setprecision(precision_)
                << std::setw(mean_width_) << mean
                << std::setw(var_width_) << std_err << "\n";
        }
    }
    
    void reset() {
        local_sum_ = local_sum_sq_ = 0.0;
        local_count_ = 0;
    }
};

class MeasurementManager {
private:
    std::vector<scalarObservable> observables_;
    std::vector<std::function<double(const std::vector<GF>&)>> calculators_;
    MPI_Comm comm_;
    int rank_;
    
public:
    MeasurementManager(MPI_Comm comm, int rank) : comm_(comm), rank_(rank) {}

    void add(const std::string& name, 
             std::function<double(const std::vector<GF>&)> calculator) {
        observables_.emplace_back(name + ".dat", rank_);
        calculators_.push_back(calculator);
    }

    void measure(const std::vector<GF>& greens) {
        for (size_t i = 0; i < calculators_.size(); ++i) {
            observables_[i] += calculators_[i](greens);
        }
    }

    void accumulate() {
        for (auto& obs : observables_) {
            obs.accumulate(comm_);
        }
    }

    void reset() {
        for (auto& obs : observables_) {
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