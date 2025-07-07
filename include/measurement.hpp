#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>  // For directory creation

#include "dqmc.hpp"

class scalarObservable {
private:
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    std::string filename;
    
    // Formatting parameters (customizable)
    const int precision = 10;
    const int mean_width = 20;
    const int var_width = 20;

    // Create directory if it doesn't exist
    bool ensure_results_dir() {
        struct stat info;
        if (stat("results", &info) != 0) {
            // Directory doesn't exist, try to create it
            #ifdef _WIN32
                int status = _mkdir("results");
            #else
                int status = mkdir("results", 0755);
            #endif
            if (status != 0) {
                std::cerr << "Error creating results directory" << std::endl;
                return false;
            }
        }
        return true;
    }

public:
    // Constructor takes MPI rank and creates rank-specific filename in results/
    scalarObservable(const std::string& base_filename, int mpi_rank) 
    {
        if (!ensure_results_dir()) {
            throw std::runtime_error("Could not create results directory");
        }
        
        // Create filename with MPI rank (padded with leading zero)
        std::ostringstream oss;
        oss << "results/" << base_filename << "_" 
            << std::setw(2) << std::setfill('0') << mpi_rank << ".dat";
        filename = oss.str();
        
        // Create file and write header
        std::ofstream outfile(filename);
        if (!outfile) {
            std::cerr << "Error creating file: " << filename << std::endl;
            throw std::runtime_error("Could not create output file");
        }
        // Aligned header
        outfile << "#" 
               << std::setw(mean_width-1) << "mean" 
               << std::setw(var_width) << "variance\n";
    }

    // += operator for accumulation
    scalarObservable& operator+=(double value) {
        sum += value;
        sum_sq += value * value;
        count++;
        return *this;
    }
    
    // Calculate and append formatted output
    void accumulate() {
        if (count == 0) {
            std::cerr << "Warning: No data accumulated\n";
            return;
        }
        
        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        
        std::ofstream outfile(filename, std::ios::app);
        if (!outfile) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }
        
        // Fixed-width output
        outfile << std::fixed << std::setprecision(precision)
               << std::setw(mean_width) << mean 
               << std::setw(var_width) << variance << "\n";
    }
    
    void reset() {
        sum = 0.0;
        sum_sq = 0.0;
        count = 0;
    }

    // Getter for the generated filename
    std::string get_filename() const { return filename; }
};

namespace Observables {

    // Density calculation
    double calculate_density(std::vector<GF>&  greens) ;

}

#endif