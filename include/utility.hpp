/*
/   This is utility library for DQMC Hubbard model
/   1. generating random values
/   2. parsing input parameters
/
/   Author: Muhammad Gaffar
*/

#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <random>
#include <fstream>
#include <armadillo>
#include <stdexcept>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <sys/stat.h>

namespace utility {
    static bool ensure_dir(const std::string& path, int rank)
    {
        if (rank == 0) {
            struct stat info;
            if (stat(path.c_str(), &info) == 0) {
                // Directory exists: remove it recursively
                #ifdef _WIN32
                    std::string cmd = "rmdir /s /q \"" + path + "\"";
                    int status = system(cmd.c_str());
                #else
                    std::string cmd = "rm -rf \"" + path + "\"";
                    int status = system(cmd.c_str());
                #endif
                if (status != 0) return false;
            }
            #ifdef _WIN32
                int status = _mkdir(path.c_str());
            #else
                int status = mkdir(path.c_str(), 0755);
            #endif
            if (status != 0) return false;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return true;
    }
    class random {
    private:
        // Static random engine getter 
        static std::mt19937& get_generator() {
            static std::mt19937 generator;
            return generator;
        }

    public:
        // Function to initialize random engine
        static void set_seed(unsigned int seed) {
            get_generator().seed(seed);
        }

        // Function that returns boolean value false or true based on probability p
        static bool bernoulli(double p) {
            std::bernoulli_distribution dist(p);
            return dist(get_generator());
        }

        static int uniform_int(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max);
            return dist(get_generator());
        }
    };

    class io {
    public:
        // Save integer fields to text file with prettier formatting
        static void save_fields(const arma::imat& fields, const std::string& filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to save fields to: " + filename);
            }

            // Write header with dimensions
            file << "{nt, ns} = {" << fields.n_rows << ", " << fields.n_cols << "}\n";

            // Write matrix with space padding for alignment
            for (arma::uword i = 0; i < fields.n_rows; ++i) {
                for (arma::uword j = 0; j < fields.n_cols; ++j) {
                    // Add space before positive numbers to align with negative
                    if (fields(i, j) >= 0) file << " ";
                    file << fields(i, j);
                    if (j < fields.n_cols - 1) file << "  "; // Double space between columns
                }
                file << "\n";
            }

            file.close();
        }

        // Load integer fields from text file
        static arma::imat load_fields(const std::string& filename) {
            arma::imat fields;
            if (!fields.load(filename, arma::arma_ascii)) {
                throw std::runtime_error("Failed to load fields from: " + filename);
            }
            return fields;
        }

        // Check if file exists using standard file operations
        static bool file_exists(const std::string& filename) {
            std::ifstream f(filename);
            return f.good();
        }

        // Print only on MPI rank 0
        template <typename... Args>
        static void print_info(Args&&... args) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank != 0) return;
            (std::cout << ... << args) << std::flush;
        }
    };
}

#endif // UTILITY_HPP
