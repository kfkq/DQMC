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

namespace utility {
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
    };
}

#endif // UTILITY_HPP