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
#include <map>
#include <sstream>
#include <algorithm>
#include <cctype>

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
        std::mt19937 generator_;

    public:
        // Function to initialize random engine
        void set_seed(unsigned int seed) {
            generator_.seed(seed);
        }

        // Function that returns boolean value false or true based on probability p
        bool bernoulli(double p) {
            std::bernoulli_distribution dist(p);
            return dist(generator_);
        }

        int rand_GHQField() {
            std::uniform_int_distribution<int> dist(0,3);
            return dist(generator_);
        }

        int rand_proposeGHQField() {
            std::uniform_int_distribution<int> dist(0,2);
            return dist(generator_);
        }
        
        // Getter for the generator (to pass to other classes)
        std::mt19937& get_generator() {
            return generator_;
        }
    };

    class parameters {
    private:
        std::map<std::string, std::map<std::string, std::string>> sections;
        
        // Helper function to trim whitespace from both ends of a string
        static void trim(std::string& str) {
            // Trim leading whitespace
            str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            
            // Trim trailing whitespace
            str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), str.end());
        }
        
        // Helper function to remove comments from a line
        static void removeComment(std::string& str) {
            // Remove everything after # or ; (comments)
            size_t comment_pos = str.find_first_of("#;");
            if (comment_pos != std::string::npos) {
                str = str.substr(0, comment_pos);
            }
        }
        
        // Helper function to check if a line is a comment
        static bool isComment(const std::string& line) {
            return !line.empty() && (line[0] == '#' || line[0] == ';');
        }
        
        // Helper function to check if a line is empty
        static bool isEmpty(const std::string& line) {
            return line.empty() || std::all_of(line.begin(), line.end(), [](unsigned char ch) {
                return std::isspace(ch);
            });
        }
        
        // Helper function to parse a line and extract key-value pair
        void parseLine(const std::string& line, std::string& current_section) {
            std::string trimmed_line = line;
            removeComment(trimmed_line);
            trim(trimmed_line);
            
            // Skip empty lines and comments
            if (isEmpty(trimmed_line) || isComment(trimmed_line)) {
                return;
            }
            
            // Check if it's a section header [section_name]
            if (trimmed_line.front() == '[' && trimmed_line.back() == ']') {
                current_section = trimmed_line.substr(1, trimmed_line.length() - 2);
                trim(current_section);
                return;
            }
            
            // Parse key = value pair
            size_t equal_pos = trimmed_line.find('=');
            if (equal_pos != std::string::npos) {
                std::string key = trimmed_line.substr(0, equal_pos);
                std::string value = trimmed_line.substr(equal_pos + 1);
                
                trim(key);
                trim(value);
                
                // Remove quotes from value if present
                if (!value.empty() && ((value.front() == '"' && value.back() == '"') ||
                                       (value.front() == '\'' && value.back() == '\''))) {
                    value = value.substr(1, value.length() - 2);
                }
                
                // Store the parameter in the current section
                sections[current_section][key] = value;
            }
        }
        
    public:
        // Constructor that parses the parameter file
        parameters(const std::string& filename) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open parameter file: " + filename);
            }
            
            std::string line;
            std::string current_section = "global"; // Default section
            
            while (std::getline(file, line)) {
                parseLine(line, current_section);
            }
            
            file.close();
        }
        
        // Method to override parameters with values from another parameter object
        void override_with(const parameters& other) {
            for (const auto& section_pair : other.sections) {
                const std::string& section_name = section_pair.first;
                const auto& other_keys = section_pair.second;

                for (const auto& key_pair : other_keys) {
                    sections[section_name][key_pair.first] = key_pair.second;
                }
            }
        }
        
        // Get string parameter
        std::string getString(const std::string& section, const std::string& key) const {
            auto section_it = sections.find(section);
            if (section_it == sections.end()) {
                throw std::runtime_error("Section '" + section + "' not found");
            }
            
            auto key_it = section_it->second.find(key);
            if (key_it == section_it->second.end()) {
                throw std::runtime_error("Key '" + key + "' not found in section '" + section + "'");
            }
            
            return key_it->second;
        }
        
        // Get string parameter with default value
        std::string getString(const std::string& section, const std::string& key, 
                             const std::string& default_value) const {
            try {
                return getString(section, key);
            } catch (...) {
                return default_value;
            }
        }
        
        // Get integer parameter
        int getInt(const std::string& section, const std::string& key) const {
            std::string value = getString(section, key);
            try {
                // Handle underscores in numbers (e.g., 10_000 -> 10000)
                std::string clean_value = value;
                clean_value.erase(std::remove(clean_value.begin(), clean_value.end(), '_'), clean_value.end());
                return std::stoi(clean_value);
            } catch (const std::exception&) {
                throw std::runtime_error("Cannot convert '" + value + "' to integer for key '" + key + "'");
            }
        }
        
        // Get integer parameter with default value
        int getInt(const std::string& section, const std::string& key, int default_value) const {
            try {
                return getInt(section, key);
            } catch (...) {
                return default_value;
            }
        }
        
        // Get double parameter
        double getDouble(const std::string& section, const std::string& key) const {
            std::string value = getString(section, key);
            try {
                // Handle underscores in numbers (e.g., 10_000 -> 10000)
                std::string clean_value = value;
                clean_value.erase(std::remove(clean_value.begin(), clean_value.end(), '_'), clean_value.end());
                return std::stod(clean_value);
            } catch (const std::exception&) {
                throw std::runtime_error("Cannot convert '" + value + "' to double for key '" + key + "'");
            }
        }
        
        // Get double parameter with default value
        double getDouble(const std::string& section, const std::string& key, double default_value) const {
            try {
                return getDouble(section, key);
            } catch (...) {
                return default_value;
            }
        }
        
        // Get boolean parameter
        bool getBool(const std::string& section, const std::string& key) const {
            std::string value = getString(section, key);
            
            // Convert to lowercase for comparison
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
                return std::tolower(c);
            });
            
            if (value == "true" || value == "1" || value == "yes" || value == "on") {
                return true;
            } else if (value == "false" || value == "0" || value == "no" || value == "off") {
                return false;
            } else {
                throw std::runtime_error("Cannot convert '" + value + "' to boolean for key '" + key + "'");
            }
        }
        
        // Get boolean parameter with default value
        bool getBool(const std::string& section, const std::string& key, bool default_value) const {
            try {
                return getBool(section, key);
            } catch (...) {
                return default_value;
            }
        }
        
        // Check if section exists
        bool hasSection(const std::string& section) const {
            return sections.find(section) != sections.end();
        }
        
        // Check if key exists in section
        bool hasKey(const std::string& section, const std::string& key) const {
            auto section_it = sections.find(section);
            if (section_it == sections.end()) {
                return false;
            }
            return section_it->second.find(key) != section_it->second.end();
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