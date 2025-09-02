/*
/   This is utility library for DQMC
/   1. Random generator
/   2. parsing input parameters
/
/   Author: Muhammad Gaffar
*/


#pragma once

#include <mpi.h>
#include <sys/stat.h>
#include <random>
#include <algorithm>
#include <fstream>

namespace utility {    
    class random {
        private:
            std::mt19937 generator_;

            // Distribution for rand_GHQField().
            std::uniform_int_distribution<int> dist_GHQField_;

        public:
            // Constructor to initialize with a specific seed
            random(unsigned int seed) 
                : generator_(seed), 
                dist_GHQField_(0, 3) // Initialize the distribution here
            {}

            // Function that returns a boolean value based on probability p
            bool bernoulli(double p) {
                std::bernoulli_distribution dist(p);
                return dist(generator_);
            }

            // Function that returns a random integer between 0 and 3 (inclusive)
            int rand_GHQField() {
                return dist_GHQField_(generator_);
            }
            
            // Getter for the generator
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

        // Get vector of doubles from a comma-separated string
        std::vector<double> getDoubleVector(const std::string& section, const std::string& key) const {
            std::string value_str = getString(section, key);
            std::vector<double> result;
            std::stringstream ss(value_str);
            std::string item;

            while (std::getline(ss, item, ',')) {
                trim(item); // Use the existing static trim function
                if (item.empty()) continue; // Skip empty entries

                try {
                    // Handle underscores for consistency
                    std::string clean_item = item;
                    clean_item.erase(std::remove(clean_item.begin(), clean_item.end(), '_'), clean_item.end());
                    result.push_back(std::stod(clean_item));
                } catch (const std::exception&) {
                    throw std::runtime_error("Cannot convert '" + item + "' to double in list for key '" + key + "'");
                }
            }
            return result;
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