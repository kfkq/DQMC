/*
/   This is utility library for DQMC
/   1. generating random values
/   2. parsing input parameters
/
/   Author: Muhammad Gaffar
*/

#pragma once

#include <random>
#include <string>
#include <map>

namespace utility {

    // --- Random Number Generation ---
    class random {
    private:
        std::mt19937 generator_;

    public:
        void set_seed(unsigned int seed);
        bool bernoulli(double p);
        std::mt19937& get_generator();
    };

    // --- Parameter File Parser ---
    class parameters {
    private:
        std::map<std::string, std::map<std::string, std::string>> sections;
        
        // Private helper for parsing
        void parseLine(const std::string& line, std::string& current_section);

    public:
        // Constructor that parses the parameter file
        explicit parameters(const std::string& filename);
        
        // Method to override parameters with values from another parameter object
        void override_with(const parameters& other);
        
        // Getters for different types
        std::string getString(const std::string& section, const std::string& key) const;
        std::string getString(const std::string& section, const std::string& key, const std::string& default_value) const;
        
        int getInt(const std::string& section, const std::string& key) const;
        int getInt(const std::string& section, const std::string& key, int default_value) const;
        
        double getDouble(const std::string& section, const std::string& key) const;
        double getDouble(const std::string& section, const std::string& key, double default_value) const;
        
        bool getBool(const std::string& section, const std::string& key) const;
        bool getBool(const std::string& section, const std::string& key, bool default_value) const;
        
        // Checkers
        bool hasSection(const std::string& section) const;
        bool hasKey(const std::string& section, const std::string& key) const;
    };

} // namespace utility