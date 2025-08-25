#include <utility.h>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace utility {

// ----------------------------------------------------------------------------
// random Class Implementation
// ----------------------------------------------------------------------------

void random::set_seed(unsigned int seed) {
    generator_.seed(seed);
}

bool random::bernoulli(double p) {
    std::bernoulli_distribution dist(p);
    return dist(generator_);
}

std::mt19937& random::get_generator() {
    return generator_;
}


// ----------------------------------------------------------------------------
// parameters Class Implementation
// ----------------------------------------------------------------------------

namespace { // Anonymous namespace for private helper functions

    // Helper to trim whitespace from both ends of a string
    void trim(std::string& str) {
        str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), str.end());
    }

    // Helper to remove comments from a line
    void removeComment(std::string& str) {
        size_t comment_pos = str.find_first_of("#;");
        if (comment_pos != std::string::npos) {
            str = str.substr(0, comment_pos);
        }
    }

} // end anonymous namespace

void parameters::parseLine(const std::string& line, std::string& current_section) {
    std::string trimmed_line = line;
    removeComment(trimmed_line);
    trim(trimmed_line);

    if (trimmed_line.empty()) {
        return;
    }

    if (trimmed_line.front() == '[' && trimmed_line.back() == ']') {
        current_section = trimmed_line.substr(1, trimmed_line.length() - 2);
        trim(current_section);
        return;
    }

    size_t equal_pos = trimmed_line.find('=');
    if (equal_pos != std::string::npos) {
        std::string key = trimmed_line.substr(0, equal_pos);
        std::string value = trimmed_line.substr(equal_pos + 1);
        
        trim(key);
        trim(value);
        
        if (!value.empty() && ((value.front() == '"' && value.back() == '"') ||
                               (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.length() - 2);
        }
        
        sections[current_section][key] = value;
    }
}

parameters::parameters(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open parameter file: " + filename);
    }
    
    std::string line;
    std::string current_section = "global";
    
    while (std::getline(file, line)) {
        parseLine(line, current_section);
    }
    
    file.close();
}

void parameters::override_with(const parameters& other) {
    for (const auto& section_pair : other.sections) {
        const std::string& section_name = section_pair.first;
        const auto& other_keys = section_pair.second;

        for (const auto& key_pair : other_keys) {
            sections[section_name][key_pair.first] = key_pair.second;
        }
    }
}

std::string parameters::getString(const std::string& section, const std::string& key) const {
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

std::string parameters::getString(const std::string& section, const std::string& key, const std::string& default_value) const {
    try {
        return getString(section, key);
    } catch (const std::runtime_error&) {
        return default_value;
    }
}

int parameters::getInt(const std::string& section, const std::string& key) const {
    std::string value = getString(section, key);
    try {
        std::string clean_value = value;
        clean_value.erase(std::remove(clean_value.begin(), clean_value.end(), '_'), clean_value.end());
        return std::stoi(clean_value);
    } catch (const std::exception&) {
        throw std::runtime_error("Cannot convert '" + value + "' to integer for key '" + key + "'");
    }
}

int parameters::getInt(const std::string& section, const std::string& key, int default_value) const {
    try {
        return getInt(section, key);
    } catch (const std::runtime_error&) {
        return default_value;
    }
}

double parameters::getDouble(const std::string& section, const std::string& key) const {
    std::string value = getString(section, key);
    try {
        std::string clean_value = value;
        clean_value.erase(std::remove(clean_value.begin(), clean_value.end(), '_'), clean_value.end());
        return std::stod(clean_value);
    } catch (const std::exception&) {
        throw std::runtime_error("Cannot convert '" + value + "' to double for key '" + key + "'");
    }
}

double parameters::getDouble(const std::string& section, const std::string& key, double default_value) const {
    try {
        return getDouble(section, key);
    } catch (const std::runtime_error&) {
        return default_value;
    }
}

bool parameters::getBool(const std::string& section, const std::string& key) const {
    std::string value = getString(section, key);
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

bool parameters::getBool(const std::string& section, const std::string& key, bool default_value) const {
    try {
        return getBool(section, key);
    } catch (const std::runtime_error&) {
        return default_value;
    }
}

bool parameters::hasSection(const std::string& section) const {
    return sections.count(section) > 0;
}

bool parameters::hasKey(const std::string& section, const std::string& key) const {
    auto section_it = sections.find(section);
    if (section_it == sections.end()) {
        return false;
    }
    return section_it->second.count(key) > 0;
}

} // namespace utility