
#ifndef LATTICE_HPP
#define LATTICE_HPP

#include <armadillo>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>

namespace lattice {

    using Matrix = arma::mat;

    // Struct to hold lattice information
    struct Lattice {
        std::string type;
        int Lx, Ly, N_sites;
        std::array<double, 2> a1, a2; 
        std::array<double, 2> b1, b2; 
    };

    // Create lattice and return lattice info
    inline Lattice create_lattice(const std::string& type, int Lx, int Ly) {
        Lattice lat;
        lat.type = type;
        lat.Lx = Lx;
        lat.Ly = Ly;
        lat.N_sites = Lx * Ly;
        
        if (type == "square") {
            // Real space vectors
            lat.a1 = {1.0, 0.0};  
            lat.a2 = {0.0, 1.0}; 
            
            // Reciprocal vectors (2π/a)
            lat.b1 = {2.0 * M_PI, 0.0};
            lat.b2 = {0.0, 2.0 * M_PI};
        } else {
            throw std::runtime_error("Unsupported lattice type: " + type);
        }
        
        return lat;
    }

    // Get nearest neighbors
    inline std::vector<int> nearest_neighbors(const Lattice& lat, int site) {
        if (lat.type == "square") {
            int x = site % lat.Lx;
            int y = site / lat.Lx;
            return {
                y * lat.Lx + ((x + 1) % lat.Lx),      // +x neighbor
                ((y + 1) % lat.Ly) * lat.Lx + x       // +y neighbor
            };
            
        } else {
            throw std::runtime_error("Unsupported lattice type: " + lat.type);
        }
    }

    // Create nearest-neighbor matrix
    inline Matrix nn_matrix(const Lattice& lat) {
        Matrix matrix(lat.N_sites, lat.N_sites);
        
        // Fill matrix based on nearest neighbor connections
        for (int site = 0; site < lat.N_sites; ++site) {
            auto nn = nearest_neighbors(lat, site);
            for (int neighbor : nn) {
                matrix(site, neighbor) = 1;
                matrix(neighbor, site) = 1; 
            }
        }
        
        return matrix;
    }

} // namespace lattice
#endif