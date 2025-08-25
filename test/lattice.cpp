#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "lattice.h" // Use the new header
#include <vector>
#include <array>
#include <cmath>

// Helper for comparing floating point numbers
bool approx_equal(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

TEST_CASE("Lattice Initialization and Basic Properties") {
    const int Lx = 4;
    const int Ly = 6;
    const int n_orb = 2;

    std::array<double, 2> a1 = {1.0, 0.0};
    std::array<double, 2> a2 = {0.0, 1.0};
    std::vector<std::array<double, 2>> orbs = {{0.0, 0.0}, {0.5, 0.5}};

    Lattice lat(a1, a2, orbs, Lx, Ly);

    SUBCASE("Dimensions are set correctly") {
        CHECK(lat.Lx() == Lx);
        CHECK(lat.Ly() == Ly);
        CHECK(lat.n_orb() == n_orb);
        CHECK(lat.n_cells() == Lx * Ly);
        CHECK(lat.n_sites() == Lx * Ly * n_orb);
    }

    SUBCASE("Reciprocal vectors for square lattice are correct") {
        // For a simple square lattice with a1=(1,0), a2=(0,1)
        // b1 should be (2pi, 0) and b2 should be (0, 2pi)
        CHECK(approx_equal(lat.b1()[0], 2.0 * M_PI));
        CHECK(approx_equal(lat.b1()[1], 0.0));
        CHECK(approx_equal(lat.b2()[0], 0.0));
        CHECK(approx_equal(lat.b2()[1], 2.0 * M_PI));
    }

    SUBCASE("Number of k-points is correct") {
        CHECK(lat.k_points().size() == Lx * Ly);
    }

    SUBCASE("Constructor throws on invalid arguments") {
        CHECK_THROWS_AS(Lattice(a1, a2, orbs, 0, Ly), std::invalid_argument);
        CHECK_THROWS_AS(Lattice(a1, a2, {}, Lx, Ly), std::invalid_argument);
        CHECK_THROWS_AS(Lattice(a1, a1, orbs, Lx, Ly), std::invalid_argument); // Collinear vectors
    }
}

TEST_CASE("Lattice Coordinate and Neighbor Finding") {
    const int Lx = 8;
    const int Ly = 8;
    const int n_orb = 1;

    std::array<double, 2> a1 = {1.0, 0.0};
    std::array<double, 2> a2 = {0.0, 1.0};
    std::vector<std::array<double, 2>> orbs = {{0.0, 0.0}};

    Lattice lat(a1, a2, orbs, Lx, Ly);

    SUBCASE("Site to cell coordinate mapping") {
        // Site 0 is at cell (0,0)
        CHECK(lat.site_to_cell_coords(0)[0] == 0);
        CHECK(lat.site_to_cell_coords(0)[1] == 0);

        // Site 7 is at cell (7,0)
        CHECK(lat.site_to_cell_coords(7)[0] == 7);
        CHECK(lat.site_to_cell_coords(7)[1] == 0);

        // Site 8 is at cell (0,1)
        CHECK(lat.site_to_cell_coords(8)[0] == 0);
        CHECK(lat.site_to_cell_coords(8)[1] == 1);

        // Last site
        int last_site = lat.n_sites() - 1;
        CHECK(lat.site_to_cell_coords(last_site)[0] == Lx - 1);
        CHECK(lat.site_to_cell_coords(last_site)[1] == Ly - 1);
    }

    SUBCASE("Cell coordinate to site mapping") {
        CHECK(lat.cell_coords_to_site(0, 0, 0) == 0);
        CHECK(lat.cell_coords_to_site(7, 0, 0) == 7);
        CHECK(lat.cell_coords_to_site(0, 1, 0) == 8);
        CHECK(lat.cell_coords_to_site(Lx - 1, Ly - 1, 0) == lat.n_sites() - 1);
    }

    SUBCASE("Neighbor finding within the lattice") {
        int site_10 = 10; // (2, 1)

        // Neighbor in +x direction should be site 11 (3, 1)
        CHECK(lat.site_to_neighbor(site_10, {1, 0}, 0) == 11);
        // Neighbor in -y direction should be site 2 (2, 0)
        CHECK(lat.site_to_neighbor(site_10, {0, -1}, 0) == 2);
    }

    SUBCASE("Neighbor finding with periodic boundary conditions") {
        int site_0 = 0; // (0, 0)
        int site_7 = 7; // (7, 0)

        // Neighbor of site 0 in -x direction should be site 7
        CHECK(lat.site_to_neighbor(site_0, {-1, 0}, 0) == 7);

        // Neighbor of site 7 in +x direction should be site 0
        CHECK(lat.site_to_neighbor(site_7, {1, 0}, 0) == 0);

        // Top-right corner
        int corner_site = lat.n_sites() - 1; // (7, 7)
        // Neighbor in +x, +y should be site 0 (0, 0)
        CHECK(lat.site_to_neighbor(corner_site, {1, 1}, 0) == 0);
    }
    
    SUBCASE("Site position calculation") {
        // Site 9 is at cell (1,1)
        auto pos = lat.site_position(9);
        CHECK(approx_equal(pos[0], 1.0));
        CHECK(approx_equal(pos[1], 1.0));
    }
}