#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <utility.h>
#include <fstream>
#include <vector>

// --- Tests for the random class ---
TEST_CASE("utility::random") {
    utility::random rng;
    rng.set_seed(42);

    SUBCASE("Bernoulli distribution") {
        int heads = 0;
        for (int i = 0; i < 1000; ++i) {
            if (rng.bernoulli(0.3)) {
                heads++;
            }
        }
        // Check if the result is roughly within expectations
        CHECK(heads > 250);
        CHECK(heads < 350);
    }

    SUBCASE("Generator access") {
        // Just check that we can get the generator and it's not null
        auto& gen = rng.get_generator();
        CHECK(sizeof(gen) > 0);
    }
}

// --- Tests for the parameters class ---
TEST_CASE("utility::parameters") {
    // Create a temporary INI file for testing
    const std::string filename = "test_params.in";
    std::ofstream test_file(filename);
    test_file << R"(
# This is a comment
[lattice]
Lx = 4
Ly = 8 ; another comment

[simulation]
beta = 5.0
n_sweeps = 10_000
is_real = true
name = "My Test"
)";
    test_file.close();

    utility::parameters params(filename);

    SUBCASE("Reading values") {
        CHECK(params.getInt("lattice", "Lx") == 4);
        CHECK(params.getInt("lattice", "Ly") == 8);
        CHECK(params.getDouble("simulation", "beta") == 5.0);
        CHECK(params.getInt("simulation", "n_sweeps") == 10000);
        CHECK(params.getBool("simulation", "is_real") == true);
        CHECK(params.getString("simulation", "name") == "My Test");
    }

    SUBCASE("Default values") {
        CHECK(params.getInt("lattice", "non_existent", 99) == 99);
        CHECK(params.getDouble("simulation", "temp", 0.2) == 0.2);
        CHECK(params.getBool("simulation", "is_complex", false) == false);
        CHECK(params.getString("lattice", "type", "square") == "square");
    }

    SUBCASE("Error handling") {
        CHECK_THROWS_AS(params.getInt("non_existent_section", "key"), std::runtime_error);
        CHECK_THROWS_AS(params.getString("lattice", "non_existent_key"), std::runtime_error);
        CHECK_THROWS_AS(params.getInt("simulation", "name"), std::runtime_error); // "My Test" is not an int
    }

    SUBCASE("Checkers") {
        CHECK(params.hasSection("lattice") == true);
        CHECK(params.hasSection("non_existent_section") == false);
        CHECK(params.hasKey("lattice", "Lx") == true);
        CHECK(params.hasKey("lattice", "non_existent_key") == false);
        CHECK(params.hasKey("non_existent_section", "key") == false);
    }

    SUBCASE("Override functionality") {
        const std::string override_filename = "override_params.in";
        std::ofstream override_file(override_filename);
        override_file << R"(
[lattice]
Lx = 16  # Override Lx
type = "honeycomb" # Add new key

[hubbard]
U = 8.0 # Add new section
)";
        override_file.close();

        utility::parameters override_params(override_filename);
        params.override_with(override_params);

        CHECK(params.getInt("lattice", "Lx") == 16); // Overridden
        CHECK(params.getInt("lattice", "Ly") == 8); // Unchanged
        CHECK(params.getString("lattice", "type") == "honeycomb"); // Added
        CHECK(params.getDouble("hubbard", "U") == 8.0); // New section added
        
        // Clean up the override file
        std::remove(override_filename.c_str());
    }

    // Clean up the main test file
    std::remove(filename.c_str());
}