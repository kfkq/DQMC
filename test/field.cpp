#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "field.h"
#include "utility.h"
#include <vector>
#include <cmath>

TEST_CASE("HSField Class Functionality") {
    // --- 1. Setup ---
    utility::random rng;
    rng.set_seed(42); // Use a fixed seed for reproducible tests

    const int nt = 10;
    const int nv = 16; // e.g., a 4x4 lattice

    // Construct the HSField object, passing the rng object.
    // This is now the only way to create an HSField.
    HSField field(nt, nv, rng);

    SUBCASE("Constructor and Dimensions") {
        CHECK(field.get_nv() == nv);
        
        const arma::imat& fields_mat = field.get_fields();
        CHECK(fields_mat.n_rows == nt);
        CHECK(fields_mat.n_cols == nv);

        // Check that initial random fields are within the correct range [0, 3]
        bool all_in_range = true;
        for (arma::uword i = 0; i < fields_mat.n_elem; ++i) {
            if (fields_mat(i) < 0 || fields_mat(i) > 3) {
                all_in_range = false;
                break;
            }
        }
        CHECK(all_in_range);
    }

    SUBCASE("GHQ Parameter Getters return correct values") {
        const double s6 = std::sqrt(6.0);
        
        // Check gamma values
        CHECK(doctest::Approx(field.get_gamma(0)) == 1.0 - s6 / 3.0);
        CHECK(doctest::Approx(field.get_gamma(1)) == 1.0 + s6 / 3.0);
        CHECK(doctest::Approx(field.get_gamma(2)) == 1.0 + s6 / 3.0);
        CHECK(doctest::Approx(field.get_gamma(3)) == 1.0 - s6 / 3.0);

        // Check eta values
        CHECK(doctest::Approx(field.get_eta(0)) == -std::sqrt(2.0 * (3.0 + s6)));
        CHECK(doctest::Approx(field.get_eta(1)) == -std::sqrt(2.0 * (3.0 - s6)));
        CHECK(doctest::Approx(field.get_eta(2)) ==  std::sqrt(2.0 * (3.0 - s6)));
        CHECK(doctest::Approx(field.get_eta(3)) ==  std::sqrt(2.0 * (3.0 + s6)));
    }

    SUBCASE("Setters modify the field correctly") {
        // Test set_field_value
        field.set_field_value(5, 10, 99); // Use a value outside the normal range
        CHECK(field.get_field_value(5, 10) == 99);

        // Test set_fields
        arma::imat new_fields(nt, nv, arma::fill::value(77));
        field.set_fields(new_fields);
        CHECK(field.get_field_value(0, 0) == 77);
        CHECK(field.get_field_value(nt - 1, nv - 1) == 77);
    }

    SUBCASE("propose_new_field proposes a valid and different field") {
        for (int old_field = 0; old_field <= 3; ++old_field) {
            // Run multiple times to ensure it's not a fluke
            for (int i = 0; i < 100; ++i) {
                int new_field = field.propose_new_field(old_field);
                
                // The new field must NOT be the same as the old one
                CHECK(new_field != old_field);
                
                // The new field must be in the valid range [0, 3]
                CHECK(new_field >= 0);
                CHECK(new_field <= 3);
            }
        }
    }
}