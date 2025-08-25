#include <field.h>
#include <cmath> // For std::sqrt

// ---------------------------------------------------------------------------- //
// HSField Class Implementation
// ---------------------------------------------------------------------------- //

HSField::HSField(int nt, int nv, utility::random& rng)
    : rng_(rng)
{    
    // Gamma and Eta are properties of the GHQ decomposition itself,
    gamma_.set_size(4);
    eta_.set_size(4);
    
    const double s6 = std::sqrt(6.0);

    // Mapping discrete field values {0,1,2,3} to GHQ parameters
    gamma_(0) = 1.0 - s6 / 3.0;
    gamma_(1) = 1.0 + s6 / 3.0;
    gamma_(2) = 1.0 + s6 / 3.0;
    gamma_(3) = 1.0 - s6 / 3.0;

    eta_(0) = -std::sqrt(2.0 * (3.0 + s6));
    eta_(1) = -std::sqrt(2.0 * (3.0 - s6));
    eta_(2) =  std::sqrt(2.0 * (3.0 - s6));
    eta_(3) =  std::sqrt(2.0 * (3.0 + s6));

    proposal_ = {{1, 2, 3},
                {0, 2, 3},
                {0, 1, 3},
                {0, 1, 2}};

    // --- Randomly initialize the field configuration ---
    // The number of columns is now nv.
    fields_.set_size(nt, nv);
    
    std::uniform_int_distribution<int> dist(0, 3);
    for (arma::uword i = 0; i < fields_.n_elem; ++i) {
        fields_(i) = dist(rng_.get_generator());
    }
}

void HSField::set_fields(const arma::imat& new_fields) {
    fields_ = new_fields;
}

void HSField::set_field_value(int l, int i, int new_value) {
    fields_(l, i) = new_value;
}

int HSField::propose_new_field(int old_field) const {
    // This implementation is for a 4-state discrete field.
    // It proposes one of the other 3 states with equal probability.
    std::uniform_int_distribution<int> dist(0, 2);
    int propose_field = dist(rng_.get_generator());

    return proposal_(old_field, propose_field);
}