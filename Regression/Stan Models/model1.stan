// An equivalent bayesian model for the NRTL model, which is easier to sample from
// This code assumes that tau_ij and it's derivatives are the parameters of the model
// instead of the a_ij, b_ij, c_ij, d_ij parameters of the NRTL model
// No idea why this model is better defined (in terms of finding modes close to the global optimum) but it is.
functions {
    vector NRTL(vector x, real T, real t12, real t21, real dt12_dT, real dt21_dT, real a) { // NRTL function in terms of tau and its derivatives
        int N = rows(x);
        real G12 = exp(-a * t12);
        real G21 = exp(-a * t21);
        vector[N] term1 = ( ( (1-x) * G12 * (1 - a*t12) + x * square(G12) ) ./ square((1-x) + x * G12) ) * dt12_dT;
        vector[N] term2 = ( ( x * G21 * (1 - a*t21) + (1-x) * square(G21) ) ./ square(x + (1-x) * G21) ) * dt21_dT;
        return -8.314 * square(T) * x .* (1-x) .* ( term1 + term2);
    }
}

data {
    int N_points;                            // Number of datapoints
    vector<lower=0, upper=1>[N_points] x;    // vector of composition
    vector[N_points] y;                      // vector of experimental excess enthalpy
    vector[4] scaling;                       // scaling factors for NRTL parameters. Not used, should probably remove
    real<lower=0> a;                         // alpha parameters for the NRTL model. Fixed to 0.3
    int N_temps;                             // Number of unique temperatures. If temperatures are close together += 0.5K it is grouped as being the same. Since this is used for initialzation, this detail does not make a big difference
    vector[N_temps] T_unique;                // Unique temperatures
    array[N_temps, 2] int T_idx;             // Start and end indices for the unique temperature for use in determining the corresponding composition and experimental excess enthalpy values to use at a given temperature
}

transformed data {
    real<lower=0> error=0.01;            // experimental error fixed to 1% of reported value
    matrix[2*N_temps, 4] tau_base;       // Mapping for the moving from NRTL parameters to tau_ij [tau_ij; dtau_ij_dT] = tau_base * [a_ij, b_ij, c_ij, d_ij]
    matrix[2*N_temps, 2*N_temps] KNN;    // Equivalent kernel for the linear mapping above

    for (i in 1:N_temps) {
        tau_base[i, :] = [1, T_unique[i], 1.0 / T_unique[i], log(T_unique[i])];             // Assign the first N_temps entries to mapping of tau_ij
        tau_base[i+N_temps, :] = [0, 1, -1.0 / square(T_unique[i]), 1.0 / T_unique[i]];     // Assign the final N_temps entries to the mapping of dtau_ij_dT
    }

    KNN = add_diag(tau_base * tau_base', 1e-8); // Compute kernel matrix by using the Mapping multiplied by its transpose and add some noise to the main diagonal for numeric stability
}

parameters {
    vector[N_temps] t12;
    vector[N_temps] t21;
    vector[N_temps] dt12_dT;
    vector[N_temps] dt21_dT;
    real<lower=0> v;
}

model {
    vector[2*N_temps] all12 = append_row(t12, dt12_dT);
    vector[2*N_temps] all21 = append_row(t21, dt21_dT);
    
    all12 ~ multi_normal(rep_vector(0, 2*N_temps), KNN);  // Prior GP on [tau_ij; dtau_ij_dT]
    all21 ~ multi_normal(rep_vector(0, 2*N_temps), KNN);
    
    v ~ exponential(2);

    {
        vector[N_points] y_means;
        for (i in 1:N_temps) { // loop through temperatures to obtain the predicted values of the excess enthalpy
            y_means[T_idx[i,1]:T_idx[i,2]] = NRTL(x[T_idx[i,1]:T_idx[i,2]], T_unique[i], t12[i], t21[i], dt12_dT[i], dt21_dT[i], a);
        }
        y ~ normal(y_means, sqrt(square(error*abs(y))+v)); // likelihood function
    }
}
