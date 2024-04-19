// Stan code for estimating the data-model mismatch parameter of the different datasets based on the adjusted Redlich-Kister polynomial
functions {
    // kernel for composition
    matrix Kx(vector x1, vector x2, int order) {
        int N = rows(x1); // Number of datapoints for the first vector x1
        int M = rows(x2); // Number of datapoints for the second vector x2
        matrix[N, order+1] X1; // Mapping of X1
        matrix[M, order+1] X2; // Mapping of X2
        for (i in 1:order) {
            X1[:,i] = x1 .^(order+2-i) - x1; // Mapping is defined as [x^2-x, x^3-x, x^4-x, ..., x^(order+1)-x] but in reverse order
            X2[:,i] = x2 .^(order+2-i) - x2;
        }
        X1[:,order+1] = 1e-1 * x1 .* sqrt(1-x1) .* exp(x1); // Append Mapping with 1e-1 * x * (1-x) * exp(x)
        X2[:,order+1] = 1e-1 * x2 .* sqrt(1-x2) .* exp(x2);
        return X1 * X2'; // Return the kernel for the 2 linear mappings
    }

    // kernel for Temperature
    matrix KT(vector T1, vector T2) {
        int N = rows(T1); // number of temperature measurements for the first set T1
        int M = rows(T2); // number of temperature measurements for the second set T2
        matrix[N, 4] TT1 = append_col(append_col(append_col(rep_vector(1.0, N), T1), 1e-2* T1.^2), 1e-4* T1.^3); // Mapping of temperature [1, T, 1e-2*T^2, 1e-4*T^3]
        matrix[M, 4] TT2 = append_col(append_col(append_col(rep_vector(1.0, M), T2), 1e-2* T2.^2), 1e-4* T2.^3);

        return TT1 * TT2';
    }

    // Combined kernel
    matrix K(vector x1, vector x2, vector T1, vector T2, int order) {
        return Kx(x1, x2, order) .* KT(T1, T2); // return combined kernel as the element-wise product of the 2 kernels 
    }
}

data {
    int N_points; // number of known mixtures
    int order; // order of the compositional polynomial
    vector[N_points] x; // experimental composition
    vector[N_points] T; // experimental temperatures
    vector[N_points] y; // experimental excess enthalpy
    real<lower=0> jitter; // jitter for stabalization
}

transformed data {
    real error=0.01;    // experimental error fixed to 0.3
    matrix[N_points,N_points] KNN = add_diag(K(x, x, T, T, order), jitter); // Compure kernel and add noise on diagnoal for numerical stability
    vector[N_points] y_err = square(error*y); // Experimental variances
}

parameters {
    real<lower=0> v;
}

model {
    matrix[N_points, N_points] y_cov = add_diag(KNN, y_err + v); // add data-model mismatch on the main diagnoal of the kernel
    v ~ exponential(2);
    y ~ multi_normal(rep_vector(0.0, N_points), y_cov); // likelihood functions with the underlying function marginalized
}
