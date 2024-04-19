
functions {
    // kernel for composition
    matrix Kx(vector x1, vector x2, int order) {
        int N = rows(x1);
        int M = rows(x2);
        matrix[N, order+1] X1;
        matrix[M, order+1] X2;
        for (i in 1:order) {
            X1[:,i] = x1 .^(order+2-i) - x1;
            X2[:,i] = x2 .^(order+2-i) - x2;
        }
        X1[:,order+1] = 1e-1 * x1 .* sqrt(1-x1) .* exp(x1);
        X2[:,order+1] = 1e-1 * x2 .* sqrt(1-x2) .* exp(x2);
        return X1 * X2';
    }

    // kernel for Temperature
    matrix KT(vector T1, vector T2) {
        int N = rows(T1);
        int M = rows(T2);
        matrix[N, 4] TT1 = append_col(append_col(append_col(rep_vector(1.0, N), T1), 1e-2* T1.^2), 1e-4* T1.^3);
        matrix[M, 4] TT2 = append_col(append_col(append_col(rep_vector(1.0, M), T2), 1e-2* T2.^2), 1e-4* T2.^3);

        return TT1 * TT2';
    }

    // Combined kernel
    matrix K(vector x1, vector x2, vector T1, vector T2, int order) {
        return Kx(x1, x2, order) .* KT(T1, T2); 
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
    real error=0.01;
    matrix[N_points,N_points] KNN = add_diag(K(x, x, T, T, order), jitter);
    vector[N_points] y_err = square(error*y);
}

parameters {
    real<lower=0> v;
}

model {
    matrix[N_points, N_points] y_cov = add_diag(KNN, y_err + v);
    v ~ exponential(2);
    y ~ multi_normal(rep_vector(0.0, N_points), y_cov);
}
