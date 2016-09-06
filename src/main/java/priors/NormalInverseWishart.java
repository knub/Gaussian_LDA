package priors;

import org.ejml.data.DenseMatrix64F;

public class NormalInverseWishart {

    /**
     * Hyperparam mean vector.
     */
    public DenseMatrix64F[] mu_0;

    /**
     * initial degrees of freedom
     */
    public double nu_0;

    /**
     * Hyperparam covariance matrix
     */
    public DenseMatrix64F sigma_0;

    /**
     * mean fraction
     */
    public double k_0;

}
