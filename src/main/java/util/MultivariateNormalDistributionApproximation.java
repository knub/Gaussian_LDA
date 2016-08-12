package util;

import org.apache.commons.math3.distribution.AbstractMultivariateRealDistribution;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.random.Well19937c;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;

/**
 * MultivariateNormalDistribution without constant factors
 */
public class MultivariateNormalDistributionApproximation extends AbstractMultivariateRealDistribution {
    /** Vector of means. */
    private final double[] means;
    /** The matrix inverse of the covariance matrix. */
    private final RealMatrix covarianceMatrixInverse;

    public MultivariateNormalDistributionApproximation(final double[] means,
                                          final double[][] covariances)
            throws SingularMatrixException,
            DimensionMismatchException,
            NonPositiveDefiniteMatrixException {
        this(new Well19937c(), means, covariances);
    }

    public MultivariateNormalDistributionApproximation(RandomGenerator rng,
                                          final double[] means,
                                          final double[][] covariances)
            throws SingularMatrixException,
            DimensionMismatchException,
            NonPositiveDefiniteMatrixException {
        super(rng, means.length);

        final int dim = means.length;

        if (covariances.length != dim) {
            throw new DimensionMismatchException(covariances.length, dim);
        }

        for (int i = 0; i < dim; i++) {
            if (dim != covariances[i].length) {
                throw new DimensionMismatchException(covariances[i].length, dim);
            }
        }

        this.means = MathArrays.copyOf(means);

        /* Covariance matrix. */
        RealMatrix covarianceMatrix = new Array2DRowRealMatrix(covariances);

        // Covariance matrix eigen decomposition.
        final EigenDecomposition covMatDec = new EigenDecomposition(covarianceMatrix);

        // Compute and store the inverse.
        covarianceMatrixInverse = covMatDec.getSolver().getInverse();

        // Eigenvalues of the covariance matrix.
        final double[] covMatEigenvalues = covMatDec.getRealEigenvalues();

        for (int i = 0; i < covMatEigenvalues.length; i++) {
            if (covMatEigenvalues[i] < 0) {
                throw new NonPositiveDefiniteMatrixException(covMatEigenvalues[i], i, 0);
            }
        }

        // Matrix where each column is an eigenvector of the covariance matrix.
        final Array2DRowRealMatrix covMatEigenvectors = new Array2DRowRealMatrix(dim, dim);
        for (int v = 0; v < dim; v++) {
            final double[] evec = covMatDec.getEigenvector(v).toArray();
            covMatEigenvectors.setColumn(v, evec);
        }

        final RealMatrix tmpMatrix = covMatEigenvectors.transpose();

        // Scale each eigenvector by the square root of its eigenvalue.
        for (int row = 0; row < dim; row++) {
            final double factor = FastMath.sqrt(covMatEigenvalues[row]);
            for (int col = 0; col < dim; col++) {
                tmpMatrix.multiplyEntry(row, col, factor);
            }
        }
    }

    @Override
    public double density(double[] x) {
        return getExponentTerm(x);
    }


    private double getExponentTerm(final double[] values) {
        final double[] centered = new double[values.length];
        for (int i = 0; i < centered.length; i++) {
            centered[i] = values[i] - means[i];
        }
        final double[] preMultiplied = covarianceMatrixInverse.preMultiply(centered);
        double sum = 0;
        for (int i = 0; i < preMultiplied.length; i++) {
            sum += preMultiplied[i] * centered[i];
        }
        return -sum;
    }

    @Override
    public double[] sample() {
        return new double[0];
    }
}
