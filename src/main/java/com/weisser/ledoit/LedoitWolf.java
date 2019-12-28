/*
 * Created on 13.09.2006
 */
package com.weisser.ledoit;

import java.io.PrintStream;

/**
 * Optimize the covariance matrix before portfolio optimization.
 *
 * This implementation is based on the Paper
 * <em>"Honey, I shrunk the sample covariance matrix"</em> by Olivier Ledoit.
 *
 * TODO Write tests.
 * TODO Warnings, when the matrix sigma holds NaN values. Mean variance optimization cannot be done in this case.
 * TODO Performance optimizations.
 */
public class LedoitWolf {

    /**
     * Toggles debug output on / off.
     */
    private final boolean debug = true;

    /**
     * Returns matrix (t by n).
     * After calling mean() it will hold the de-meaned returns.
     */
    private double[][] x;

    /**
     * Elementwise squared return matrix.
     * y = x.^2
     */
    private double[][] y;

    /**
     *
     */
    private double[][] v;

    /**
     * Sample covariance matrix (n by n).
     */
    private double[][] sample;

    /**
     * Shrinkage target matrix (n by n).
     */
    private double[][] shrinkage;

    /**
     * The resulting matrix sigma (n by n).
     */
    private double[][] sigma;

    /**
     * Return means of each asset.
     */
    private double[] meanx;

    /**
     * The number of assets.
     */
    private int n;

    /**
     * The number of returns in matrix x.
     */
    private int t;

    /**
     * Rho.
     */
    private double rho;

    /**
     * Program entry point (small test routine with sample data).
     * @param args Commandline arguments (not used.)
     */
    public static void main(String[] args) {
        // Matrix of prices.
        double[][] x = {
            {1, 2, 3, 4},
            {3, 4, 5, 6},
            {6, 7, 8, 9},
            {3, 4, 5, 3},
            {5, 6, 7, 7},
            {3, 3, 8, 7},
            {4, 5, 6, 6}};

        // 4 assets, 7 price samples each.
        LedoitWolf l = new LedoitWolf(x, 7, 4);

        // The matrix sigma.
        double[][] mySigma = l.getSigma();
    }

    /**
     * Creates a new Ledoit object.
     * @param x The (t x n) matrix of returns.
     * @param t Number of assets.
     * @param n Number of samples.
     */
    public LedoitWolf(double[][] x, int t, int n) {
        this.x = x;
        this.n = n;
        this.t = t;

        sample = covarianceMatrix(this.x, n, t);
        shrinkage = shrinkageTarget(sample, n);

        double d = _comp_d();
        double r2 = _comp_r2();
        double phidiag = _comp_phidiag();
        _comp_v();
        double phioff = _comp_phioff();
        double phi = phidiag + rho * phioff;

        if (debug) {
            System.out.println("phi = " + phi);
        }

        double s = Math.max(0, Math.min(1, (r2 - phi) / d));

        computeSigma(s);
    }

    /**
     * Compute mean for each of the n series and de-mean returns.
     * @param x The (t x n) matrix of returns.
     * @param t Number of assets.
     * @param n Number of samples.
     */
    private void mean(double[][] x, int t, int n) {
        int i, j;
        meanx = new double[n];

        for (j = 0; j < n; j++) {
            meanx[j] = 0.0;
            for (i = 0; i < t; i++) {
                meanx[j] += x[i][j];
            }
            meanx[j] /= t;

            for (i = 0; i < t; i++) {
                x[i][j] -= meanx[j];
            }
        }
    }

    /**
     * Computes the sample covariance matrix from the given time series.
     * @param x The (t x n) matrix of returns.
     * @param t Number of assets.
     * @param n Number of samples.
     * @return
     */
    public final double[][] covarianceMatrix(double[][] x, int n, int t) {
        int i, j, k;

        sample = new double[n][n];

        // Compute means and de-mean matrix x.
        mean(x, t, n);

        if (debug) {
            //System.out.println("x de-meaned:");
            //showMatrix(x, t, n);

            showMatrix(System.out, "x de-meaned:", x);
        }

        //
        // Compute sample covariance matrix.
        //
        for (k = 0; k < n; k++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                for (i = 0; i < t; i++) {
                    sum += x[i][k] * x[i][j];
                }
                sample[k][j] = sum / t;
            }
        }

        if (debug) {
            //System.out.println("sample covariance:");
            //showMatrix(sample, n, n);

            showMatrix(System.out, "sample covariance:", sample);
        }

        return sample;
    }

    /**
     * Computes rho and the shrinkage target.
     *
     * @param sample The n by n sample covariance matrix.
     * @param n
     * @return
     */
    public final double[][] shrinkageTarget(double[][] sample, int n) {
        int i, j;
        shrinkage = new double[n][n];

        rho = 0.0;

        // TODO compute rho more efficient, precalc the sqrt's
        for (i = 0; i < n - 1; i++) {
            for (j = i + 1; j < n; j++) {
                rho += sample[i][j] / Math.sqrt(sample[i][i] * sample[j][j]);
            }
        }

        rho *= 2.0 / ((n - 1) * n);

        if (debug) {
            System.out.println("rho = " + rho);
        }

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (i == j) {
                    shrinkage[i][i] = sample[i][i];
                } else {
                    shrinkage[i][j] = rho * Math.sqrt(sample[i][i] * sample[j][j]);
                }
            }
            System.out.println();
        }

        if (debug) {
            //System.out.println("shrinkage:");
            //showMatrix(shrinkage, n, n);

            showMatrix(System.out, "shrinkage:", shrinkage);
        }

        return shrinkage;
    }


//-------------------------------------------------
// 1:1 implementation of the matlab code
//	-------------------------------------------------
/*
    % compute shrinkage parameters
    d=1/n*norm(sample-prior,'fro')^2; % 0.30877
    y=x.^2; % quadriert die einzelelemente der matrix x
    r2=1/n/t^2*sum(sum(y'*y))-1/n/t*sum(sum(sample.^2)); % 12.765
    phidiag=1/n/t^2*(sum(sum(y.^2))-1/n/t*sum(diag(sample).^2));
    v=((x.^3)'*x)/(t^2)-(var(:,ones(1,n)).*sample)/t;
    v(logical(eye(n)))=zeros(n,1);
    phioff=sum(sum(v.*(sqrtvar(:,ones(n,1))./sqrtvar(:,ones(n,1))')))/n;
    phi=phidiag+rho*phioff;
    % compute the estimator
    shrinkage=max(0,min(1,(r2-phi)/d));
    sigma=shrinkage*prior+(1-shrinkage)*sample;
     */
    /**
     *
     */
    private double _comp_d() {
        int i, j;

        double d = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double tmp = sample[i][j] - shrinkage[i][j];
                d += tmp * tmp;
            }
        }

        return d / n;
    }

    /**
     *
     */
    private void _comp_y() {
        int i, j;

        // y=x.^2; % quadriert die einzelelemente der matrix x
        y = new double[t][n];
        for (i = 0; i < t; i++) {
            for (j = 0; j < n; j++) {
                y[i][j] = x[i][j] * x[i][j];
            }
        }

        if (debug) {
            //System.out.println("y");
            //showMatrix(y, t, n);

            showMatrix(System.out, "y", y);
        }
    }

    /**
     * r2=1/n/t^2*sum(sum(y'*y))-1/n/t*sum(sum(sample.^2)); % 12.765
     * @return
     */
    private double _comp_r2() {
        int i, j, k;

        _comp_y();

        // y : [t x n] Matrix
        // sum(sum(y'*y))
        double sum1 = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                for (k = 0; k < t; k++) {
                    sum1 += y[k][i] * y[k][j];   // not y[i][k], because we need y' * y
                }
            }
        }

        // sum(sum(sample.^2))
        double sum2 = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                sum2 += sample[i][j] * sample[i][j];
            }
        }

        double r2 = 1.0 / n / (t * t) * sum1 - 1.0 / n / t * sum2;

        if (debug) {
            System.out.println("r2 = " + r2);
        }
        return r2;
    }

    /**
     * Compute phidiag.
     * @return phidiag=1/n/t^2*(sum(sum(y.^2))-1/n/t*sum(diag(sample).^2))
     */
    private double _comp_phidiag() {
        int i, j;

        double sum3 = 0.0;
        for (i = 0; i < t; i++) {
            for (j = 0; j < n; j++) {
                sum3 += y[i][j] * y[i][j];
            }
        }

        double sum4 = 0.0;
        for (i = 0; i < n; i++) {
            sum4 += sample[i][i] * sample[i][i];
        }

        double phidiag = 1.0 / n / (t * t) * (sum3 - 1.0 / n / t * sum4);

        if (debug) {
            System.out.println("phidiag = " + phidiag);
        }

        return phidiag;
    }

    /**
     * v=((x.^3)'*x)/(t^2)-(var(:,ones(1,n)).*sample)/t;
     *
     */
    private void _comp_v() {
        int i, j, k;

        double tmp1 = t * t;

        v = new double[n][n];

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {

                v[i][j] = 0.0;
                for (k = 0; k < t; k++) {
                    v[i][j] += x[k][i] * x[k][i] * x[k][i] * x[k][j];  // (x.^3)'*x / t^2
                }
                v[i][j] /= tmp1;

                v[i][j] -= sample[i][j] * sample[i][i] / t; // (var(:,ones(1,n)).*sample)/t
            }
            v[i][i] = 0.0; // v(logical(eye(n)))=zeros(n,1);
        }

        if (debug) {
            //System.out.println("v:");
            //showMatrix(v, n, n);

            showMatrix(System.out, "v", v);
        }
    }

    private double _comp_phioff() {
        int i, j;
        double phioff = 0.0;

        for (i = 0; i < n; i++) {
            double tmp1 = Math.sqrt(sample[i][i]);

            for (j = 0; j < n; j++) {
                double tmp2 = Math.sqrt(sample[j][j]);
                phioff += v[i][j] * tmp1 / tmp2;
            }
        }

        phioff /= n;

        if (debug) {
            System.out.println("phioff = " + phioff);
        }

        return phioff;
    }

    /**
     * Computes the shrinked matrix <em>sigma</em>.
     * <pre>
     * sigma = shrinkage * prior + (1 - shrinkage) * sample.
     * </pre>
     * @param s Shrinkage intensity.
     */
    private void computeSigma(double s) {
        int i, j;

        double s1 = 1 - s;
        sigma = new double[n][n];
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                sigma[i][j] = shrinkage[i][j] * s + sample[i][j] * s1;
            }
        }
    }

    /**
     * Returns the optimized matrix <em>sigma</em>.
     * @return The desired matrix <em>sigma</em>.
     */
    public double[][] getSigma() {
        return sigma;
    }

    /**
     * Displays the given matrix of doubles on a PrintStream.
     * @param out The PrintStream.
     * @param name The name of matrix (It's just used to identify it in the print stream).
     * @param m The matrix of doubles.
     */
    public static void showMatrix(PrintStream out, String name, double[][] m) {
        out.println(name);
        for (int i = 0; i < m.length; i++) {
            out.print("[ ");
            for (int j = 0; j < m[i].length; j++) {
                out.print(m[i][j] + "     ");
            }
            out.println(" ]");
        }
    }
}
