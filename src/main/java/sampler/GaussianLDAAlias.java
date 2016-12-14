package sampler;

import java.io.*;
import java.util.*;

import com.carrotsearch.hppc.IntArrayList;
import com.carrotsearch.hppc.cursors.IntCursor;
import knub.master_thesis.util.CorpusReader;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.special.Gamma;
import org.ejml.alg.dense.decomposition.TriangularSolver;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.CholeskyDecomposition;
import org.ejml.ops.CommonOps;

import priors.NormalInverseWishart;
import util.Util;
import util.VoseAlias;
import data.Data;


/**
 * Implementation of the collapsed gibbs sampler for Dirichlet Process Mixture Models where the distribution of each table are multivariate gaussian
 * with unknown mean and covariances. I have extensively referred Frank Wood's matalab implementation (http://www.robots.ox.ac.uk/~fwood/code/index.html)
 *
 * @author rajarshd
 */
public class GaussianLDAAlias implements Runnable {

    public static FileWriter iterationProgressWriter;


    private static List<String> vocabulary;

    /**
     * The embedding associated with each word of the vocab.
     */
    private static DenseMatrix64F[] dataVectors;
    /**
     * The corpus of documents
     */
    private static List<IntArrayList> corpus;

    /**
     * Number of iterations of Gibbs sweep
     */
    private static int numIterations;

    /**
     * Number of tables in the current iteration
     */
    private static int K;
    /**
     * Number of documents
     */
    private static int N;
    /*
     * save every x iterations
     */
    private static int saveStep;

    /**
     * In the current iteration, map of table_id's to number of customers. ****Table id starts from 0.****
     */
    private static HashMap<Integer, Integer> tableCounts = new HashMap<Integer, Integer>();
    /**
     * tableCountstableCountsPerDoc is a K X N array. tableCounts[i][j] represents how many words of document j are present in topic i.
     */
    private static int[][] tableCountsPerDoc;
    /**
     * map of table id to id of customers
     */

    //private static HashMap<Integer,Set<Integer>> tableMembers = new HashMap<Integer,Set<Integer>>();


    /**
     * Stores the table (topic) assignment of each customer in each iteration. tableAssignments[i][j] gives the table assignment of customer j of the ith document.
     */
    private static ArrayList<ArrayList<Integer>> tableAssignments;


    /**
     * mean vector associated with each table in the current iteration. This is the bayesian mean (i.e has the prior part too)
     */
    private static ArrayList<DenseMatrix64F> tableMeans = new ArrayList<DenseMatrix64F>();

    /**
     * Cholesky Lower Triangular Decomposition of covariance matrix associated with each table.
     */
    private static ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat = new ArrayList<DenseMatrix64F>();

    /**
     * log-determinant of covariance matrix for each table. Since 0.5*logDet is required in (see logMultivariateTDensity), therefore that value is kept.
     */
    private static ArrayList<Double> logDeterminants = new ArrayList<Double>();

    /**
     * the normal inverse wishart prior
     */
    private static NormalInverseWishart prior;
    private static CholeskyDecomposition<DenseMatrix64F> decomposer = DecompositionFactory.chol(Data.D, true);

    /**
     * Caching the choelsky of prior sigma0
     */
    private static DenseMatrix64F CholSigma0;
    /**
     * file path for reading vocab (to form mapping) and the initial cluster assignment
     */
    private static String resultsFolder;

    //the dirichlet hyperparam.
    private static double alpha;

    /**
     * stores the alias table for each word
     */
    private static VoseAlias[] q;

    public static boolean done = false;
    private static int MH_STEPS = 2;

    public GaussianLDAAlias() throws IOException {
        iterationProgressWriter.write("iteration\tlikelihood\ttime%n");
    }
/************************************Member Declaration Ends***********************************/
    /**
     * updates params -- mean and the cholesky decomp of covariance matrix using rank1 update (customer added) or downdate (customer removed)
     *
     * @param tableId
     * @param custId
     * @param isRemoved
     */
    private static void updateTableParams(int tableId, int custId, boolean isRemoved) {
        int count = tableCounts.get(tableId);
        double k_n = prior.k_0 + count;
        double nu_n = prior.nu_0 + count;
        double scaleTdistrn = (k_n + 1) / (k_n * (nu_n - Data.D + 1));

        DenseMatrix64F oldLTriangularDecomp = tableCholeskyLTriangularMat.get(tableId);
        if (isRemoved) {
            /**
             * Now use the rank1 downdate to calculate the cholesky decomposition of the updated covariance matrix
             * the update equaltion is \Sigma_(N+1) =\Sigma_(N) - (k_0 + N+1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
             * therefore x = sqrt((k_0 + N - 1)/(k_0 + N)) (X_{n} - \mu_{n})
             * Note here \mu_n will be the mean before updating. After updating sigma_n, we will update \mu_n.
             */
            DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
            CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
            double coeff = Math.sqrt((k_n + 1) / k_n);
            CommonOps.scale(coeff, x);
            Util.cholRank1Downdate(oldLTriangularDecomp, x);
            tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map
            //updateMean(tableId);
            DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
            CommonOps.scale(k_n + 1, tableMeans.get(tableId), newMean);
            CommonOps.subEquals(newMean, dataVectors[custId]);
            CommonOps.divide(k_n, newMean);
            tableMeans.set(tableId, newMean);

        } else //new customer is added
        {
            DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
            CommonOps.scale(k_n - 1, tableMeans.get(tableId), newMean);
            CommonOps.addEquals(newMean, dataVectors[custId]);
            CommonOps.divide(k_n, newMean);
            tableMeans.set(tableId, newMean);
            /**
             * The rank1 update equation is
             * \Sigma_{n+1} = \Sigma_{n} + (k_0 + n + 1)/(k_0 + n) * (x_{n+1} - \mu_{n+1})(x_{n+1} - \mu_{n+1})^T
             */
            DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
            CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
            double coeff = Math.sqrt(k_n / (k_n - 1));
            CommonOps.scale(coeff, x);
            Util.cholRank1Update(oldLTriangularDecomp, x);
            tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map
        }
        //calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
        double logDet = 0.0;
        for (int l = 0; l < Data.D; l++)
            logDet = logDet + Math.log(oldLTriangularDecomp.get(l, l));
        logDet += Data.D * Math.log(scaleTdistrn) / (double) 2;

        if (tableId < logDeterminants.size())
            logDeterminants.set(tableId, logDet);
        else
            logDeterminants.add(logDet);

    }

    /**
     * Initialize the gibbs sampler state. I start with log N tables and randomly initialize customers to those tables.
     *
     * @throws IOException
     */
    public static void initialize() throws IOException {
        //first check the prior degrees of freedom. It has to be >= num_dimension
        if (prior.nu_0 < (double) Data.D) {
            System.out.println("The initial degrees of freedom of the prior is less than the dimension!. Setting it to the number of dimension: " + Data.D);
            prior.nu_0 = Data.D;
        }
        //storing zeros in sumTableCustomers and later will keep on adding each customer. Also initialize tableInverseCovariances and determinants
        double scaleTdistrn = (prior.k_0 + 1) / (double) (prior.k_0 * (prior.nu_0 - Data.D + 1));
        for (int i = 0; i < K; i++) {
            DenseMatrix64F priorMean = new DenseMatrix64F(prior.mu_0[i]);
            DenseMatrix64F initialCholesky = new DenseMatrix64F(CholSigma0);
            //calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
            double logDet = 0.0;
            for (int l = 0; l < Data.D; l++)
                logDet = logDet + Math.log(CholSigma0.get(l, l));
            logDet += Data.D * Math.log(scaleTdistrn) / (double) 2;
            logDeterminants.add(logDet);
            tableMeans.add(priorMean);
            tableCholeskyLTriangularMat.add(initialCholesky);
        }
        //randomly assign customers to tables.
        Random gen = new Random();
        for (int d = 0; d < N; d++) //for each document
        {
            IntArrayList doc = corpus.get(d);
            int wordCounter = 0;
            tableAssignments.add(new ArrayList<Integer>());
            for (IntCursor c : doc) //for each word in the document.
            {
                int i = c.value;
                int tableId = gen.nextInt(K);
                tableAssignments.get(d).add(tableId);
                if (tableCounts.containsKey(tableId)) {
                    int prevCount = tableCounts.get(tableId);
                    tableCounts.put(tableId, prevCount + 1);
                } else
                    tableCounts.put(tableId, 1);
                tableCountsPerDoc[tableId][d]++; //because in table 'tableId', one more customer is sitting
                //Now table 'tableId' has a new word 'i'. Therefore we will have to update the params of the table (topic)
                updateTableParams(tableId, i, false);
                wordCounter++;
            }
        }
        //double check again
        for (int i = 0; i < K; i++)
            if (!tableCounts.containsKey(i)) {
                System.out.println("Still some tables are empty....exiting!");
                System.exit(1);
            }

        //calculate initial avg ll
        double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K, N, prior, tableCountsPerDoc);
        iterationProgressWriter.write(String.format("0\t%.4f\0%n", avgLL));
    }

    private static double logMultivariateTDensity(DenseMatrix64F x, int tableId) {
        double logprob = 0.0;
        int count = tableCounts.get(tableId);
        double k_n = prior.k_0 + count;
        double nu_n = prior.nu_0 + count;
        double scaleTdistrn = Math.sqrt((k_n + 1) / (k_n * (nu_n - Data.D + 1)));
        double nu = prior.nu_0 + count - Data.D + 1;
        //Since I am storing lower triangular matrices, therefore it is easy to calculate the value of (x-\mu)^T\Sigma^-1(x-\mu)
        //therefore I am gonna use triangular solver
        //first calculate (x-mu)
        DenseMatrix64F x_minus_mu = new DenseMatrix64F(Data.D, 1);
        CommonOps.sub(x, tableMeans.get(tableId), x_minus_mu);
        //now scale the lower triangular matrix
        DenseMatrix64F lTriangularChol = new DenseMatrix64F(Data.D, Data.D);
        CommonOps.scale(scaleTdistrn, tableCholeskyLTriangularMat.get(tableId), lTriangularChol);
        TriangularSolver.solveL(lTriangularChol.data, x_minus_mu.data, Data.D); //now x_minus_mu has the solved value
        //Now take xTx
        DenseMatrix64F x_minus_mu_T = new DenseMatrix64F(1, Data.D);
        CommonOps.transpose(x_minus_mu, x_minus_mu_T);
        DenseMatrix64F mul = new DenseMatrix64F(1, 1);
        CommonOps.mult(x_minus_mu_T, x_minus_mu, mul);
        double val = mul.get(0, 0);
        logprob = Gamma.logGamma((nu + Data.D) / 2) - (Gamma.logGamma(nu / 2) + Data.D / 2 * (Math.log(nu) + Math.log(Math.PI)) + logDeterminants.get(tableId) + (nu + Data.D) / 2 * Math.log(1 + val / nu));
        return logprob;
    }

    private static void sample(Util writer) throws IOException, InterruptedException {
        initRun();
        Thread t1 = (new Thread(new GaussianLDAAlias()));
        t1.start();
        writer.printTopWords(tableMeans, tableCholeskyLTriangularMat, dataVectors, vocabulary, 0, 10);
        for (int currentIteration = 1; currentIteration <= numIterations; currentIteration++) {
            long startTime = System.currentTimeMillis();
            for (int d = 0; d < corpus.size(); d++) {
                IntArrayList document = corpus.get(d);
                int wordCounter = 0;
                for (IntCursor c : document) {
                    int custId = c.value;
                    //remove custId from his old_table
                    int oldTableId = tableAssignments.get(d).get(wordCounter);
                    tableAssignments.get(d).set(wordCounter, -1);
                    int oldCount = tableCounts.get(oldTableId);
                    tableCounts.put(oldTableId, oldCount - 1); //decrement count
                    tableCountsPerDoc[oldTableId][d]--; //topic 'oldTableId' has one member less.
                    //now recalculate table parameters for this table
                    updateTableParams(oldTableId, custId, true);
                    //Now calculate the prior and likelihood for the customer to sit in each table and sample.
                    ArrayList<Double> posterior = new ArrayList<Double>();
                    ArrayList<Integer> nonZeroTopicIndex = new ArrayList<Integer>();
                    Double max = Double.NEGATIVE_INFINITY;
                    double pSum = 0;
                    //go over each table
                    for (int k = 0; k < K; k++) {
                        if (tableCountsPerDoc[k][d] > 0) {
                            //Now calculate the likelihood
                            //double count = tableCountsPerDoc[k][d]+alpha;//here count is the number of words of the same doc which are sitting in the same topic.
                            double logLikelihood = logMultivariateTDensity(dataVectors[custId], k);
                            //System.out.println(custId+" "+k+" "+logLikelihood);
                            //add log prior in the posterior vector
                            //double logPosterior = Math.log(count) + logLikelihood;
                            double logPosterior = Math.log(tableCountsPerDoc[k][d]) + logLikelihood;
                            nonZeroTopicIndex.add(k);
                            posterior.add(logPosterior);
                            if (logPosterior > max)
                                max = logPosterior;
                        }

                    }
                    //to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                    //the log p_max cancels put and prevents overflow in the exponentiating phase.
                    for (int k = 0; k < posterior.size(); k++) {
                        double p = posterior.get(k);
                        p = p - max;
                        double expP = Math.exp(p);
                        pSum += expP;
                        posterior.set(k, pSum);
                    }
                    //now sample an index from this posterior vector. The sample method will normalize the vector
                    //so no need to normalize now.
                    double select_pr = pSum / (pSum + alpha * q[custId].wsum);

                    //MHV to draw new topic
                    Random rand = new Random();
                    int newTableId = -1;
                    for (int r = 0; r < MH_STEPS; ++r) {
                        //1. Flip a coin
                        if (rand.nextDouble() < select_pr) {
                            double u = rand.nextDouble() * pSum;
                            int temp = Util.binSearchArrayList(posterior, u, 0, posterior.size() - 1);
                            newTableId = nonZeroTopicIndex.get(temp);
                        } else {
                            newTableId = q[custId].sampleVose();
                        }

                        if (oldTableId != newTableId) {
                            //2. Find acceptance probability
                            double temp_old = logMultivariateTDensity(dataVectors[custId], oldTableId);
                            double temp_new = logMultivariateTDensity(dataVectors[custId], newTableId);
                            double acceptance = (tableCountsPerDoc[newTableId][d] + alpha) / (tableCountsPerDoc[oldTableId][d] + alpha)
                                    * Math.exp(temp_new - temp_old)
                                    * (tableCountsPerDoc[oldTableId][d] * temp_old + alpha * q[custId].w[oldTableId])
                                    / (tableCountsPerDoc[newTableId][d] * temp_new + alpha * q[custId].w[newTableId]);

                            //3. Compare against uniform[0,1]
                            double u = rand.nextDouble();
                            if (u < acceptance)
                                oldTableId = newTableId;
                        }
                    }
                    tableAssignments.get(d).set(wordCounter, newTableId);
                    tableCounts.put(newTableId, tableCounts.get(newTableId) + 1);
                    tableCountsPerDoc[newTableId][d]++;
                    updateTableParams(newTableId, custId, false);
                    wordCounter++;
                }
//                if (d % 1000 == 0) {
//                    System.out.println(String.format("Finished %d. Took %d s", d, (System.currentTimeMillis() - iterationStartTime) / 1000));
//                    iterationStartTime = System.currentTimeMillis();
//                }
            }
            //Printing stuffs now
            long stopTime = System.currentTimeMillis();
            long elapsedTime = (stopTime - startTime) / 1000;

            //calculate perplexity
            double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K, N, prior, tableCountsPerDoc);
            String iterationOutput = String.format("%d\t%.4f\t%d%n", currentIteration, avgLL, elapsedTime);
            System.out.println(iterationOutput + " " + new Date());
            iterationProgressWriter.write(iterationOutput);

            if (currentIteration == numIterations || currentIteration % saveStep == 0) {
                if (currentIteration == numIterations) {
                    writer.printGaussians(tableMeans, tableCholeskyLTriangularMat, currentIteration);
                    writer.printDocumentTopicDistribution(tableCountsPerDoc, alpha, currentIteration);
                    writer.printTableAssignments(tableAssignments, currentIteration);
                    writer.printNumCustomersPerTopic(tableCountsPerDoc, currentIteration);
                    writer.printTopWords(tableMeans, tableCholeskyLTriangularMat, dataVectors, vocabulary, currentIteration, 500);
                }
                writer.printTopWords(tableMeans, tableCholeskyLTriangularMat, dataVectors, vocabulary, currentIteration, 10);
            }
        }
        done = true;
        iterationProgressWriter.close();
        t1.join();
    }


    private static List<String> readVocabulary(String vocabularyFile) throws IOException {
        List<String> vocabulary = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(vocabularyFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                vocabulary.add(line);
            }
        }
        return vocabulary;
    }

    @Override
    public void run() {

        VoseAlias temp = new VoseAlias();
        temp.init(K);
        //temp.init_temp();
        do {
            for (int w = 0; w < Data.numVectors; ++w) {
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < K; k++) {
                    double logLikelihood = logMultivariateTDensity(dataVectors[w], k);
                    //posterior.add(logLikelihood);
                    temp.w[k] = logLikelihood;
                    if (logLikelihood > max)
                        max = logLikelihood;
                }
                //to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
                //the log p_max cancels put and prevents overflow in the exponentiating phase.
                temp.wsum = 0.0;
                for (int k = 0; k < K; k++) {
                    double p = temp.w[k];
                    p = p - max;
                    double expP = Math.exp(p);
                    temp.wsum += expP;
                    temp.w[k] = expP;
                }
                temp.generateTable();
                q[w].copy(temp);
            }
        } while (!done);

    }


    public static void initRun() {
        VoseAlias temp = new VoseAlias();
        temp.init(K);
        //temp.init_temp();
        for (int w = 0; w < Data.numVectors; ++w) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < K; k++) {
                double logLikelihood = logMultivariateTDensity(dataVectors[w], k);
                //posterior.add(logLikelihood);
                temp.w[k] = logLikelihood;
                if (logLikelihood > max)
                    max = logLikelihood;
            }
            //to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
            //the log p_max cancels put and prevents overflow in the exponentiating phase.
            temp.wsum = 0.0;
            for (int k = 0; k < K; k++) {
                double p = temp.w[k];
                p = p - max;
                double expP = Math.exp(p);
                temp.wsum += expP;
                temp.w[k] = expP;
            }
            temp.generateTable();
            q[w].copy(temp);
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        long startTime = System.currentTimeMillis();

        Data.embeddingFileName = args[0];

        int D = Integer.parseInt(args[1]);
        Data.D = D;

        numIterations = Integer.parseInt(args[2]);

        K = Integer.parseInt(args[3]);

        resultsFolder = args[4];
        //noinspection ResultOfMethodCallIgnored
        new File(resultsFolder).mkdirs();

        iterationProgressWriter = new FileWriter(resultsFolder + "/iterations.txt");

        DenseMatrix64F data = Data.readData();
        dataVectors = new DenseMatrix64F[data.numRows]; //splitting into vectors
        CommonOps.rowsToVector(data, dataVectors);
        //Read corpus
        String inputCorpusFile = args[5];
//        corpus = Data.readCorpus(inputCorpusFile);
        corpus = CorpusReader.readCorpus(inputCorpusFile).documents();
        vocabulary = readVocabulary(args[6]);
        saveStep = Integer.parseInt(args[7]);
        alpha = Double.parseDouble(args[8]);
        System.out.println(String.format("vector-dimensions: %d, iterations: %d, num-topics: %d, alpha: %s",
                D, numIterations, K, alpha));
        N = corpus.size();
        System.out.println(String.format("num-documents: %d, num-vectors: %d", N, data.numRows));
        //initialize the prior
        prior = new NormalInverseWishart();

        prior.mu_0 = new DenseMatrix64F[K];
        List<String> lines = FileUtils.readLines(new File(new File(inputCorpusFile).getParent() + "/model.ssv"));
        int k = 0;
        for (String line : lines) {
            if (line.contains("topic-count"))
                continue;
            String[] split = line.split(" ");
            int lenSplit = split.length;
            List<DenseMatrix64F> topicVectors = new ArrayList<>(10);
//            System.out.println(line);
            for (int i = 0; i < 10; i += 1) {
                String word = split[lenSplit - 1 - i];
//                System.out.print(word + " ");
                int idx = vocabulary.indexOf(word);
                if (idx > -1)
                    topicVectors.add(dataVectors[idx]);
            }
//            System.out.println();
            prior.mu_0[k] = Util.getSampleMean(dataVectors); //topicVectors.toArray(new DenseMatrix64F[topicVectors.size()]));
            k += 1;
        }
        assert k == 50 || k == 250;

        prior.nu_0 = Data.D; //initializing to the dimension
        prior.sigma_0 = CommonOps.identity(Data.D); //setting as the identity matrix
        CommonOps.scale(3 * Data.D, prior.sigma_0);
        prior.k_0 = 0.1;
        CholSigma0 = new DenseMatrix64F(Data.D, Data.D);
        CommonOps.addEquals(CholSigma0, prior.sigma_0);
        if (!decomposer.decompose(CholSigma0))//cholesky decomp
        {
            System.out.println("Matrix couldnt be Cholesky decomposed");
            System.exit(1);
        }
        //Now initialize each datapoint (customer)
        tableAssignments = new ArrayList<ArrayList<Integer>>();
        tableCountsPerDoc = new int[K][N];

        q = new VoseAlias[Data.numVectors];
        for (int w = 0; w < Data.numVectors; w++) {
            q[w] = new VoseAlias();
            q[w].init(K);
        }

        Util writer = new Util(resultsFolder, N, K);

        /**************** Initialize ***********/
        initialize();
        /******sample*********/
        sample(writer);
        long stopTime = System.currentTimeMillis();
        long elapsedTime = (stopTime - startTime) / 1000;
        System.out.println("Time: " + elapsedTime + " s");
        FileUtils.writeStringToFile(new File(resultsFolder + "/runtime.txt"), String.valueOf(elapsedTime));
    }
}
