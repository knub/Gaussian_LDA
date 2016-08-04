package util;

import data.Data;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.util.FastMath;
import org.ejml.alg.dense.decomposition.TriangularSolver;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.RandomMatrices;
import priors.NormalInverseWishart;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.Map.Entry;

@SuppressWarnings({"WeakerAccess", "unused"})
public class Util {


    /**
     * This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A) where
     * A' = A + x*x^T.
     * Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
     */
    public static void cholRank1Update(DenseMatrix64F L, DenseMatrix64F x) {
        //L should be a square lower triangular matrix (although not checking for triangularity here explicitly)
        //Data.numDimensions = 2;
        assert L.numCols == Data.numDimensions;
        assert L.numRows == Data.numDimensions;
        //x should be a vector
        assert x.numCols == 1;
        assert x.numRows == Data.numDimensions;

        for (int k = 0; k < Data.numDimensions; k++) {
            double r = Math.sqrt(Math.pow(L.get(k, k), 2) + Math.pow(x.get(k, 0), 2));
            double c = r / L.get(k, k);
            double s = x.get(k, 0) / L.get(k, k);
            L.set(k, k, r);
            for (int l = k + 1; l < Data.numDimensions; l++) {
                double val = (L.get(l, k) + s * x.get(l, 0)) / c;
                L.set(l, k, val);
                val = c * x.get(l, 0) - s * val;
                x.set(l, 0, val);
            }
        }
    }

    /**
     * This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A) where
     * A' = A - x*x^T.
     * Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
     */
    public static void cholRank1Downdate(DenseMatrix64F L, DenseMatrix64F x) {
        //L should be a square lower triangular matrix (although not checking for triangularity here explicitly)
        //Data.numDimensions = 2;
        assert L.numCols == Data.numDimensions;
        assert L.numRows == Data.numDimensions;
        //x should be a vector
        assert x.numCols == 1;
        assert x.numRows == Data.numDimensions;

        for (int k = 0; k < Data.numDimensions; k++) {
            double r = Math.sqrt(L.get(k, k) * L.get(k, k) - x.get(k, 0) * x.get(k, 0));
            double c = r / L.get(k, k);
            double s = x.get(k, 0) / L.get(k, k);
            L.set(k, k, r);
            for (int l = k + 1; l < Data.numDimensions; l++) {
                double val = (L.get(l, k) - s * x.get(l, 0)) / c;
                L.set(l, k, val);
                val = c * x.get(l, 0) - s * L.get(l, k);
                x.set(l, 0, val);
            }
        }
    }

    public static int sample(List<Double> probs) {
        double[] cumulative_probs = new double[probs.size()];
        double sum_probs = 0;
        int counter = 0;
        for (double prob : probs) {
            sum_probs = sum_probs + prob;
            cumulative_probs[counter] = sum_probs;
            counter++;
        }
        if (sum_probs != 1)        //normalizing
            for (int i = 0; i < probs.size(); i++) {
                probs.set(i, probs.get(i) / sum_probs);
                //cumulative_probs.set(i, cumulative_probs.get(i)/sum_probs);
                cumulative_probs[i] = cumulative_probs[i] / sum_probs;
            }
        //cumulative_probs should be sorted, therefore do binary search
        Random r = new Random();
        double nextRandom = r.nextDouble();
        @SuppressWarnings("UnnecessaryLocalVariable")
        int sample = binSearch(cumulative_probs, nextRandom, 0, cumulative_probs.length - 1);
        return sample;

    }

    public static int binSearch(double[] cumProb, double key, int start, int end) {
        if (start > end)
            return start;

        int mid = (start + end) / 2;
        if (key == cumProb[mid])
            return mid + 1;
        if (key < cumProb[mid])
            return binSearch(cumProb, key, start, mid - 1);
        if (key > cumProb[mid])
            return binSearch(cumProb, key, mid + 1, end);
        return -1;
    }

    public static int binSearchArrayList(ArrayList<Double> cumProb, double key, int start, int end) {
        if (start > end)
            return start;

        int mid = (start + end) / 2;
        if (key == cumProb.get(mid))
            return mid + 1;
        if (key < cumProb.get(mid))
            return binSearchArrayList(cumProb, key, start, mid - 1);
        if (key > cumProb.get(mid))
            return binSearchArrayList(cumProb, key, mid + 1, end);
        return -1;
    }


    private static void getClusterNum() {
        try {
            BufferedReader reader1 =
                    new BufferedReader(
                            new InputStreamReader(new FileInputStream("data/test/bergsma_vocab.txt"), "UTF-8"));

            BufferedReader reader2 =
                    new BufferedReader(
                            new InputStreamReader(new FileInputStream("data/train/cluster_all_tweets.txt"), "UTF-8"));

            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("data/test/bergsma_cluster_nums.txt"), "UTF-8"));

            String word;
            while ((word = reader1.readLine()) != null) {
                int flag = 0;
                String cluster_line;
                while ((cluster_line = reader2.readLine()) != null) {
                    String[] word_clusternum = cluster_line.split(" ");
                    String c_word = word_clusternum[0];
                    String c = word_clusternum[1];
                    if (c_word.equals(word)) {
                        System.out.println(word);
                        out.write(c + "\n");
                        flag = 1;
                        break;
                    }
                }
                if (flag == 0) {
                    System.out.println("Couldnot find the word " + word);
                    System.exit(1);
                }
            }
            reader1.close();
            out.close();
            reader2.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * mean of the data
     */
    public static DenseMatrix64F getSampleMean(DenseMatrix64F[] data) {
        DenseMatrix64F mean = new DenseMatrix64F(Data.numDimensions, 1);//initialized to 0
        for (DenseMatrix64F vec : data)
            CommonOps.addEquals(mean, vec);

        CommonOps.divide(data.length, mean);

        return mean;
    }

    /**
     * returns the sample covariance
     *
     * @param mean mean of the data; can be calculated by calling getSampleMean (see above)
     */
    public static DenseMatrix64F getSampleCovariance(DenseMatrix64F[] data, DenseMatrix64F mean) {
        DenseMatrix64F sampleCovariance = new DenseMatrix64F(Data.numDimensions, Data.numDimensions);
        for (int i = 0; i < Data.numVectors; i++) {
            DenseMatrix64F x_minus_x_bar = new DenseMatrix64F(Data.numDimensions, 1);
            CommonOps.add(data[i], x_minus_x_bar, x_minus_x_bar);
            CommonOps.sub(x_minus_x_bar, mean, x_minus_x_bar); //(x_i - x_bar)
            DenseMatrix64F x_minus_x_bar_T = new DenseMatrix64F(1, Data.numDimensions);
            x_minus_x_bar_T = CommonOps.transpose(x_minus_x_bar, x_minus_x_bar_T); //(x_i - x_bar)^T
            DenseMatrix64F mul = new DenseMatrix64F(Data.numDimensions, Data.numDimensions);
            CommonOps.mult(x_minus_x_bar, x_minus_x_bar_T, mul);//(x_i - x_bar)(x_i - x_bar)^T
            CommonOps.add(mul, sampleCovariance, sampleCovariance);
            CommonOps.divide(Data.numVectors - 1, sampleCovariance);
        }
        return sampleCovariance;
    }

    static class WordProb {
        public int wordId;
        public double prob;

        public WordProb(int wordId, double prob) {
            this.wordId = wordId;
            this.prob = prob;
        }
    }

    static class WordProbComparator implements Comparator<WordProb> {
        @Override
        public int compare(WordProb wp1, WordProb wp2) {
            return (int) FastMath.signum(wp1.prob - wp2.prob);
        }
    }

    public static void printTopics(ArrayList<DenseMatrix64F> tableMeans, ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat, DenseMatrix64F[] dataVectors, int K, String dirName, int iteration) throws IOException {
        System.out.println("Starting printTopics: " + new Date());
        assert dataVectors[0].numRows == 200 : "dataVectors nr rows should be 200 and not " + dataVectors[0].numRows;
        assert dataVectors[0].numCols == 1 : "dataVectors nr cols should be 1 and not " + dataVectors[0].numCols;
        System.out.println("Calculating topics");
        PrintWriter output = new PrintWriter(new OutputStreamWriter(new FileOutputStream(String.format("%sgaussian-%03d.id-topics", dirName, iteration)), StandardCharsets.UTF_8), true);
        for (int i = 0; i < K; i++) {
            System.out.println("Topic " + i);
            DenseMatrix64F mean = tableMeans.get(i);
            assert mean.numRows == 200 : "there should be 50 rows instead of " + mean.numRows;
            assert mean.numCols == 1 : "there should be 1 column instead of " + mean.numCols;
//            output.write(tableMeans.get(i).get(l, 0) + " ");

            // get the covariance matrix
            DenseMatrix64F chol = tableCholeskyLTriangularMat.get(i);
            DenseMatrix64F cholT = new DenseMatrix64F(chol.numRows, chol.numCols);
            CommonOps.transpose(chol, cholT);
            DenseMatrix64F covar = new DenseMatrix64F(chol.numRows, chol.numCols);
            CommonOps.mult(chol, cholT, covar);
            assert covar.numRows == 200 : "there should be 200 rows in covariance matrix instead of " + covar.numRows;
            assert covar.numCols == 200 : "there should be 200 columns in covariance matrix instead of " + covar.numCols;

            double[][] covariance = new double[200][200];
            for (int j = 0; j < covar.numRows; j += 1) {
                for (int k = 0; k < covar.numCols; k += 1) {
                    covariance[j][k]= covar.get(j, k);
                }
            }
            // do stuff row-wise
            MultivariateNormalDistributionApproximation nd = new MultivariateNormalDistributionApproximation(mean.getData(), covariance);

            int TOP_WORDS = 10;
            PriorityQueue<WordProb> queue = new PriorityQueue<>(TOP_WORDS, new WordProbComparator());
            for (int l = 0; l < dataVectors.length; l += 1) {
                DenseMatrix64F vector = dataVectors[l];
                double currentProb = nd.density(vector.data);
                if (queue.size() < TOP_WORDS) {
                    System.out.println(String.format("Adding %d with prob %f", l, currentProb));
                    queue.add(new WordProb(l, currentProb));
                    continue;
                }
                double leastProb = queue.peek().prob;
                if (leastProb < currentProb) {
                    WordProb removedWordProb = queue.remove();
                    assert removedWordProb.prob == leastProb : "least element should be removed";
                    queue.add(new WordProb(l, currentProb));
                    assert queue.size() == 10 : "queue size should be 10 and not " + queue.size();
                }
           }

            output.print(queue.remove().wordId);
            while (queue.size() > 0) {
                output.print(" " + queue.remove().wordId);
            }
            output.write("\n");
            output.flush();
        }
        output.close();
        System.out.println("Finishing printTopics: " + new Date());
    }

    /**
     * Prints the multivariate distributions (the word|topic distribution)
     */
    public static void printGaussians(ArrayList<DenseMatrix64F> tableMeans, ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat, int K, String dirName, int currentIteration) throws IOException {
        for (int i = 0; i < K; i++) {
            BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(String.format("%sgaussians-%03d.txt", dirName, currentIteration), false), "UTF-8"));
            //first write the mean
            for (int l = 0; l < tableMeans.get(i).numRows; l++)
                output.write(tableMeans.get(i).get(l, 0) + " ");
            output.write("\n");
            //write the covariance matrix
            //first recover it from the cholesky
            DenseMatrix64F chol = tableCholeskyLTriangularMat.get(i);
            DenseMatrix64F cholT = new DenseMatrix64F(chol.numRows, chol.numCols);
            CommonOps.transpose(chol, cholT);
            DenseMatrix64F covar = new DenseMatrix64F(chol.numRows, chol.numCols);
            CommonOps.mult(chol, cholT, covar);

            //write the covar now
            for (int r = 0; r < covar.numRows; r++) {
                for (int c = 0; c < covar.numCols; c++)
                    output.write(covar.get(r, c) + " ");
                output.write("\n");
            }
            output.close();
        }
    }

    public static void printNumCustomersPerTopic(int[][] tableCountsPerDoc, String dirName, int numTopics, int numDocuments) throws IOException {
        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "topic_counts" + ".txt", false), "UTF-8"));
        for (int k = 0; k < numTopics; k++) {
            int n_k = 0;
            for (int n = 0; n < numDocuments; n++)
                n_k = n_k + tableCountsPerDoc[k][n];
            output.write(n_k + "\n");
        }
        output.close();
    }

    public static void printDocumentTopicDistribution(int[][] tableCountsPerDoc, int numDocs, int K, String dirName, double alpha) {
        //for each document, print the normalized count
        try {
            BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(dirName + "document_topic" + ".txt", false), "UTF-8"));
            System.out.println(numDocs);
            for (int i = 0; i < numDocs; i++) {
                double sum = 0;
                for (int k = 0; k < K; k++)
                    sum += tableCountsPerDoc[k][i];
                for (int k = 0; k < K; k++)
                    output.write((tableCountsPerDoc[k][i] + alpha) / (sum + K * alpha) + " ");
                output.write("\n");
            }
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void printTopicAssignments(ArrayList<ArrayList<Integer>> tableAssignments, String dirName) throws IOException {
        BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(dirName + "table_assignments" + ".txt", true), "UTF-8"));
        for (ArrayList<Integer> eachDoc : tableAssignments) {
            for (int assignment : eachDoc)
                output.write(assignment + " ");
            output.write("\n");
        }
        output.close();
    }

    /**
     * calculates corpus perplexity (avg. log-likelihood)
     *
     */
    public static double calculateAvgLL(ArrayList<ArrayList<Integer>> corpus, ArrayList<ArrayList<Integer>> tableAssignments,
                                        DenseMatrix64F[] dataVectors, ArrayList<DenseMatrix64F> tableMeans, ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat, int K, int N, NormalInverseWishart prior, int[][] tableCountsPerDoc) {
        //first divide the cholesky's by the scale term
        //but before that calculate the number of words sitting in each table
        int[] N_k = new int[K];
        for (int k = 0; k < K; k++) {
            int n_k = 0;
            for (int n = 0; n < N; n++)
                n_k = n_k + tableCountsPerDoc[k][n];
            N_k[k] = n_k;
//			if(n_k == 0)
//			{
//				System.out.println("Table "+k+" is empty....exiting");
//				System.exit(1);
//			}
            //System.out.println(n_k);
        }
        //nu_0 + N_k - numDimensions
        double[] scalar = new double[K];
        for (int k = 0; k < K; k++) {
            scalar[k] = prior.nu_0 + N_k[k] - Data.numDimensions;
            //System.out.println(scalar[k]);
        }

        //now divide the cholesky's by sqrt(scalar)
        ArrayList<DenseMatrix64F> scaledCholeskies = new ArrayList<>();
        for (int k = 0; k < K; k++) {
            DenseMatrix64F scaledCholesky = new DenseMatrix64F(Data.numDimensions, Data.numDimensions);
            CommonOps.divide(Math.sqrt(scalar[k]), tableCholeskyLTriangularMat.get(k), scaledCholesky);
            scaledCholeskies.add(scaledCholesky);
        }

        //System.out.println(tableAssignments);

        //logDensity of mulitvariate normal is given by -0.5*(log numDimensions + K*log(2*\pi)+(x-\mu)^T\Sigma^-1(x-\mu))
        //calculate log numDimensions for all table from cholesky
        ArrayList<Double> logDeterminants = new ArrayList<>();
        for (int i = 0; i < K; i++) {
            double logDet = 0.0;
            for (int l = 0; l < Data.numDimensions; l++)
                logDet = logDet + Math.log(scaledCholeskies.get(i).get(l, l));
            logDeterminants.add(logDet);
        }
        int docCounter = 0;
        int totalWordCounter = 0;
        double totalLogLL = 0;
        for (ArrayList<Integer> eachDoc : corpus) {
            int wordCounter = 0;
            for (int word : eachDoc) {
                DenseMatrix64F x = dataVectors[word];
                int tableId = tableAssignments.get(docCounter).get(wordCounter);
                //calculate (x-\mu)
                DenseMatrix64F x_minus_mu = new DenseMatrix64F(Data.numDimensions, 1);
                CommonOps.sub(x, tableMeans.get(tableId), x_minus_mu);
                DenseMatrix64F lTriangularChol = scaledCholeskies.get(tableId);
                TriangularSolver.solveL(lTriangularChol.data, x_minus_mu.data, Data.numDimensions); //now x_minus_mu has the solved value
                DenseMatrix64F x_minus_mu_T = new DenseMatrix64F(1, Data.numDimensions);
                CommonOps.transpose(x_minus_mu, x_minus_mu_T);
                DenseMatrix64F mul = new DenseMatrix64F(1, 1);
                CommonOps.mult(x_minus_mu_T, x_minus_mu, mul);
                double val = mul.get(0, 0);
                double logDensity = 0.5 * (val + Data.numDimensions * Math.log(2 * Math.PI)) + logDeterminants.get(tableId);
                totalLogLL = totalLogLL - logDensity;
                wordCounter++;
                totalWordCounter++;
            }
            docCounter++;
        }
        //to get the average, divide by the totalWordCounter
        @SuppressWarnings("UnnecessaryLocalVariable")
        double avgDensity = totalLogLL / (double) totalWordCounter;
        return avgDensity;
    }


    /**
     */
    public static HashMap<Integer, String> getCustomerIdWordMappings(String vocabFile) {
        HashMap<Integer, String> map = new HashMap<>();
        try {
            BufferedReader reader1 =
                    new BufferedReader(
                            new InputStreamReader(new FileInputStream(vocabFile), "UTF-8"));
            String word;
            int counter = 0;
            while ((word = reader1.readLine()) != null) {
                map.put(counter, word);
                counter++;
            }
            reader1.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

    private static void makeClusterNumbersContinuous() {
        try {
            BufferedReader reader1 =
                    new BufferedReader(
                            new InputStreamReader(new FileInputStream("data/test/bergsma_vocab.txt"), "UTF-8"));

            BufferedReader reader2 =
                    new BufferedReader(
                            new InputStreamReader(new FileInputStream("data/test/bergsma_cluster_nums.txt"), "UTF-8"));

            BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream("data/test/bergsma_cluster_nums_continuous.txt"), "UTF-8"));

            String cluster_line;
            int counter = 0;
            HashMap<Integer, Integer> idMapping = new HashMap<>();
            while ((cluster_line = reader2.readLine()) != null) {
                String c_word = reader1.readLine();
                int c = Integer.parseInt(cluster_line);
                if (!idMapping.containsKey(c)) {
                    idMapping.put(c, counter);
                    out.write(counter + "\n");
                    counter++;
                } else {
                    int map = idMapping.get(c);
                    out.write(map + "\n");
                }
            }
            reader2.close();
            reader1.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void readClusterPrintAsHTML() {
        //first read the all_word_langid_map.txt file to create a map of word -> lang -> softcounts
        HashMap<String, HashMap<String, Double>> wordLangMap = new HashMap<>();
        String wordLangFile = "data/test/all/all_word_langid_map.txt";
        //read cluster file
        String clusterFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/final_clusters/last_iteration_table_members.txt";
        String outputFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/final_clusters/last_iteration_table_members.html";

        try {

            PrintStream out = new PrintStream(outputFileName, "UTF-8");
            out.println("<html>");
            out.println("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />");

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(new FileInputStream(wordLangFile), "UTF-8"));
            String line;
            while ((line = reader.readLine()) != null)//for eadh word
            {
                String[] split = line.split(" ");
                String word = split[0];
                String langsCounts = split[1];
                split = langsCounts.split(",");
                double sum = 0;
                for (String langCount : split)//for each language a word belongs to
                {
                    String[] splits = langCount.split(":");
                    String lang = splits[0];
                    int count = Integer.parseInt(splits[1]);
                    sum = sum + count;
                    if (!wordLangMap.containsKey(word)) {
                        HashMap<String, Double> langSoftCountMap = new HashMap<>();
                        langSoftCountMap.put(lang, count * 1.0);
                        wordLangMap.put(word, langSoftCountMap);
                    } else {
                        HashMap<String, Double> langSoftCountMap = wordLangMap.get(word);
                        langSoftCountMap.put(lang, count * 1.0);
                        wordLangMap.put(word, langSoftCountMap);
                    }
                }
                for (Entry<String, Double> e : wordLangMap.get(word).entrySet()) {
                    e.setValue(e.getValue() / sum);
                }
                //String maxLang = split[0];
                //wordLangMap.put(word, maxLang);
            }
            reader.close();
            reader = new BufferedReader(new InputStreamReader(new FileInputStream(clusterFileName), "UTF-8"));
            while ((line = reader.readLine()) != null)//for each cluster
            {
                HashMap<String, Double> langCountMap = new HashMap<>();
                String[] split = line.split(":");
                int clusterNum = Integer.parseInt(split[0]);
                String[] clusterWords = split[1].split(" ");
                int count = 0;
                for (int m = 1; m < clusterWords.length; m++) //starting from 1 because first word is space
                {
                    String word = clusterWords[m];
                    if (wordLangMap.containsKey(word)) {
                        count++;
                        HashMap<String, Double> langSoftCountMap = wordLangMap.get(word);
                        Set<Entry<String, Double>> langSoftCountSet = langSoftCountMap.entrySet();
                        for (Entry<String, Double> e : langSoftCountSet) {
                            String lang = e.getKey();
                            double c = e.getValue();
                            if (!langCountMap.containsKey(lang))
                                langCountMap.put(lang, c);
                            else {
                                double langCount = langCountMap.get(lang);
                                langCountMap.put(lang, langCount + c);
                            }
                        }
                    }
                }

                Set<Entry<String, Double>> set = langCountMap.entrySet();
                ArrayList<Entry<String, Double>> list = new ArrayList<>();
                list.addAll(set);
                //sort in descending order
                Collections.sort(list, (o1, o2) -> {
                    double count1 = o1.getValue();
                    double count2 = o2.getValue();
                    if (count2 > count1)
                        return -1;
                    else
                        return 0;
                });

                //normalize the list
                double sum = 0;
                for (Entry<String, Double> e : list)
                    sum = sum + e.getValue();

                for (Entry<String, Double> e : list)
                    e.setValue(e.getValue() / sum);

                System.out.println("Cluster " + clusterNum + " count " + count);
                for (Entry<String, Double> aList : list) System.out.println(aList.getKey() + " " + aList.getValue());


            }
            reader.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }


    /***************
     * Commented code
     ***********************/
    public static void main(String[] args) {
        //getClusterNum();
        //makeClusterNumbersContinuous();
//		Data.numDimensions = 4;
//		DenseMatrix64F L = CommonOps.identity(4);
//		Random rand = new Random();
//		DenseMatrix64F x = RandomMatrices.createRandom(4, 1, -1, 1, rand);
//		DenseMatrix64F x_copy = new DenseMatrix64F(x);
//		System.out.println("X = ");
//		System.out.println(x);
//		Util.cholRank1UpdateUpper(L, x);
//		System.out.println("After cholRank1DowndateUpper ");
//		System.out.println("L =");
//		//System.out.println(x);
//		System.out.println(L);
//		System.out.println("x =");
//		System.out.println(x);
//		L = CommonOps.identity(4);
//		System.out.println("X_copy = ");
//		System.out.println(x_copy);
//		Util.cholRank1Update(L, x_copy);
//		System.out.println("After cholRank1Downdate ");
//		System.out.println("L =");
//		//System.out.println(x);
//		System.out.println(L);
//		System.out.println("x_copy =");
//		System.out.println(x_copy);

        Data.numDimensions = 4;
        DenseMatrix64F L = CommonOps.identity(4);
        Random rand = new Random();
        DenseMatrix64F x = RandomMatrices.createRandom(4, 1, -1, 1, rand);
        DenseMatrix64F x_copy = new DenseMatrix64F(x);
        System.out.println("Initial L =");
        System.out.println(L);
        Util.cholRank1Update(L, x);
        System.out.println("After update");
        System.out.println(L);
        Util.cholRank1Downdate(L, x_copy);
        System.out.println("After downdate");
        System.out.println(L);
    }

//	/**
//	 * This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A) where
//	 * A' = A - x*x^T. 
//	 * Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
//	 */
//	public static void cholRank1DowndateUpper(DenseMatrix64F L, DenseMatrix64F x)
//	{
//		//L should be a square lower triangular matrix (although not checking for triangularity here explicitly)
//		//Data.numDimensions = 2;
//		assert L.numCols == Data.numDimensions;
//		assert L.numRows == Data.numDimensions;
//		//x should be a vector
//		assert x.numCols == 1;
//		assert x.numRows == Data.numDimensions;
//		
//		for(int k=0;k<Data.numDimensions;k++)
//		{
//			double r = Math.sqrt(L.get(k, k)*L.get(k, k) - x.get(k, 0)*x.get(k, 0));
//			double c = r/(double)L.get(k, k);
//			double s = x.get(k, 0)/L.get(k, k);
//			L.set(k, k, r);
//			for(int l=k+1;l<Data.numDimensions;l++)
//			{
//				double val = (L.get(k, l) - s*x.get(l, 0))/(double)c ;
//				L.set(k, l, val);
//				val = c*x.get(l, 0) - s*L.get(k, l);
//				x.set(l, 0, val);				
//			}
//		}
//	}

//	/**
//	 * Does a partial cholesky update i.e just updates the diagonal element. This is sufficient to do when we have to just find the determinant of the original matrix. This operation
//	 * unlike  the above 2 questions doesnot change the original L matrix. I am returning the determinant itself, removes the step where we have to make copy of L which might be O(numDimensions^2)
//	 * @param L
//	 * @param x
//	 */
//	public static double cholRank1UpdatePartialAndReturnDet(DenseMatrix64F L, DenseMatrix64F x)
//	{
//		//DenseMatrix64F LPartial = new DenseMatrix64F(L);//copying the matrix	
//		//L should be a square lower triangular matrix (although not checking for triangularity here explicitly)
//		//Data.numDimensions = 2;
//		assert L.numCols == Data.numDimensions;
//		assert L.numRows == Data.numDimensions;
//		//x should be a vector
//		assert x.numCols == 1;
//		assert x.numRows == Data.numDimensions;
//		double logDet = 0;
//		for(int k=0;k<Data.numDimensions;k++)
//		{
//			double r = 0.5 * Math.log(L.get(k, k)*L.get(k, k) + x.get(k, 0)*x.get(k, 0));
//			logDet = logDet + r;			
//		}
//		return logDet;
//	}

//	/**
//	 * This function computes the lower triangular cholesky decomposition L' of matrix A' from L (the cholesky decomp of A) where
//	 * A' = A + x*x^T. 
//	 * Based on the pseudocode in the wiki page https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
//	 */
//	public static void cholRank1UpdateUpper(DenseMatrix64F L, DenseMatrix64F x)
//	{
//		//L should be a square upper triangular matrix (although not checking for triangularity here explicitly)
//		//Data.numDimensions = 2;
//		assert L.numCols == Data.numDimensions;
//		assert L.numRows == Data.numDimensions;
//		//x should be a vector
//		assert x.numCols == 1;
//		assert x.numRows == Data.numDimensions;
//		
//		for(int k=0;k<Data.numDimensions;k++)
//		{
//			double r = Math.sqrt(L.get(k, k)*L.get(k, k) + x.get(k, 0)*x.get(k, 0));
//			double c = r/(double)L.get(k, k);
//			double s = x.get(k, 0)/L.get(k, k);
//			L.set(k, k, r);
//			for(int l=k+1;l<Data.numDimensions;l++)
//			{
//				double val = (L.get(k, l) + s*x.get(l, 0))/(double)c ;
//				L.set(k, l, val);
//				val = c*x.get(l, 0) - s*L.get(k, l);
//				x.set(l, 0, val);				
//			}
//		}
//	}

}
