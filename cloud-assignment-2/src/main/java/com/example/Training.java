package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;

import java.io.IOException;
import java.util.Arrays;

public class Training {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().appName("Training").getOrCreate();

        // Load and process data
        Dataset<Row> trainingDataset = loadAndProcessData(spark, "s3://dm752bucket/cloud-assignment-2/TrainingDataset.csv");
        Dataset<Row> validationDataset = loadAndProcessData(spark, "s3://dm752bucket/cloud-assignment-2/ValidationDataset.csv");

        // Logistic Regression Model
        LogisticRegression logisticRegression = new LogisticRegression().setLabelCol("quality").setFeaturesCol("normalisedFeatures")
                .setMaxIter(25).setRegParam(0.5);
        LogisticRegressionModel logisticRegressionModel = logisticRegression.fit(trainingDataset);

        // Evaluate Logistic Regression Model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("quality")
                .setPredictionCol("prediction");
        double trainingAccuracy = evaluator.evaluate(logisticRegressionModel.transform(trainingDataset));
        double validationAccuracy = evaluator.evaluate(logisticRegressionModel.transform(validationDataset)
        );

        System.out.println("Train Accuracy: " + trainingAccuracy + "\nValidation Accuracy: " + validationAccuracy);

        // Random Forest Model with Cross Validation
        RandomForestClassifier randomForest = new RandomForestClassifier().setLabelCol("quality").setFeaturesCol("normalisedFeatures");

        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        ParamMap[] paramGrid = paramGridBuilder.addGrid(randomForest.numTrees(), new int[] { 10, 25, 50 })
                .addGrid(randomForest.maxDepth(), new int[] { 5, 10, 15 }).build();

        CrossValidator cv = new CrossValidator().setEstimator(randomForest).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        CrossValidatorModel cvModel = cv.fit(trainingDataset);
        double bestScore = cvModel.avgMetrics()[0];

        System.out.println("Best F1 Score: " + bestScore);

        String bestModelPath = "s3://dm752bucket/cloud-assignment-2/saved_model";
        RandomForestClassificationModel bestModel = (RandomForestClassificationModel) cvModel.bestModel();
        bestModel.write().overwrite().save(bestModelPath);
        System.out.println("Best model saved at:  " + bestModelPath);

        spark.stop();
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        Dataset<Row> df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv(filePath);

        String[] columns = new String[] { "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                "quality" };
        df = df.toDF(columns);

        VectorAssembler assembler = new VectorAssembler().setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");
        StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("normalisedFeatures")
                .setWithStd(true).setWithMean(true);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { assembler, scaler });
        PipelineModel pipelineModel = pipeline.fit(df);
        return pipelineModel.transform(df);
    }
}
