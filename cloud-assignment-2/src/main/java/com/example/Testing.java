package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

import java.util.Arrays;

public class Testing {
    public static void main(String[] args) {
        String modelPath = args.length > 0 ? args[0] : "s3://dm752bucket/cloud-assignment-2/saved_model";
        String testFilePath = args.length > 1 ? args[1] : "s3://dm752bucket/cloud-assignment-2/ValidationDataset.csv";

        SparkSession spark = SparkSession.builder().appName("Wine Quality Prediction").getOrCreate();

        Dataset<Row> validationData = loadAndProcessData(spark, testFilePath);
        validationData.select("normalisedfeatures").show(false);

        RandomForestClassificationModel loadedModel = RandomForestClassificationModel.load(modelPath);
        Dataset<Row> predictions = loadedModel.transform(validationData);

        predictions.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double score = evaluator.evaluate(predictions);
        System.out.println("F1 Score on the given test data is : " + score);

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
