package com.mmakowski.svpnm

import com.opencsv.CSVWriter
import me.tongfei.progressbar.ProgressBar
import org.slf4j.LoggerFactory
import weka.classifiers.evaluation.EvaluationUtils
import weka.classifiers.evaluation.Prediction
import weka.classifiers.evaluation.TwoClassStats
import weka.classifiers.trees.RandomForest
import weka.core.Instances
import weka.core.converters.ConverterUtils
import weka.filters.Filter
import weka.filters.supervised.attribute.Discretize
import weka.filters.unsupervised.attribute.RemoveUseless
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths


val log = LoggerFactory.getLogger("DrfClassifier")!!

fun main(args: Array<String>) {
    val trainFile = Paths.get(args[0])
    val testFile = Paths.get(args[1])
    val outputDir = Paths.get(args[2])
    Files.createDirectories(outputDir)
    evaluate(trainFile, testFile, outputDir)
}

fun evaluate(trainFile: Path, testFile: Path, outputDir: Path) {
    log.info("evaluating {}:{}", trainFile.fileName, testFile.fileName)

    val trainInstances = readInput(trainFile)
    val testInstances = readInput(testFile)

    trainInstances.setClassIndex(trainInstances.numAttributes() - 1)
    log.info("number of attributes in input: " + trainInstances.numAttributes())

    val discretize = Discretize()
    // "In the end, we figured out that the first filter to use was “Discretize” with the options “Kononeko”,
    // makes binary and use bin number activated"
    discretize.useKononenko = true
    discretize.makeBinary = true
    discretize.useBinNumbers = true

    val removeUseless = RemoveUseless()

    val preprocessedTrain = fitFilter(removeUseless, fitFilter(discretize, trainInstances))
    val preprocessedTest = applyFilter(removeUseless, applyFilter(discretize, testInstances))
    log.info("number of attributes after preprocessing: " + preprocessedTrain.numAttributes())

    val randomForest = RandomForest()
    // in Weka 3.8 the default number of iterations (= number of trees) is 100, as in Scandariato's paper
    log.info("building the classifier")
    randomForest.buildClassifier(preprocessedTrain)
    log.info("evaluating on the test set")
    val testPredictions = EvaluationUtils().getTestPredictions(randomForest, preprocessedTest)

    writeResults(outputDir, testPredictions)
}

fun writeResults(outputDir: Path, predictions: Iterable<Prediction>) {
    val predictionsFile = outputDir.resolve("predictions.csv")
    CSVWriter(Files.newBufferedWriter(predictionsFile)).use { csvWriter ->
        csvWriter.writeNext(arrayOf("prediction", "label"), false)
        predictions.forEach {
            csvWriter.writeNext(arrayOf(it.predicted().toString(), it.actual().toString()), false)
        }
    }

    val stats = evaluationStats(predictions)
    File(outputDir.toFile(), "stats.txt").bufferedWriter().use {
        it.write("TP: ${stats.truePositive.toInt()}\n")
        it.write("TN: ${stats.trueNegative.toInt()}\n")
        it.write("FP: ${stats.falsePositive.toInt()}\n")
        it.write("FN: ${stats.falseNegative.toInt()}\n")
        it.write("precision: ${stats.precision}\n")
        it.write("recall: ${stats.recall}\n")
        it.write("F1: ${stats.fMeasure}\n")
    }
}

fun readInput(datasetFile: Path): Instances {
    val dataSource = ConverterUtils.DataSource(datasetFile.toAbsolutePath().toString())
    return dataSource.dataSet
}

fun fitFilter(filter: Filter, trainSet: Instances): Instances {
    log.info("training filter $filter on a data set containing " + trainSet.size + " elements")
    filter.setInputFormat(trainSet)
    trainSet.forEach { instance -> filter.input(instance)}
    filter.batchFinished()
    val filteredTrainSet = filter.outputFormat
    var processed = filter.output()
    while (processed != null) {
        filteredTrainSet.add(processed)
        processed = filter.output()
    }
    log.info("filter trained")
    return filteredTrainSet
}

fun applyFilter(filter: Filter, dataSet: Instances): Instances {
    log.info("applying filter $filter to a data set containing " + dataSet.size + " elements")
    val filteredDataSet = filter.outputFormat
    ProgressBar.wrap(dataSet, "applying").forEach { instance ->
        assert(filter.input(instance))
        filteredDataSet.add(filter.output())
    }
    log.info("filter applied")
    return filteredDataSet
}

fun evaluationStats(predictions: Iterable<Prediction>): TwoClassStats {
    var tp = 0
    var tn = 0
    var fp = 0
    var fn = 0
    predictions.forEach { prediction -> when (prediction.actual()) {
        0.0 -> when (prediction.predicted()) {
            0.0 -> tn += 1
            1.0 -> fp += 1
            else -> error("unsupported predicted value: " + prediction.predicted())
        }
        1.0 -> when (prediction.predicted()) {
            0.0 -> fn += 1
            1.0 -> tp += 1
            else -> error("unsupported predicted value: " + prediction.predicted())
        }
        else -> error("unsupported actual value: " + prediction.actual())
    }}
    return TwoClassStats(tp.toDouble(), fp.toDouble(), tn.toDouble(), fn.toDouble())
}
