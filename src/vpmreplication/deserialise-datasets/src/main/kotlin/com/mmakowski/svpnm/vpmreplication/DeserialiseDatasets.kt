package com.mmakowski.svpnm.vpmreplication

import com.opencsv.CSVWriter
import lu.jimenez.research.bugsandvulnerabilities.model.extension.experiment.ExperimentalSets
import lu.jimenez.research.bugsandvulnerabilities.model.internal.Document
import lu.jimenez.research.bugsandvulnerabilities.model.internal.DocumentType
import lu.jimenez.research.bugsandvulnerabilities.utils.Serialization
import org.slf4j.LoggerFactory
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.Instant
import java.util.stream.Stream

val log = LoggerFactory.getLogger("dataset-flatten")!!

enum class Label {
    VULNERABLE, NOT_VULNERABLE
}


data class SourceFile(
        val id: String,
        val content: String,
        val label: DocumentType,
        val sourceRepositoryVersion: String,
        val path: Path,
        val commitTime: Instant)


fun main(args: Array<String>) {
    val collectionsDir = Paths.get(args[0])
    val splitsDir = Paths.get(args[1])
    val outputDir = Paths.get(args[2])

    val catFile = collectionsDir.resolve("real_MapOfIdCat.obj")
    log.debug("loading label data from {}", catFile)
    @Suppress("UNCHECKED_CAST")
    val hashToCat = Serialization.loadMapHashData(catFile.toString()) as Map<Int, DocumentType>
    log.info("{} label entries loaded", hashToCat.size)

    writeContents(outputDir, readContents(collectionsDir, hashToCat))
    writeSplits(outputDir, splitsDir, hashToCat)
}


fun writeContents(outputDir: Path, files: Stream<SourceFile>) {
    val metadataFile = outputDir.resolve("metadata.csv")
    val contentDir = outputDir.resolve("content")
    Files.createDirectories(contentDir)

    CSVWriter(Files.newBufferedWriter(metadataFile)).use { metadataWriter ->
        metadataWriter.writeNext(arrayOf("id", "label", "commit hash", "path", "commit time"), false)
        files.forEach {
            writeContent(contentDir, it.id, it.content)
            appendMetadata(metadataWriter, it)
            log.trace("wrote {}", it.id)
        }
    }
    log.info("contents written to {}", outputDir)
}


fun writeSplits(outputDir: Path, splitsDir: Path, hashToCat: Map<Int, DocumentType>) {
    val splitsFile = splitsDir.resolve("real_MapOfIdTime.obj")
    log.debug("loading split data from {}", splitsFile)
    @Suppress("UNCHECKED_CAST")
    val timeToSets = Serialization.loadMapHashData(splitsFile.toString()) as Map<String, ExperimentalSets>
    log.info("{} split entries loaded", timeToSets.size)
    timeToSets.forEach {
        val time = it.key.replace("Time_", "")
        writeDataset(outputDir.resolve(time + "_train.csv"), it.value.trainingset, hashToCat)
        writeDataset(outputDir.resolve(time + "_test.csv"), it.value.testingset, hashToCat)
        log.info("wrote split {}", time)
    }
    log.info("splits written to {}", outputDir)
}

fun readContents(collectionsDir: Path, hashToCat: Map<Int, DocumentType>): Stream<SourceFile> {
    val docFile = collectionsDir.resolve("real_MapOfIdDoc.obj")
    log.debug("loading document data from {}", docFile)
    @Suppress("UNCHECKED_CAST")
    val hashToDoc = Serialization.loadMapHashData(docFile.toString()) as Map<Int, Document>
    log.info("{} document entries loaded", hashToDoc.size)

    return hashToDoc.entries.stream()
            .filter {hashToCat.containsKey(it.key)}
            .map {
                SourceFile(
                        docId(it.key),
                        it.value.content,
                        hashToCat[it.key]!!,
                        it.value.commitHash,
                        Paths.get(it.value.fullPath),
                        Instant.ofEpochSecond(it.value.time.toLong())
                )
            }
}

fun writeDataset(outputFile: Path, hashes: List<Int>, hashToCat: Map<Int, DocumentType>) {
    CSVWriter(Files.newBufferedWriter(outputFile)).use { csvWriter ->
        csvWriter.writeNext(arrayOf("file", "label"), false)
        hashes.forEach {
            csvWriter.writeNext(arrayOf(docId(it), translateLabel(hashToCat[it]!!).toString()), false)
        }
    }
}


fun writeContent(contentDir: Path, id: String, content: String) =
    Files.write(contentDir.resolve(id), content.toByteArray())!!


fun docId(docHash: Int): String = "%016x".format(docHash)


fun translateLabel(sourceLabel: DocumentType): Label = when (sourceLabel) {
    DocumentType.VULNERABLE_FILE -> Label.VULNERABLE
    DocumentType.PATCHED_VULNERABLE_FILE -> throw RuntimeException("cannot generate label for %s".format(sourceLabel))
    else                         -> Label.NOT_VULNERABLE
}


fun appendMetadata(metadataWriter: CSVWriter, metadata: SourceFile) = metadataWriter.writeNext(arrayOf(
        metadata.id,
        metadata.label.toString(),
        metadata.sourceRepositoryVersion,
        metadata.path.toString(),
        metadata.commitTime.epochSecond.toString()
), false)
