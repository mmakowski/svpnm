.PHONY: all data results

all: data results

data: data/raw data/processed data/generated

results: results/01-uaf-drf/stats.txt \
         results/02-uaf-lstm/run_ids.txt \
         results/03-uaf-noisy-data-stats/summary.txt \
         results/04-crnn-uaf-noisy/run_ids.txt \
		 results/05-vpmreplication/1412467200/stats.txt \
		 results/06-crnn-linux/splits/1412467200/run_ids.txt \
		 results/07-lmc-linux/splits/1412467200/run_ids.txt \
		 results/report/01-crnn-uaf-pr.pdf \
		 results/report/02-drf-linux-splits-pr.pdf \
		 results/report/03-crnn-linux-splits-pr.pdf \
		 results/report/04-lmc-linux-splits-pr.pdf \
		 results/report/linux-splits-comparison.pdf \
		 results/report/lmc-alert-volume.pdf \
		 results/report/crnn-diagram.eps

# download data from external sources
data/raw: data/raw/nvdcve \
		  data/raw/linux \
		  data/raw/c-corpus-projects

# download the NVD-CVE JSON and XML files
data/raw/nvdcve: src/01-download-nvdcve-data.sh
	rm -rf data/raw/nvdcve
	mkdir -p data/raw/nvdcve/json
	mkdir -p data/raw/nvdcve/xml
	src/01-download-nvdcve-data.sh data/raw/nvdcve

# clone linux git repo
data/raw/linux:
	git clone https://github.com/torvalds/linux.git data/raw/linux

# download snapshots of open source c projects
data/raw/c-corpus-projects: src/14-download-c-corpus-projects.sh
	src/14-download-c-corpus-projects.sh

# process the raw data to construct the dataset
data/processed: data/processed/cve-blame.json \
				data/processed/relational.db \
				data/processed/vpmreplication \
				data/processed/c-corpus.txt

# initial parsing of CVE files and merging with repo information
data/processed/cve-blame.json: data/raw/nvdcve/json data/raw/linux src/02-cve-blame.py
	mkdir -p data/processed
	mkdir -p log
	src/02-cve-blame.py | tee log/02-cve-blame.log

# construction of sqlite database
data/processed/relational.db: data/processed/cve-blame.json src/03-create-relational-db.py
	rm -f data/processed/relational.db
	src/03-create-relational-db.py | tee log/03-create-relational-db.log

# reproducing the dataset from Vulnerability Prediction Models paper
data/processed/vpmreplication: data/processed/vpmreplication/collection \
							   data/processed/vpmreplication/splits \
							   data/processed/vpmreplication/deserialised \
							   data/processed/vpmreplication/arff

# running the VPM collector
data/processed/vpmreplication/collection: data/raw/linux \
										  src/vpmreplication/collector/target/bugsandvulnerabilities-collector-1.0.jar
	mkdir -p data/processed/vpmreplication/collection && \
	cd src/vpmreplication/collector && \
	mvn exec:java -Dlinux.repo=../../../data/raw/linux/ -Doutput.dir=../../../data/processed/vpmreplication/collection/ ; \
	cd ../../..

# running the VPM splitting
data/processed/vpmreplication/splits: data/processed/vpmreplication/collection \
									  src/vpmreplication/splitting/target/bugsandvulnerabilities-experiments-splitting-1.0.jar
	mkdir -p data/processed/vpmreplication/splits && \
	cd src/vpmreplication/splitting && \
	mvn exec:java -Dinput.dir=../../../data/processed/vpmreplication/collection/ -Doutput.dir=../../../data/processed/vpmreplication/splits/ ; \
	cd ../../..

# deserialising VPM datasets
data/processed/vpmreplication/deserialised: \
		data/processed/vpmreplication/collection \
		data/processed/vpmreplication/splits \
		src/vpmreplication/deserialise-datasets/target/deserialise-datasets-1.0.0-SNAPSHOT-jar-with-dependencies.jar
	java -jar src/vpmreplication/deserialise-datasets/target/deserialise-datasets-1.0.0-SNAPSHOT-jar-with-dependencies.jar \
		data/processed/vpmreplication/collection \
		data/processed/vpmreplication/splits \
		data/processed/vpmreplication/deserialised

# converting VPM datasets to arff format
data/processed/vpmreplication/arff: \
		data/processed/vpmreplication/deserialised \
		src/05-make-arff.py
	for time in `ls -1 data/processed/vpmreplication/deserialised | grep '_train' | cut -d '_' -f 1`; do \
		src/05-make-arff.py data/processed/vpmreplication/deserialised/$${time}_train.csv \
							data/processed/vpmreplication/deserialised/$${time}_test.csv \
							data/processed/vpmreplication/arff/$${time} ; \
	done

# preprocessing available project snapshots to build c corpus
data/processed/c-corpus.txt: data/raw/c-corpus-projects src/15-preprocess-c-corpus.py
	src/15-preprocess-c-corpus.py data/raw/c-corpus-projects data/processed/c-corpus.txt

# learning a c language model
data/processed/c-lm/model.hdf5: data/processed/c-corpus.txt src/16-train-c-language-model.py
	src/16-train-c-language-model.py data/processed/c-corpus.txt data/processed/c-lm

# vectorising the VPM examples using the c language model
data/processed/vpmreplication/lmvec.pkl: \
		data/processed/vpmreplication/deserialised \
		data/processed/c-lm/model.hdf5 \
		src/17-lm-vectorise.py
	src/17-lm-vectorise.py data/processed/vpmreplication/deserialised \
	                       data/processed/c-lm \
	                       data/processed/vpmreplication/lmvec.pkl

# building the VPM collector
src/vpmreplication/collector/target/bugsandvulnerabilities-collector-1.0.jar: \
		data/raw/nvdcve/xml \
		src/vpmreplication/collector/pom.xml \
		${HOME}/.m2 \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-utils/1.0-SNAPSHOT/bugsandvulnerabilities-utils-1.0-SNAPSHOT.jar \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-model/1.0/bugsandvulnerabilities-model-1.0.jar
	cp data/raw/nvdcve/xml/*.xml.gz src/vpmreplication/collector/src/main/resources/cve_XML/ && \
	gunzip src/vpmreplication/collector/src/main/resources/cve_XML/*.xml.gz && \
	cd src/vpmreplication/collector && \
	mvn -B clean package ; \
	cd ../../..

# building the VPM splitting
src/vpmreplication/splitting/target/bugsandvulnerabilities-experiments-splitting-1.0.jar: \
		src/vpmreplication/splitting/pom.xml \
		${HOME}/.m2 \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-utils/1.0-SNAPSHOT/bugsandvulnerabilities-utils-1.0-SNAPSHOT.jar \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-model/1.0/bugsandvulnerabilities-model-1.0.jar
	cd src/vpmreplication/splitting && mvn -B clean package ; cd ../../..

# buillding the dataset deserialisation tool
src/vpmreplication/deserialise-datasets/target/deserialise-datasets-1.0.0-SNAPSHOT-jar-with-dependencies.jar: \
		${HOME}/.m2 \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-model/1.0/bugsandvulnerabilities-model-1.0.jar \
		src/vpmreplication/deserialise-datasets/src/main/kotlin/com/mmakowski/svpnm/vpmreplication/DeserialiseDatasets.kt \
		src/vpmreplication/deserialise-datasets/pom.xml
	cd src/vpmreplication/deserialise-datasets && mvn -B clean package ; cd ../..

# building the VPM utils
${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-utils/1.0-SNAPSHOT/bugsandvulnerabilities-utils-1.0-SNAPSHOT.jar: \
		${HOME}/.m2 \
		src/vpmreplication/utils/pom.xml
	cd src/vpmreplication/utils && mvn -B clean install ; cd ../../..

# building the VPM model
${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-model/1.0/bugsandvulnerabilities-model-1.0.jar: \
		src/vpmreplication/utils/pom.xml \
		${HOME}/.m2 \
		${HOME}/.m2/repository/lu/jimenez/research/bugsandvulnerabilities-utils/1.0-SNAPSHOT/bugsandvulnerabilities-utils-1.0-SNAPSHOT.jar
	cd src/vpmreplication/model && mvn -B clean install ; cd ../../..

# link ${HOME}/.m2 to .m2 in the current directory, that will be persistent across container restarts
${HOME}/.m2: .m2
	ln -s `pwd`/.m2 ${HOME}/.m2

.m2:
	mkdir -p .m2

# generation of data
data/generated: data/generated/uaf/test.csv \
				data/generated/uaf-noisy/test.csv \
				data/generated/uaf-distant/test.csv

# generation of use-after-free vulnerabilities
data/generated/uaf/test.csv: src/04-generate-uaf-data.py
	src/04-generate-uaf-data.py | tee log/04-generate-uaf-data.log

# generation of noisy use-after-free vulnerabilities
data/generated/uaf-noisy/test.csv: src/08-generate-uaf-noisy-data.py
	src/08-generate-uaf-noisy-data.py | tee log/08-generate-uaf-noisy-data.log

# generation of distant use-after-free vulnerabilities
data/generated/uaf-distant/test.csv: src/11-generate-uaf-distant-data.py
	src/11-generate-uaf-distant-data.py | tee log/11-generate-uaf-distant-data.log

# vectorisation and conversion to .arff files of the use-after-free data
data/processed/uaf/test.arff: data/generated/uaf/test.csv src/05-make-arff.py
	src/05-make-arff.py data/generated/uaf/train.csv data/generated/uaf/test.csv data/processed/uaf

# reproduction of dicretisation+random forest on the use-after-free synthetic dataset
results/01-uaf-drf/stats.txt: \
		data/processed/uaf/test.arff \
		src/06-drf-classifier/target/drf-classifier-1.0.0-SNAPSHOT-jar-with-dependencies.jar
	java -jar src/06-drf-classifier/target/drf-classifier-1.0.0-SNAPSHOT-jar-with-dependencies.jar \
		data/processed/uaf/train.arff \
		data/processed/uaf/test.arff \
		results/01-uaf-drf

# results of CRNN model for the use-after-free synthetic dataset
results/02-uaf-lstm/run_ids.txt: data/generated/uaf/test.csv src/07-uaf-lstm.py
	src/07-uaf-lstm.py data/generated/uaf/train.csv data/generated/uaf/test.csv results/02-uaf-lstm

# data statistic for checking if the data is generated sensibly
results/03-uaf-noisy-data-stats/summary.txt: data/generated/uaf-noisy/test.csv src/09-uaf-data-stats.py
	src/09-uaf-data-stats.py data/generated/uaf-noisy/test.csv results/03-uaf-noisy-data-stats

# results of CRNN model for the noisy use-after-free synthetic dataset
results/04-crnn-uaf-noisy/run_ids.txt: data/generated/uaf-noisy/test.csv src/10-crnn-uaf-noisy.py
	src/10-crnn-uaf-noisy.py data/generated/uaf-noisy/train.csv data/generated/uaf-noisy/test.csv results/04-crnn-uaf-noisy

# reproduction of discretisaiont+random forest on the VPM paper dataset
results/05-vpmreplication/1412467200/stats.txt: \
		data/processed/vpmreplication/arff \
		src/06-drf-classifier/target/drf-classifier-1.0.0-SNAPSHOT-jar-with-dependencies.jar
	for time in `ls -1 data/processed/vpmreplication/arff`; do \
		java -jar src/06-drf-classifier/target/drf-classifier-1.0.0-SNAPSHOT-jar-with-dependencies.jar \
			data/processed/vpmreplication/arff/$${time}/train.arff \
			data/processed/vpmreplication/arff/$${time}/test.arff \
			results/05-vpmreplication/$${time} ; \
	done

# optimal hyperparameters for CRNN model on Linux dataset
results/06-crnn-linux/hyperopt/best_hyperparameters.json: \
		data/processed/vpmreplication/deserialised/1412467200_train.csv \
		src/12-crnn-linux-hyperopt.py \
		src/crnn.py
	src/12-crnn-linux-hyperopt.py data/processed/vpmreplication/deserialised/1412467200_train.csv \
	                     		  results/06-crnn-linux/hyperopt

# results of CRNN model for Linux dataset
results/06-crnn-linux/splits/1412467200/run_ids.txt: \
		data/processed/vpmreplication/deserialised/1412467200_test.csv \
		results/06-crnn-linux/hyperopt/best_hyperparameters.json \
		src/13-crnn-linux.py \
		src/crnn.py
	src/13-crnn-linux.py data/processed/vpmreplication/deserialised \
	                     results/06-crnn-linux/splits \
	                     results/06-crnn-linux/hyperopt/best_hyperparameters.json

# optimal hyperparameters for LM+classifier on Linux dataset
results/07-lmc-linux/hyperopt/best_hyperparameters.json: \
		data/processed/vpmreplication/deserialised/1412467200_train.csv \
		data/processed/vpmreplication/lmvec.pkl \
		src/18-lmc-linux-hyperopt.py
	src/18-lmc-linux-hyperopt.py data/processed/vpmreplication/deserialised/1412467200_train.csv \
	                             data/processed/vpmreplication/lmvec.pkl \
	                             results/07-lmc-linux/hyperopt

# results of LM+classifier model for Linux dataset
results/07-lmc-linux/splits/1412467200/run_ids.txt: \
		data/processed/vpmreplication/deserialised/1412467200_test.csv \
		results/07-lmc-linux/hyperopt/best_hyperparameters.json \
		data/processed/vpmreplication/lmvec.pkl \
		src/19-lmc-linux.py
	src/19-lmc-linux.py data/processed/vpmreplication/deserialised \
	                    results/07-lmc-linux/hyperopt/best_hyperparameters.json \
	                    data/processed/vpmreplication/lmvec.pkl \
	                    results/07-lmc-linux/splits

# CRNN model for use-after-free -- figures
results/report/01-crnn-uaf-pr.pdf: results/01-uaf-drf/stats.txt results/02-uaf-lstm/run_ids.txt src/make-figures-experiment1.py
	src/make-figures-experiment1.py

# DRF model for Linux -- figures
results/report/02-drf-linux-splits-pr.pdf: results/05-vpmreplication/1412467200/stats.txt src/make-figures-experiment2.py
	src/make-figures-experiment2.py data/processed/vpmreplication/deserialised results/05-vpmreplication/

# CRNN model for Linux -- figures
results/report/03-crnn-linux-splits-pr.pdf: \
		results/05-vpmreplication/1412467200/stats.txt \
		results/06-crnn-linux/splits/1412467200/run_ids.txt \
		src/make-figures-experiment3.py
	src/make-figures-experiment3.py results/05-vpmreplication/ \
	                                results/06-crnn-linux/splits/ \
	                                results/report/03-crnn-linux-splits-pr.pdf \
	                                CRNN

# LM+classifier model for Linux -- figures
results/report/04-lmc-linux-splits-pr.pdf: \
		results/05-vpmreplication/1412467200/stats.txt \
		results/07-lmc-linux/splits/1412467200/run_ids.txt \
		src/make-figures-experiment3.py
	src/make-figures-experiment3.py results/05-vpmreplication/ \
	                                results/07-lmc-linux/splits/ \
	                                results/report/04-lmc-linux-splits-pr.pdf \
	                                LMRF

# make alert volume diagram for the report
results/report/linux-splits-comparison.pdf: \
		results/05-vpmreplication/1412467200/stats.txt \
		results/06-crnn-linux/splits/1412467200/run_ids.txt \
		results/07-lmc-linux/splits/1412467200/run_ids.txt \
		src/make-linux-comparison-figure.py
	src/make-linux-comparison-figure.py results/05-vpmreplication/ \
	                                    results/06-crnn-linux/splits/ \
	                                    results/07-lmc-linux/splits/ \
	                                    results/report/linux-splits-comparison.pdf

# make alert volume diagram for the report
results/report/lmc-alert-volume.pdf: \
		results/07-lmc-linux/splits/1412467200/run_ids.txt \
		src/make-alert-volume-figure.py
	src/make-alert-volume-figure.py results/07-lmc-linux/splits/1412467200/run_ids.txt

# make neural network diagrams for the report
results/report/crnn-diagram.eps: \
		src/crnn.py \
		results/06-crnn-linux/hyperopt/best_hyperparameters.json \
		src/make-nn-diagrams.py
	src/make-nn-diagrams.py results/06-crnn-linux/hyperopt/best_hyperparameters.json


# building the DRF classifier
src/06-drf-classifier/target/drf-classifier-1.0.0-SNAPSHOT-jar-with-dependencies.jar: \
		${HOME}/.m2 \
		src/06-drf-classifier/src/main/kotlin/com/mmakowski/svpnm/DrfClassifier.kt \
		src/06-drf-classifier/pom.xml
	cd src/06-drf-classifier && mvn -B clean package ; cd ../..

# remove everything except data obtained from external, potentially mutable sources
clean:
	cd src/06-drf-classifier && mvn -B clean ; cd ../..
	# TODO: other java artifacts
	rm -rf src/vpmreplication/collector/src/main/resources/cve_XML/*.xml
	rm -rf data/generated
	rm -rf data/processed
	rm -rf results/
	rm -rf log/
	rm -rf .m2/

# Remove everything, including raw downloaded data -- this data might be impossible to reproduce if it has changed at source!
# Note that the NVDCVE files *do* change over time.
deep-clean: clean
	rm -rf data
