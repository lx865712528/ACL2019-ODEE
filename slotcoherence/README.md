# ODEE Slot Coherence Evaluation
This package is modified from [Jey Han Lau's "topic_interpretability"](https://github.com/jhlau/topic_interpretability) to competite with python3 and ODEE.

This package contains the scripts and various python tools for computing the semantic 
interpretability of topics via: (1) PMI/NPMI/LCP-based observed 
coherence.

python3 for slot coherence

## Prepare Reference Corpus
Parallel processing for sampling the word counts can be achieved by splitting the reference corpus 
into multiple partitions. The format of the reference corpus is one line per document, and the words 
should be tokenised (separated by white space). An example reference 
corpus is given in `ref_corpus/`.

Detailed steps is:
1. run `cd gn && split ../../dataprocessor/corpus corpus. -d -l 5500` to split the corpus into pieces

## Running the System
Pairwse PMI/NPMI/LCP observed coherence:
* Set up the parameters in `run-oc.sh`
* If run the first time, just run `sh run-oc.sh`
* Else run with previously computed and cached values `sh run-oc.sh --cached`

## Output
* Debug OFF (in ComputeObservedCoherence.py): one score per line, each score corresponds to the topic of the same line
* Debug ON (in ComputeObservedCoherence.py): score and topics are displayed

## Licensing
* MIT license - http://opensource.org/licenses/MIT.

## Publications
### Original Paper
* Jey Han Lau, David Newman and Timothy Baldwin (2014). Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality. In Proceedings of the 14th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2014), Gothenburg, Sweden, pp. 530—539.

### Other Related Papers
* David Newman, Jey Han Lau, Karl Grieser and Timothy Baldwin (2010). Automatic Evaluation of Topic
Coherence. In Proceedings of Human Language Technologies: The 11th Annual Conference of the North
American Chapter of the Association for Computational Linguistics (NAACL HLT 2010), Los Angeles,
USA, pp. 100—108.
* Jey Han Lau, Timothy Baldwin and David Newman (2013). On Collocations and Topic Models. ACM 
Transactions on Speech and Language Processing 10(3), pp. 10:1—10:14.
* Jey Han Lau and Timothy Baldwin (2016). The Sensitivity of Topic Coherence Evaluation to Topic Cardinality. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics — Human Language Technologies (NAACL HLT 2016), San Diego, USA, pp. 483—487.
