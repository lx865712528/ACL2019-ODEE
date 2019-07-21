#!/bin/bash

#script that computes the observed coherence (pointwise mutual information, normalised pmi or log 
#conditional probability)
#steps:
#1. sample the word counts of the topic words based on the reference corpus
#2. compute the observed coherence using the chosen metric

#parameters
metric="npmi" #evaluation metric: pmi, npmi or lcp
#input
topic_file="slot_head_words.txt"
ref_corpus_dir="gn"
#output
wordcount_file="results/wc-oc.txt"
oc_file="results/slots-oc.txt"

if [ "$1" == "--cached" ] # always keep the spaces around ops
then
    echo "Using cached word occurrence..."
else
    #compute the word occurrences
    echo "Computing word occurrence..."
    python ComputeWordCount.py $topic_file $ref_corpus_dir > $wordcount_file
fi

#compute the topic observed coherence
echo "Computing the observed coherence..."
python ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file
