#! /bin/bash

set -e

OUTPUT=$1

mkdir -p $OUTPUT

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip -O $OUTPUT/wiki-news-300d-1M.vec.zip
