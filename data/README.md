# Download Scripts
MRQA have provided a convenience script to download all of the training and development data (that is released).

Please run:

`./download_train.sh path/to/store/downloaded/directory`

To download the development data of the training datasets (in-domain), run:

`./download_in_domain_dev.sh path/to/store/downloaded/directory`

To download the out-of-domain development data, run:

`./download_out_of_domain_domain_dev.sh path/to/store/downloaded/directory`

To extract files, use `gunzip -k filename.jsonl.gz`
