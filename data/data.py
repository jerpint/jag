import gzip
import json


def dataset_generator(fname='data/SQuAD.jsonl.gz'):
    """Simple dataset generator for MRQA.

    Extended description of function.

    Args:
        fname (str): File to be processed

    Returns:
        obj: Object wtih fields defined by MRQA

    """

    with gzip.open(fname, 'rb') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)

            # Skip headers.
            if i == 0 and 'header' in obj:
                continue

            else:
                yield obj
