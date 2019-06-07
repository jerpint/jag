from jag.utils import data_fetcher


def test_mrqa_text_fetcher():
    data_fetcher.get_file(data_src='./jag/data/mrqa_urls_sample.txt',
                          cache_dir='./unit_tests/data')
