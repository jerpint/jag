import utils.data_fetcher


def test_mrqa_text_fetcher():
    utils.data_fetcher.get_file(data_src='./data/mrqa_urls_sample.txt',
                                cache_dir='./unit_tests/data')
