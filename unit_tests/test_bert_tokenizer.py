def test_basic_tokenizer():
    '''Unit test of tokenizer utility'''

    from utils.bert_tokenizer import BasicTokenizer

    basic_tokenizer = BasicTokenizer()

    test_string = "This is a basic test for our tokenizer!"

    output_tokens = basic_tokenizer.tokenize(test_string)

    assert type(output_tokens) == list

    for token in output_tokens:
        assert type(token) == str

    expected_out = ['this', 'is', 'a', 'basic', 'test', 'for',
                    'our', 'tokenizer', '!']

    assert output_tokens == expected_out

def test_bert_tokenizer():
    '''Unit test of Bert tokenizer utility'''

    from utils.bert_tokenizer import BertTokenizer

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                   do_lower_case=False,
                                                   cache_dir='/tmp')

    test_string = "This is a basic test for our BERT tokenizer!"

    output_tokens = bert_tokenizer.tokenize(test_string)

    assert type(output_tokens) == list

    for token in output_tokens:
        assert type(token) == str

    expected_out = ['This', 'is', 'a', 'basic', 'test', 'for',
                    'our', 'BE', '##RT', 'tok', '##eni', '##zer', '!']

    assert output_tokens == expected_out
