def test_remove_html_tags():
    '''Unit test of data utility'''
    from jag.utils.data_utils import remove_html_tags

    text = '<div class="tab0">CSS code formatter</div>' + \
        '<div class="tab2">CSS code compressor</div>'

    target = ' CSS code formatter  CSS code compressor '
    result = remove_html_tags(text)

    assert target == result
