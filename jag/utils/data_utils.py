import six
import re
import string


def printable_text(text):
    """Returns text encoded in a way suitable for print.
    Args:
        text: `str` for both Python2 and Python3, but in one case
              it's a Unicode string and in the other it's a byte string.
    Output:
        result: corresponding text encoded in a way suitable for print.
                it has the same type as the input text.

    """

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.

    if six.PY3:
        unicode = str
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def remove_html_tags(text):
    """Returns text from which html tags with related attributes
       have been removed.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding text in which html tags with
        related attributes have been removed.

    """
    return re.sub(r'<(.|\n)*?>', ' ', text)


def remove_punc(text):
    """Returns text from which punctuations have been removed.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding text in which punctuations
        have been removed.

    """
    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    exclude = set(string.punctuation)
    exclude.add('’')
    exclude.add('“')
    exclude.add('”')
    exclude.add('‘')
    return ''.join(ch for ch in text if ch not in exclude)


def white_space_fix(text):
    """Returns text from which succesive white spaces characters
       are replaced by a single white space character.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding text in which succesive white
        spaces characters are replaced by a single white space
        character.

    """
    return ' '.join([v.strip() for v in text.strip().split()])


def remove_articles(text):
    """Returns text from which articles a|an|the are replaced by white spaces.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding text in which articles a|an|the are
        replaced by white spaces.

    """
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def lower(text):
    """Returns text that has been lower.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding text which has been lower.

    """
    return text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace.
    Args:
        text (str): input text to process.
    Output:
        result (str): corresponding processed text.

    """

    return white_space_fix(remove_articles(remove_punc(lower(text))))
