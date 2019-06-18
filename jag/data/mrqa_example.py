from jag.utils.data_utils import printable_text


class MRQAExample(object):
    """A single training/test example.
       For examples without an answer,
       the start and end position are [].

       Args:
            qas_id (str): the id of the question
            question_text (str): the (processed) text representing the question
            question_tokens (List[str]): the tokens of the `question_text`
            context_text (str): the (processed) text representing the context
            context_tokens (List[str]): the tokens of the `context_text`
            ds_id (str): id of the dataset from which the question is retrieved.
                Default: None.
            answers_text (List[str]): the list of all possible answers to
                the question. Default: None
            contextualized_answers (List[str]): the list of all possible
                answers to the question as extracted from the (preprocessed)
                `context_text`. Default: None
            start_position (List[List[int]]): The i-th entry of this list
                contains a list of indexes corresponding to the start positions
                of the i-th answer in `context_tokens`. Default: None
            end_position (List[List[int]]): The i-th entry of this list
                contains a list of indexes corresponding to the end positions
                (aligned with the `start_position` elements) of the i-th answer
                in `context_tokens`. Default: None

    """

    def __init__(self,
                 qas_id,
                 question_text,
                 question_tokens,
                 context_text,
                 context_tokens,
                 ds_id=None,
                 answers_text=None,
                 contextualized_answers=None,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.ds_id = ds_id
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.context_text = context_text
        self.context_tokens = context_tokens
        self.answers_text = answers_text
        self.contextualized_answers = contextualized_answers
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (
            printable_text(self.question_text))
        s += ", context_tokens: [%s]" % (" ".join(self.context_tokens))
        if self.start_position:
            s += ", start_position: {}".format(self.start_position)
        if self.end_position:
            s += ", end_position: {}".format(self.end_position)
        return s
