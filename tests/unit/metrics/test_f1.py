from src.metrics.f1 import F1ForExtractiveQuestionAnswering
from src.taskmodules.extractive_question_answering import (
    ExtractiveAnswer,
    ExtractiveQADocument,
    Question,
)


def test_f1_for_eqa_exact_match():
    metric = F1ForExtractiveQuestionAnswering()

    # create a test document
    # sample edit
    doc = ExtractiveQADocument(text="This is a test document.")
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=8, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == str(doc.answers[0])

    metric._update(doc)
    # assert the tp, fp, fn counts
    # TODO: adjust when the metric is correctly implemented
    assert metric.counts == {"MICRO": (0, 0, 0)}
    metric_values = metric._compute()
    # assert the f1, p, r values
    # TODO: adjust when the metric is correctly implemented
    assert metric_values == {"MICRO": {"f1": 0.0, "p": 0.0, "r": 0.0}}


def test_f1_for_eqa_span_mismatch():
    metric = F1ForExtractiveQuestionAnswering()

    # create a test document
    doc = ExtractiveQADocument(text="This is a test document.")
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer for q1
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer for q1
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=10, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == "test document"
    # the spans are not the same!
    assert str(doc.answers.predictions[0]) != str(doc.answers[0])

    metric._update(doc)
    # assert the tp, fp, fn counts
    # TODO: adjust when the metric is correctly implemented
    assert metric.counts == {"MICRO": (0, 0, 0)}
    metric_values = metric._compute()
    # assert the f1, p, r values
    # TODO: adjust when the metric is correctly implemented
    assert metric_values == {"MICRO": {"f1": 0.0, "p": 0.0, "r": 0.0}}


# TODO: maybe add more tests
