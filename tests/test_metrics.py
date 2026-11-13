from types import SimpleNamespace

from neorag.eval.metrics import citation_precision, evaluate_agent, token_f1


def test_token_f1_identical():
    assert token_f1("the cat sat", "the cat sat") == 1.0


def test_token_f1_disjoint():
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_token_f1_partial():
    score = token_f1("the cat sat on the mat", "the cat sat")
    assert 0.0 < score < 1.0


def make_result(answer, citations, passages):
    return SimpleNamespace(answer=answer, citations=citations, passages=passages, steps_used=1, gave_up=False)


def test_citation_precision():
    r = make_result("ans", [1, 2, 9], {1: "p1", 2: "p2"})
    assert abs(citation_precision(r) - 2 / 3) < 1e-9


def test_evaluate_agent_summary():
    gold1 = {"answer": "the sky is blue"}
    r1 = make_result("the sky is blue [1]", [1], {1: "p"})
    gold2 = {"answer": "water boils at 100c"}
    r2 = make_result("no idea", [], {})
    summary = evaluate_agent([(gold1, r1), (gold2, r2)])
    assert summary["n"] == 2
    assert summary["gave_up_rate"] == 0.0
    assert summary["answer_rate"] == 1.0
    assert summary["token_f1"] > 0.0
    assert summary["citation_precision"] > 0.0
