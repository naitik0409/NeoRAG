from neorag.agent.react import extract_json_object, parse_action


def test_extract_plain_json():
    obj = extract_json_object('{"action": "search", "action_input": {"query": "x"}}')
    assert obj == {"action": "search", "action_input": {"query": "x"}}


def test_extract_json_with_fences_and_noise():
    text = 'Sure! Here you go:\n```json\n{"action": "answer", "action_input": {"answer": "42"}}\n```\nDone.'
    obj = extract_json_object(text)
    assert obj and obj["action"] == "answer"


def test_extract_nested_braces_in_strings():
    text = '{"thought": "use { and } inside", "action": "search", "action_input": {"query": "a{b}c"}}'
    obj = extract_json_object(text)
    assert obj and obj["action_input"]["query"] == "a{b}c"


def test_extract_returns_none_on_garbage():
    assert extract_json_object("no json at all") is None
    assert extract_json_object("{broken") is None


def test_parse_action_valid():
    parsed = parse_action('{"thought": "t", "action": "SEARCH", "action_input": {"query": "q"}}')
    assert parsed["action"] == "search"
    assert parsed["action_input"] == {"query": "q"}


def test_parse_action_invalid_shape_falls_back():
    parsed = parse_action("I think the answer is probably yes honestly")
    assert parsed["action"] == "invalid"
    assert parsed["action_input"] == {}


def test_parse_action_non_dict_action_input():
    parsed = parse_action('{"action": "answer", "action_input": "the answer text"}')
    assert parsed["action"] == "answer"
    assert parsed["action_input"] == {"value": "the answer text"}
