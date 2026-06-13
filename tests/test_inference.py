import os


def test_input_text_env_can_be_set():
    os.environ["INPUT_TEXT"] = "sample text"
    assert os.getenv("INPUT_TEXT") == "sample text"
