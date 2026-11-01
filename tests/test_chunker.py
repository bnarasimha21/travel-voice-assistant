from voice_agent.chunker import SentenceChunker


def run(text: str, step: int = 3, **kw) -> list[str]:
    """Feed `text` in small deltas, like a token stream, then flush."""
    c = SentenceChunker(**kw)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += c.push(text[i : i + step])
    return out + c.flush()


def test_sentence_ends():
    assert run("It is sunny. Bring a hat! Want more?") == ["It is sunny.", "Bring a hat!", "Want more?"]


def test_sentence_end_waits_for_following_space():
    c = SentenceChunker()
    assert c.push("Fares start at 3.") == []
    assert c.push("5 dollars. Next") == ["Fares start at 3.5 dollars."]
    assert c.flush() == ["Next"]


def test_abbreviations_do_not_split():
    text = "Head to St. Louis with Dr. Smith and Mr. Lee. Then rest."
    assert run(text) == ["Head to St. Louis with Dr. Smith and Mr. Lee.", "Then rest."]


def test_initials_and_dotted_abbreviations():
    assert run("Ask J. R. Ewing, e.g. on Monday. Done.", first_clause_words=50) == [
        "Ask J. R. Ewing, e.g. on Monday.",
        "Done.",
    ]


def test_decimal_numbers():
    assert run("The flight is 3.5 hours and costs 129.99 dollars. Great.") == [
        "The flight is 3.5 hours and costs 129.99 dollars.",
        "Great.",
    ]


def test_final_chunk_without_punctuation():
    assert run("Sure. I can look that up for you") == ["Sure.", "I can look that up for you"]


def test_clause_boundary_after_enough_words():
    text = "Well, if you fly out of LAX early on Friday, you will land before lunch and skip the traffic."
    chunks = run(text, first_clause_words=5)
    # "Well," is too short to stand alone; the clause after it is long enough.
    assert chunks[0] == "Well, if you fly out of LAX early on Friday,"
    assert chunks[1] == "you will land before lunch and skip the traffic."


def test_first_chunk_is_shorter_than_later_ones():
    text = "One two three four five, six seven eight nine, ten eleven twelve thirteen fourteen, end."
    assert run(text, first_clause_words=5, clause_words=8)[:2] == [
        "One two three four five,",
        "six seven eight nine, ten eleven twelve thirteen fourteen,",
    ]


def test_hard_word_limit():
    text = " ".join(f"w{i}" for i in range(12)) + " "
    c = SentenceChunker(max_words=5)
    assert c.push(text) == ["w0 w1 w2 w3 w4", "w5 w6 w7 w8 w9"]
    assert c.flush() == ["w10 w11"]


def test_quote_after_period_and_newline():
    assert run('He said "go." Then left.\nNew line') == ['He said "go."', "Then left.", "New line"]


def test_empty_stream():
    assert run("") == []
    assert run("   ") == []
