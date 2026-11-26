import unicodedata

from augmentation.corrupt import corrupt
from common.preprocessing import AGGRESSIVE, DEFAULT, NormConfig, is_noise_only, normalize, script_profile


def test_bangla_preserved_exactly():
    s = "আপনার পার্সেল আটকে আছে। ডেলিভারি চার্জ ৫০ টাকা"
    assert normalize(s) == unicodedata.normalize("NFC", s)


def test_bangla_zwj_kept_inside_bangla_but_removed_in_latin():
    bangla = "র‍্যাব"
    assert "‍" in normalize(bangla)
    assert normalize("Gr​oup e jоin") == "group e join"
    assert normalize("pa‍y") == "pay"


def test_homoglyphs_folded_only_in_latin_tokens():
    assert normalize("what iste sаlаry?") == "what iste salary?"
    assert normalize("tаkа") == "taka"
    assert normalize("Москва", NormConfig(lowercase=False)) == "Москва"  # pure Cyrillic word untouched


def test_banglish_and_mixed_keep_numbers_urls_emoji():
    s = "Apni 5 lakh taka jitechen 🎁 hxxps://bd-winner[.]top/claim  01XXXXXXXXX"
    n = normalize(s)
    for part in ("5", "lakh", "🎁", "hxxps://bd-winner[.]top/claim", "01xxxxxxxxx"):
        assert part in n
    assert "  " not in n
    mixed = "Eid offer! সব item e ৩০% discount."
    assert "সব" in normalize(mixed) and "৩০%" in normalize(mixed)


def test_optional_views():
    assert normalize("PIIIN", NormConfig(squeeze_repeats=True)) == "piin"
    assert normalize("PIIIN") == "piiin"
    assert normalize("hxxps://x[.]top", AGGRESSIVE) == "https://x.top"


def test_empty_and_noise():
    assert normalize("") == ""
    assert normalize(None) == ""
    assert normalize("   ​ ", NormConfig(empty_token="<empty>")) == "<empty>"
    assert is_noise_only("lkjsdfoiu xcvbnm")
    assert is_noise_only("zzqpwoeiru")
    assert not is_noise_only("Bhai kal exam er routine ta pathaiso?")
    assert not is_noise_only("আপনার")


def test_script_profile():
    assert script_profile("আপনার পার্সেল")["script"] == "bangla"
    assert script_profile("Bus ta late")["script"] == "latin"
    assert script_profile("Eid offer সব পণ্যে ছাড়")["script"] == "mixed"
    assert script_profile("12345")["script"] == "none"


def test_corruption_deterministic_and_normalizable():
    s = "Your bKash PIN will expire today, reply with your PIN"
    assert corrupt(s, 42) == corrupt(s, 42)
    assert corrupt(s, 42) != s
    assert normalize(corrupt(s, 1))  # never crashes, never empty
