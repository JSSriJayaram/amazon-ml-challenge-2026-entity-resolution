import pytest

from ber.metrics import entity_f05, macro_f05
from ber.normalize import normalize_address, normalize_name, skeleton


def test_pdf_worked_example():
    f = entity_f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert f == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert entity_f05([], []) == 1.0
    assert entity_f05(["S2-1"], []) == 0.0
    assert entity_f05([], ["S2-1"]) == 0.0


def test_macro_counts_missing_as_empty():
    truth = {"a": {"S2-1"}, "b": set()}
    assert macro_f05({"a": ["S2-1"]}, truth) == 1.0


@pytest.mark.parametrize("a,b", [
    ("Starbucks Coffee Pvt. Ltd.", "STAR BUCKS COFFEE PRIVATE LIMITED"),
    ("Pediatric Denta1 Associates", "pediatric dental associates"),
    ("UNITED [(CENTER)]", "United Centre"),
    ("halcyonproductions.com", "Halcyon Productions"),
])
def test_name_views_agree(a, b):
    na, nb = normalize_name(a, "US"), normalize_name(b, "US")
    assert na["name_nospace"] == nb["name_nospace"]


def test_transliterated_legal_form():
    n = normalize_name("भारत इंजीनियरिंग प्राइवेट लिमिटेड", "India")
    assert n["name_legal"] == "ltd pvt"
    assert "prvt" not in n["name_skel"]


def test_skeleton():
    assert skeleton("praaivett") == skeleton("private") == "prvt"
    assert skeleton("laxmi") == skeleton("lkssmii")


def test_address_views():
    a = normalize_address("01335 FIFTH ST, PORT ANGELES, WA", "US")
    b = normalize_address("1335-C 5th Street, Port Angeles, Washington", "US")
    assert a["addr_nums"] == b["addr_nums"] == "1335 5"
    assert a["addr_city"] == b["addr_city"] == "port angeles"
    c = normalize_address("Cotton Trail Dr, <NULL>, Rossville, Tennessee", "US")
    assert c["addr_clean"] == "cotton trl dr rossville tn"
    d = normalize_address("123, M.G. Road, Near Metro Stn, Bangalore - 560001", "India")
    assert d["addr_city"] == "bangalore"
