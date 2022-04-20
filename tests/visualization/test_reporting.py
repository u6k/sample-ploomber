def test_classification_report():
    with open("/var/output/reports/classification_report.txt", "r") as f:
        cls_report = f.read()

    assert len(cls_report) > 0


def test_confusion_matrix():
    with open("/var/output/reports/confusion_matrix.txt", "r") as f:
        con_matrix = f.read()

    assert len(con_matrix) > 0
