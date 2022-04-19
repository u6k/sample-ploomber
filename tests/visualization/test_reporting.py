def test_product():
    # Load file
    with open("/var/output/reports/classification_report.txt", "r") as f:
        cls_report = f.read()
    with open("/var/output/reports/confusion_matrix.txt", "r") as f:
        con_matrix = f.read()

    # Assert
    assert len(cls_report) > 0
    assert len(con_matrix) > 0
