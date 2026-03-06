import pytest
from src.processor.element_classifier import ElementClassifier
from src.processor.table_extractor import TableExtractor

def test_element_classifier():
    classifier = ElementClassifier()
    
    element = {"type": "text", "content": "Sample text"}
    assert classifier.classify(element) == "text"
    
    element = {"type": "image", "content": b"image_data"}
    assert classifier.classify(element) == "image"

    chart_element = {"type": "image", "caption": "Revenue chart by quarter"}
    assert classifier.classify(chart_element) == "chart"

def test_table_extractor():
    extractor = TableExtractor()
    
    table_data = {
        "data": [
            ["Name", "Age"],
            ["Alice", 30],
            ["Bob", 25]
        ]
    }
    
    result = extractor.extract(table_data)
    assert "summary" in result
    assert "dataframe" in result
    assert result["shape"] == [2, 2]
