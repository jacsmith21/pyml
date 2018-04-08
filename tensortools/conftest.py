import json
import os
import pytest


@pytest.fixture
def voc_annotations():
    path = os.path.join(os.path.dirname(__file__), '../resources/voc_annotations.txt')
    with open(path) as f:
        lines = f.read().splitlines()
        annotations = [list(map(float, line.split(','))) for line in lines]

    return annotations


@pytest.fixture
def random_annotations():
    path = os.path.join(os.path.dirname(__file__), '../resources/annotations.json')
    with open(path) as f:
        annotations = json.load(f)

    return annotations
