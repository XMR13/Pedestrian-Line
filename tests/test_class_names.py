import json

from pedestrian_line_counter.config import infer_track_class_ids_from_class_names, load_class_names


def test_load_class_names_from_yolo_data_yaml(tmp_path) -> None:
    p = tmp_path / "data.yaml"
    p.write_text("names:\n  - truck\n  - pickup\n  - trailer\n")

    m = load_class_names(p)
    assert m == {0: "truck", 1: "pickup", 2: "trailer"}


def test_load_class_names_from_json_dict(tmp_path) -> None:
    p = tmp_path / "names.json"
    p.write_text(json.dumps({"0": "truck", "2": "trailer"}))

    m = load_class_names(p)
    assert m[0] == "truck"
    assert m[2] == "trailer"


def test_infer_track_class_ids_from_class_names() -> None:
    assert infer_track_class_ids_from_class_names({2: "trailer", 0: "truck", 1: "pickup"}) == [0, 1, 2]

