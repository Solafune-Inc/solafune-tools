from solafune_tools.competition_tools import solafune_dataset

def test_segmentation_submission_from_var():
    # From variable
    submission_dict = {
        "images": [
            {
                "file_name" : "test_0.tif",
                "width" : 100,
                "height" : 100,
                "annotations" : [
                    {
                        "class": "class_name",
                        "segmentation": [
                            0,
                            0,
                            1,
                            1,
                            2,
                            2,
                        ]
                    },
                    {
                        "class": "class_name",
                        "segmentation": [
                            3,
                            3,
                            4,
                            4,
                            5,
                            5,
                        ]
                    }
                ]
            }
        ]
    }
    validated_submission = solafune_dataset.json_submission_validator(pred_dict = submission_dict, ann_type = "segmentation")
    assert validated_submission == "Valid"

def test_segmentation_submission_from_file():
    # From file
    submission_file = "tests/data-test/sample_segmentation.json"
    validated_submission = solafune_dataset.json_submission_validator(pred_file_path = submission_file, ann_type = "segmentation")
    assert validated_submission == "Valid"

def test_bbox_submission_from_var():
    # From variable
    submission_dict = {
        "images": [
            {
                "file_name" : "test_0.tif",
                "width" : 100,
                "height" : 100,
                "annotations" : [
                    {
                        "class": "class_name",
                        "bbox": [
                            0,
                            0,
                            1,
                            1
                        ]
                    },
                    {
                        "class": "class_name",
                        "bbox": [
                            3,
                            3,
                            4,
                            4
                        ]
                    }
                ]
            }
        ]
    }
    validated_submission = solafune_dataset.json_submission_validator(pred_dict = submission_dict, ann_type = "bbox")
    assert validated_submission == "Valid"

def test_bbox_submission_from_file():
    # From file
    submission_file = "tests/data-test/sample_bbox.json"
    validated_submission = solafune_dataset.json_submission_validator(pred_file_path = submission_file, ann_type = "bbox")
    assert validated_submission == "Valid"