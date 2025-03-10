from solafune_tools.competition_tools import solafune_dataset

def test_compute_pq_from_var():
    # From variable
    submission_dict = {
        "images": [
            {
                "file_name" : "test_0.tif",
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
    validated_submission, num_images = solafune_dataset.submission_validator(pdict = submission_dict)
    assert validated_submission == "Valid"
    assert num_images == 1

def test_compute_pq_from_file():
    # From file
    submission_file = "tests/data-test/sample.json"
    validated_submission, num_images = solafune_dataset.submission_validator(file_path = submission_file)
    assert validated_submission == "Valid"
    assert num_images == 50
