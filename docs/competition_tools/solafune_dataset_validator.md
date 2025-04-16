# Solafune Dataset Submission Validator

Solafune Dataset Submission Validator is a tool we can use for evaluating user submissions to the solafune's submission website. This tool can be used to help you validate whether your submission complies or not with our JSON schema.

## Usage

Using cli-base method

```bash
python solafune_dataset.py --file_path sample_submission.json --ann_type segmentation # Insert your JSON file
```

Using library-base function

```python
from solafune_tools.competition_tools import solafune_dataset

# From variable
submission_dict = {
    "images": [
        {
            "file_names" : "test_xxx.tif",
            "annotations" : [
                {
                    "class": "class_name",
                    "segmentation": [
                        polygon1_x1,
                        polygon1_y1,
                        polygon1_x2,
                        polygon1_y2,
                        polygon1_xn,
                        polygon1_yn,
                    ]
                },
                {
                    "class": "class_name",
                    "segmentation": [
                        polygon2_x1,
                        polygon2_y1,
                        polygon2_x2,
                        polygon2_y2,
                        polygon2_xn,
                        polygon2_yn,
                    ]
                }
            ]
        }
    ]
}
validated_submission, number_of_images = solafune_dataset.json_submission_validator(pdict = submission_dict, ann_type = "segmentation")
print(validated_submission)

# From saved file
submission_file = "path/to/your/file.json"
validated_submission, number_of_images = pq_scoring_submission.submission_validator(file_path = submission_file, ann_type = "segmentation")
print(validated_submission)
```
