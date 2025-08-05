# Solafune Dataset Submission Validator

Solafune Dataset Submission Validator is a tool used for evaluating user submissions to the Solafune submission platform. This validator ensures that your submission complies with the expected JSON schema and optionally performs additional checks based on dimensions, resolution, scene type, and confidence scores.

---

## Features

- Validate prediction submissions in JSON format.
- Supports both **CLI** and **library** usage.
- Optional arguments to control validation rules:
  - `--use_dimensions`
  - `--use_scene`
  - `--use_resolution`
  - `--use_confidence`

---

## Installation

Place `solafune_dataset.py` and necessary modules (e.g., `check_dict`, `return_error_message`) inside your project folder or install as a package if wrapped.

---

## Usage

### Using CLI

```bash
python solafune_dataset.py \
  --pred_file_path path/to/prediction.json \
  --gt_file_path path/to/groundtruth.json \
  --ann_type segmentation \
  --use_dimensions \
  --use_scene \
  --use_resolution \
  --use_confidence
```

Usage:

- --pred_file_path: Path to your predicted JSON file (required).
- --gt_file_path: Optional ground truth JSON file for extended validation.
- --ann_type: Type of annotation, e.g., "segmentation" or "bbox" (required).
- --use_dimensions: Enable validation on width and height of annotations.
- --use_scene: Enable validation on scene type.
- --use_resolution: Enable resolution checks.
- --use_confidence: Enable confidence score checks.

### Using library-base function

```python
from solafune_tools.competition_tools import solafune_dataset

# From variable dictionary
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
validated_submission = solafune_dataset.json_submission_validator(pred_dict = submission_dict, ann_type = "segmentation")
print(validated_submission)

# From saved file
submission_file = "path/to/your/file.json"
validated_submission = pq_scoring_submission.submission_validator(pred_file_path = submission_file, ann_type = "segmentation")
print(validated_submission)
```
