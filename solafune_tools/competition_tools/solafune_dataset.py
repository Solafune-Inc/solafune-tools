import json
import os
import shutil
from typing import Tuple, Union
from solafune_tools.competition_tools.quality_checker import check_dict, return_error_message

def load_file(file_path) -> Union[dict, str]:
    """
    Loads a JSON file from the given file path.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        str: The content of the JSON file as a dictionary, or an error message if the file is not found or not in JSON format.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        return "File not found or not a json format"

def json_submission_validator(gt_file_path: str = "", pred_file_path: str = "",
                              gt_dict: dict = {}, pred_dict: dict = {},
                              ann_type: str = "",
                              use_dimensions: bool = False, 
                              use_scene: bool = False,
                              use_resolution: bool = False,
                              use_confidence: bool = False) -> str:
    """
    Validates a submission either from a file path or a dictionary.

    Args:
        gt_file_path (str, optional): The path to the ground truth JSON file.
        pred_file_path (str): The path to the prediction JSON file.
        gt_dict (dict, optional): The ground truth dictionary.
        pred_dict (dict): The prediction dictionary.
        ann_type (str): The type of annotation, e.g., 'segmentation' or 'bbox'.

    Returns:
        str: The validation result message.
        int: The number of images in the dictionary.

    Raises:
        ValueError: If neither file_path nor pdict is provided.
    """
    if pred_file_path != "" and ann_type != "":
        if gt_file_path:
            gtdict = load_file(gt_file_path)
            if isinstance(gtdict, str):
                gtdict = {}
        else:
            gtdict = {}
        pdict = load_file(pred_file_path)
        if isinstance(pdict, str):
            return pdict
        err_code = check_dict(gt_dict=gtdict, pd_dict=pdict, ann_type=ann_type,
                               use_dimensions=use_dimensions, use_scene=use_scene,
                               use_resolution=use_resolution, use_confidence=use_confidence)
        check = return_error_message(err_code, filetype="json")
        return check

    elif pred_dict != "" and ann_type != "":
        err_code = check_dict(gt_dict=gt_dict, pd_dict=pred_dict, ann_type=ann_type,
                               use_dimensions=use_dimensions, use_scene=use_scene,
                               use_resolution=use_resolution, use_confidence=use_confidence)
        check = return_error_message(err_code, filetype="json")
        return check

    raise ValueError("No file_path or pdict provided or ann_type is not provided")

def main(args):
    gt_file_path = args.gt_file_path
    pred_file_path = args.pred_file_path
    ann_type = args.ann_type
    use_dimensions = args.use_dimensions
    use_scene = args.use_scene
    use_resolution = args.use_resolution
    use_confidence = args.use_confidence

    if gt_file_path:
        gtdict = load_file(gt_file_path)
        if isinstance(gtdict, str):
            gtdict = {}
    else:
        gtdict = {}
    pdict = load_file(pred_file_path)
    if isinstance(pdict, str):
        return pdict
    err_code = check_dict(gt_dict=gtdict, pd_dict=pdict, ann_type=ann_type,
                            use_dimensions=use_dimensions, use_scene=use_scene,
                            use_resolution=use_resolution, use_confidence=use_confidence)
    check = return_error_message(err_code, filetype="json")
    return check

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file_path', type=str, default="", help='Path to the ground truth JSON file')
    parser.add_argument('--pred_file_path', type=str, default="", help='Path to the prediction JSON file')
    parser.add_argument('--ann_type', type=str, required=True, help='Type of annotation (e.g., "segmentation", "bbox")')
    parser.add_argument('--use_dimensions', action='store_true', help='Use dimensions in validation')
    parser.add_argument('--use_scene', action='store_true', help='Use scene type in validation')
    parser.add_argument('--use_resolution', action='store_true', help='Use resolution in validation')
    parser.add_argument('--use_confidence', action='store_true', help='Use confidence score in validation')

    args = parser.parse_args()
    result = main(args)
    print(result)
    if isinstance(result, str):
        print(result)
    else:
        print("Validation completed successfully.")