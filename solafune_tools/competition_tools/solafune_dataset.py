import json
import os
import shutil

def load_file(file_path) -> str:
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

def check_dict(p_dict: dict = None, ann_type: str = "segmentation"):
    """
    Check the format of the prediction dictionaries.
    Error codes:
        0: No errors
        1: Invalid format, missing "images" key, or "images" is not a list
        2: Invalid format, image file name not found in ground truth from the prediction
        3: Invalid format, missing "annotations" key, or "annotations" is not a list in one of the images
        4: Invalid format, missing "bbox/segmentation" key, or "bbox/segmentation" is not a list in one of the annotations
        5: Invalid format, one of bbox/segmentation list length is less than 4 or not even in one of the annotations
        6: Invalid format, missing "class" key in one of the annotations
        7: Invalid format, missing "width" or "height" key in one of the images
    """
    if not p_dict:
        return 1, None
    
    if p_dict is None:
        return 1, None
    
    if not "images" in p_dict:
        return 1, None
    
    if not isinstance(p_dict["images"], list):
        return 1, None
    
    for image in p_dict["images"]:
        if not "file_name" in image:
            return 2, None
        
        if not "annotations" in image:
            return 3, None
        
        if not isinstance(image["annotations"], list):
            return 3, None
        
        for anno in image["annotations"]:
            if not ann_type in anno:
                return 4, None
            
            if not isinstance(anno[ann_type], list):
                return 4, None
            
            if len(anno[ann_type]) % 2 != 0:
                return 5, None
            
            if ann_type == "bbox" and len(anno[ann_type]) != 4:
                return 5, None
            
            if len(anno[ann_type]) < 4:
                return 5, None
            
            if not "class" in anno:
                return 6, None
            
        if not "width" in image:
            return 7, None
        
        if not "height" in image:
            return 7, None
    
    return 0, len(p_dict["images"])

def return_error_message(error_code: int, filetype: str = "json"):
    """
    Set the error messages of the prediction dictionaries or zipfile.
    Error codes:
        0: No errors
        1: Invalid format, missing "images" key, or "images" is not a list
        2: Invalid format, no file_name key found in one of the images
        3: Invalid format, missing "annotations" key, or "annotations" is not a list in one of the images
        4: Invalid format, missing "bbox/segmentation" key, or "bbox/segmentation" is not a list in one of the annotations
        5: Invalid format, one of bbox/segmentation list length is less than 4 or not even in one of the annotations
        6: Invalid format, missing "class" key in one of the annotations
        7: Invalid format, missing "width" or "height" key in one of the images
    """
    if filetype == "json":
        if error_code == 0:
            return "Valid"
        elif error_code == 1:
            return "Invalid format, missing 'images' key, or 'images' is not a list."
        elif error_code == 2:
            return "Invalid format, missing 'file_name' key in one of the images."
        elif error_code == 3:
            return "Invalid format, missing 'annotations' key, or 'annotations' is not a list in one of the images."
        elif error_code == 4:
            return "Invalid format, missing 'bbox/segmentation' key, or 'bbox/segmentation' is not a list in one of the annotations."
        elif error_code == 5:
            return "Invalid format, one of bbox/segmentation list length is less than 4 or not even in one of the annotations."
        elif error_code == 6:
            return "Invalid format, missing 'class' key in one of the annotations."
        elif error_code == 7:
            return "Invalid format, missing 'width' or 'height' key in one of the images."
        
    elif filetype == "zip":
        if error_code == 1:
            return "Error extracting zip file."
        elif error_code == 2:
            return "Either bbox.json or segmentation.json not found."
        elif error_code == 9:
            return "Error extracting zip file."

def json_submission_validator(file_path: str = None, pdict: dict = None, ann_type: str = None) -> str:
    """
    Validates a submission either from a file path or a dictionary.

    Args:
        file_path (str, optional): The path to the JSON file to validate.
        pdict (dict, optional): The dictionary to validate.

    Returns:
        str: The validation result message.
        int: The number of images in the dictionary.

    Raises:
        ValueError: If neither file_path nor pdict is provided.
    """
    if file_path is not None and ann_type is not None:
        pdict = load_file(file_path)
        if pdict == "File not found":
            return "File not found", 0
        err_code, num_image = check_dict(pdict, ann_type = ann_type)
        check = return_error_message(err_code, filetype = "json")
        return check, num_image
    
    if pdict is not None and ann_type is not None:
        err_code, num_image = check_dict(pdict, ann_type = ann_type)
        check = return_error_message(err_code, filetype = "json")
        return check, num_image

    raise ValueError("No file_path or pdict provided or ann_type is not provided")

def main(args):
    file_path = args.file_path
    ann_type = args.ann_type
    pdict = load_file(file_path)
    if pdict == "File not found":
        print("File not found")
        return
    
    check = check_dict(pdict, ann_type = ann_type)
    print(check)
    return check

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the file")
    parser.add_argument("--ann_type", type=str, help="Type of annotation, eg. 'segmentation' or 'bbox'")
    args = parser.parse_args()
    main(args)