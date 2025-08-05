def check_dict(pd_dict: dict = {}, gt_dict: dict = {}, 
               ann_type: str = "segmentation",
               use_dimensions: bool = False, 
               use_scene: bool = False,
               use_resolution: bool = False,
               use_confidence: bool = False) -> int:
    """
    Check the format of the prediction dictionaries.
    Error codes:
        0: Valid
        1: Incorrect format, missing "images" key, or "images" is not a list
        2: Incorrect format, image file name not found in ground truth from the prediction
        3: Incorrect format, missing "scene_type" key in one of the images
        4: Incorrect format, missing "cm_resolution" key in one of the images
        5: Incorrect format, missing "annotations" key, or "annotations" is not a list in one of the images
        6: Incorrect format, missing "bbox/segmentation" key, or "bbox/segmentation" is not a list in one of the annotations
        7: Incorrect format, missing "confidence_score" key in one of the annotations
        8: Incorrect format, one of the bbox/segmentation list length is less than 4 or not even in one of the annotations
        9: Incorrect format, missing "class" key in one of the annotations
        10: Incorrect format, missing "width" or "height" key in one of the images
        11: Incorrect format, image files length mismatch between prediction and ground truth
        12: No ground truth provided, but prediction dictionary is empty
    """
    if gt_dict:
        if pd_dict == None and gt_dict == None:
            return 12
        image_names = [image["file_name"] for image in gt_dict["images"]]

        if not len(image_names) == 0:
            return 11
        
    else:
        image_names = [image["file_name"] for image in pd_dict["images"]]

    if not pd_dict:
        return 1
    
    if pd_dict is None:
        return 1
    
    if not "images" in pd_dict:
        return 1
    
    if not isinstance(pd_dict["images"], list):
        return 1
    
    for image in pd_dict["images"]:
        if not "file_name" in image:
            return 2
        if not image["file_name"] in image_names:
            return 2
        image_names.remove(image["file_name"])
        
        if use_scene and not "scene_type" in image:
            return 3
        
        if use_resolution and not "cm_resolution" in image:
            return 4

        if not "annotations" in image:
            return 5
        
        if not isinstance(image["annotations"], list):
            return 5
        
        for anno in image["annotations"]:
            if not ann_type in anno:
                return 6
            
            if not isinstance(anno[ann_type], list):
                return 6

            if use_confidence and not "confidence_score" in anno:
                return 7

            if ann_type == "bbox" and len(anno[ann_type]) != 4:
                return 8

            if len(anno[ann_type]) % 2 != 0:
                return 8
            
            if len(anno[ann_type]) < 4:
                return 8
            
            if not "class" in anno:
                return 9
            
        if use_dimensions and ("width" not in image or "height" not in image):
            return 10

    return 0

def return_error_message(error_code: int, filetype: str = "json") -> str:
    """
    Set the error messages of the prediction dictionaries or zipfile.
    Error codes for dictionaries:
        0: Valid
        1: Incorrect format, missing "images" key, or "images" is not a list
        2: Incorrect format, image file name not found in ground truth from the prediction
        3: Incorrect format, missing "scene_type" key in one of the images
        4: Incorrect format, missing "cm_resolution" key in one of the images
        5: Incorrect format, missing "annotations" key, or "annotations" is not a list in one of the images
        6: Incorrect format, missing "bbox/segmentation" key, or "bbox/segmentation" is not a list in one of the annotations
        7: Incorrect format, missing "confidence_score" key in one of the annotations
        8: Incorrect format, one of the bbox/segmentation list length is less than 4 or not even in one of the annotations
        9: Incorrect format, missing "class" key in one of the annotations
        10: Incorrect format, missing "width" or "height" key in one of the images
        11: Incorrect format, image files length mismatch between prediction and ground truth
        12: No ground truth provided, but prediction dictionary is empty
    Error codes for zip files:
        1: Error extracting zip file.
        2: Either bbox.json or segmentation.json not found.
        9: Error extracting zip file.
    """
    if filetype == "json":
        if error_code == 0:
            return "Valid"
        elif error_code == 1:
            return "Incorrect format, missing 'images' key, or 'images' is not a list."
        elif error_code == 2:
            return "Incorrect format, image file name not found in ground truth from the prediction."
        elif error_code == 3:
            return "Incorrect format, missing 'scene_type' key in one of the images."
        elif error_code == 4:
            return "Incorrect format, missing 'cm_resolution' key in one of the images."
        elif error_code == 5:
            return "Incorrect format, missing 'annotations' key, or 'annotations' is not a list in one of the images."
        elif error_code == 6:
            return "Incorrect format, missing 'bbox/segmentation' key, or 'bbox/segmentation' is not a list in one of the annotations."
        elif error_code == 7:
            return "Incorrect format, missing 'confidence_score' key in one of the annotations."
        elif error_code == 8:
            return "Incorrect format, one of the bbox/segmentation list length is less than 4 or not even in one of the annotations."
        elif error_code == 9:
            return "Incorrect format, missing 'class' key in one of the annotations."
        elif error_code == 10:
            return "Incorrect format, missing 'width' or 'height' key in one of the images."
        elif error_code == 11:
            return "Incorrect format, image files length mismatch between prediction and ground truth."
        elif error_code == 12:
            return "No ground truth provided, but prediction dictionary is empty."
        
    elif filetype == "zip":
        if error_code == 1:
            return "Error extracting zip file."
        elif error_code == 2:
            return "Either bbox.json or segmentation.json not found."
        elif error_code == 9:
            return "Error extracting zip file."
        
    return "Unknown error, contact the developer for more information."