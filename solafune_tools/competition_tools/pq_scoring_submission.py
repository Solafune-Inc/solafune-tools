import json

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


def check_dict(pdict : dict) -> str:
    """
    Validates the structure of a dictionary containing image and annotation data.

    Args:
        pdict (dict): The dictionary to validate. Expected to have an "images" key with a list of image dictionaries.

    Returns:
        str: A message indicating the validation result.
        int: The number of images in the dictionary.

    The function checks for the following:
    - The dictionary is not empty.
    - The dictionary contains an "images" key.
    - The "images" key maps to a list.
    - Each image dictionary contains a "file_name" key with a valid value.
    - Each image dictionary contains an "annotations" key with a list of annotations.
    - Each annotation contains a "segmentation" key with a list of coordinates.
    - The "segmentation" list has an even number of elements and at least 4 elements.
    """
    
    if not pdict:
        return "Empty dictionary"
    
    if pdict is None:
        return "Empty dictionary"
    
    if not "images" in pdict:
        return "No \"images\" key in dictionary"
    
    if not isinstance(pdict["images"], list):
        return "images is not a list"
    
    if str(pdict["images"][0]["file_name"]).split("_")[0] == "evaluation":
        image_names = [f'evaluation_{x}.tif' for x in range(len(pdict["images"]))]
    else:
        image_names = [f'test_{x}.tif' for x in range(len(pdict["images"]))]
    
    for image in pdict["images"]:
        if not "file_name" in image:
            return "No \"file_name\" key in image" , len(pdict["images"])
        if not image["file_name"] in image_names:
            return "Invalid file_name in image", len(pdict["images"])
        image_names.remove(image["file_name"])
        
        if not "annotations" in image:
            return "No \"annotations\" key in image", len(pdict["images"])
        
        if not isinstance(image["annotations"], list):
            return "annotations is not a list", len(pdict["images"])
        
        for anno in image["annotations"]:
            if not "segmentation" in anno:
                return "No \"segmentation\" key in annotation", len(pdict["images"])
            
            if not isinstance(anno["segmentation"], list):
                return "segmentation is not a list", len(pdict["images"])
            
            if len(anno["segmentation"]) % 2 != 0:
                return "segmentation format is invalid, the number of ssegmnets should be even", len(pdict["images"])
            
            if len(anno["segmentation"]) < 4:
                return "segmentation format is invalid, the number of segmnets should be at least 4", len(pdict["images"])
    
    #if not len(image_names) == 0:
    #    return "Some images are missing"
    
    return "Valid", len(pdict["images"])

def submission_validator(file_path: str = None, pdict: dict = None) -> str:
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
    if file_path is not None:
        pdict = load_file(file_path)
        if pdict == "File not found":
            return "File not found", 0
        check, num_image = check_dict(pdict)
        return check, num_image
    
    if pdict is not None:
        check, num_image = check_dict(pdict)
        return check, num_image

    raise ValueError("No file_path or pdict provided")

def main(args):
    file_path = args.file_path
    pdict = load_file(file_path)
    if pdict == "File not found":
        print("File not found")
        return
    
    check = check_dict(pdict)
    print(check)
    return check

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the file")
    args = parser.parse_args()
    main(args)