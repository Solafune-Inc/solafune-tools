import json

def load_file(file_path) -> str:
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        return "File not found or not a json format"


def check_dict(pdict : dict) -> str:
    image_names = [f'test_{x}.tif' for x in range(0, 50)]
    if not pdict:
        return "Empty dictionary"
    
    if pdict is None:
        return "Empty dictionary"
    
    if not "images" in pdict:
        return "No \"images\" key in dictionary"
    
    if not isinstance(pdict["images"], list):
        return "images is not a list"
    
    
    for image in pdict["images"]:
        if not "file_name" in image:
            return "No \"file_name\" key in image"
        if not image["file_name"] in image_names:
            return "Invalid file_name in image"
        image_names.remove(image["file_name"])
        
        if not "annotations" in image:
            return "No \"annotations\" key in image"
        
        if not isinstance(image["annotations"], list):
            return "annotations is not a list"
        
        for anno in image["annotations"]:
            if not "segmentation" in anno:
                return "No \"segmentation\" key in annotation"
            
            if not isinstance(anno["segmentation"], list):
                return "segmentation is not a list"
            
            if len(anno["segmentation"]) % 2 != 0:
                return "segmentation format is invalid, the number of ssegmnets should be even"
            
            if len(anno["segmentation"]) < 4:
                return "segmentation format is invalid, the number of segmnets should be at least 4"
    
    if not len(image_names) == 0:
        return "Some images are missing"
    
    return "Valid"

def submission_validator(file_path: str = None, pdict: dict = None) -> str:
    if file_path is not None:
        pdict = load_file(file_path)
        if pdict == "File not found":
            return "File not found"
        check = check_dict(pdict)
        return check
    
    if pdict is not None:
        check = check_dict(pdict)
        return check

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