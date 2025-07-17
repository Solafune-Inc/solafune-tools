import os

def move_weights():
    """
    Executes the final steps of the training process.

    This function performs the following actions:
    - Move the trained model to the 'weights' directory.
    """
    # Move the trained model to the 'weights' directory
    if not os.path.exists("weights"):
        os.makedirs("weights")
    
    list_of_weights_files = [f for f in os.listdir('working') if f.endswith('.pth') or f.endswith('.ckpt')]

    for file in list_of_weights_files:
        source_path = os.path.join('working', file)
        destination_path = os.path.join('weights', file)
        
        if os.path.exists(destination_path):
            os.remove(destination_path)
        os.rename(source_path, destination_path)
    print("Trained model has been moved to the 'weights' directory.")