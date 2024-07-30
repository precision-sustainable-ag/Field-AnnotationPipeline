import cv2

def add_padding_to_cropout(cropout_image):
    """
    Add padding to the image to make size (1024, 1024)
    """
    image = cv2.cvtColor(cropout_image, cv2.COLOR_BGR2RGB)
    
    height, width, _ = image.shape

    # calculate padding needed to reach 1024 x 1024
    top = (1024 - height) // 2 if height < 1024 else 0
    bottom = 1024 - height - top if height < 1024 else 0
    left = (1024 - width) // 2 if width < 1024 else 0
    right = 1024 - width - left if width < 1024 else 0   

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]) 
    
    cv2.imwrite("test_images/MDA06501_padded.jpg", padded_image)

    padding = {'top': top, 'bottom': bottom, 'left': left, 'right': right}

    return padded_image, padding