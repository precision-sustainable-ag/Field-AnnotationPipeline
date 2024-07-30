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

def shifting_bbox_coordinates(padded_image, bbox):
    height, width, _ = image.shape
    crop_height, crop_width = crop_size

    # Ensure the original image is larger than the crop size
    if height < crop_height or width < crop_width:
        raise ValueError("The original image is smaller than the crop size.")

    # Calculate the starting point of the crop
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    # Crop the image
    cropped_image_bgr = image[start_y:start_y + crop_height, start_x:start_x + crop_width]    
    cropped_image = cv2.cvtColor(cropped_image_bgr, cv2.COLOR_BGR2RGB)
    # x_min, y_min, x_max, y_max = map(int, bbox) # bbox coordintates

    # Shift the bounding box coordinates
    # x_min = x_min - start_x
    # y_min = y_min - start_y
    # x_max = x_max - start_x
    # y_max = y_max - start_y

    # # Ensure the bounding box is within the cropped image bounds
    # new_x1 = max(0, min(new_x1, crop_width))
    # new_y1 = max(0, min(new_y1, crop_height))
    # new_x2 = max(0, min(new_x2, crop_width))
    # new_y2 = max(0, min(new_y2, crop_height))

    # # Only adjust the bounding box if it is within the cropped image
    # if x_min < x_max and y_min < y_max:
    #     adjusted_bbox_cropped_image = (x_min, y_min, x_max, y_max)
    # else:
    #     adjusted_bbox_cropped_image = None

    # return cropped_image, adjusted_bbox_cropped_image, x_min, y_min, x_max, y_max
    return cropped_image