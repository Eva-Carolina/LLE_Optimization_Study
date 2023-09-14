import os
import cv2

def crop_and_save_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define the crop ranges
    crop_ranges = [
        {"name": "left", "x_start": 420, "x_end": 670, "y_start": 220, "y_end": 720},
        {"name": "right", "x_start": 840, "x_end": 1090, "y_start": 220, "y_end": 720}
    ]

    # Perform cropping and save the images
    for crop_range in crop_ranges:
        crop_name = crop_range["name"]
        x_start = crop_range["x_start"]
        x_end = crop_range["x_end"]
        y_start = crop_range["y_start"]
        y_end = crop_range["y_end"]

        # Crop the image
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Get the directory and filename of the original image
        image_directory, image_filename = os.path.split(image_path)

        # Construct the output path for the cropped image
        output_path = os.path.join(image_directory, crop_name + "_" + image_filename)

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)

        print("Cropped image saved as:", output_path)

def crop_and_save_images_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Filter files with names starting with "frame_" and ending with ".jpg"
    image_files = [file for file in file_list if file.startswith("frame_") and file.endswith(".jpg")]

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        crop_and_save_image(image_path)

# Test the function with the folder path containing the images
folder_path = r"C:\Users\Asus\Desktop\final videos coluna\final\c1_1"
crop_and_save_images_in_folder(folder_path)