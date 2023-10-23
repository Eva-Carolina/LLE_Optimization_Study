from PIL import Image

def count_size(image_path):
    try:
        # Open the image using Pillow
        img = Image.open(image_path)

        # Get the dimensions of the image (width and height)
        width, height = img.size

        return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None

print(count_size("coluna.jpg")[0])

