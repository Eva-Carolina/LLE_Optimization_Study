import csv
import cv2
import os


class CircleDrawer:
    def __init__(self, image):
        self.image = image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.circles = []

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            radius = int(((x - self.ix) ** 2 + (y - self.iy) ** 2) ** 0.5)
            self.circles.append((self.ix, self.iy, radius))
            cv2.circle(self.image, (self.ix, self.iy), radius, (0, 255, 0), 2)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_img = self.image.copy()
                radius = int(((x - self.ix) ** 2 + (y - self.iy) ** 2) ** 0.5)
                cv2.circle(temp_img, (self.ix, self.iy), radius, (0, 255, 0), 2)
                cv2.imshow('image', temp_img)

    def draw_circles_on_image(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.draw_circle)

        while True:
            cv2.imshow('image', self.image)
            key = cv2.waitKey(1)

            if key == ord('q') or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

    def save_circles_to_csv(self, image_filename):
        filename, _ = os.path.splitext(image_filename)
        csv_filename = filename + '.csv'
        data = [(x, y, r) for x, y, r in self.circles]
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

    def get_circles(self):
        return self.circles


# Load the original image
original_image_filename = 'left_frame_288.jpg'
original_image = cv2.imread(original_image_filename)

# Resize the image to the desired dimensions
resized_image = cv2.resize(original_image, (1000, 2000))

# Create an instance of the CircleDrawer class
circle_drawer = CircleDrawer(resized_image)

# Draw circles on the image interactively
circle_drawer.draw_circles_on_image()

# Get the list of circles
circles = circle_drawer.get_circles()

# Print the circles in the desired format
formatted_circles = [(x, y, r) for x, y, r in circles]
print(formatted_circles)

# Save circles to CSV file
circle_drawer.save_circles_to_csv(original_image_filename)


'''
#isto é para usar o ficheiro cvs que estamos a criar aqui noutro 
#código. 

import csv

def load_circles_from_csv(csv_filename):
    circles = []
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x, y, r = map(int, row)
            circles.append((x, y, r))
    return circles

# Example usage
csv_filename = 'your_file.csv'
circles = load_circles_from_csv(csv_filename)

# Now you can use the 'circles' list in your code
for circle in circles:
    x, y, r = circle
'''