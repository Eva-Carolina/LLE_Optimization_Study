# import the necessary packages
from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import cv2
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

#original é uma imagem a cores
#image é uma imagem a preto e branco
def detect_circles(img_color, img_canny, dp=1.5, center_dist=10, param2=50, minR= 15, maxR= 2):

    image=img_canny.copy()

    # Get the dimensions of the input image
    h, w = image.shape

    # Apply Hough Circle Transform to detect circles in the input image
    
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, int(w / center_dist), param2=param2, minRadius=int(minR),
    maxRadius=int(maxR))
    output = img_color.copy()
    green_circles=[]

    # ensure at least some circles we  found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        #print('circles', circles)
        circles_with_votes = np.hstack((circles, circles[:, 2:3]))
        #print(circles_with_votes)
        #print(circles)
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            output=cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            green_circles.append([x,y,r-10])
        #print('green circles', green_circles)
    return [output,green_circles]
    
#esta é a que avalia se as bolhas que eu la coloquei foram ou não encontradas 
#e se não foram encontradas então desenha um circulo verde
def detect_circles_score(img_color, original_centers, original_radii, detected_circles, tolerance=0.5, radius_margin=0.01):
    # Extract the centers and radii from the detected circles
    detected_centers = detected_circles[:, :2]
    detected_radii = detected_circles[:, 2]

    # Initialize arrays to store which circles were correctly detected
    correct_centers = np.zeros(len(original_centers))
    flag=0
    img=img_color.copy()
    percentage=0

    # Check if each original circle has a corresponding detected circle within the given tolerance
    for i, (oc, orad) in enumerate(zip(original_centers, original_radii)):
        distances = np.linalg.norm(detected_centers - np.array(oc), axis=1)
        #print(distances)
        if list(distances)==[]:
            #print('Hello?')
            break

        matches = np.argmin(distances)
        #if distances[matches]>10:
            #img=cv2.line(imagem,oc,detected_centers[matches], (255,0,0), 9)
        if distances[matches]<= tolerance*orad:

            flag=flag+1

            # If there is a match, check if the radii match within the given margin
            detected_radius = detected_radii[matches]
            if abs((orad-detected_radius))<=radius_margin*orad:
                # If the radii match within the margin, mark the original circle and detected circle as correct
                correct_centers[i] = 1

                # Remove the detected circle so it can't be matched to another original circle
                detected_centers = np.delete(detected_centers, matches, axis=0)
                detected_radii = np.delete(detected_radii, matches)
            else:
                img=cv2.circle(img, oc, int(orad), (255, 0, 0), 4)
        else:
            img=cv2.circle(img, oc, int(orad), (255, 0, 0), 4)

    # Calculate the percentage of correct detections
    percentage = np.sum(correct_centers) / len(correct_centers) * 100

    return percentage,img

#deteta os circulos na imagem, se o circulo detetado estiver vazio então desenha-o a vermelho e não entra 
#para o score de ser um circuloo
'''
def detect_circles_color(original,image,tolerancia, dp=1.5, center_dist=10, param2=50, minR= 15, maxR= 2):

    # Get the dimensions of the input image
    h, w = image.shape

    # Apply Hough Circle Transform to detect circles in the input image

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, int(w / center_dist), param2=param2, minRadius=int(minR),
    maxRadius=int(maxR))
    output = original.copy()
    tresh=original.copy()
    tresh=otsu_threshold(tresh)
    percent=[]


    # ensure at least some circles we  found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        circles_with_votes = np.hstack((circles, circles[:, 2:3]))
        #print(circles_with_votes)
        #print(circles)
        # loop over the (x, y) coordinates and radius of the circles
        total_circles=len(circles)
        count=0
        green_circles=[]
        for (x, y, r) in circles:
            mask = np.zeros_like(tresh)
            cv2.circle(mask, (x,y), r, 255, -1)
            #aqui estamos a contar o numero de pixeis no circulo da mascara
            number_of_pixels=np.count_nonzero(mask)
            #print('number of pixels',number_of_pixels)


            masked_img = cv2.bitwise_and(tresh, mask)

            num_white = np.count_nonzero(masked_img == 255)
            #print('number of white',num_white)
            
            percent_white = round(num_white /  number_of_pixels* 100, 2)
            percent.append(percent_white)
            if percent_white<=tolerancia:
                count=count+1
                output=cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                green_circles.append(r)
            else:
                output=cv2.circle(output, (x, y), r, (255, 165, 0), 4)
                output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #print(percent)
    score=count/total_circles*100
    return [output,circles,score,green_circles]
'''

#leva uma imagem a cores
def otsu_threshold(img_color):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    

    # Get the threshold using Otsu's method
    otsu_threshold, image_result = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #print(cv2.THRESH_BINARY)
    #print(cv2.THRESH_OTSU)
    # Return the thresholded image
    return image_result

def otsu_threshold_teste(img,value):
    # Convert the image to grayscale
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the threshold using Otsu's method
    ret, image_result = cv2.threshold(
        image_gray, value, 255, cv2.THRESH_BINARY)

    # Return the thresholded image
    return image_result



def threshold_otsu_impl(image, nbins=1):
    
    #validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return
    
    #validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return
    
    image=image[image!=0]
    
    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image)+nbins, np.max(image)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg*variance_fg + weight_bg*variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
        #print("trace:", within_class_variance, color_threshold)
            
    return least_variance_threshold

#nova função que acrescenta circulos laranjas
def detect_circles_color(img_color, img_canny, tolerancia=50,tolerancia_int=1, dp=1.5, center_dist=10, param2=50, minR= 15, maxR= 2):
    image=img_canny.copy()
    # Get the dimensions of the input image
    h, w = image.shape

    # Apply Hough Circle Transform to detect circles in the input image

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp, int(w / center_dist), param2=param2, minRadius=int(minR),
    maxRadius=int(maxR))
    output = img_color.copy()
    tresh=img_color.copy()
    tresh=otsu_threshold(tresh)
    percent=[]
    count=0
    score=1
    green_circles=[]
    tall=1
    # ensure at least some circles we  found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        circles_with_votes = np.hstack((circles, circles[:, 2:3]))
        #print(circles_with_votes)
        #print(circles)
        # loop over the (x, y) coordinates and radius of the circles
        total_circles=len(circles)
        
        
        for (x, y, r) in circles:
            mask = np.zeros_like(tresh)
            
            mask2= np.zeros_like(tresh)
            cv2.circle(mask, (x,y), r, 255, -1)
            #aqui estamos a contar o numero de pixeis no circulo da mascara
            number_of_pixels=np.count_nonzero(mask)
            #print('number of pixels',number_of_pixels)
            masked_img = cv2.bitwise_and(tresh, mask)
            num_white = np.count_nonzero(masked_img == 255)
            #print('number of white',num_white)
            percent_white = round(num_white /  number_of_pixels* 100, 2)
            percent.append(percent_white)
            if percent_white<=tolerancia:
                cv2.circle(mask2, (x,y), int(r/2), 255, -1)
                number_of_pixels2=np.count_nonzero(mask2)
                masked_img2 = cv2.bitwise_and(tresh, mask2)
                num_white2 = np.count_nonzero(masked_img2 == 255)
                percent_white2 = round(num_white2 /  number_of_pixels2* 100, 2)
                if percent_white2<=tolerancia_int:
                    #é considerado um verdadeiro circulo
                    count=count+1
                    output=cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    green_circles.append([x,y,r-5])
                else:
                    tall+=1
                    #output=cv2.circle(output, (x, y), r, (255, 165, 0), 4)
                    #output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                
            else:
                tall+=1
                #output=cv2.circle(output, (x, y), r, (255, 165, 0), 4)
                #output=cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    #print(percent)
    #score=count/total_circles*100
    return [output,circles,score,green_circles]


def detect_circles_score_mod(img_color, original_centers, original_radii, detected_circles, tolerance=0.5, radius_margin=0.01):
    # Extract the centers and radii from the detected circles
    detected_centers = detected_circles[:, :2]
    detected_radii = detected_circles[:, 2]

    # Initialize arrays to store which circles were correctly detected
    correct_centers = np.zeros(len(original_centers))
    flag=0
    img=img_color.copy()
    percentage=0

    # Check if each original circle has a corresponding detected circle within the given tolerance
    for i, (oc, orad) in enumerate(zip(original_centers, original_radii)):
        distances = np.linalg.norm(detected_centers - np.array(oc), axis=1)
        #print(distances)
        if list(distances)==[]:
            #print('Hello?')
            break

        matches = np.argmin(distances)
        #if distances[matches]>10:
            #img=cv2.line(imagem,oc,detected_centers[matches], (255,0,0), 9)
        if distances[matches]<= tolerance*orad:

            flag=flag+1

            # If there is a match, check if the radii match within the given margin
            detected_radius = detected_radii[matches]
            if abs((orad-detected_radius))<=radius_margin*orad:
                # If the radii match within the margin, mark the original circle and detected circle as correct
                correct_centers[i] = 1

                # Remove the detected circle so it can't be matched to another original circle
                detected_centers = np.delete(detected_centers, matches, axis=0)
                detected_radii = np.delete(detected_radii, matches)
            #else:
                #img=cv2.circle(img, oc, int(orad), (255, 0, 0), 4)
       # else:
           # img=cv2.circle(img, oc, int(orad), (255, 0, 0), 4)

    # Calculate the percentage of correct detections
    percentage = np.sum(correct_centers) / len(correct_centers) * 100

    return percentage,img

def remove_repeated_circles_mod(list1, list2, tolerance=0.5):
    merged_list = list1 + list2  # Combine the two input lists
    #print(merged_list)
    unique_circles = []  # List to store the unique circles
    

    # Iterate over each circle in the merged list
    for circle in merged_list:
        # Flag to indicate if the circle is a repeated circle
        is_repeated = False

        # Check if the circle is within the tolerance distance of any previously added unique circles
        for unique_circle in unique_circles:
            distance = ((circle[0] - unique_circle[0]) ** 2 + (circle[1] - unique_circle[1]) ** 2) ** 0.5
            #print (distance)
            #print (unique_circle[2]*tolerance)
            big=max(unique_circle[2],circle[2])
            #Neste caso dois circulos com o mesmo centro foram identificados, entra no loop
            if distance  <=  tolerance*big:
                #print('Hello')
                if unique_circle[2]<=circle[2]:
                    unique_circles.remove(unique_circle)
                    is_repeated = False
                    break
                else:
                    is_repeated = True
                    break
        # If the circle is not a repeated circle, add it to the unique circles list
        if not is_repeated:
            unique_circles.append(circle)


    return unique_circles

def draw_green_circles(green_circles, img_color):
    image=img_color.copy()
    for circle in green_circles:
        x, y, r = circle
        image = cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        image = cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return image



def draw_blue_circles(green_circles, img_color):
    image=img_color.copy()
    for circle in green_circles:
        x, y, r = circle
        image = cv2.circle(image, (int(x), int(y)), int(r), (255, 125, 125), 4)
        image = cv2.rectangle(image, (int(x) - 5, int(y) - 5), (int(x) + 5, int(y) + 5), (0, 128, 255), -1)
    return image


#detect circles
def detect_circles_final(img_color,img_canny,tolerancia,tolerancia_int,minR, maxR):
    sum=int((maxR-minR)/6)
    one=minR
    two=minR+sum
    tree=two+sum
    four=maxR

    first=detect_circles_color(img_color,img_canny,tolerancia=50,tolerancia_int=1,dp=1.5,center_dist=10,param2=20,minR=one,maxR=two)
    second=detect_circles_color(img_color,img_canny,tolerancia=35,tolerancia_int=1,dp=1.5,center_dist=10,param2=40,minR=two,maxR=tree)
    third=detect_circles_color(img_color,img_canny,tolerancia=10,tolerancia_int=1,dp=1.5,center_dist=10,param2=50,minR=tree,maxR=four)
    small=first[3]
    small_image=first[0]
  
    medium=second[3]
    medium_image=second[0]
    big=third[3]
    big_image=third[0]

    inicial=remove_repeated_circles_mod(small,medium,tolerance=1)
    final=remove_repeated_circles_mod(big,inicial,tolerance=1)
    
    return[final,small_image,medium_image,big_image]