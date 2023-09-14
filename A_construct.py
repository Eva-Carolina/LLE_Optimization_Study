import cv2
import os
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import math

def generate_random_num(prob=0.2,min=0.7,max=1):
    rand_num = random.random()
    if rand_num < prob:
        output = random.uniform(min, max)
    else:
        output=1
    return output

def overlay_transparent(bg_img, overlay, x, y, feather_size=51, sigma=5, alpha=0.5):
        overlay_h, overlay_w, _ = overlay.shape
        if overlay.shape[2] == 3:
            overlay = np.dstack([overlay, np.ones((overlay_h, overlay_w, 1), dtype=overlay.dtype)] * 255)

        # Create a region of interest (ROI) for the overlay image
        roi = bg_img[y:y+overlay_h, x:x+overlay_w]

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0
    
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGBA2BGR)
        #overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

        # Feather the edges of the mask
        kernel_size = feather_size * 2 + 1
        #kernel = cv2.getGaussianKernel(kernel_size, sigma)
        mask = cv2.GaussianBlur(mask, (kernel_size,kernel_size),sigma)

        # Expand dimensions of mask and alpha arrays to match overlay_image
        mask = np.expand_dims(mask, axis=2)
        alpha = alpha * mask

        # Blend the overlay and the background image
        bg_img[y:y+overlay_h, x:x+overlay_w] = (1.0 - alpha) * roi + alpha * overlay_image

        return bg_img



def rot_n_increase(image, maxD, minD, P_F, scale, shape):

    old_w=image.shape[1]
    #print('este é o w antigo',old_w)
    old_radious=old_w*0.8
    #print('este é o raio da bolha quando foi capturada',old_radious)
    im = np.rot90(image, k=np.random.randint(0, 3))

    

    # Generate a value from the normal distribution
    size= 2*np.random.lognormal(mean=np.log(scale), sigma=shape)

    # Generate a new random number if the size is larger than maxD or smaller than minD
    while size > maxD or size < minD:
        
        size = 2*np.random.lognormal(mean=np.log(scale), sigma=shape)

    # Convert the size to an integer
    size = int(size)
    

    # Generate a random value for the second dimension
    rand_val = generate_random_num(prob=P_F, min=0.7, max=1)

    # Resize the image with the updated size
    im = cv2.resize(im, (size,int( size*rand_val)))
    return im


def rot_n_increase_old(image, maxD, minD, P_F, deviation):

    old_w=image.shape[1]
    #print('este é o w antigo',old_w)
    old_radious=old_w*0.8
    #print('este é o raio da bolha quando foi capturada',old_radious)
    im = np.rot90(image, k=np.random.randint(0, 3))
    sigma = (maxD - minD) / 8
    mu = ((maxD + minD) / 2)-(deviation*sigma)
    

    # Generate a value from the normal distribution
    size = np.random.lognormal(mean=np.log(mu), sigma=sigma)

    # Generate a new random number if the size is larger than maxD or smaller than minD
    while size > maxD or size < minD:
        
        size = np.random.normal(mu, sigma)

    # Convert the size to an integer
    size = int(size)
    

    # Generate a random value for the second dimension
    rand_val = generate_random_num(prob=P_F, min=0.7, max=1)

    # Resize the image with the updated size
    im = cv2.resize(im, (size,int( size*rand_val)))
    return im

def get_random_cell(cell_path):
        directory = cell_path
        filenames = os.listdir(directory)
        cell_filename = os.path.join(directory, random.choice(filenames))
        return cv2.imread(cell_filename, cv2.IMREAD_UNCHANGED)
    

def insert_cell(bground_path, cell_path,num_cells, overlap_percentage,min_d,max_d,p_trans,p_flat,s=95.56,m=0.198):
        #random.seed(54)
        min_d=min_d
        max_d=max_d
        bground = cv2.imread(bground_path)
        bground = cv2.cvtColor(bground, cv2.COLOR_BGR2RGB)
        new_height = bground.shape[0]
        new_width = bground.shape[1]
        used_positions = []
        centers=[]
        radii = []
        for i in range(num_cells):
            cell_type = get_random_cell(cell_path)
            
            cell = rot_n_increase(cell_type, max_d, min_d,p_flat,scale=s,shape=m)
            shape = cell.shape
            h = shape[0]
            w = shape[1]
            #print('este é o w', w)

            radius = (w*0.8)/2  # save radius of the image for later use
            #print('este é o r',radius)
            x_tl = random.randint(0, new_width - w)
            x=x_tl+int(w/2)
            y_tl = random.randint(0, new_height - h)
            y=y_tl+int(h/2)
            overlap = overlap_percentage * radius
            while True:
                found_overlap = False
                for pos, r in zip(used_positions, radii):
                    px, py = pos
                    dist = math.sqrt((x-px)**2 + (y-py)**2)
                    if dist < (radius + r - overlap):  # break if overlap found
                        found_overlap = True
                        break
                if not found_overlap:
                    
                    break
                x_tl = random.randint(0, new_width - w)
                x=x_tl+int(w/2)
                y_tl = random.randint(0, new_height - h)
                y=y_tl+int(h/2)
            used_positions.append((x, y))
            centers.append((x, y))
            radii.append(radius) 
            #print(radii)
                #rand_num_sigma=random.randint(2,100)
            rand_num_alpha = alpha_number(h/max_d + p_trans)
            #rand_num_alpha = random.uniform(0.8, 1)
            rand_num_feather=int(150*(1-h/max_d))
            
            bground = overlay_transparent(bground, cell, x_tl, y_tl,feather_size=rand_num_feather, alpha=rand_num_alpha)
        #print('end')
        return [bground, centers,radii]


def insert_cell_mod(bground_path, cell_path,num_cells, overlap_percentage,min_d,max_d,p_trans,p_flat):
        #random.seed(54)
        min_d=min_d+40
        max_d=max_d+40
        bground = cv2.imread(bground_path)
        bground = cv2.cvtColor(bground, cv2.COLOR_BGR2RGB)
        bground = cv2.resize(bground, (1000, 2000))

        new_height = bground.shape[0]
        new_width = bground.shape[1]
        used_positions = []
        centers=[]
        radii = []
        for i in range(num_cells):
            cell_type = get_random_cell(cell_path)
            cell = rot_n_increase(cell_type, max_d, min_d,p_flat)
            shape = cell.shape
            h = shape[0]
            w = shape[1]

            radius = (w-40) / 2  # save radius of the image for later use
            
            x_tl = random.randint(0, new_width - w)
            x=x_tl+int(w/2)
            y_tl = random.randint(0, new_height - h)
            y=y_tl+int(h/2)
            overlap = overlap_percentage * radius
            while True:
                found_overlap = False
                for pos, r in zip(used_positions, radii):
                    px, py = pos
                    dist = math.sqrt((x-px)**2 + (y-py)**2)
                    if dist < (radius + r - overlap):  # break if overlap found
                        found_overlap = True
                        break
                if not found_overlap:
                    
                    break
                x_tl = random.randint(0, new_width - w)
                x=x_tl+int(w/2)
                y_tl = random.randint(0, new_height - h)
                y=y_tl+int(h/2)
            used_positions.append((x, y))
            centers.append((x, y))
            radii.append(radius) 
                #rand_num_sigma=random.randint(2,100)
            rand_num_alpha = alpha_number(h/max_d + p_trans)
            #rand_num_alpha = random.uniform(0.8, 1)
            rand_num_feather=int(150*(1-h/max_d))
            
            bground = overlay_transparent(bground, cell, x_tl, y_tl,feather_size=rand_num_feather, alpha=rand_num_alpha)
        return [bground, centers,radii]



def alpha_number(num):
    if num < 1:
        return num
    else:
        return 1







caminho_cell=r"C:\Users\Asus\OneDrive - Universidade de Lisboa\TESE\Python\saved_el"
caminho_bg=r"C:\Users\Asus\OneDrive - Universidade de Lisboa\TESE\Python\saved_rect\claro.png"
n_cell=20
ov_percentage=0
diam_min=20
diam_max=300
probab_trans=0.2
probab_flat=0.2


#final=insert_cell(caminho_bg,caminho_cell,n_cell,ov_percentage,diam_min,diam_max,probab_trans,probab_flat)

#plt.imshow(final)
#plt.show()