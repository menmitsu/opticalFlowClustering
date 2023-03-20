# This program takes an input image and computes clusters of the most dominant colors as defined by the user
# It saves the clusters into a csv file along with the HSV and hue information

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
import csv,os
import re
import colorsys

def parse_arguments():
    """
    Parse command-line arguments and return them as a dictionary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
                    help="Path to the image")
    ap.add_argument("-c", "--clusters", required=True, type=int,
                    help="# of clusters")
    ap.add_argument("-f", "--csv", required=True, type=str,
                    help="# of clusters")
    
    return vars(ap.parse_args())

def read_image(image_path):
    """
    Read an image from disk and return it as a NumPy array in RGB format.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    """
    Preprocess an image by converting it to grayscale, applying thresholding, and
    merging the result with the original image to create a four-channel RGBA image.
    """


     # Threshold black pixels
    image[image < 30] = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    alpha[alpha > 0] = 255
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]


    return cv2.merge(rgba, 4)

def cluster_colors(image, n_clusters, image_path,csv_file):
    """
    Cluster the colors in an image using K-means clustering and return the
    resulting color bar image as a NumPy array.
    """

    # print(image.shape)

    # # # Filter out pixels with alpha value of 0
    # image = image[image[:, :, 3] > 0]

    flattened_image = image.reshape(image.shape[0] * image.shape[1], 4)
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(flattened_image)


    # hist = utils.centroid_histogram(clt)
    # print(hist)
    
    # bar = utils.plot_colors(hist, clt.cluster_centers_)

    # print("Centroid clusters:",clt.cluster_centers_)

    # get labels for all points
    labels = clt.predict(flattened_image)

    # get counts for each unique label
    label_counts = np.bincount(labels)



    # print("Label",labels)

     # Get percentages of each cluster and sort by percentage
    label_percentages = label_counts.astype(float) / len(flattened_image)
    label_info = []
    
      
    for i, centroid in enumerate(clt.cluster_centers_):
        label_info.append((label_percentages[i], f"Cluster {i+1}", centroid))

    # Sort the list of tuples by label percentages in descending order
    label_info = sorted(label_info, key=lambda x: x[0], reverse=True)

    # Print the sorted list of tuples
    for label_percentage, label_name, cluster_center in label_info:
        # print(f"{label_name}: {label_percentage*100:.2f}%\nCluster Center: {np.round(cluster_center,decimals=2)}\n")
        # print(f"{label_name}: {label_percentage*100:.2f}%\nCluster Center: {np.rint(cluster_center)}\n")
        pass


    # Save top 2 cluster centers to CSV file
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if os.stat('cluster_centers.csv').st_size == 0:
            # writer.writerow(["File name", "Cluster 1", "Cluster 2", "HSV Cluster 1","HSV Cluster 2, Hue 0, Hue 1"])
            writer.writerow(["File name", "Cluster 1", "HSV Cluster 1","Hue 0"])
        clusters = []
        for i in range(1):
            clusters.append(np.rint(label_info[i][2]))

        r0, g0, b0,a0 =clusters[0]
        # r1, g1, b1,a1 =clusters[1]

        rgb0 = np.array([[[r0, g0, b0]]], dtype=np.uint8)
        # rgb1 = np.array([[[r1, g1, b1]]], dtype=np.uint8)

        # Convert RGB to HSV using OpenCV
        hsv0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2HSV)

        # print(rgb0)

        # hsv1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2HSV)

        # print("HSVs",hsv0[0][0],"  ",hsv1[0][0])
        # print("HSVs",hsv0[0][0])

        # hsv1 = cv2.cvtColor([[ [r1, g1, b1] ]], cv2.COLOR_RGB2HSV)

        # writer.writerow([os.path.basename(image_path), clusters[0], clusters[1], hsv0,hsv1,hsv0[0][0][0],hsv1[0][0][0]])
        # writer.writerow([os.path.basename(image_path), clusters[0], hsv0,hsv0[0][0][0]])
        writer.writerow([image_path, clusters[0], hsv0,hsv0[0][0][0]])

    return None

def get_number(filename):
    pattern = re.compile(r'(\d+)')
    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    else:
        return None

if __name__ == "__main__":
    args = parse_arguments()
    dirs = args['dir']
    for contentFolder in sorted(os.listdir(dirs), key=get_number):
        for img_path in sorted(os.listdir(dirs+contentFolder), key=get_number):
            image = read_image(dirs+contentFolder+'/'+img_path)

            # print("\n\n\n Image Name",args["image"])
            processed_image = preprocess_image(image)

            # print("Dimensions",processed_image.ndim)
            bar = cluster_colors(processed_image, args["clusters"], contentFolder+'/'+img_path, args["csv"])
        print(contentFolder)
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    # ax1.imshow(image)
    # ax1.axis("off")
    # ax2.imshow(bar)
    # ax2.axis("off")
    
    # plt.show()

# python .\color_kmeans.py -i images\601_3_50x50\0001.png -c 1 -f add.csv
# python .\color_kmeans.py -d OutImgs\ -c 1 -f add.csv
# python .\color_kmeans.py -d OutImgs\ -c 1 -f add.csv

# python -W ignore .\color_kmeans.py -d OutImgs\video_lq\ -c 1 -f add.csv