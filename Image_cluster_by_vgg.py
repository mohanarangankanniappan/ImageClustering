############################################################################
# File : image_clustering_vgg.py
#
# Description : Feature extracted with VGG Model, Clustered using K-means
#
#
# Author : Mohanarangan
#
# Email : mohanaranganphd@gmail.com
# 
# Date : 28 / 01 / 2021
#############################################################################


import cv2
import os
import numpy as np
from keras.models import load_model, Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.image as mpimg


IN_UNCLUSTERED_IMG_DIR = "./images/unClustered/"
OUT_CLUSTERED_IMG_DIR = "./images/clustered/"
RESIZED_IMG_SIZE = (224,224) 
NO_OF_CLUSTERS = 4


def get_model(layer='fc2'):
  """Keras Model of the VGG16 network, with the output layer set to `layer`.
    The default layer is the second-to-last fully connected layer 'fc2' of
    shape (4096,).
    Parameters
    ----------
    layer : str
        which layer to extract (must be of shape (None, X)), e.g. 'fc2', 'fc1'
        or 'flatten'
    """
    # base_model.summary():
    #     ....
    #     block5_conv4 (Conv2D)        (None, 15, 15, 512)       2359808
    #     _________________________________________________________________
    #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
    #     _________________________________________________________________
    #     flatten (Flatten)            (None, 25088)             0
    #     _________________________________________________________________
    #     fc1 (Dense)                  (None, 4096)              102764544
    #     _________________________________________________________________
    #     fc2 (Dense)                  (None, 4096)              16781312
    #     _________________________________________________________________
    #     predictions (Dense)          (None, 1000)              4097000
    #
  base_model = VGG16(weights='imagenet', include_top=True)
  model = Model(inputs=base_model.input,
                outputs=base_model.get_layer(layer).output)
  return model


#print(len([file for file in os.listdir('/home/mohan/Projects/ResearchPaper/assigntet/data/')]))

def get_files(path_to_files, size):
  fn_imgs = []
  files = [file for file in os.listdir(path_to_files)]
  for file in files:
      print ("Full path trying to read ", path_to_files+"/" +file)
      img = cv2.resize(cv2.imread(path_to_files+"/" +file), size)
  #         img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
      fn_imgs.append([file, img])
  return dict(fn_imgs)

# path_to_files = '/content/drive/My Drive/Google Photos/All Pics/'
# size = (224, 224, 3)

def feature_vector(img_arr, model):
  if img_arr.shape[2] == 1:
    img_arr = img_arr.repeat(3, axis=2)

  # (1, 224, 224, 3)
  arr4d = np.expand_dims(img_arr, axis=0)  
  arr4d_pp = preprocess_input(arr4d)
  return model.predict(arr4d_pp)[0,:]

def feature_vectors(imgs_dict, model):
  f_vect = {}
  for fn, img in imgs_dict.items():
    f_vect[fn] = feature_vector(img, model)
    print("Image :" + fn + "\t " + "Feature Vector " + str(len(f_vect[fn])) + "\t" + "FV :" + str(f_vect[fn]))
  return f_vect

# path_to_files = '/content/drive/My Drive/Google Photos/All Pics/'
# size = (224, 224)





def main():

        # Gel all the image file names in the dictionary   
	imgs_dict = get_files(path_to_files = IN_UNCLUSTERED_IMG_DIR,size = RESIZED_IMG_SIZE)
	print("The total number of unclustered file ", len(imgs_dict))

	# Create Keras NN model.
	model = get_model()

	# Feed images through the model and extract feature vectors.
	img_feature_vector = feature_vectors(imgs_dict, model)

	images = list(img_feature_vector.values())
	fns = list(img_feature_vector.keys())
	sum_of_squared_distances = []
	K = range(1, NO_OF_CLUSTERS)
	for k in K:
	  km = KMeans(n_clusters=k)
	  km = km.fit(images)
	  sum_of_squared_distances.append(km.inertia_)
	plt.plot(K, sum_of_squared_distances, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Sum_of_squared_distances')
	plt.title('Elbow Method For Optimal k')
	plt.show()

	kmeans = KMeans(n_clusters=3, init='k-means++')
	kmeans.fit(images)
	y_kmeans = kmeans.predict(images)
	file_names = list(imgs_dict.keys())

	n_clusters = NO_OF_CLUSTERS -1
	cluster_path = OUT_CLUSTERED_IMG_DIR
	path_to_files = IN_UNCLUSTERED_IMG_DIR

	for c in range(0,n_clusters):
	  if not os.path.exists(cluster_path+str(c)):
	    os.mkdir(cluster_path+str(c))
	    
	for fn, cluster in zip(file_names, y_kmeans):
            image = cv2.imread(path_to_files+fn)
            print(path_to_files + fn)
            
            print(cluster_path+  str(cluster)+"/"+fn)
            cv2.imwrite(cluster_path + str(cluster)+"/"+fn, image)



if __name__=="__main__":
    main()
    
