import torch
import faiss
import numpy as np
import time
import cv2
import argparse
import os
from Matcher import Dinov2Matcher

# #Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #Convert to float32 numpy
    vector = np.float32(embedding)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)
 
 
#Create Faiss index using FlatL2 type with 768 dimensions (DINOv2-base) as this

def build_database(base_path, database_save_path):
    
    index = faiss.IndexFlatL2(768)
    # Extract features
    t0 = time.time()
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            # 检查文件是否是图像类型
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                image_paths.append(file_path)        

    for image_path in image_paths:
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            image_tensor, _, _ = dm.prepare_image(image)
            global_features = dm.extract_global_features(image_tensor)
        add_vector_to_index(global_features, index)

    # Store the index locally
    print('Extraction done in :', time.time()-t0)
    faiss.write_index(index, database_save_path)


######### Retrieve the image to search
def retrieval_from_database(image_path, database_path):
    #Extract the features
    with torch.no_grad():
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image_tensor, _, _ = dm.prepare_image(image)
        global_features = dm.extract_global_features(image_tensor)        
        
    vector = np.float32(global_features)
    faiss.normalize_L2(vector)
    
    #Read the index file and perform search of top-3 images
    index = faiss.read_index(database_path)
    d,i = index.search(vector, 5)
    print('distances:', d, 'indexes:', i)

if __name__=='__main__':
    dm = Dinov2Matcher(half_precision=False)
    
    parser = argparse.ArgumentParser(description='Global_trerieval')
    parser.add_argument('--database_path', type=str, required=True, help='Path to save the database')
    parser.add_argument('--data_path', type=str, required=True, help='Path of non-polyp reference images')
    parser.add_argument('--image_path', type=str, required=True, help='Path of Query image')
    args = parser.parse_args()

    build_database(args.data_path, args.database_path)
    retrieval_from_database(args.image_path, args.database_path)