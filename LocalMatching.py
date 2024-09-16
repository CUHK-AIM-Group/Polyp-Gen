from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.cluster import DBSCAN
from Matcher import Dinov2Matcher

patch_size = 14


def plot_matching_figure(image1, image2, xyA_list, xyB_list, save_path):
  fig = plt.figure(figsize=(11,5))
  ax1 = fig.add_subplot(121)
  ax2 = fig.add_subplot(122)

  ax1.imshow(image1)
  ax2.imshow(image2)

  for xyA, xyB in zip(xyA_list, xyB_list):
    con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                          axesA=ax2, axesB=ax1, color="green")
    ax2.add_artist(con)

  fig.tight_layout()
  ax1.axis('off')
  ax2.axis('off')
  plt.subplots_adjust(wspace=0.05)
  fig.show()
  fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=400)



def MaskProposer(origin_image, origin_mask, target_image, target_mask, matching_figure_save_path=None):
# Init Dinov2Matcher
  dm = Dinov2Matcher(half_precision=False)  
  # Extract image1 features
  image1 = cv2.cvtColor(cv2.imread(origin_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  mask1 = cv2.imread(origin_mask, cv2.IMREAD_COLOR)[:,:,0] > 127
  image_tensor1, grid_size1, resize_scale1 = dm.prepare_image(image1)
  features1 = dm.extract_local_features(image_tensor1)
  print(features1.shape)
  # Extract image2 features
  image2 = cv2.cvtColor(cv2.imread(target_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  image_tensor2, grid_size2, resize_scale2 = dm.prepare_image(image2)
  features2 = dm.extract_local_features(image_tensor2)

  # Build knn using features from image1, and query all features from image2
  knn = NearestNeighbors(n_neighbors=1)
  knn.fit(features1)
  distances, match2to1 = knn.kneighbors(features2)
  match2to1 = np.array(match2to1)

  xyA_list = []
  xyB_list = []
  distances_list = []

  for idx2, (dist, idx1) in enumerate(zip(distances, match2to1)):
    row, col = dm.idx_to_source_position(idx1, grid_size1, resize_scale1)
    xyA = (col, row)
    if not mask1[int(row), int(col)]: continue # skip if feature is not on the object
    row, col = dm.idx_to_source_position(idx2, grid_size2, resize_scale2)
    xyB = (col, row)
    xyB_list.append(xyB)
    xyA_list.append(xyA)
    distances_list.append(dist[0])

  #Filter by distance
  if len(xyA_list) > 30:
    zip_list = list(zip(distances_list, xyA_list, xyB_list))
    zip_list.sort(key=lambda x: x[0])
    distances_list, xyA_list, xyB_list = zip(*zip_list)
    xyA_list = xyA_list[:30]
    xyB_list = xyB_list[:30]


  if matching_figure_save_path is not None:
    plot_matching_figure(image1, image2, xyA_list, xyB_list, matching_figure_save_path)

  # DBSCAN clustering 
  X = np.array(xyB_list)
  clustering = DBSCAN(eps=2*patch_size+1 , min_samples=1).fit(X)
  labels = clustering.labels_

  # find the cluster with the most number of points
  unique_labels, counts = np.unique(labels, return_counts=True)
  max_label = unique_labels[np.argmax(counts)]
  new_list = [xyB for i, xyB in enumerate(xyB_list) if labels[i] == max_label]

  #find the min-col and max-col of the cluster
  min_col = np.min([xy[0] for xy in new_list]) - patch_size//2
  max_col = np.max([xy[0] for xy in new_list]) + patch_size//2
  #find the min-row and max-row of the cluster
  min_row = np.min([xy[1] for xy in new_list]) - patch_size//2
  max_row = np.max([xy[1] for xy in new_list]) + patch_size//2

  mask = np.zeros((image2.shape[0], image2.shape[1]))
  mask[int(min_row):int(max_row), int(min_col):int(max_col)] = 255
  mask = mask.astype(np.uint8)
  mask = Image.fromarray(mask).convert('L')
  mask.save(target_mask)
  return mask


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser(description='LocalMatching')
  parser.add_argument('--ref_image', type=str, required=True, help='Path of Reference image')
  parser.add_argument('--ref_mask', type=str, required=True, help='Path of Reference mask')
  parser.add_argument('--query_image', type=str, required=True, help='Path of Query image')
  parser.add_argument('--mask_proposal', type=str, required=True, help='Save Path of Mask proposal')
  parser.add_argument('--save_fig', type=str, default=None, help='Save the Matching Figure')
  
  args = parser.parse_args()

  mask = MaskProposer(origin_image=args.ref_image,
                      origin_mask=args.ref_mask, 
                      target_image=args.query_image, 
                      target_mask=args.mask_proposal,
                      matching_figure_save_path=args.save_fig
  )