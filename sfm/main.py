import os
import cv2
import numpy as np
import open3d as o3d

images_path = "images/door"
list_images = os.listdir(images_path)
nbImages = len(list_images) - 1

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

img_shape = (300, 600)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   flags=2)

goods = []
matched_images = []
keypoints = []
centroids = []
all_images = []
# traiter les images 2 par 2 ( on utilise les 12 images )
for i in range(1, nbImages, 2):
    # premiere image
    o_img1 = cv2.imread(os.path.join(images_path, list_images[i]))
    o_img1 = cv2.resize(o_img1, img_shape)
    all_images.append(o_img1)
    o_gray1 = cv2.cvtColor(o_img1, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = sift.detectAndCompute(o_gray1, None)
    keypoints.append(keypoints1)

    # 2e image
    o_img2 = cv2.imread(os.path.join(images_path, list_images[i + 1]))
    o_img2 = cv2.resize(o_img2, img_shape)
    all_images.append(o_img2)
    o_gray2 = cv2.cvtColor(o_img2, cv2.COLOR_BGR2GRAY)
    keypoints2, descriptors2 = sift.detectAndCompute(o_gray2, None)
    keypoints.append(keypoints2)

    # matches entre les 2 images
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    # affichage des matches
    img = cv2.drawMatches(o_img1, keypoints1, o_img2, keypoints2, good, None, **draw_params)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    goods.append(good)
    matched_images.append(img)

# calcule des centroid pour chaque image
for i in range(len(all_images)):
    kp = keypoints[i]
    x_coords = [k.pt[0] for k in kp]
    y_coords = [k.pt[1] for k in kp]
    centroid = (int(np.mean(x_coords)), int(np.mean(y_coords)))
    centroids.append(centroid)
    # affichage des centroid sur les images
    cv2.circle(all_images[i], centroid, 5, (0, 0, 255), 2)
    cv2.putText(all_images[i], "Centroid", centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Image', all_images[i])
    cv2.waitKey(0)

# matrice W:
X = np.zeros((nbImages, len(goods[0]), 2))

for i in range(len(goods)):
    for j, match in enumerate(goods[i]):
        if j != 393:
            X[i, j] = np.array(keypoints[i][match.queryIdx].pt) - np.array(centroids[i])
        else:
            break

X_centered = X - np.mean(X, axis=0)  # matrice centrée (changement du repere vers barycentre)
W = X_centered.reshape(-1, X_centered.shape[-1])
U, s, Vt = np.linalg.svd(W, full_matrices=True, compute_uv=True, hermitian=False)  # W = U @ np.diag(s) @ Vt

point_cloud = o3d.io.read_point_cloud('Point_Cloud/sparse.ply')
o3d.visualization.draw_geometries([point_cloud])
