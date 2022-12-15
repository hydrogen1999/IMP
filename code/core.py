import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    alpha = 0.06
    threshold = 0.01
    stride = 2
    sigma = 0.1
    min_distance = 3
    sigma0 = 0.1

    print(f'alpha: {alpha}, threshold: {threshold}, stride: {stride}, sigma: {sigma}, min_distance: {min_distance}')

    #Step1: blur image (optional)
    filtered_image = filters.gaussian(image, sigma=sigma)

    # Step2: calculate gradient of image
    I_x = filters.sobel_v(filtered_image)
    I_y = filters.sobel_h(filtered_image)

    # Step3: calculate Gxx, Gxy, Gyy
    I_xx = np.square(I_x)
    I_xy = np.multiply(I_x, I_y)
    I_yy = np.square(I_y)

    I_xx = filters.gaussian(I_xx, sigma=sigma0)
    I_xy = filters.gaussian(I_xy, sigma=sigma0)
    I_yy = filters.gaussian(I_yy, sigma=sigma0)

    listC = np.zeros_like(image)

    # Step4: caculate C matrix
    for y in range(0, image.shape[0]-feature_width, stride):
        for x in range(0, image.shape[1]-feature_width, stride):
            # matrix 17x17
            Sxx = np.sum(I_xx[y:y+feature_width+1, x:x+feature_width+1])
            Syy = np.sum(I_yy[y:y+feature_width+1, x:x+feature_width+1])
            Sxy = np.sum(I_xy[y:y+feature_width+1, x:x+feature_width+1])

            detC = (Sxx * Syy) - (Sxy**2)
            traceC = Sxx + Syy
            C = detC - alpha*(traceC**2)
            
            if C > threshold:
                listC[y+feature_width//2, x+feature_width//2] = C

    # Step5: using non-maximal suppression
    ret = feature.peak_local_max(listC, min_distance=min_distance, threshold_abs=threshold)
    return ret[:, 1], ret[:, 0]


def get_features(image, x, y, feature_width):

    x = np.round(x).astype(int)
    y = np.round(y).astype(int)

    sigma_gradient_image = 0.1
    sigma_16x16 = 0.4
    threshold = 0.2

    print(f'sigma_gradient_image: {sigma_gradient_image}, sigma_16x16: {sigma_16x16}, threshold: {threshold}')

    features = np.zeros((len(x), 4, 4, 8))
    
    # step0: blur image (optional)
    filtered_image = filters.gaussian(image, sigma=sigma_gradient_image)
    
    # step1: compute the gradient of image
    d_im_x = filters.sobel_v(filtered_image)
    d_im_y = filters.sobel_h(filtered_image)

    magnitude_gradient = np.sqrt(np.add(np.square(d_im_x), np.square(d_im_y)))
    direction_gradient = np.arctan2(d_im_y, d_im_x)
    direction_gradient[direction_gradient < 0] += 2 * np.pi

    # step2:
    # image.shape[0] = 1024
    # image.shape[1] = 768
    # x (0 -> 768)
    # y (0 -> 1024)
    for n, (x_, y_) in enumerate(zip(x, y)):
        # get windows of key point(x, y)
        rows = (y_ - feature_width//2, y_ + feature_width//2 + 1)
        cols = (x_ - feature_width//2, x_ + feature_width//2 + 1)

        if rows[0] < 0:
            rows = (0, feature_width+1)
        if rows[1] > image.shape[0]:
            rows = (image.shape[0]-feature_width-1, image.shape[0]-1)

        if cols[0] < 0:
            cols = (0, feature_width+1)
        if cols[1] > image.shape[1]:
            cols = (image.shape[1]-feature_width-1, image.shape[1]-1)

        # get gradient and angle of key point
        magnitude_window = magnitude_gradient[rows[0]:rows[1], cols[0]:cols[1]]
        direction_window = direction_gradient[rows[0]:rows[1], cols[0]:cols[1]]

        # Gaussian filter on window
        magnitude_window = filters.gaussian(
            magnitude_window, sigma=sigma_16x16)
        direction_window = filters.gaussian(
            direction_window, sigma=sigma_16x16)

        for i in range(feature_width//4):
            for j in range(feature_width//4):
                current_magnitude = magnitude_window[i*feature_width//4: (
                    i+1)*feature_width//4, j*feature_width//4:(j+1)*feature_width//4]

                current_direction = direction_window[i*feature_width//4: (
                    i+1)*feature_width//4, j*feature_width//4:(j+1)*feature_width//4]

                features[n, i, j] = np.histogram(current_direction.reshape(
                    -1), bins=8, range=(0, 2*np.pi), weights=current_magnitude.reshape(-1))[0]

    # Extract 8 x 16 values into 128-dim vector
    features = features.reshape((len(x), -1,))

    # Normalize vector to [0...1]
    norm = np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)
    features = features / norm

    # Clamp all vector values > 0.2 to 0.2
    features[features >= threshold] = threshold

    # Re-normalize
    norm = np.sqrt(np.square(features).sum(axis=1)).reshape(-1, 1)
    features = features / norm

    return features


def match_features(im1_features, im2_features):
    threshold = 0.8
    print(f'Match threshold: {threshold}')

    matches = []
    confidences = []

    for i in range(im1_features.shape[0]):
        distances = np.sqrt(np.square(np.subtract(
            im1_features[i, :], im2_features)).sum(axis=1))
        index_sorted = np.argsort(distances)
        if distances[index_sorted[0]] / distances[index_sorted[1]] < threshold:
            matches.append([i, index_sorted[0]])
            confidences.append(
                1.0 - distances[index_sorted[0]]/distances[index_sorted[1]])
    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    return matches, confidences
