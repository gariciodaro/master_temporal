import cv2


def calculate_histogram(imagePath1,imagePath2):
    image1 = cv2.imread(imagePath1)
    image2 = cv2.imread(imagePath2)
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [8, 8, 8],
        [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

    return d
