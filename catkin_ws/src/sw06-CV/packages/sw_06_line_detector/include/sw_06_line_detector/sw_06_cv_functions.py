import cv2
import numpy as np
import rospy

## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
def inRange(hsv_image, low_range, high_range):

	return cv2.inRange(hsv_image, low_range, high_range)

	'''
    # hsv_image is a HSV HxWx3 image, low_range and high_range
    # are 3-dim vectors for each of the channels

    # Start the mask off as an empty array
    mask = np.zeros(dtype = np.uint8, shape=hsv_image.shape[0:2])
    #print(mask)

    # For each pixel, check if it's in the correct range
    for i, row in enumerate(hsv_image):
        for j, pixel in enumerate(row):
            if (
                    pixel[0] >= low_range[0] and pixel[0] <= high_range[0]
                and pixel[1] >= low_range[1] and pixel[1] <= high_range[1]
                and pixel[2] >= low_range[2] and pixel[2] <= high_range[2]
            ):
                mask[i, j] = 255

    #assert np.array_equal(cv2.inRange(hsv_image, low_range, high_range), mask)
    return mask
	'''

def bitwise_or(bitwise1, bitwise2):

	return cv2.bitwise_or(bitwise1, bitwise2)

	'''
	# Note: there is also a numpy function to do exactly this
    # np.bitwise_or(bitwise1, bitwise2)
    btw_or = np.zeros(dtype = np.uint8, shape=bitwise1.shape)
    
    # Do a logical or pixel-by-pixel
    for i, row in enumerate(bitwise2):
        for j, _ in enumerate(row):
            if bitwise1[i, j] or bitwise2[i, j]:
                btw_or[i, j] = 1
                
    #assert np.array_equal(btw_or, cv2.bitwise_or(bitwise1, bitwise2))
    return btw_or
	'''
	
def bitwise_and(bitwise1, bitwise2):

	return cv2.bitwise_and(bitwise1, bitwise2)

	'''
	# Note: there is also a numpy function to do exactly this
    # np.bitwise_and(bitwise1, bitwise2)
    btw_and = np.zeros(dtype = np.uint8, shape=bitwise1.shape)
    
    # Do a logical and pixel-by-pixel
    for i, row in enumerate(bitwise2):
        for j, _ in enumerate(row):
            if bitwise1[i, j] and bitwise2[i, j]:
                btw_and[i, j] = 1

    #assert np.array_equal(btw_and, cv2.bitwise_and(bitwise1, bitwise2))
    return btw_and
	'''

def getStructuringElement(shape, size):
	# Note: if it's even dimension,
    # just have to chop off right/bottom accordingly
    # ^integer division handles this automatically^

	return cv2.getStructuringElement(shape, size)
    
	'''
    output_array = np.zeros(dtype = np.uint8, shape=(size[1], size[0]))
    
    # Morph Rect
    if shape == cv2.MORPH_RECT:
        output_array = np.ones(dtype = np.uint8, shape=(size[1], size[0]))
    
    # Morph Cross
    elif shape == cv2.MORPH_CROSS:
        mid_x = size[0] // 2
        mid_y = size[1] // 2
        for i, col in enumerate(output_array):
            for j, _ in enumerate(col):
                if i == mid_y or j == mid_x:
                    output_array[i, j] = 1
    
    # Morph Ellipse
    elif shape == cv2.MORPH_ELLIPSE:
        mid_x = size[0] // 2
        mid_y = size[1] // 2        
        for y, row in enumerate(output_array):
            for x, _ in enumerate(row):
                # If the point is in/on the ellipse, add it
                rel_x = x - mid_x
                rel_y = y - mid_y

                if rel_x * rel_x / (mid_x * mid_x) + rel_y * rel_y / (mid_y * mid_y) <= 1:
                    output_array[y, x] = 1
    
    # Shape not found
    else:
        raise ValueError("""I don't know how to make the shape called {}.
                            Please check you have spelled it correctly""".format(shape))
    #assert np.equal_array(output_array, cv2.getStructuringElement(shape, size))
    return output_array
	'''

def dilate(bitwise, kernel):

	return cv2.dilate(bitwise, kernel)

	'''
    # In order to properly perform dilation, it is convenient
    # to first pad the image so we can easily slide
    # the kernel around
    def pad(bitwise, kernel):
    
        # Extract the shape and centre of the kernel
        h, w = kernel.shape
        cw, ch = w//2, h//2
    
        # Based on kernel, find how much padding we need
        left_pad = cw
        top_pad = ch
        right_pad = w - cw - 1
        bot_pad = h - ch - 1

        # Pad the image. We also return the size of the padding
        # so it's easier to remove later
        padding = ((top_pad, bot_pad), (left_pad, right_pad))
        return np.pad(bitwise, padding, 'constant'), padding
    
    # Get the kernel dimensions, and check if they are even
    k_height, k_width = kernel.shape
    even_width = 1 - k_width % 2
    even_height = 1 - k_height % 2
    
    # Pad the image and extract how much padding we used on each side
    padded, padding = pad(bitwise, kernel)
    ((top_pad, bot_pad), (left_pad, right_pad)) = padding
    p_height, p_width = padded.shape
    
    # Create an empty image which we'll use to store the result
    empty_padded = np.zeros(shape=padded.shape, dtype=np.uint8)
    
    # Loop through the pixels, masking the kernel and image at each step
    for i, row in enumerate(padded):
        for j, _ in enumerate(row):
            
            # We stop looping when the 'centre' of the kernel would be off
            # the original image (otherwise we would go out of bounds in
            # the bitwise_and calculation)
            if i < p_height - bot_pad - even_height and j < p_width - right_pad - even_width:
                
                # If the bitwise and is nonzero, we want to store that as a 1
                result = bitwise_and(kernel, padded[i : k_height + i, j : k_width + j]).any()
                
                # Record in the correct pixel
                empty_padded[i + top_pad, j + left_pad] = result * 255

	# Remove the padding from the dilated image
	padding_removed = empty_padded[top_pad : p_height - bot_pad, left_pad : p_width - right_pad]

	# assert(padding_removed, cv2.dilate(bitwise, kernel)))
    return padding_removed
	'''

## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)