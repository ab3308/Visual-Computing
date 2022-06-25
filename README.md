# Visual-Computing
A collection of small implementations of visual computing principles and techniques.

## Basic Convolution
basic_convolution.py shows my implementation of the following pseudocode:

    for each image row in input image:

        for each pixel in image row:

            set accumulator to zero

            for each kernel row in kernel:

                for each element in kernel row:

                    if element position <corresponding to> pixel position then
                        multiply element value <corresponding to> pixel value
                        add result to accumulator

                    endif      

            set output image pixel to accumulator
        
This is used to take an input grayscale image (2D matrix) and filtering kernel (2D matrix), 
returning the convolved image result as a greyscale image with the same size and datatype as the input image.

## Convolution theorem
convolution_theorem.py shows my application of the convolution theorem, used to speed up the above convolution. This is carried out using NumPy's functions for 2D Fast Fourier Transform and its inverse. This is essentially where convolution becomes multiplication in the fourier domain, which is significantly more efficient.

## Image warping
image_warping.py contains many methods for various applications with regard to image warping.

### transform_pixel_nn()
The purpose of this method is to transform a point (x, y) using a homogenous 2D transform matrix.

### forward_mapping()

This method implements forward mapping using the transform_pixel_nn() method above.
Forward mapping is where p' = Tp, where p is the input pixel. This applies to all pixel in an input image.

### backward_mapping()

This method implements backward mapping using the same transformation.
p=T^-1 p'

### backward_mapping_bilinear()

This method is an extension of the backward mapping to sample pixel colours from the source image using bilinear interpolation, whilst also handling edge cases (fading to black on the boundary)

### undistort_point()

This method implements the following steps for computing the location (u', v') to sample from the source image for a target image location (u, v).
![image](https://github.bath.ac.uk/storage/user/4594/files/6f3b88a8-3ae2-45ab-b46d-5f37ddb6f20c)

### undistort_image()

This implements polynomial lens undistortion for a given image using the undistort_point function alongside a variation of backward mapping with bilinear interpolation.

### undistort_image_vectorised()

This is an optimised implementation of undistort_image, where no loops are used. Instead, it uses vectorisation.
It is worth noting that this code can be further optimised, the key principle was to implement a version of undistort_image without loops.
