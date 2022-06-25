def basic_convolution(image, kernel, verbose=False):
    output = np.zeros_like(image)
    
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    
    kernelY, kernelX = kernel.shape
    y, x = image.shape
    
    if(kernelY%2 == 0):
        midY = kernelY/2
    else:
        midY = (kernelY+1)/2
        
    if(kernelX%2 == 0):
        midX = kernelX/2
    else:
        midX = (kernelX+1)/2
        
    for image_row in range(0, y-1):
        for pixel in range(0, x-1):
            for kernel_row in range(0, kernelY): 
                padY = int(image_row - midY + kernel_row)
                if(padY<0 or padY>=y):
                    pass
                for element in range(0, kernelX):
                    padX = int(pixel - midX + element)
                    if(padX<0 or padX >= x):
                        pass
                    else:
                        output[image_row][pixel] += kernel[kernel_row][element]*image[padY][padX]
                    
                    
    return output