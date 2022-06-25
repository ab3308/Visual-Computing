def fft_convolution(image, kernel):
    
    kernel_x, kernel_y = kernel.shape[0], kernel.shape[1]
    image=np.pad(image, ((kernel_x, kernel_x),(kernel_y, kernel_y)), 'edge')
    
    padded_k = np.zeros_like(image)
    padding_kx, padding_ky  = int((padded_k.shape[0] - kernel_x)/2), int((padded_k.shape[1] - kernel_y)/2)
    if(kernel_x%2!=0):
        padding_kx+=1
    if(kernel_y%2!=0):
        padding_ky+=1
    padded_k[padding_kx:padding_kx + kernel.shape[0], padding_ky:padding_ky+kernel.shape[1]] = kernel
    padded_k = np.fft.fftshift(padded_k)
    
    output = np.fft.ifft2(np.fft.fft2(image)*np.fft.fft2(padded_k))

    output = output[kernel_x:-kernel_x, kernel_y:-kernel_y]

    return np.real(output)