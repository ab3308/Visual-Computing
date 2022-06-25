#Forward mapping
def transform_pixel_nn(x, y, transform):
    """Transforms a source pixel coordinate (x, y) using 'transform', and rounds to the nearest pixel
    coordinate. Returns a tuple (x', y')."""
    
    result = transform@np.array((x,y,1)).T
    
    return (round(result[0]/result[2]), round(result[1]/result[2]))
  
def forward_mapping(source, transform):
    """Warps the 'source' image by the given 'transform' using forward mapping."""
    output = np.zeros_like(source)
    rowCnt = source.shape[0]
    colCnt = source.shape[1]
    
    for row in range(0, rowCnt):
        for col in range(0, colCnt):
            newX, newY = transform_pixel_nn(col, row, transform)
            newX, newY = int(newX), int(newY)
            if(newX>=0 and newX<colCnt and newY>=0 and newY<rowCnt):
                output[newY][newX] = source[row][col]
    
    return output
 
#Backward mapping
def backward_mapping(source, transform):
    """Warps the 'source' image by the given 'transform' using backward mapping with nearest-neighbour interpolation."""
    output = np.zeros_like(source)
    invTransform = np.linalg.inv(transform)
    rowCnt = source.shape[0]
    colCnt = source.shape[1]
    
    for row in range(0, rowCnt):
        for col in range(0, colCnt):
            originX, originY = transform_pixel_nn(col, row, invTransform)
            originX, originY = int(originX), int(originY)
            if(originX>=0 and originX<colCnt and originY>=0 and originY<rowCnt):
                output[row][col] = source[originY][originX]
    
    return output
  
  
def backward_mapping_bilinear(source, transform):
    """Warps the 'source' image by the given 'transform' using backward mapping with bilinear interpolation."""
    output = np.zeros_like(source)
    invTransform = np.linalg.inv(transform)
    rowCnt = source.shape[0]
    colCnt = source.shape[1]

    for row in range(0, rowCnt):
        for col in range(0, colCnt):
            result = invTransform@np.array((col,row,1)).T
            originX, originY = result[0], result[1]

            #if border, set f1 etc to 0
            f1x, f1y = np.floor(originX), np.floor(originY)
            f1 = source[int(f1y)][int(f1x)]
            f2x, f2y = np.ceil(originX), f1y
            f2 = source[int(f2y)][int(f2x)]
            f3x, f3y = f1x, np.ceil(originY)
            f3 = source[int(f3y)][int(f3x)]
            f4x, f4y = f2x, f3y
            f4 = source[int(f4y)][int(f4x)]

            alpha = originX-f1x
            beta = originY-f1y


            if(f1x<0):
                f1, f3 = 0, 0
                if(f2x<0):
                    f2, f4 = 0, 0

            if(f1y<0):
                f1, f2 = 0, 0
                if(f3y<0):
                    f3, f4 = 0, 0
            if(f2x>colCnt):
                f2, f4 = 0, 0
                if(f1x>colCnt):
                    f2, f4 = 0, 0
            if(f3y>rowCnt):
                f3, f4 = 0, 0
                if(f1y>rowCnt):
                    f1, f2 = 0, 0

            f12 = (1-alpha)*f1 + alpha*f2
            f34 = (1-alpha)*f3 + alpha*f4

            f1234 = (1-beta)*f12 + beta*f34

            output[row][col] = f1234

    return output

#Lens undistortion
def undistort_point(u, v, camera_matrix, dist_coeffs):
    """Undistorts a pixel's coordinates (u, v) using the given camera matrix and
    distortion coefficients. Returns a tuple (u', v')."""
    fx, fy, px, py = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2] 
    k1, k2, k3 = dist_coeffs[0], dist_coeffs[1], dist_coeffs[2]
    
    x = (u-px)/fx
    y= (v-py)/fy
    rSq = x**2 + y**2
    xDash = x*(1+(k1*rSq)+(k2*(rSq*rSq))+(k3*(rSq*rSq*rSq)))
    yDash = y*(1+(k1*rSq)+(k2*(rSq*rSq))+(k3*(rSq*rSq*rSq)))
    uDash = xDash*fx + px
    vDash = yDash*fy + py
    
    return uDash, vDash
  
def undistort_image(image, camera_matrix, dist_coeffs):
    """Undistorts an image using the given camera matrix and distortion coefficients."""
    
    output = np.zeros_like(source)
    rowCnt = source.shape[0]
    colCnt = source.shape[1]
    
    for row in range(0, rowCnt):
        for col in range(0, colCnt):
            originX, originY = undistort_point(col, row, camera_matrix, dist_coeffs)
            
            f1x, f1y = np.floor(originX), np.floor(originY)
            f1 = image[int(f1y)][int(f1x)]
            
            f2x, f2y = np.ceil(originX), f1y
            f2 = image[int(f2y)][int(f2x)]
            
            f3x, f3y = f1x, np.ceil(originY)
            f3 = image[int(f3y)][int(f3x)]
            
            f4x, f4y = f2x, f3y
            f4 = image[int(f4y)][int(f4x)]
            
            alpha = originX-f1x
            beta = originY-f1y
            
            f12 = (1-alpha)*f1 + alpha*f2
            f34 = (1-alpha)*f3 + alpha*f4
            
            f1234 = (1-beta)*f12 + beta*f34
            
            output[row][col] = f1234
    
    return output
  
def undistort_image_vectorised(image, camera_matrix, dist_coeffs):
    """Undistorts an image using the given camera matrix and distortion coefficients.
    Use vectorised operations to avoid slow for loops."""
    
    rowCnt = image.shape[0]
    colCnt = image.shape[1]
    
    fx, fy, px, py = camera_matrix[0][0], camera_matrix[1][1], camera_matrix[0][2], camera_matrix[1][2] 
    k1, k2, k3 = dist_coeffs[0], dist_coeffs[1], dist_coeffs[2]
    
    coordinates = np.mgrid[0:rowCnt, 0:colCnt]
    x = (coordinates[1]-px)/fx
    y = (coordinates[0]-py)/fy
    rSq = x**2 + y**2
    xDash = x*(1+(k1*rSq)+(k2*(rSq*rSq))+(k3*(rSq*rSq*rSq)))
    yDash = y*(1+(k1*rSq)+(k2*(rSq*rSq))+(k3*(rSq*rSq*rSq)))
    originX = xDash*fx + px
    originY = yDash*fy + py

    
    f1x, f1y = np.array(list(map(np.floor, originX))), np.array(list(map(np.floor, originY)))
    f1 = image[f1y.astype(int), f1x.astype(int)]
        
    f2x, f2y = np.array(list(map(np.ceil, originX))), f1y
    f2 = image[f2y.astype(int), f2x.astype(int)]

    f3x, f3y = f1x, np.array(list(map(np.ceil, originY)))
    f3 = image[f3y.astype(int), f3x.astype(int)]
    
    f4x, f4y = f2x, f3y
    f4 = image[f4y.astype(int), f4x.astype(int)]

    alpha = originX-f1x
    nAlpha = 1-alpha
    beta = originY-f1y
    nBeta = 1-beta
    
    f12 = (f1.T*nAlpha.T).T + (alpha.T*f2.T).T
    f34 = (nAlpha.T*f3.T).T + (alpha.T*f4.T).T
    
    f1234 = (nBeta.T*f12.T).T + (beta.T*f34.T).T
    image[coordinates[0], coordinates[1]] = f1234

    return image
