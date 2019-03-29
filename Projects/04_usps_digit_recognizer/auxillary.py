import numpy as np

def isolate_image(x):

    # reshape into square matrix
    x = x.reshape([28,28])

    # find the non-zeros
    nz = x.nonzero()

    # get the row and column indicies that contain the image
    i,j = nz[0].min(), nz[0].max()
    m,n = nz[1].min(), nz[1].max()

    # make sure the length and width are even so they can be split into equal halves
    height = j - i
    width = n - m
    if height % 2 == 1: j += 1
    if width % 2 == 1: m += 1

    return x[ i:j , m:n ]

def vertical_symmetry(x):

    # grab the isolated image
    x = isolate_image(x)

    # get the half width of image in pixels
    l = x.shape[1]//2

    # get left and right halves
    xleft = x[ : , :l ]
    xright = x[:,l:]

    # invert right half
    xright = xright[:,::-1]

    # multiply left and inverted right elementwise to get a measure of symmetry
    S = xleft * xright

    # output
    return S.sum()

def horizonal_symmetry(x):

    # grab the isolated image
    x = isolate_image(x)

    # get the half length of image in pixels
    l = x.shape[0]//2

    # top and bottom halves
    xtop = x[:l,:]
    xbot = x[l:,:]

    # invert bottom half
    xbot = xbot[::-1,:]

    # multiply top and inverted bottom elementwise to get a measure of symmetry
    S = xtop * xbot

    # output
    return S.sum()

def how_many_empty_pixels(x):
    return x.shape[0] - x.nonzero()[0].shape[0]

def sum_quadrant1(x):

    # grab the isolated image
    x = isolate_image(x)

    # find bifurcating indicies
    i,j = x.shape[0]//2,x.shape[1]//2

    # return the sum of the first quadrant
    return x[:i,j:].sum()

def sum_quadrant2(x):

    # grab the isolated image
    x = isolate_image(x)

    # find bifurcating indicies
    i,j = x.shape[0]//2,x.shape[1]//2

    # return sum of second quadrant
    return x[:i,:j].sum()

def sum_quadrant3(x):

    # grab the isolated image
    x = isolate_image(x)

    # find the bifurcating indicies
    i,j = x.shape[0]//2,x.shape[1]//2

    # return the sum of third quadrant
    return x[i:,:j].sum()

def sum_quadrant4(x):

    # grab the isolated image
    x = isolate_image(x)

    # find the bifurcating indicies
    i,j = x.shape[0]//2,x.shape[1]//2

    # return the sum of the fourth quadrant
    return x[i:,j:].sum()
