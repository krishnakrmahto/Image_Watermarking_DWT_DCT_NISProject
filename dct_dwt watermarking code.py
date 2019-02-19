import numpy as np
import pywt
import os
from PIL import Image
from scipy.fftpack import dct
from scipy.fftpack import idct

current_path = str(os.path.dirname(__file__))

image = 'image.png'
watermark = 'watermark2.png'

def convert_image(image_name, size):
    with open('log/function_calls.txt', 'a') as logfile:
        logfile.write('convert_image\n')

    # open image returns image object. resize(required_size_in_tuple), resampling_factor)
    img = Image.open('./pictures/' + image_name).resize((size, size), 1)
    # img.covert('L') returns grayscale image -- 'L' is for grayscale
    img = img.convert('L')
    img.save('./dataset/' + image_name)


    image_array = np.array(img.getdata(), dtype=np.float).reshape((size, size))
    print(image_array[0][0])
    print(image_array[10][10])

    return image_array

def apply_dwt(imArray, model, level):
    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('apply_dwt\n')
    dwt_coefficients=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    # print(coeffs[0].__len__())

    return dwt_coefficients



def embed_watermark(watermark_array, transformed_image):
    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('embed_watermark\n')
    watermark_array_size = watermark_array[0].__len__()
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range (0, transformed_image.__len__(), 8):
        for y in range (0, transformed_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = transformed_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                transformed_image[x:x+8, y:y+8] = subdct
                ind += 1

    return transformed_image



def apply_dct(image_array):
    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('apply_dct\n')
    size = image_array[0].__len__()
    dct_coefficients = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            dct_coefficients[i:i+8, j:j+8] = subdct

    return dct_coefficients


def inverse_dct(dct_image):
    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('inverse_dct\n')
    size = dct_image[0].__len__()
    idct_image = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(dct_image[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            idct_image[i:i+8, j:j+8] = subidct

    return idct_image


def get_watermark(dct_watermarked_coeff, watermark_size):

    # print("kya", dct_watermarked_coeff.shape)

    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('get_watermark\n')

    subwatermarks = []

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark


def recover_watermark(image_array, model='haar', level = 1):
    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('recover_watermark\n')

    coeffs_watermarked_image = apply_dwt(image_array, model, level=level)
    coeffs_watermarked_image2 = apply_dwt(coeffs_watermarked_image[0], model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image2[0])

    watermark_array = get_watermark(dct_watermarked_coeff, 64)

    watermark_array =  np.uint8(watermark_array)

#Save result
    img = Image.fromarray(watermark_array)
    img.save('./result/recovered_watermark.jpg')


def print_image_from_array(image_array, name):

    with open('./log/function_calls.txt', 'a') as logfile:
        logfile.write('print_image_from_array\n')

    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save('./result/' + name)



def watermarker():
    with open('./log/function_calls.txt', 'w') as logfile:
        logfile.write('watermarker\n')
    wavelet_type = 'haar'
    level = 1
    image_array = convert_image(image, 2048) # return grayscale of size 2048*2048
    watermark_array = convert_image(watermark, 64) # return grayscale of size 64*64

    dwt_image = apply_dwt(image_array, wavelet_type, level=level)
    dwt_image2 = apply_dwt(dwt_image[0],wavelet_type, level=level)
    dct_image = apply_dct(dwt_image2[0]) # returns dct of entire image
    # print("bhag", len(dwt_image2[0]))
    dct_image = embed_watermark(watermark_array, dct_image)
    dwt_image2[0] = inverse_dct(dct_image)


# construct the watermarked image
    idwt_image2 = pywt.waverec2(dwt_image2, wavelet_type)
    dwt_image[0] = idwt_image2
    idwt_image = pywt.waverec2(dwt_image, wavelet_type)
    print_image_from_array(idwt_image, 'image_with_watermark.jpg')

# recover images
    # print("bhai", idwt_image.shape)
    recover_watermark(image_array = idwt_image, model=wavelet_type, level = level)


if __name__ == '__main__':
    watermarker()
