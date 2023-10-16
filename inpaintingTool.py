import cv2
import tensorflow.keras as keras
import numpy as np
import sys

loc=sys.argv[1]
task=sys.argv[2] #inpaint or sures


img=cv2.imread(loc)

#super resolution

def super_res(image):
    if image.shape!=(64,64,3):
        image=cv2.resize(image,(64,64))
    img=image/127.5 -1
    img=img.resize(1,img.shape[0],img.shape[1],img.shape[2])

    #Load Super res Gen model
    json_file=open("suRe/generator.json","r")
    loaded_model_json=json_file.read()
    json_file.close()
    superres_genModel=keras.models.model_from_json(loaded_model_json)
    superres_genModel.load_weights("suRe/generator_weights.hdf5")

    predimg= superres_genModel.predict(img)
    predimg=predimg.reshape(predimg.shape[1:])

    #convert 0-255
    predimg=(predimg+1)*127.5
    predimg=predimg.astype('uint8')

    cv2.imwrite("originalImage.jpg",image)
    cv2.imwrite("predimgsuperres.jpg",predimg)

def imageInpainting(image):

    #Load gen file
    json_file=open("Impainting/generator.json","r")
    loaded_model_json=json_file.read()
    json_file.close()
    #Load Weight
    inpainting_genmodel=keras.models.model_from_json(loaded_model_json)
    inpainting_genmodel.load_weights("Impainting/generator_weights.hdf5")

    def mask_randomly(imgs):
        img_rows=32
        img_cols=32
        mask_height=8
        mask_width=8

        y1=np.random.randint(0,img_rows-mask_height,img.shape[0])
        y2=y1+mask_height
        x1=np.random.randint(0,img_rows-mask_width,img.shape[0])
        x2=x1+mask_width

        masked_imgs=np.empty_like(imgs)
        missing_parts=np.empty((img.shape[0],mask_height,mask_width,3))

        masked_img=imgs[0].copy()
        _y1,_y2,_x1,_x2=y1[0], y2[0], x1[0], x2[0]
        missing_parts[0]=masked_img[_y1:_y2,_x1:_x2,:].copy()
        masked_img[_y1:_y2,_x1:_x2]=0 
        masked_imgs[0]=masked_img

        return(masked_imgs,missing_parts,(y1,y2,x1,x2))
    if image.shape!=(32,32,3):
        image=cv2.resize(image,(32,32))
        img2=image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        img2=img2/127.5 -1
        
        masked_imgs, missing_parts, (y1,y2,x1,x2)=mask_randomly(img2)

        predimg=inpainting_genmodel.predict(masked_imgs)
        print(predimg.shape)
        predimgreshape=predimg.reshape(predimg.shape[1],predimg.shape[2],predimg.shape[3])
        newpred=(predimgreshape+1)*127.5
        newpreduint=newpred.astype('uint8')

        masked_imgsreshape=masked_imgs.reshape(masked_imgs.shape[1],masked_imgs.shape[2],masked_imgs.shape[3])
        maskednewpred=(masked_imgsreshape+1)*127.5
        maskednewpreduint=maskednewpred.astype('uint8')

        cv2.imwrite("maskedimage.jpg",maskednewpreduint)

        maskednewpreduint[y1[0]:y2[0],x1[0]:x2[0],:]=newpreduint

        cv2.imwrite("predictedImageinpainting.jpg",maskednewpreduint)


if task=="inpaint":
    print("Performing inpainting on image...")
    imageInpainting(img)
elif task=="sures":
    print("Performing Superresolution")
    super_res(img)
else:
    print("Enter the correct task, 'inpaint' or 'sures'. ")






