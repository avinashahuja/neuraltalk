import numpy as np
import cv2
import nltk
import os
import json


def create_region_img(region_info, orig_image):
    """
    :param region_info: A matrix of MX4 where M denotes the number of regions inside the image for which one needs to extract regions.
    :param orig_image: Orignial Image
    :return: A list of images which has to be subsequently given a unique id each and stored in a folder for further processing by neural network.
    """
    # Assuming that the first coordinate is x , 2nd y 3 rd width (along x-axis) and 4th height along y-axis
    imgs = []
    for i  in range(region_info.shape[0]):
        imgs+=[orig_image[region_info[i,0]:region_info[i,2], region_info[i,1]:region_info[i,3]]]
    return imgs

def main(folder_imgs, region_vgenome, NumImgs, output_folder, output_json):
    """
    :param folder_name: Name of the folder where all the image files and image metadata is stored.
    :param region_vgenome : Region JSON file
    :param NumImgs : Number of Images from which region descriptions need to be proposed--Size of Dataset
    :param output_folder: Destination folder where the images will be stored--assuming that it already exists
    :param output_json : Output JSON File containing the data for training the neural network
    :return:  Create another directory which contains the image files having the region data and also retirn a JSON file for processing in the format that is consodered by neuraltalk
    """
    with open(region_vgenome) as data_file:
         region_data = json.load(data_file)   # May need to use IJSON as the file size is huge.
    i=0
    feature_size = 224
    output_dict = {}
    output_dict["dataset"] = "viz_genome"
    output_dict["images"] = []
    train_test_split = 0.7
    split_val = "train"
    while (i<NumImgs):
         image_attr = region_data[i]
         img = cv2.imread('%s/%s.jpg'%(folder_imgs,image_attr["image_id"]))
         img_regions_info, img_regions_id, img_regions_sentence = zip(*[([region["x"] region["y"] region["width"] region["height"]],
                        region["region_id"], region["phrase"])  for region in image_attr["regions"] if (region["width"] > 0.5*feature_size and region["height"]>0.5*feature_size) ])
                              #Filtering based on the features sizeof VGG-16
         if(len(img_regions_info)==0):
             continue
         new_images = create_region_img(np.asarray(img_regions_info),img)
         for im in range(len(new_images)):
             filename = image_attr["image_id"]+"_%s"%(img_regions_id)+".jpg"
             with open('all_imgs.txt', 'a') as fl:
                  fl.write(filename)
             cv2.imwrite(os.path.join(output_folder,filename), new_images[im])
             tokens = nltk.word_tokenize(img_regions_sentence[im])
             text_tokens = nltk.Text(tokens)
             words = [w.lower() for w in text_tokens if w.isalpha()]
             tmp ={}
             tmp["sentids"] = [0]  # Since we have a single caption per image region
             tmp["imgid"] =i
             tmp["sentences"] = [{"tokens":words, "raw":img_regions_sentence[im], "imgid":i, "sentid":0}]
             if (i>0.7*NumImgs):
                split_val = "test"
             tmp["split"] = split_val
             tmp["filename"] = filename
             i+=1
             output_dict["images"]+=[tmp]
    with open(output_json, 'w') as outfile:
         json.dump(output_dict, outfile)
if(__name__=='__main__'):
    main('VG_100K','region_descriptions.json', 120000, 'viz_genome', 'dataset_viz_genome.txt')












