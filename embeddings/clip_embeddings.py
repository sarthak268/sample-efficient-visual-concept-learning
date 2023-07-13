import torch
import clip
import glob
import random
from PIL import Image

def makeLabelDict(path):
    file = open(path, 'r')
    lines = file.readlines()
    
    # nodedict = {}
    nodelist = []
    for line in lines:  
        data = line.split(':')
        index = int(data[0])
        name = str(data[1])[:-1]
        # nodedict[index] = name
        nodelist.append(name)

    return nodelist

def test_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_path = 'path to images'
    image_name = image_path + 'VG_100K/2335928.jpg'
    # print (image_name)
    # image_name = image_path + 'VG_100K/2356221.jpg'

    image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)

    node_name_list = makeLabelDict('nodename2index_corrected.txt')
    node_name_list.append('kitchen')
    
    text = clip.tokenize(node_name_list).to(device)
    # text = clip.tokenize(["kitchen", "dog", "cat", "microwave", "bedroom", "cup", "ice"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

def find_relevant(query_word, word_list, clip_threshold=1e-2, max_samples=100, device='cuda'):
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_path = 'path to images'
    # image_name = image_path + 'VG_100K/2335928.jpg'
    # image_name = image_path + 'VG_100K/2356221.jpg'
    all_images_1 = glob.glob(image_path + 'VG_100K/*jpg')
    all_images_2 = glob.glob(image_path + 'VG_100K_2/*jpg')
    # all_images = all_images_1 + all_images_2
    all_images = all_images_1
    random.shuffle(all_images, random.random)

    relevant_images = []

    for image_name in all_images:

        image_pil = Image.open(image_name)
        image = preprocess(image_pil).unsqueeze(0).to(device)
        text = clip.tokenize(word_list).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            if probs[0][-1] > clip_threshold:
                relevant_images.append(image_name)
                image_pil.save('./scenes/random/{}.jpg'.format(image_name.split('/')[-1].split('.')[0]))

                print ('adding...')
                if len(relevant_images) % 10 == 0:
                    print ('{} matches for relevant images found'.format(len(relevant_images)))
        
        if len(relevant_images) >= max_samples:
            break

    print ('Found {} relevant matches in dataset'.format(len(relevant_images)))

    return relevant_images

if (__name__ == '__main__'):
    # test_clip()
    query_word = 'cat'
    word_list = makeLabelDict('nodename2index.txt')
    find_relevant(query_word, word_list)