from PIL import Image
# Process each images and convert to 1280 x 960 

# input image filenames
# img_array = ["Ablation_ANN.png", "budget_hist.png", "feature_importance_updated.png", 
#              "feature_importance.png", "rating_hist.png", "ann_nerual.jpg", "ann_nerual.jpg", "ann_weight.jpg"] 
img_array = ["hot-encoding.png", "ann_weight.png"]
new_size = (1280, 960)  # width, height

# Process each image
for filename in img_array:
    try:
        ext = filename.lower().rsplit('.', 1)[-1]
        
        if ext == 'png':
            img = Image.open(filename).convert("RGBA")  # Keep alpha channel
            output_name = filename.rsplit('.', 1)[0] + "_adj.png"
            img_resized = img.resize(new_size)
            img_resized.save(output_name)
            print(f"Processed PNG: {output_name}")

        elif ext == 'jpg' or ext == 'jpeg':
            img = Image.open(filename).convert("RGB")  # Remove alpha if any
            output_name = filename.rsplit('.', 1)[0] + "_adj.jpg"
            img_resized = img.resize(new_size)
            img_resized.save(output_name)
            print(f"Processed JPG: {output_name}")

        else:
            print(f"Unsupported file format: {filename}")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")