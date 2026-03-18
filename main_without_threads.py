import gradio as gr
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import os

# Pulls my key from Hugging Face Secrets (uses my test key if not found)
api_key = os.getenv("ROBOFLOW_API_KEY", "zboym1fZcR8kXWX1nlZ1") 

CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
MODEL_ID = "deepfashion2-m-10k/2"

def analyze_inspo(files, color_count):
    all_clothing_pixels = []
    
    if files is None:
        return "No images uploaded.", None

    # Now I loop through the files and get all my pixels needed
    for file in files:
        raw_img = cv2.imread(file.name)
        if raw_img is None: 
            continue

        # IF the image is unnecessarily big, I scale down the largest dimension
        max_dim = max(raw_img.shape[0], raw_img.shape[1])
        if max_dim > 800:
            scale_factor = 800 / max_dim
            new_size = (int(raw_img.shape[1] * scale_factor), int(raw_img.shape[0] * scale_factor))
            raw_img = cv2.resize(raw_img, new_size, interpolation=cv2.INTER_AREA)

        # I added this later to scale down the image immediately after loading it if it is unnecessarily big
       
        rgb_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) # OpenCV uses BGR by default, so I convert to RGB 

        cv2.imwrite("shrunk_raw_img.jpg", raw_img) # I save the image to run it through Roboflow's Inference SDK, this is a temporary workaround

        model_res = CLIENT.infer("shrunk_raw_img.jpg", model_id=MODEL_ID) # function fron Roboflow's Inference SDK to run the model on each file
        
        for item_res in model_res['predictions']:
            
            # We need a mask for each of the images to cut out the clothing parts
            mask = np.zeros(raw_img.shape[:2], dtype=np.uint8) # ':2' gives us the height and width, as we don't need the color channels info
            # I just used uint8 because each pixel can only be 0-255, this is also pretty standard for image masks

            # Now I find the points for the polygon mask, this is giving me flashbacks to my Photoshop era
            
            mask_coords = []
            for coord in item_res['points']:
                mask_coords.append([int(coord['x']), int(coord['y'])]) # This works out with how Roboflow saves the coordinates, 
                                                                       # checked their docs for results schema
            
            # Filling the mask
            mask_array = np.array(mask_coords, dtype=np.int32)
            cv2.fillPoly(mask, [mask_array], 255) # I made the mask black so I'm coloring in white (255)
            
            # Snip snip
            masked_pix = rgb_image[mask > 0] # this is shorthand to keep things fast, called Boolean Indexing
                                             # All we are doing is Coordinate mapping but faster
        

            # Sample 1000 pixels so we don't crash the computer, this is for each identified clothing item
            if len(masked_pix) > 1000:
                indexes = np.random.choice(len(masked_pix), 1000, replace=False)
                masked_pix = masked_pix[indexes]
            
            all_clothing_pixels.extend(masked_pix) # This is where I collect all the pixels from all the clothing items across all the images

        if os.path.exists("shrunk_raw_img.jpg"):
            os.remove("shrunk_raw_img.jpg")
        
    if len(all_clothing_pixels) == 0:
        return "No clothes found.", None # Just in case the model finds nothing, obv not good unless the picture is really really bad


    # NOW I RUN K MEANS!!

    # K-Means needs the data to be 'float32' (decimals)
    pixel_data_float = np.float32(all_clothing_pixels)

    # We need to tell the computer when to stop calculating.

    # Stop if we reach 10 iterations OR if the accuracy improves by less than 1.0
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


    # How many times should the algorithm start over with random spots? Need to play around with this
    # 10 attempts helps find the 'best' possible colors.
    num_attempts = 10

    # How should it choose the very first random colors?
    initial_choice_method = cv2.KMEANS_RANDOM_CENTERS

    # Run the clustering algorithm
    # compactness: (The '_') A score of how 'tight' the color groups are (we don't need this)
    # result_labels: Which pixel belongs to which color group (0, 1, 2...)
    # color_centers: The actual RGB values of our top colors
    kmeans_res = cv2.kmeans(
        pixel_data_float, 
        int(color_count), 
        None, 
        stop_criteria, 
        num_attempts, 
        initial_choice_method
    )

    result_labels = kmeans_res[1]
    color_centers = kmeans_res[2]


    # NOW TO COMPILE THE RESULTS
    
    # I count how many pixels are in each color group
    # labels.flatten() turns the data into one long list of IDs
    pix_counts = np.bincount(result_labels.flatten())
    total_samples = len(result_labels)
    
    # Convert colors back to normal whole numbers (0-255)
    final_rgb_colors = color_centers.astype(int)
    
    report_text = "COLOR PALETTE RESULTS\n"
    visual_bar = np.zeros((100, 500, 3), dtype=np.uint8)
    
    current_position = 0
    for i in range(len(final_rgb_colors)):
        rgb = final_rgb_colors[i]
        
        # Calculate percentage of this color in the whole set
        percent = (pix_counts[i] / total_samples)
        percentage_text = percent * 100
        
        hexcode = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        report_text += f"Color {i+1}: {hexcode} ({percentage_text:.1f}%)\n"
        
        # Calculate stripe width (Percentage of the 500-pixel wide bar)
        stripe_width = int(percent * 500)
        end_position = current_position + stripe_width
        
        # Draw the color stripe
        visual_bar[:, current_position:end_position] = rgb
        current_position = end_position

    return report_text, visual_bar

# MY Gradio "FRONT-END"

with gr.Blocks() as demo:
    gr.Markdown("Pre-Internship Demo: Color Palette Finder")
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="multiple", label="Upload Your Photos")
            count = gr.Slider(2, 10, value=5, label="Colors to Identify")
            btn = gr.Button("Analyze My Pics for Main Colors", variant="primary")
        with gr.Column():
            text = gr.Textbox(label="Analysis")
            img = gr.Image(label="Your Color Palette")

    btn.click(fn=analyze_inspo, inputs=[files, count], outputs=[text, img])

demo.launch()