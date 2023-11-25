from PIL import Image
import io

import os
import base64
import requests
import openai

openai.organization = "org-thCr4LFe4RGs65Rdh4YKMLC8"
# First save your openai gpt key to a txt file called gptkey.txt and restrict reading privileges
# then run the following in the terminal $ export OPENAI_API_KEY=$(cat /path/to/gptkey.txt)
# Make sure you do this in the same session as which you launch this notebook. You will probably
# need to restart this jupyter session in order to set the key.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image):
    """
    Takes a uint8 ndarray or an image file path
    """
    if type(image)==type(np.zeros(1)):
        if image.dtype!="uint8":
            image = np.uint8(image*255)
        img = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode('utf-8')
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
async def async_request(payload, headers):
    return requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

async def image_request(image, text, max_tokens=300): 
    # Getting the base64 string
    base64_image = encode_image(image)
    
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai.api_key}"
    }
    
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text,
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": max_tokens,
    }
    
    return await async_request(payload, headers)