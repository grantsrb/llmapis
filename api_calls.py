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

def get_img_url(image):
    base64_img = encode_image(image)
    img_type = "jpeg"
    if type(image)==str:
        img_type = image.split(".")[-1]
        if img_type=="jpg":
            img_type = "jpeg" #unsure if necessary
    return f"data:image/{img_type};base64,{base64_img}"

async def async_request(payload, headers):
    return requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload)

def get_headers():
    return {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai.api_key}"
    }

def make_payload(
        history=None,
        model_type="gpt-3.5-turbo",
        max_tokens=300,
        n=1,
        seed=None,
        stop_tokens=None,
        temperature=None):
    """
    text: str
        optionally argue text to accompany the image
    history: none or list
        a list of the message history
    model_type: str
        the model type for the api call
    max_tokens: int
        the maximum response length
    n: int
        the number of completions to return.
    stop_tokens: str, array, or None
        the sequences that if they occur, the model should stop and
        send its message
    temperature: None or float 0-2 inclusive
        the sampling temperature. larger values increase randomness
    """
    payload = {
      "model": model_type,
      "messages": history,
      "max_tokens": max_tokens,
    }
    if n is not None and n>1:
        payload["n"] = n
    if stop_tokens is not None:
        payload["stop"] = stop_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload

async def text_request(
        text,
        history=None,
        model_type="gpt-3.5-turbo",
        max_tokens=300,
        n=1,
        seed=None,
        stop_tokens=None,
        temperature=None): 
    """
    text: str
        optionally argue text to accompany the image
    history: none or list
        a list of the message history
    model_type: str
        the model type for the api call
    max_tokens: int
        the maximum response length
    n: int
        the number of completions to return.
    stop_tokens: str, array, or None
        the sequences that if they occur, the model should stop and
        send its message
    temperature: None or float 0-2 inclusive
        the sampling temperature. larger values increase randomness
    """
    assert n is None or n==1, "not implemented yet"
    # Get the appropriate headers
    headers = get_headers()

    # create new message from user
    new_message = {
      "role": "user",
      "content": [{ "type": "text", "text": text }],
    }

    # add to message history if message history or make new history
    if history is None: history = []
    history.append(new_message)

    payload = make_payload(
        model_type=model_type,
        history=history,
        max_tokens=max_tokens,
        n=n,
        stop_tokens=stop_tokens,
        temperature=temperature,)

    # collect response
    resp = await async_request(payload, headers)

    # add response to existing history for future calls
    history.append(resp["choices"][0]["message"])

    return resp, history

async def image_request(
        image,
        text=None,
        history=None,
        model_type="gpt-4-vision-preview",
        max_tokens=300,
        n=1,
        seed=None,
        stop_tokens=None,
        temperature=None): 
    """
    image: ndarray or str
        if string, should be a valid path to an image
    text: str
        optionally argue text to accompany the image
    history: none or list
        a list of the message history
    model_type: str
        the model type for the api call
    max_tokens: int
        the maximum response length
    n: int
        the number of completions to return.
    stop_tokens: str, array, or None
        the sequences that if they occur, the model should stop and
        send its message
    temperature: None or float 0-2 inclusive
        the sampling temperature. larger values increase randomness
    """
    # Getting the base64 string of image
    img_url_dict = { "url": get_img_url(image) }
    # Get the appropriate headers
    headers = get_headers()

    # create new message from user
    content = []
    if type(text)==str:
        content.append({ "type": "text", "text": text })
    content.append({ "type": "image_url", "image_url": img_url_dict, })
    new_message = { "role": "user", "content": content, }

    # add to message history if message history or make new history
    if history is None: history = []
    history.append(new_message)

    # make final payload
    payload = make_payload(
        model_type=model_type,
        history=history,
        max_tokens=max_tokens,
        n=n,
        stop_tokens=stop_tokens,
        temperature=temperature,)

    # collect response
    resp = await async_request(payload, headers)

    # add response to existing history for future calls
    history.append(resp["choices"][0]["message"])

    return resp, history

async def multiple_text_requests(prompts, histories=None, *args, **kwargs):
    """
    Include the sepcific fields for the api call in args and kwargs

    Args:
        prompts: list of str
        histories: None or list of lists of dicts
            the chat histories if continuing a conversation
    Returns:
        responses: list of dicts
            the raw responses
        histories: list of lists of dicts
            the updated chat histories for future calls
    """
    if histories:
        tups = await asyncio.map([
            text_request(
                text=p, history=h, *args, **kwargs
            ) for p,h in zip(prompts, histories)
        ])
    else:
        tups = await asyncio.map([
          text_request(text=p,history=h,*args,**kwargs) for p in prompts
        ])
    responses, histories = zip(tups)
    return responses, histories

async def multiple_requests(
        images=None,
        prompts=None,
        histories=None,
        *args, **kwargs):
    """
    Include the sepcific fields for the api call in args and kwargs

    Args:
        images: None or list of str or list of ndarray
        prompts: None or list of str
        histories: None or list of lists of dicts
            the chat histories if continuing a conversation
    Returns:
        responses: list of dicts
            the raw responses
        histories: list of lists of dicts
            the updated chat histories for future calls
    """
    if images is not None: n_loops = len(images)
    elif prompts is not None: n_loops = len(prompts)
    requests = []
    for i in range(n_loops):
        inpts = { }
        if prompts is not None:
            inpts["text"] = prompts[i]
        if histories is not None:
            inpts["history"] = histories[i]
        if images is not None:
            inpts["image"] = images[i]
            requests.append(image_request(**inpts, *args, **kwargs))
        else:
            requests.append(text_request(**inpts, *args, **kwargs))
    tups = await asyncio.map(requests)
    responses, histories = zip(tups)
    return responses, histories


