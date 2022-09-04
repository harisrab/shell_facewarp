# Delaunay Triangulation based Facewarp Tool for CMD

This is a partial implementation of the paper titled "Bringing Portraits to Life" without the fine details transfer. It's been tailored to run without GUI optionality. This helps this to be encapsulted into an API and run on backend systems where GUI is not always available. 

**Paper Abstract:**
> We present a technique to automatically animate a still portrait, making it possible for the subject in the photo to come to life and express various emotions. We use a driving video (of a different subject) and develop means to transfer the expressiveness of the subject in the driving video to the target portrait. In contrast to previous work that requires an input video of the target face to reenact a facial performance, our technique uses only a single target image. We animate the target image through 2D warps that imitate the facial transformations in the driving video. As warps alone do not carry the full expressiveness of the face, we add fine-scale dynamic details which are commonly associated with facial expressions such as creases and wrinkles. Furthermore, we hallucinate regions that are hidden in the input target face, most notably in the inner mouth. Our technique gives rise to reactive profiles, where people in still images can automatically interact with their viewers. We demonstrate our technique operating on numerous still portraits from the internet.

**Setup Details:**
Clone the repo
'''
git clone https://github.com/harisrab/shell_facewarp/
'''

Navigate to the repo
'''
cd shell_facewarp
'''

Install all the relevant packages.
'''
pip install -r requirements.txt
'''

**Usage Details:**

Open main.py and make these edits.

'''
video = "<video used for transfer>.avi"
image = "<name of the image to be used>.jpg"
'''

Put your file src image into the folder "src_images"
Put your source video into the folder "src_videos"

