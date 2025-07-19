# Import kivy dependencies

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers2 import L1Dist
import os
import numpy as np

#Build App
class CamApp(App):

    def build(self):
        self.title = 'Insurance eKYC'
        self.web_cam = Image(
            size_hint=(1, 0.88),
            allow_stretch=True,      # Allows image to stretch to fit the widget
            keep_ratio=False         # Disables aspect ratio constraint
        )
        self.button = Button(text="Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        #Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model, might need to change
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        #video capture
        self.capture = cv2.VideoCapture(0)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    #Run continuously to get webcam feed by using the Clock function
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[80:80+250, 160:160+250, :]

        # Flip horizontal and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    #preprocessing - Scale and REsize
#We will be applying the preprocessor using the iterator, so we can use it on all values
    
    #Load image from file and convert to 100x100px
    def preprocess(self, file_path):
        
        #Read in image from file path
        byte_img = tf.io.read_file(file_path)#load the image, decode, resize and / 255. to rescale
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100,100))#resize images to 100x100x3
        img = img / 255.0# so that the values out of this is 0 or 1, since all our images are 255(value)
        return img

    #Verification function to verify person
    def verify(self, *args):
        detection_threshold = 0.6
        verification_threshold = 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[80:80+250, 160:160+250, :]
        cv2.imwrite(SAVE_PATH, frame)


        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        # Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        #Log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.5))
        Logger.info(np.sum(np.array(results) > 0.8))


        return results, verified

if __name__ == '__main__':
    CamApp().run()