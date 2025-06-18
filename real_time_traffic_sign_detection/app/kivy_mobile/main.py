from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2

class TrafficApp(App):
    def build(self):
        self.img = Image()
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        self.capture = cv2.VideoCapture(0)
        return self.img

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tostring()
            self.img.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

TrafficApp().run()
