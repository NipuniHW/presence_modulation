import qi

class Pepper:
    def __init__(self):

        self.session          = qi.Session()
        self.behavior_service = None
        self.tts_service      = None
        self.led_service      = None

    def connect(self, ip='localhost', port='36383'):
        print("Connecting to the robot...")

        try:
            self.session.connect("tcp://{0}:{1}".format(ip, port))
            print("Session Connected....!")
            return self.session
        
        except Exception as e:
            print("Could not connect to Pepper:", e)
            exit(1)

    def setup_behaviour_manager_service(self):
        self.behavior_service = self.session.service("ALBehaviorManager")
        print("Connected to ALBehaviorManager")

    def setup_text_to_speech_service(self):
        self.tts_service = self.session.service("ALTextToSpeech")
        print("Connected to ALTextToSpeech")

    def setup_led_service(self):
        self.led_service = self.session.service("ALLeds")
        print("Connected to ALLeds")

    def trigger_led_intensity(self, value):
        self.led_service.setIntensity("Face/Led/Blue/Left/225Deg/Actuator/Value", value)
        self.led_service.setIntensity("Face/Led/Blue/Left/270Deg/Actuator/Value", value)            
        self.led_service.setIntensity("Face/Led/Green/Left/225Deg/Actuator/Value", value)
        self.led_service.setIntensity("Face/Led/Green/Left/270Deg/Actuator/Value", value)
        self.led_service.setIntensity("Face/Led/Red/Left/270Deg/Actuator/Value", value)

    def trigger_movement(self, action):
        self.behavior_service.stopAllBehaviors()
        self.behavior_service.startBehavior("modulated_actions/" + str(action)) 

    def trigger_text_to_speech(self, volume, text):
        self.tts_service.setVolume(volume)
        self.tts_service.say(text)