import numpy as np
from multiprocessing.connection import Connection,Listener
from multiprocessing import AuthenticationError
import threading


#mmwave_model_integrator code
from mmwave_model_integrator.input_encoders._radar_range_az_encoder import _RadarRangeAzEncoder
from mmwave_model_integrator.decoders._lidar_pc_polar_decoder import _lidarPCPolarDecoder
from mmwave_model_integrator.model_runner._model_runner import _ModelRunner

class realTimeRngAzToPCRunner:

    def __init__(
            self,
            input_encoder:_RadarRangeAzEncoder,
            model_runner:_ModelRunner,
            prediction_decoder:_lidarPCPolarDecoder,
            adc_cube_addr=0,
            input_encoding_addr=0,
            predicted_pc_addr=0

    ):
        
        #initialize model attributes
        self.input_encoder:_RadarRangeAzEncoder = input_encoder
        self.model_runner:_ModelRunner = model_runner
        self.prediction_decoder:_lidarPCPolarDecoder = prediction_decoder

        #initializing listerner objects
        self.adc_cube_addr = adc_cube_addr
        self.input_encoding_addr = input_encoding_addr
        self.predicted_pc_addr=predicted_pc_addr

        #connection status
        self.adc_cube_conn_enabled = False
        self.input_encodding_conn_enabled = False
        self.predicted_pc_conn_enabled = False

        #connection objects
        self.adc_cube_conn:Connection = None
        self.input_encodding_conn:Connection = None
        self.predicted_pc_conn:Connection = None

        self._init_listeners()

    ####################################################################
    #Initializing listener objects
    ####################################################################
    def _init_listeners(self)->bool:
        
        try:

            threads = []

            if self.adc_cube_addr > 0:
                print("connect adc cube client")
                t = threading.Thread(
                    target=self._init_adc_cube_listener
                )
                threads.append(t)
                t.start()
            if self.input_encoding_addr > 0:
                print("connect input encoding client")
                t = threading.Thread(
                    target=self._init_input_encoding_listener
                )
                threads.append(t)
                t.start()
            if self.predicted_pc_addr > 0:
                print("connect predicted point cloud client")
                t = threading.Thread(
                    target=self._init_predicted_pc_listener
                )
                threads.append(t)
                t.start()
            
            #wait to join all of the other threads
            for t in threads:
                t.join()
            
            print("Listener connection status: ADC ({}),Input ({}), PC ({})".format(
                self.adc_cube_conn_enabled,
                self.input_encodding_conn_enabled,
                self.predicted_pc_conn_enabled
            ))
            
        except AuthenticationError:
            print("experienced authentication error when attempting to connect to listeners")
            return False
        
    def _init_adc_cube_listener(self):

        # get the authentication string
        authkey_str = "DCA1000_client"
        authkey = authkey_str.encode()

        addr = ("localhost", self.adc_cube_addr)
        listener = Listener(addr, authkey=authkey)
        self.adc_cube_conn = listener.accept()
        self.adc_cube_conn_enabled = True

        return
    
    def _init_input_encoding_listener(self):

        # get the authentication string
        authkey_str = "DCA1000_client"
        authkey = authkey_str.encode()

        addr = ("localhost", self.input_encoding_addr)
        listener = Listener(addr, authkey=authkey)
        self.input_encodding_conn = listener.accept()
        self.input_encodding_conn_enabled = True

        return
    
    def _init_predicted_pc_listener(self):

        # get the authentication string
        authkey_str = "DCA1000_client"
        authkey = authkey_str.encode()

        addr = ("localhost", self.predicted_pc_addr)
        listener = Listener(addr, authkey=authkey)
        self.predicted_pc_conn = listener.accept()
        self.predicted_pc_conn_enabled = True

        return
    
    ####################################################################
    #Run loop
    ####################################################################
    def run(self):

        if self.adc_cube_conn_enabled:
            while True:
                
                #wait to receive a valid adc cube
                try:
                    adc_cube:np.ndarray = self.adc_cube_conn.recv()
                    print("ADC cube: {}".format(adc_cube.shape))
                except EOFError:
                    print("adc cube connection closed")
                    break

                encoded_data = self.input_encoder.encode(adc_cube)
                if self.input_encoder.full_encoding_ready:

                    if self.input_encodding_conn_enabled:
                        try:
                            self.input_encodding_conn.send(encoded_data)
                        except ConnectionResetError:
                            print("input encoder connection reset")
                            break

                    pred = self.model_runner.make_prediction(encoded_data)
                    pc_cart = self.prediction_decoder.convert_polar_to_cartesian(
                        self.prediction_decoder.decode(pred)
                    )

                    if self.predicted_pc_conn_enabled:
                        try:
                            self.predicted_pc_conn.send(pc_cart)
                        except ConnectionResetError:
                            print("predicted point cloud connection reset")
                            break

        else:
            print("adc cube connection not enabled, exiting run loop")