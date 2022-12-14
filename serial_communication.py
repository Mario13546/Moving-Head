# Created by Alex Pereira

# Imports
import serial
import serial.tools.list_ports

# Creates the SerialComms Class
class SerialComms:
    # Constructor
    def __init__(self, portNum = None, baudRate = 9600):
        """
        Constructor for the SerialComms class.
        @param serialPortNumber
        @param baudRate
        """
        # Sets variables
        connected = False

        # Port number not given
        if portNum is None:
            try:
                # Lists all available devices
                ports = list(serial.tools.list_ports.comports())

                # Looks for an Arduino in the list of devices
                for port in ports:
                    if "Arduino" in port.description:
                        print(f'{port.description} Connected')
                        self.ser = serial.Serial(port.device, baudRate)
                        connected = True
            except:
                pass

            # Device no found
            if not connected:
                print("Arduino not found")
                self.testMode = True
        # Port number given
        else:
            try:
                # Sucessful connection
                self.ser = serial.Serial(portNum, baudRate)
                print("Connected to Serial Device")
            except:
                # Failed connection
                self.testMode = True
                print("Failed to connect to Serial Device")

        # Bypasses serial device creation
        if (self.testMode == True):
            # Makes the serial object blank
            self.ser = None

    def sendData(self, data):
        """
        Sends the data over serial.
        @param dataArray
        """
        # Creates the holder string
        myString = ""

        # Adds the data to myString
        myString += str(int(data)).zfill(2)

        # Invalid or no data present
        if (myString == "-1" or myString == "91" or myString == ""):
            myString = "91"

        # Adds the terminating character to the end
        myString += "\r"

        # Writes data to serial if testMode is disabled
        if (self.testMode == False):
            self.ser.write(myString.encode())
        else:
            myString = myString.encode()

        # Returns the string for debugging
        return myString

    def getData(self):
        """
        Gets the data from serial.
        @return recievedData
        """
        # Reads data from serial if testMode is disabled
        if (self.testMode == False):
            data = bytes(self.ser.read())
        else:
            data = "No Value".encode()

        # Decodes the recieved data 
        data = data.decode("utf-8")
        
        # Return the data
        return data