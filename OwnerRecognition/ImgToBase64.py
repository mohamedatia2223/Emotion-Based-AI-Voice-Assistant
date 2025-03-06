import base64 
class ImgBase64 : 
    def __init__(self):
         pass

    @staticmethod  
    def ConvertToBase64(img_path:str): 

        # convert the img into binary base64 
        # and then base64 string 

        # 'rb' : r is read mode , and b is binary mode 
        # both of them are neceassary 
        with open(img_path,"rb") as img : 
            imgfile = base64.b64encode(img.read()).decode()

        return imgfile
    
    @staticmethod  
    def ConvertToBiary(base64String:str,outputPath):
        # convert the base64 into a binary img

        # 'wb' : w is write mode , and b is binary mode 
        # both of them are neceassary 
        with open(outputPath,"wb") as file : 
            binaryImg = base64.b64decode(base64String)
            file.write(binaryImg)


# base64String = ImgBase64.ConvertToBase64("C:\\1.jpg")
# ImgBase64.ConvertToBiary(base64String,"img.png")

base64String = ImgBase64.ConvertToBase64("C:\\1.jpg")
with open("output.txt","w") as file :
    file.write(base64String)