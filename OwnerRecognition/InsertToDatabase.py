import pyodbc
from ImgToBase64 import ImgBase64

class Owner:
    def __init__(self, name, age, imageBase64):
        self.name = name
        self.age = age
        self.imageBase64 = imageBase64


Connection_String = "Driver={SQL Server};Server=LAPTOP-2665Q733\\SQLEXPRESS02;Database=Ai-Dog;Trusted_Connection=yes;"

conn = pyodbc.connect(Connection_String)
cursor = conn.cursor()

def addOwner(owner: Owner):
    query = "INSERT INTO Owners (name, age, image_Base64) VALUES (?, ?, ?)"
    cursor.execute(query, (owner.name, owner.age, owner.imageBase64))
    conn.commit()  

def getOwners():
    query = "SELECT * FROM Owners"
    cursor.execute(query)
    
    return cursor.fetchall()


# image_base64 = ImgBase64.ConvertToBase64("C:\\1.jpg")

# owner = Owner("Mohamed El-Gazzar", 20, image_base64)

# addOwner(owner)
# print(getOwners())

cursor.close()
conn.close()
