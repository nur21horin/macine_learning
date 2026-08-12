class Car:
    def __init__(self,brand,year):
        self.brand=brand
        self.year=year

    def show_info(self):
        print("Brand:",self.brand)
        print("Year:",self.year)

car1=Car("Toyota1",2020)
car1.show_info()