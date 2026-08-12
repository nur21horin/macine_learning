class Account:
    def __init__(self,owner,balance):
        self.owner=owner
        self.__balance=balance
    def deposit(self,amount):
        self.__balance+=amount
        print("Deposit Accepted")
    def withdraw(self,amount):
        if amount<=self.__balance:
            self.__balance-=amount
            print("Withdrawal Accepted")
        else:
            print("Insufficient Funds")
acc=Account("John",1000)
acc.deposit(500)
print("Balance:",acc._Account__balance)
acc.withdraw(200)
print("Balance:",acc._Account__balance)