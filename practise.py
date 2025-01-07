# # l = list(map(int,input("Enter the list of your choice:").split(',')))
# # largest = l[0]
# # for i in l:
# #     if(i > largest):
# #         largest = i


# # print((f"The largest number in this list is {largest}"))
    
# #FACTORIAL OF A NUMBER 

# # def fact (n):
   

# #     if(n==1 or n==0):

# #         return 1
# #     else:
# #         return n* fact(n-1)
# # print(fact(5))


# # num = 7
# # for i in range(2,num+1
# # ):
# #     if num%i==0:
# #         print('Prime')
# #         break
# # else:
# #         print("not prime")
# # x = 5

# # def my_func():
# #     y = 10
# #     print(y)
# #     global x
# #     x = 10

# # my_func()
# # print(x)
# # f = open("my_txt2",'w')
# # f.write("Hello there you are")
# # f.write("I was searching you everywhere")
# # f.write("And i found you here")
# # f.close

# # f = open ("my_txt2",'r')
# # while True:
# #     line = f.readline()

# #     if not line:
# #         break
# #     print(line)

# # f = open("my_txt2",'w')
# # lines = ['Hello 1/n','Hello 2/n','Hello 3/n']
# # f.writelines(lines)
# # f.close()

# #READING A FILE
# # f = open("my_txt",'r')
# # text = f.read()
# # print(text)
# # f.close()

# # #WRITING A FILE
# # f = open("my_txt",'w')
# # f.write("This is the new text")
# # f.close()


# # with open("my_txt",'w') as f:
# #     f.write("Hello i am Kalash/n I am a Python student /n I love python programming /n")


# # f = open("my_txt",'r')
# # while True:
# #     line = f.readline()
# #     if not line:
# #         break
# #     print(line)

# # f = open("my_txt",'w')
# # lines =["123456789ABCDEFGHIJKLM"]
# # f.writelines(lines)
# # f.close()

# # with open("my_txt",'r') as f:
# #     f.seek(10)
# #     data = f.read()
# #     print(data)

# # double = lambda x : x*2
# # print(double(5))

# # def double (x):
# #     return x*2
# # print(double(5))

# # def cube (fx, value):
# #     return 3*fx(value)
# # print(cube(double,5))
# # def cube(x):
# #     return x*x*x
# # # print(cube(3))
# # # l = [1,2,3,4,1,2,3]
# # # newl = []
# # # for i in l:
# # #     newl.append(cube(i))
# # # print(newl)
# #MAP FUNCTION
# # def cube(x):
# #     return x*x*x
# # print(cube(2))
# l = [1,2,3,4,1,2,3]
# newl = list(map(lambda x : x*x*x,l))
# print(newl)
# # newl = list(map(cube,l))
# # print(newl)

# #FILTER FUNCTION
# # def doub(a):
# #     return a>2
# # new_l = list(filter(doub,l))
# new_l = list(filter(lambda a:a>2,l))
# print(new_l)

# #REDUCE FUNCTIOn
# from functools import reduce
# # num = [1,2,3,4,5]
# # # def mysum(x,y):
# # #     return x+y
# # # sum = reduce(mysum,num)
# # print(sum)

# sum = reduce(lambda x,y:x+y,l)
# print(sum)

# ROCK PAPER SCISSOR
# import random
# def game():
#     player = input("Enter your choice(rock,paper,sissor):")
#     computer = random.choice(seq=["rock","paper","sissor"])

#     if player == computer:
#         print('The match is a draw')
#     elif player == 'rock' and computer == 'paper':
#         print("You won")
#     elif player == "paper" and computer == "rock":
#         print("You loose")
#     elif player == "sissor" and computer =="rock":
#         print("You loose")
#     elif player == "rock" and computer == "sissor":
#         print("You won")
#     elif player == "paper" and computer == "sissor":
#         print("You loose")
#     elif player == "sissor" and computer == "paper":
#         print("You won")
#     else:
#         print("Please choose the choices from rock paper and sissor, now try again please")

# game()

# num = int(input("Enter a number "))
# try:
#     for i in range(1,11):
#         # return i*num
#         print(f"{num} X {i}  = {num*i} ")
# except:
#     print("Some error occured")
# finally:
#     print("I am always printed")

# print("Something top secrert")
    
# marks = [12,34,5,67]

# for index,marks in enumerate(marks):
#     print(f"The marks are {marks}")
#     if(index==2):
#         print("Rohan's marks")

#Classes and Objects in python
# class Person:
#     name = "Kalash"
#     occupation = "Student"
#     age = 20
#     def info(self):
#         print(f"The peson's name is {self.name}, his age is {self.age} and he is a {self.occupation}")
        
# a = Person()
# # b = Person()
# # a.name = "Virat"
# # a.age = 36
# # a.occupation = "Crickter"
# # b.name = "Harry"
# # b.age = 32
# # b.occupation = "Teacher"
# # print(b.info())
# print(a.info())
# class Person:
#     def __init__(self,name,age,occ):
#         self.name = name
#         self.age = age
#         self.occupation = occ
#     def info(self):
#         print(f"The name of the person is {self.name}, his age is {self.age} and he is a {self.occupation}")

# a = Person("Kalash",20,"Student")
# b = Person("Virat",35,"Crickter")
# b.info()
# a.info()

# class Employee:
#     def __init__(self,name,id,salary):
#         self.name = name
#         self.id = id
#         self.salary = salary
#     def showDetails(self):
#         print(f"The name of the employee is {self.name} and id is {self.id} and his salary is {self.salary}")
# class Manager(Employee):
#     def getInfo(self):
        
#         print("He is very funloving kind of a person")

# P = Employee("Kalash",400,40000)
# P.showDetails()
# M = Manager("Rohit",45,295000)
# M.showDetails()
# M.getInfo()


#Private modifier
# class Student:
#     def __init__(self,name):
#         self.__name = name
# s = Student("Kalash")
# # print(s.__name)
# print(s._Student__name)

#protected modifier
# class Student:
#     def __init__(self):
#         self._name = "Kalash"
#     def _newFun(self):
#         return "Kalash is good"

# class Subject(Student):
#     pass

# obj = Student()
# obj2 = Subject()
# print(obj2._name)
# print(obj2._newFun())

# class Maths:
#     def __init__(self,num):
#         self.num = num

#     def addnum(self,num1):
#         self.num1 = num1

#     def getinfo(self):
#         print(f"The numbers are {self.num} and {self.num1}")

#     @staticmethod
#     def add(a,b):
#         return a+b
    

# a = Maths(5)
# a.addnum(6)
# a.getinfo()
# print(a.add(5,9))
# class Employee:
#     companyName = "Google" # class variable
#     def __init__(self,name):
#         self.name = name
#         self.raise_amount = 0.25 #instance variable
#     def showDetails(self):
#         print(f"The name of the Employee is {self.name} and raise amount in his company {self.companyName} is {self.raise_amount}")

# emp = Employee("Kalash")
# emp.raise_amount = 0.36
# emp.showDetails()
# emp2 = Employee("Virat")
# emp2.companyName = "Microsoft"
# emp2.showDetails()


# class library:
    
#     Book_num = 3
#     def __init__(self,name):
#         self.name = name 
#         library.Book_num +=1
#         self.thriller_Book = []
#         self.sci_fi_Book = []
#     def Books(self):
#                 try:
#                     instert = input("Enter the book name: ")
#                     type = input("IS it a Thriller or Sci-fi: ")
#                     if  type == "Thriller":
#                         self.thriller_Book.append(instert)
#                         library.Book_num +=1

#                     elif  type == "Sci-fi":
#                         self.sci_fi_Book.append(instert)
#                         library.Book_num +=1
#                     else:
#                           print("Please enter from Thriller and Sci-fi please")
#                 except:
#                      print("There was an error please resolve it first")
                

#     def getInfo(self):
#             print(f"The name of the books you have  entered is {self.name} and now the new count of books is {library.Book_num} and the new thriller list of book is {self.thriller_Book} and for the sci-fi the new list is {self.sci_fi_Book}")

# l = library("My own way")
# l.Books()
# l.getInfo()

# Filter function
# a = [11,12,22,33,34,44,55,56]
# def func(a):
#     return a%2==0
# myList = list(filter(func,a))
# print(myList)
#Map Function
# def func(a):
#     return a*a
# myList = list(map(func,a))
# print(myList)
#Reduce function
# from functools import reduce

# newlist = reduce(lambda x,y:x+y,a)
# print(newlist)
# class Employee:
#     Company = "Apple" # class variable 
#     def show(self):
#         print(f"The name of the employee is {self.name} and the company is {self.Company}")
#     @classmethod
#     def changeCompany(self,newCompany):
#         self.Company = newCompany

# e = Employee()
# e.name = "Kalash"
# e.show()
# e2 = Employee()
# e2.name = "Kalash"
# e2.changeCompany("Google")
# e2.show()
# print(Employee.Company)
# class Enpmployee:
#     def __init__(self,name,salary):
#         self.name = name
#         self.salary = salary
#     @classmethod
#     def fromstr(self,string):
#         name,salary = string.split("-")
#         return self(name,int(salary))

# e1 = Enpmployee("Kalash",12000)
# print(e1.name)
# print(e1.salary)
# string ="Lalit-12000"
# e2 = Enpmployee.fromstr(string)
# print(e2.name)
# print(e2.salary)


# row = 5
# for i in range(1,row+1):
#     for j in range(1,i+1):
#         print("*" ,end="")
#     print()
        
# import numpy as np

# # Activation function (Sigmoid)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Derivative of the sigmoid function
# def sigmoid_derivative(x):
#     return x * (1 - x)

# # Input data (4 samples with 2 features each)
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# # Output labels (XOR problem)
# y = np.array([[0], [1], [1], [0]])

# # Seed for reproducibility
# np.random.seed(42)

# # Initialize weights and biases
# input_layer_neurons = X.shape[1]  # 2 input features
# hidden_layer_neurons = 2          # 2 neurons in the hidden layer
# output_neurons = 1                # 1 output

# # Weights and biases
# hidden_weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
# hidden_bias = np.random.uniform(size=(1, hidden_layer_neurons))
# output_weights = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
# output_bias = np.random.uniform(size=(1, output_neurons))

# # Learning rate
# learning_rate = 0.1

# # Training the network
# for epoch in range(10000):  # 10,000 iterations
#     # Forward pass
#     hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
#     hidden_layer_activation = sigmoid(hidden_layer_input)

#     output_layer_input = np.dot(hidden_layer_activation, output_weights) + output_bias
#     predicted_output = sigmoid(output_layer_input)

#     # Backpropagation
#     error = y - predicted_output
#     d_predicted_output = error * sigmoid_derivative(predicted_output)

#     error_hidden_layer = d_predicted_output.dot(output_weights.T)
#     d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)

#     # Update weights and biases
#     output_weights += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
#     output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate

#     hidden_weights += X.T.dot(d_hidden_layer) * learning_rate
#     hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# # Display final output
# print("Final Predicted Output:\n", predicted_output)

# import tensorflow as tf

# x = tf.constant(10.0)
# y = tf.constant(15.0)
# z = x+y
# print(z)

# a = tf.constant(10)
# b = tf.constant(20)
# c = a * b
# print(c)


# k = tf.constant(25)
# l = tf.constant(5)
# s = k/l
# print(s)



# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Example dataset (features and target)
# X = [[1], [2], [3], [4], [5]]  # Independent variable
# y = [2, 4, 6, 8, 10]           # Dependent variable

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("Predictions:", y_pred)
# vector = tf.constant(5)
# print("Vector",vector)

# scalar = tf.constant([1,2])
# print("Scalar",scalar)

# matrix = tf.constant([[1,2],[3,4],[5,6]])
# print("Matrix",matrix) 

# import numpy as np

# myarr = np.array([[1,2,3,4]],np.int64)
# print(myarr[0,1])
# print(myarr.shape)

# lspace = np.linspace(1,3,5)
# print(lspace)

# rnge = np.arange(12)
# print(rnge)

# zeros = np.zeros((2,12))
# print(zeros)

# emp = np.empty((4,6))
# print(emp)

# id = np.identity(3)
# print(id)
# print(id.shape)

# arr = np.arange(20)
# print(arr)

# newarr = arr.reshape(4,5)
# print(newarr)

# print(newarr.sum(axis =0))

# import tensorflow as tf

# x = tf.constant(10.0)
# y = tf.constant(15.0)
# z = x+y
# print(z)

# a = tf.constant(10)
# b = tf.constant(20)
# c = a * b
# print(c)


# k = tf.constant(25)
# l = tf.constant(5)
# s = k/l
# print(s)

# x = tf.Variable(initial_value = 5  ,dtype = tf.float32)
# x.assign(10)
# print(x)
# a = int(input("Enter the valye of a: "))
# b =  int(input("Enter the value of b: "))
# sum = a+b
# print(sum)

# num = int(input("Enter the value of number: "))
# z = int(input("Enter the value of z: "))

# a = num%z
# print("The remainder is :",a)


# a = 34
# b= 80
# if(a>b):
#     print("Greater ")
# else:
#     print("Smaller")

# a = int(input("Enter the value of a: "))
# b = int(input("Enter the value of b:  ")) 
# print("The avg of these two number is:",(a+b)/2)


# x = int(input("Enter the value of x: "))
# print("The squar of the numer is: ",x**2)

# if __name__ == '__main__':
#     a = int(input("Enter the value of a: "))
#     b = int(input("Enter the value of b: "))
    
# print(a+b)
# print(a-b)
# print(a*b)


# num1 = int(input("Enter the value of the 1st numbner: "))
# num2 = int(input("Enter the value of the 2st numbner: "))
# num3 = int(input("Enter the value of the 3st numbner: "))
# num4 = int(input("Enter the value of the 4st numbner: "))
# if(num1>num2):
#     print(num1)
# elif(num2>num3):
#     print(num2)
# elif(num3>num4):
#     print(num3)
# else:
#     print(num4)

# sub1 = int(input("Enter the marks of Subject 1: "))
# sub2 = int(input(("Enter the marks of Subject 2: ")))
# sub3 = int(input(("Enter the marks of Subject 3: ")))

# percent = (sub1+sub2+sub3)/300*100

# if (percent>33):
#     print("You have passed and you have got",percent)
# else:
#     print("You have failed the exam and you have got",percent)

# email =" Tamasha bana na or group after first gaining their trust Make a lot of money Confidence tricks exploit victims using a combination of the victim's credulity, naivety, compassion, vanity, confidence, irresponsibility, and greed. Researchers have defined confidence tricks as a distinctive species of fraudulent conduct ... intending to further voluntary exchanges that are not mutually beneficial, as they benefit con operators con men at the expense of their victims the 'marks"
# spam_words1 = "Make a lot of Money"
# spam_words2 ="buy now"
# spam_words3 ="subscribe this"
# spam_words4 = "click this"
# if((spam_words1 in email) or(spam_words2 in email) or (spam_words3 in email) or (spam_words4 in email)):
#     print("This is a spam emai")
# else:
#     print("This is a normal email")


# user_name = input("Enter the usre name: ")

# if len( user_name)<10:
#     print("Your Character in the name contains less than 10 character")
# else:
#     print("Your Character in the name contains more than 10 character")

names = ["Aryan Malhotra",
"Sanya Kapoor",
"Ishaan Verma",
"Ritika Sharma",
"Kabir Mehta"]
your_name = input("Enter your name to check if it is in the list or not: ")

if(your_name in names):
    print("Your name is in the list")
else:
    print("Your name is not in the list")


