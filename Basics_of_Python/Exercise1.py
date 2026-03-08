# 1. Write a Python program to display "Welcome to Python Programming!!!
print('Welcome to Python Programming')
# 2. Write a Python program to ask the user to enter two numbers, perform all the mathematical operations (addition, subtraction, multiplication, division) and then display the results.
'''n1=int(input("Enter a number: "))
n2=int(input("Enter a number: "))
print(n1+n2)
print(n1-n2)
print(n1*n2)
print(n1/n2)'''

# 3. Wnte a Python program to calculate the simple interest and display it when the user enters the principal, rate, and time.

# 4. Write a Python program to display the area and perimeter of a circle when the user enters the radius.

# 5. Write a Python program that asks the user to enter the length of the square and display its area and perimeter

# 6. Write a Python program that asks the user to enter the length and breadth of a rectangle and then display its area and perimeter.

# 7. Write a Python program to display the volume of a cuboid when the length, breadth, and height are entered by the user.

# 8. Write a Python program to display the following escape sequence:
print('S.No\tProduct\n----\t-------\n001\tProcessor\n0002\tRam\n0003\tHard Drive')
# 9. Write a Python program to display the temperature in Fahrenheit when the user enters the temperature in Celsius.

# 10. Write a Python program that will accept days as input and then display it in terms of years, months, and days. (Example: If the user enters 400, it should display 1 year, 1 month, and 5 days.)
date=int(input('Enter the number of days'))
year=0
month=0
days=0
while date//365 > 0:
    year=year+date//365
    date=date%365
    while date//30 > 0:
        month=month+date//30
        date=date%30
print(year,'Year',month,'Month',date,'days')    

# 11. Write a Python program that will swap two numbers entered by the user without using a temporary variable.
x=10
y=20
print("Before Swapping x:",x,"y:",y)
x,y=y,x
print("After Swapping x:",x,"y:",y)
# 12. Write a Python program that asks the name of a student and the marks obtained by him/her in 5 subjects. Display the percentage of the student assuming full marks are 100 for each subject.

# 13. Write a Python program to find compound interest for a given principal amount, time, year, and rate.

# 14. Write a Python program to read the height and base of a triangle and find the area of it.

# 15. Write a Python program to read the three sides of a triangle and calculate the area.
s1=int(input('Enter 1st side'))
s2=int(input('Enter 2nd side'))
s3=int(input('Enter 3rd side'))
s=(s1+s2+s3)/2

# 16. Write a Python program to convert dollars to rupees.

# 17. Write a Python program to convert centimeters into meters.

# 18. Write a Python program to convert hours into minutes.

# 19. Write a Python program which reads the radius of a sphere from the keyboard and calculates its volume and area. (Hint: Volume = 4mm³/3 and Area = 4m²).

# 20. Write a Python program that defines two variables with the same value and uses the id() function to print their memory addresses. Change the value of one variable and print the memory addresses again.

# 21. Write a Python program that takes two boolean values and prints the result of and, or, and not operations.

# 22. Write a Python program that takes two integers and prints the result of bitwise AND, OR, XOR, left shift, and right shift operations.