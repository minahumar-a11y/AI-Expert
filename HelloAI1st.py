#GREET THE USER 
print("Hello! I am your chatbot. \nWhat is your name? ")
name = input()

#Respond to the users name 
print(f"Nice to meet you,{name}:)")

#Ask question from the user
feeling = input("How are you feeling today? (Good/Bad) : ").lower()

if feeling == "good":
    print("I'm very glad to hear that!")
elif feeling == "bad":
    print("Yes,sometimes its okay to feel bad.")
else:
    print("I hope your day gets better and i get that sometimes its very hard to put feelings in words.")

print(f"Its very nice to talk to you,{name}")


