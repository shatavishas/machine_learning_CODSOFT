from tkinter import *

# Create the main window
root = Tk()
root.title("Calculator")

# Create the display
display = Entry(root, width=30)
display.grid(row=0, column=0, columnspan=5)

# Create the buttons
buttons = [
    "7", "8", "9", "/",
    "4", "5", "6", "*",
    "1", "2", "3", "-",
    "0", ".", "=", "+"
]

# Create a function for each button
def button_click(button):
    display.insert(END, button)

# Create the buttons
for button in buttons:
    Button(root, text=button, command=lambda b=button: button_click(b)).grid(row=buttons.index(button) // 4 + 1, column=buttons.index(button) % 4)

# Start the main loop
root.mainloop()