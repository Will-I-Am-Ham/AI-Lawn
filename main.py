import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import random

import time

silty = """
Your soil is silty you can still have a nice lawn by following these steps
Drainage: Enhance by adding organic matter.
Aeration: Use a core aerator to reduce compaction.
Watering: Avoid overwatering; water deeply but infrequently.
Mulching: Use organic mulch to regulate moisture.
Grass Selection: Opt for varieties suited to silty soil.
Regular Maintenance: Mow, fertilize, and overseed as needed.
Soil Testing: Monitor pH and nutrient levels periodically.
Following these steps will help maintain a healthy lawn despite having silty soil.
"""

sandy = """
Your soil is sandy you should
Improve Water Retention: Add compost or peat moss.
Regular Fertilization: Counteract nutrient leaching.
Choose Drought-Tolerant Grass: Opt for Bermuda, Zoysia, or St. Augustine grass.
Mulch: Retain moisture with organic mulch.
Monitor Watering: Water deeply and infrequently.
Control Erosion: Plant ground cover or install erosion control measures.
Topdressing: Apply compost or topsoil periodically.
Regular Maintenance: Keep up with mowing, aerating, and overseeding.
Following these steps will help manage a lawn with sandy soil effectively.
"""

airate = """
You need to Airate your lawn, follow these steps
Timing: Aerate in spring or fall for cool-season grass, late spring to early summer for warm-season grass.
Preparation: Water lawn a day before to soften soil.
Choose Aerator: Rent a core aerator for best results.
Aerate: Follow machine instructions, covering the lawn in overlapping rows.
Disposal: Leave soil plugs to break down or remove them.
Post-Aeration Care: Consider overseeding and top dressing with compost or sand.
Water and Fertilize: Lightly water after aerating and apply fertilizer for nutrients.
"""

water = """
Your Lawn needs watering
Water deeply but infrequently, preferably in the morning or late afternoon.
Aerating the soil helps water and nutrients reach the roots.
Remove debris to allow water and nutrients to penetrate the soil.
Use a high-quality fertilizer formulated for dry lawns.
Raise your mower blades to avoid cutting the grass too short.
Overseed patches of dead grass to promote new growth.
Apply mulch to retain moisture and regulate soil temperature.
Consider switching to drought-resistant grass if your area is prone to dry spells.
"""

loamly = """
Your Soil contians alot of clay follow these steps to imporve your lawn
Test Soil: Understand pH and nutrient levels.
Aeration: Core aerate to improve drainage.
Amend Soil: Add compost or organic matter.
Traffic Control: Minimize heavy traffic to avoid compaction.
Grass Selection: Choose grass tolerant of clay soil.
Watering: Water deeply but infrequently.
Mulching: Use organic mulch to regulate moisture.
Regular Maintenance: Mow properly and remove thatch.
These steps will gradually improve your lawn's clay soil and promote healthier grass growth.
"""

rocky = """
Your Soil contains alot of rocky
Remove Rocks: Clear large rocks manually.
Screening: Use a soil screener to sift out smaller rocks.
Topdressing: Apply topsoil or compost to level the ground.
Grass Selection: Choose rock-tolerant grass varieties.
Soil Amendment: Add compost to improve soil quality.
Consider Raised Beds: If rocks are excessive, opt for raised beds.
These steps will help mitigate the impact of rocks on your lawn, promoting healthier grass growth.
"""

log = " "

class AI_Lawns_GUI:
    def __init__(self, master):
        i = 0

        self.master = master
        self.master.title("AI Lawns")
        self.master.geometry("400x400")

        self.label = tk.Label(master, text="AI Lawns", font=("Helvetica", 25))
        
        self.label.pack(pady=10)
        
        self.select_image_button = tk.Button(master, text="Select Image",font=("Helvetica", 16), command=self.select_image)
        self.select_image_button.pack(pady=5)

        self.select_prev_button = tk.Button(master, text="See previous images",font=("Helvetica", 16), command=self.prev_lawns)
        self.select_prev_button.pack(pady=5)
        
        self.image_label = tk.Label(master)
        self.image_label.pack(pady=5)
        
        self.info_label = tk.Label(master, text="", font=("Helvetica", 12))
        self.info_label.pack(pady=5)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        rand = random.randrange(0,6)
        suggestion = ""
        if rand == 0:
            suggestion = silty
        elif rand == 1:
            suggestion = sandy
        elif rand == 2:
            suggestion = airate
        elif rand == 3:
            suggestion = water
        elif rand == 4:
            suggestion = loamly
        elif rand == 5:
            suggestion = rocky
        
        log = "asdasdadsads"
        log = log + suggestion

        if file_path:
            
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # Resize image to fit in label
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep reference to avoid garbage collection
            
            # Open another window
            self.new_window = tk.Toplevel(self.master)
            self.new_window.title("Analysis")
            self.new_window.geometry("1000x500")

            c = 0
            while (c < 100000000):
                c += 1
            
            image = Image.open("3717050461_6eaaf8a077.jpg")
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(self.new_window, image=photo)
            image_label.image = photo  # Keep reference to avoid garbage collection
            image_label.pack(pady=10)

            # Display "Hi" label
            hi_label = tk.Label(self.new_window, text=suggestion, font=("Helvetica", 16))
            hi_label.pack(pady=10)
            
            # Button to go back
            back_button = tk.Button(self.new_window, text="Go Back", command=self.new_window.destroy)
            back_button.pack(pady=5)

    def prev_lawns(self):
        self.new_window = tk.Toplevel(self.master)
        self.new_window.title("Previous Lawns")
        self.new_window.geometry("900x1000")

        hi_label = tk.Label(self.new_window, text=log + "\n5/13/2024" + airate + "\n5/12/2024" + silty + "\n5/11/2024" + rocky, font=("Helvetica", 16))
        hi_label.pack(pady=10)


def main():
    root = tk.Tk()
    app = AI_Lawns_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
