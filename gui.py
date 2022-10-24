from tkinter import Tk, IntVar, Label, Entry, StringVar, Button

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from pendulum import Pendulum, Bob  # noqa: E402


class Gui(Tk):
    def __init__(self):
        super().__init__()
        self.geometry("250x150")

        self.bob_length = IntVar(self, 1)
        self.bob_mass = IntVar(self, 1)
        self.bob_initial_angle = IntVar(self, 1)
        self.bob_damping = StringVar(self, "1")
        self.pendulum_simulation = IntVar(self, 30)
        self.pendulum_dt = StringVar(self, "0.1")

        self.create_labels()
        self.create_entries()
        self.create_button()

    def call_simulation(self):
        bob_instance = Bob(
            length=self.bob_length.get(),
            mass=self.bob_mass.get(),
            initial_angle=self.bob_initial_angle.get(),
            damping=float(self.bob_damping.get())
        )
        pendulum_instance = Pendulum(
            bob_instance,
            len_of_simulation=self.pendulum_simulation.get(),
            dt=float(self.pendulum_dt.get()),
        )
        pendulum_instance.run_simulation()

    def create_button(self):

        start_button = Button(text="Start", command=self.call_simulation)
        start_button.grid(row=6, column=0, columnspan=2)

    def create_labels(self):

        bob_length_label = Label(self, text="Bob length")
        bob_length_label.grid(row=0, column=0)

        bob_mass_label = Label(self, text="Bob mass")
        bob_mass_label.grid(row=1, column=0)

        bob_initial_angle_label = Label(self, text="Bob initial angle")
        bob_initial_angle_label.grid(row=2, column=0)

        bob_damping_label = Label(self, text="Bob damping")
        bob_damping_label.grid(row=3, column=0)

        pendulum_simulation_label = Label(self, text="Simulation length")
        pendulum_simulation_label.grid(row=4, column=0)

        pendulum_dt_label = Label(self, text="Simulation dt")
        pendulum_dt_label.grid(row=5, column=0)

    def create_entries(self):

        bob_length_entry = Entry(self, textvariable=self.bob_length)
        bob_length_entry.grid(row=0, column=1)

        bob_mass_entry = Entry(self, textvariable=self.bob_mass)
        bob_mass_entry.grid(row=1, column=1)

        bob_initial_angle_entry = Entry(self, textvariable=self.bob_initial_angle)
        bob_initial_angle_entry.grid(row=2, column=1)

        bob_damping_entry = Entry(self, textvariable=self.bob_damping)
        bob_damping_entry.grid(row=3, column=1)

        pendulum_simulation_entry = Entry(self, textvariable=self.bob_initial_angle)
        pendulum_simulation_entry.grid(row=4, column=1)

        pendulum_dt_entry = Entry(self, textvariable=self.pendulum_dt)
        pendulum_dt_entry.grid(row=5, column=1)


gui = Gui()
gui.mainloop()
