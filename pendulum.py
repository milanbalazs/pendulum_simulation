"""
Pendulum simulation.
Diagrams:
    - Move of simple pendulum
    - Phase space
    - Theta/t
    - Energy/t
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

from typing import Union, List, Tuple

GRAVITY = 9.8


class Bob:
    """
    This class holds all of pendulum variables:
        - Length of rod/string
        - Mass of the Bob
        - Theta angle (between rod and origin)
        - Velocity
        - X and Y position of Bob
        - Acceleration
        - Damping
        - Energy (Pe+Kp)
        - Kinetic Energy (Ke)
        - Potential Energy (Pe)
    """

    def __init__(
        self, length: Union[int, float], mass: Union[int, float], initial_angle: Union[int, float]
    ) -> None:
        self.length: Union[int, float] = length
        self.mass: Union[int, float] = mass
        self.theta: Union[int, float] = initial_angle
        # If it's 1 then there is no damping. If it's 9.99 then 1% is the damping.
        self.damping: Union[int, float] = 1
        self.velocity: Union[int, float] = 0
        self.x_position: Union[int, float] = 0
        self.y_position: Union[int, float] = 0
        self.acceleration: Union[int, float] = 0
        self.energy: Union[int, float] = 0
        self.ke: Union[int, float] = 0
        self.pe: Union[int, float] = 0


class Pendulum:
    """
    Simulate the Pendulum movement and visalize the most relevant diagrams.
    """

    def __init__(
        self, bob: Bob, len_of_simulation: Union[int, float] = 30, dt: Union[int, float] = 0.1
    ) -> None:
        """
        Init method of Pendulum class. It initialises the instance variables.

        Args:
            bob: An instance of Bob class.
            len_of_simulation: Length of the simulation.
            dt: Dt (Simulation step size)
        """

        self.bob: Bob = bob
        self.len_of_simulation: Union[int, float] = len_of_simulation
        self.dt: Union[int, float] = dt

    def run_simulation(self) -> None:
        """
        This is the Simulation starter method.
        This method has to be called to start the complete simulation.

        Returns: None
        """

        self.initialize()
        fig, axes, data = self.initialize_plots()

        # Create the array of dts.
        curr_time: np.ndarray = np.arange(int(self.len_of_simulation / self.dt) + 1) * self.dt

        for ti in curr_time[1:]:
            self.integrator_wrapper()
            self.get_positions()
            self.get_energies()
            self.update_plots(ti, axes, data)

    def update_plots(self, current_time, axes, data) -> None:
        """
        Update all of the plots.

        # TODO: Visualize the Ke and Pe Energies.

        Args:
            current_time: The current time (time+dt).
            axes: List of axes.
            data: The data array.

        Returns: None
        """

        axes[0, 0].set_title("t = %f" % current_time)
        line, line1 = data[0]
        line.set_xdata([0, self.bob.x_position])
        line.set_ydata([0, self.bob.y_position])
        line1.set_xdata(np.append(line1.get_xdata(), self.bob.x_position))
        line1.set_ydata(np.append(line1.get_ydata(), self.bob.y_position))

        axes[1, 0].set_title("Theta: {:.2f} {}".format(self.bob.theta, r"${\pi}$"))
        line1 = data[1][0]
        line1.set_xdata(np.append(line1.get_xdata(), current_time))
        line1.set_ydata(np.append(line1.get_ydata(), self.bob.theta))

        if current_time > axes[1, 0].get_xlim()[1]:
            axes[1, 0].set_xlim(0, 2 * current_time)

        line1, _ = data[2]
        axes[0, 1].set_title("(Cords: {:.2f};{:.2f})".format(self.bob.theta, self.bob.velocity))
        line1.set_xdata(np.append(line1.get_xdata(), self.bob.theta))
        line1.set_ydata(np.append(line1.get_ydata(), self.bob.velocity))

        y_axe_max_1_1 = np.amax(line1.get_ydata()) + 1
        y_axe_min_1_1 = np.amin(line1.get_ydata()) - 1
        axes[0, 1].set_ylim(y_axe_min_1_1, y_axe_max_1_1)

        line1 = data[3][0]

        axes[1, 1].set_title("Total energy: {:.2f}".format(self.bob.energy))
        line1.set_xdata(np.append(line1.get_xdata(), current_time))
        line1.set_ydata(np.append(line1.get_ydata(), self.bob.energy))

        y_axe_max_1_1 = np.amax(line1.get_ydata()) + 0.1
        y_axe_min_1_1 = np.amin(line1.get_ydata()) - 0.1
        axes[1, 1].set_ylim(y_axe_min_1_1, y_axe_max_1_1)

        if current_time > axes[1, 1].get_xlim()[1]:
            axes[1, 1].set_xlim(0, 2 * current_time)

        plt.pause(1e-5)

    def acceleration_wrapper(self, data_array, _) -> np.array:
        """
        Wrapper method for the kick method (Acceleration calculation).

        Args:
            data_array: The data array ([Theta, Velocity])
            _: Unused parameter. It's needed because it's a call back function.

        Returns: Array of velocity and acceleration

        """

        self.bob.theta, self.bob.velocity = data_array
        acceleration: Union[int, float] = self.acceleration_calculation()
        result_array: np.array = np.array([self.bob.velocity, acceleration])
        return result_array

    def integrator_wrapper(self) -> None:
        """
        This is acceleration wrapper to the odeint integrator.

        Please see the details:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

        Returns: None
        """

        theta_and_velocity_array: np.array = np.array([self.bob.theta, self.bob.velocity])

        integration_result_array: np.array = odeint(
            self.acceleration_wrapper, theta_and_velocity_array, [0, self.dt]
        )
        self.bob.theta, self.bob.velocity, = integration_result_array[1]
        self.bob.velocity *= self.bob.damping
        if self.bob.theta > np.pi:
            while self.bob.theta > np.pi:
                self.bob.theta -= 2 * np.pi
        if self.bob.theta < -np.pi:
            while self.bob.theta < -np.pi:
                self.bob.theta += 2 * np.pi

    def initialize_plots(self) -> Tuple[mpl.figure.Figure, np.ndarray, list]:
        """
        Init the plots which will be used for the simulation. This method has to be called to get
        the list of generated plots and exes.

        Returns: The Figure and axes and the data in Tuple (Please see the return type annotation).
        """

        fig: mpl.figure.Figure
        axes: np.ndarray

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.tight_layout(pad=5)

        data: list = []

        xlist: List[Union[int, float]] = [0, self.bob.x_position]  # Grab the locations of the bob.
        ylist: List[Union[int, float]] = [0, self.bob.y_position]

        axes[0, 0].plot([-0.5, 0.5], [0, 0], "-k", linewidth=5)

        (line,) = axes[0, 0].plot(xlist, ylist, "-bo", markersize=10, linewidth=3)

        (line1,) = axes[0, 0].plot(self.bob.x_position, self.bob.y_position, "-b", linewidth=2)

        axes[0, 0].set_xlim(-self.bob.length - 0.5, self.bob.length + 0.5)

        axes[0, 0].set_ylim(-self.bob.length - 0.5, self.bob.length + 0.5)
        axes[0, 0].set_title("t = 0", fontsize=20)
        axes[0, 0].set_xlabel("x_position", fontsize=20)
        axes[0, 0].set_ylabel("y_position", fontsize=20)
        data.append([line, line1])

        (line2,) = axes[1, 0].plot(0, self.bob.theta, "-b")
        axes[1, 0].set_ylim(-np.pi - 0.5, np.pi + 0.5)

        unit: float = 0.25
        y_tick: np.arange = np.arange(-1, 1 + unit, unit)

        y_label2: List[str] = [r"$" + format(r, ".2g") + r"\pi$" for r in y_tick]
        axes[1, 0].set_yticks(y_tick * np.pi)
        axes[1, 0].set_yticklabels(y_label2)

        axes[1, 0].set_xlabel("t", fontsize=20)
        axes[1, 0].set_ylabel("$\\theta$", fontsize=20)
        data.append([line2])

        (line1,) = axes[0, 1].plot(self.bob.theta, self.bob.velocity, "b.")

        axes[0, 1].set_xlabel("$\\theta$", fontsize=20)
        axes[0, 1].set_ylabel("$\\dot{\\theta}$", fontsize=20)
        axes[0, 1].set_xlim(-3, 3)
        axes[0, 1].set_ylim(-4, 4)

        axes[1, 1].set_xlabel("t", fontsize=20)
        axes[1, 1].set_ylabel("Energy (Ke+Pe)", fontsize=20)

        data.append([line1, line2])

        (line1,) = axes[1, 1].plot(0, self.bob.energy, "-b")

        data.append([line1])
        axes[0, 0].plot(xlist, ylist, "-o", color="grey", linewidth=3, markersize=10)

        # plt.show()
        return fig, axes, data

    def initialize(self) -> None:
        """
        Init the simulation:
            - Get the initial positions.
            - Get the initial energies.
            - Get the initial acceleration.

        Returns: None
        """
        self.get_positions()
        self.get_energies()
        self.bob.acceleration = self.acceleration_calculation()

    def acceleration_calculation(self) -> Union[int, float]:
        """
        Calculate the acceleration of bob.

        FYI:
            It's minus because the acceleration vector position is opposite of bob movement.

        Returns: Acceleration of bob.
        """

        acceleration = -(
            (
                GRAVITY
                + (self.bob.acceleration * np.sin((2 * np.pi * self.dt) / self.len_of_simulation))
            )
            / self.bob.length
        ) * np.sin(self.bob.theta)

        return acceleration

    def get_positions(self) -> Tuple[float, float]:
        """
        Calculate the X and Y positions of bob.

        Returns: X and Y positions in Tuple.
        """

        length: Union[int, float] = self.bob.length
        theta: Union[int, float] = self.bob.theta

        x_pos: Union[int, float] = length * np.sin(theta)
        y_pos: Union[int, float] = -length * np.cos(theta)

        self.bob.x_position = x_pos
        self.bob.y_position = y_pos

        return x_pos, y_pos

    def get_energies(self) -> None:
        """
        Calculate the kinetic, potential energies and the summa of them (total energy) of Bob.

        The current calculation has been made based on:
            - https://blogs.bu.edu/ggarber/interlace/pendulum/energy-in-a-pendulum/

        The previous (probably bad calculation):

            x_pos: float
            y_pos: float
            x_pos, y_pos = self.get_positions()
            x_velocity: Union[int, float] = -y_pos * self.bob.velocity
            y_velocity: Union[int, float] = x_pos * self.bob.velocity
            self.bob.ke = 0.5 * self.bob.mass * (x_velocity ** 2 + y_velocity ** 2)
            self.bob.pe = self.bob.mass * GRAVITY * y_pos

        Returns: None
        """

        self.bob.ke = 0.5 * self.bob.mass * self.bob.velocity ** 2
        self.bob.pe = (self.bob.mass * GRAVITY * self.bob.length) * (1 - np.cos(self.bob.theta))
        self.bob.energy = self.bob.ke + self.bob.pe


if __name__ == "__main__":
    bob_instance = Bob(length=1, mass=1, initial_angle=1)
    pendulum_instance = Pendulum(bob_instance, len_of_simulation=30)
    pendulum_instance.run_simulation()

