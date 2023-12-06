# Predicting NBA Draft Outcomes

B365 Final Project

By: Nandha Gopikannan, Richie Fleming, Dillon New

This project aims to predict whether a basketball player will be drafted to the NBA based on their statistics. The prediction is made using classifier models trained on historical data.

## Getting Started

These instructions will guide you on how to run the project on your local machine.

### Prerequisites

You need to have Python installed on your machine. You can install Python here: https://www.python.org/downloads/.

### Installation

1. **Clone the repository**:
   Open a terminal and run the following git command:

   ```
   git clone https://github.com/nandhag29/nbadraftprediction.git
   ```

2. **Navigate to the project directory**:

   ```
   cd nbadraftprediction
   ```

3. **Install the required libraries**:
   ```
   pip install pandas numpy sklearn
   ```
   If you're using Python 3, replace `pip` with `pip3`.

### Usage

To run the project, run the main.py file.

Here's an example of how to use the program with a specific player:

1. When prompted, enter the Sports Reference College Basketball URL for the player. For example:

Enter Sports Reference College Basketball URL: https://www.sports-reference.com/cbb/players/anthony-edwards-2.html

2. The program will scrape the player's data from the provided URL, use the machine learning model to predict whether the player will be drafted, and display the prediction.

If you'd like to run the prediction for another player, close and run the main.py file again, and enter a new Sports Reference link.
