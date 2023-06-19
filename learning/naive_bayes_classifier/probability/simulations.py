import numpy as np

__all__ = ['Dice', 'FairDice', 'LoadedDice']

class Dice:
    def __init__(self, sides=6):
        self.sides = sides

    def roll(self):
        """ Roll the dice once. This is an abstract method to be implemented in subclasses."""
        pass

    def roll_many(self, n, condition=None):
        """
        Roll the dice n times. If a condition is specified, perform a second roll when the condition is met.
        n (int): The number of rolls.
        condition (function, optional): A condition function that takes the result of a roll and returns a boolean.

        Returns:
        rolls (np.array): The array of sums of the rolls.
        """
        rolls = np.array([self.roll() for _ in range(n)])
        if condition is not None:
            second_rolls = np.where(condition(rolls), np.array([self.roll() for _ in range(n)]), 0)
            rolls += second_rolls
        return rolls

    def side_prob(self, n):
        """
        Compute the probabilities of each side showing up after n rolls.
        n (int): The number of rolls.

        Returns:
        side_probs (dict): A dictionary mapping each side to its probability.
        """
        rolls = self.roll_many(n)
        unique, counts = np.unique(rolls, return_counts=True)
        return dict(zip(unique, counts / n))

    def stats(self, n, condition=None):
        """
        Calculate statistics (mean, variance, and median) of the roll results.
        n (int): The number of rolls.
        condition (function, optional): A condition function that takes the result of a roll and returns a boolean.

        Returns:
        mean, var, median (tuple): A tuple of the mean, variance, and median of the roll results.
        """
        rolls = self.roll_many(n, condition)
        mean = np.mean(rolls)
        var = np.var(rolls)
        median = np.median(rolls)
        return mean, var, median

    def covariance(self, r1, r2):
        """
        Calculate the covariance of the joint distribution of two dice.
        r1 (int): The number of rolls of the first die.
        r2 (int): The number of rolls of the second die.

        Returns:
        cov (float): The covariance of the joint distribution.
        """
        rolls1 = self.roll_many(r1)
        rolls2 = self.roll_many(r2)
        return np.cov(rolls1, rolls2)[0][1]



class FairDice(Dice):
    def __init__(self, sides=6):
        """
        Initialize a fair dice with the number of sides.
        sides (int): The number of sides of the dice.
        """
        super().__init__(sides)
        self.probs = np.ones(sides) / sides

    def roll(self):
        """
        Roll the fair dice once and return the result.

        Returns:
        roll (int): The result of the roll.
        """
        return np.random.choice(np.arange(1, self.sides + 1), p=self.probs)


class LoadedDice(Dice):
    def __init__(self, sides=6, loaded_side=1, bias=0.5):
        """
        Initialize a loaded dice with the number of sides, the loaded side, and the bias.
        sides (int): The number of sides of the dice.
        loaded_side (int): The side that the dice is loaded towards.
        bias (float): The bias towards the loaded side.
        """
        super().__init__(sides)
        self.probs = np.ones(sides) * (1 - bias) / (sides - 1)
        self.probs[loaded_side - 1] = bias
        if not np.isclose(np.sum(self.probs), 1):
            raise ValueError("All probabilities must add up to 1")

    def roll(self):
        """
        Roll the loaded dice once and return the result.

        Returns:
        roll (int): The result of the roll.
        """
        return np.random.choice(np.arange(1, self.sides + 1), p=self.probs)


# if __name__ == '__main__':
#     fair_dice = FairDice(6)
#     loaded_dice = LoadedDice(6, loaded_side=3, bias=0.3)
    
#     print(fair_dice.roll_many(10))
#     print(loaded_dice.roll_many(10))
#     print(fair_dice.side_prob(10))
#     print(loaded_dice.side_prob(10))
#     print(fair_dice.stats(10))
#     print(loaded_dice.stats(10))
#     print(fair_dice.stats(10, lambda x: x % 2 == 0))
#     print(loaded_dice.stats(10, lambda x: x % 2 == 0))

#     print("Fair dice probabilities: ", fair_dice.side_prob(10000))
#     print("Loaded dice probabilities: ", loaded_dice.side_prob(10000))

#     print("Stats (mean, variance, median) for fair dice: ", fair_dice.stats(10000, condition=lambda x: x <= 3))
#     print("Stats (mean, variance, median) for loaded dice: ", loaded_dice.stats(10000, condition=lambda x: x <= 3))