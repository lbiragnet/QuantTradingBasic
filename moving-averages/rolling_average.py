import sys

class AnalyticsCalculator:
    def __init__(self, nums, k):
        """
        Initialise the AnalyticsCalculator with the list of numbers and the window size

        Args
            | nums (list) - the list of numbers
            | k (int) - the window size
        """
        try:
            # Validate inputs
            if not isinstance(nums, list):
                raise TypeError("The input 'nums' must be a list.")
            if not all(isinstance(x, (int, float)) for x in nums):
                raise ValueError("All elements in 'nums' must be integers or floats.")
            if not isinstance(k, int):
                raise TypeError("The window size 'k' must be an integer")
            if k <= 0:
                raise ValueError("The window size 'k' must be greater than zero.")
            self.nums = nums
            self.k = k
        except Exception as e:
            print(f"Error: {e}")
            self.nums = []
            self.k = 0

    def rolling_average(self):
        """
        Calculate the rolling average of the numbers with the specified window size
        
        Returns:
            | result - a list of rolling averages
        """
        # Make sure nums is set
        if not self.nums:
            return []
        
        result = []
        window_sum = 0
        for i in range(len(self.nums)):
            # Add the current number to the sum of numbers in the window
            window_sum += self.nums[i]
            # If window limit is reached, calculate the rolling average
            if i >= self.k - 1:
                average = window_sum/self.k
                # Cast to int if possible
                if average % 1 == 0:
                    average = int(average)
                result.append(average)
                # Remove the element that is past the window
                window_sum -= self.nums[i - self.k + 1]

        return result
    
    def reset_calculator(self, nums, k):
        """
        Reset the calculator with a new list of numbers and a new window size

        Args:
            | nums (list) - the new list of numbers
            | k (int) - the new window size
        """
        self.__init__(nums, k)


def main():
    # Check if correct number of arguments is passed
    if len(sys.argv) != 3:
        print("Usage: python rolling_average.py '<list_of_numbers>' <window_size>")
        print("Make sure the list of numbers does not contain any spaces")
        sys.exit(1)

    try:
        # Parse arguments from sys.argv
        nums_str = sys.argv[1]
        k = int(sys.argv[2])
        # Convert string of numbers to a list 
        nums = [int(x.strip()) for x in nums_str.strip("[]").split(",")]
        calculator = AnalyticsCalculator(nums, k)
        # Calculate the rolling average
        result = calculator.rolling_average()
        print("Rolling averages: ", result)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    