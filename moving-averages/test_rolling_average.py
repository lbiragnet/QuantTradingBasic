from rolling_average import AnalyticsCalculator

def test_rolling_average_basic():
        calculator = AnalyticsCalculator([1, 2, 3, 4, 5], 3)
        assert calculator.rolling_average() == [2, 3, 4]
        calculator.reset_calculator([1, 2, 3, 4, 5], 1)
        assert calculator.rolling_average() == [1, 2, 3, 4, 5]
        calculator.reset_calculator([1, 2, 3, 4, 5], 5)
        assert calculator.rolling_average() == [3]
        calculator.reset_calculator([1, 2, 3], 3)
        assert calculator.rolling_average() == [2]

def test_rolling_average_empty():
        calculator = AnalyticsCalculator([], 3)
        assert calculator.rolling_average() == []
        calculator.reset_calculator([1, 2, 3, 4, 5], 0)
        assert calculator.rolling_average() == []

def test_rolling_average_bigwindow():
        calculator = AnalyticsCalculator([1, 2, 3, 4, 5], 6)
        assert calculator.rolling_average() == []

def run_tests():
        test_rolling_average_basic()
        test_rolling_average_empty()
        test_rolling_average_bigwindow()
        print("All tests passed.")

if __name__ == "__main__":
        run_tests()