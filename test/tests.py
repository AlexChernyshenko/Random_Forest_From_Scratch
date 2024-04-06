import re

from hstest import StageTest, CheckResult, dynamic_test, TestedProgram

# The source data I will test on
true_data = 0.839
pattern = r"^[0-1][.][0-9]{1,3}$"


class ForestTest(StageTest):

    @dynamic_test()
    def test1(self):
        t = TestedProgram()
        reply = t.start()

        if len(reply) == 0:
            return CheckResult.wrong("No output was printed!")

        match = re.match(pattern=pattern, string=reply)

        if not match:
            return CheckResult.wrong("The result should be a decimal number rounded to three decimal places!")

        reply = float(reply)
        tolerance = 0.1

        # Getting the student's results from the reply

        if tolerance:
            if not (abs((reply - true_data) / true_data) < tolerance):
                return CheckResult.wrong('Incorrect value.')

        return CheckResult.correct()


if __name__ == '__main__':
    ForestTest.run_tests()
